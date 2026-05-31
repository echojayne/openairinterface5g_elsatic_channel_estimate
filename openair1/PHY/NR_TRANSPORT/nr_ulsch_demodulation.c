/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "PHY/defs_gNB.h"
#include "PHY/phy_extern.h"
#include "nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/defs_nr_common.h"
#include "PHY/nr_phy_common/inc/nr_phy_common.h"
#include "common/utils/nr/nr_common.h"
#include <openair1/PHY/TOOLS/phy_scope_interface.h>
#include "PHY/sse_intrin.h"
#include "T.h"
#include "T_messages_creator.h"
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#if T_TRACER
#include "SIMULATION/TOOLS/sim.h"
#endif

#if T_TRACER
static void copy_c16_data_to_slot_memory(c16_t *src, c16_t *dst_slot, int nb_re_pusch, int symbol)
{
  memcpy(&dst_slot[nb_re_pusch * symbol], src, nb_re_pusch * sizeof(c16_t));
}

static int16_t clip_true_channel_sample(double value)
{
  if (value > 32767.0)
    return 32767;
  if (value < -32768.0)
    return -32768;
  return (int16_t)lround(value);
}

static channel_desc_t *get_rfsim_uplink_channel_desc(void)
{
  static bool warned_missing_channel = false;
  static bool lookup_disabled = false;
  if (lookup_disabled)
    return NULL;
  channel_desc_t *desc = find_channel_desc_fromname("rfsimu_channel_ue0");
  if (desc == NULL)
    desc = find_channel_desc_fromname("rfsimu_channel_enB0");
  if (desc == NULL && !warned_missing_channel) {
    LOG_W(PHY, "RFsim channel descriptor not found; true-channel trace will be zero\n");
    warned_missing_channel = true;
    lookup_disabled = true;
  }
  return desc;
}

static void fill_rfsim_true_channel_slot(c16_t *dst_slot,
                                         int nb_re_pusch,
                                         const NR_DL_FRAME_PARMS *frame_parms,
                                         const nfapi_nr_pusch_pdu_t *rel15_ul)
{
  channel_desc_t *desc = get_rfsim_uplink_channel_desc();
  if (desc == NULL || desc->ch == NULL || desc->channel_length == 0 || desc->sampling_rate <= 0.0)
    return;

  const int nb_rx_ant = frame_parms->nb_antennas_rx;
  const int nb_layer = rel15_ul->nrOfLayers;
  if (desc->nb_rx < nb_rx_ant || desc->nb_tx < nb_layer)
    return;

  const double scs_hz = 15000.0 * (double)(1U << rel15_ul->subcarrier_spacing);
  const double sample_rate_hz = desc->sampling_rate * 1e6;
  const double path_loss_linear = pow(10.0, desc->path_loss_dB / 20.0);
  const double ce_unit_scale = 364.0 * path_loss_linear;
  const double two_pi = 6.28318530717958647692;
  const int start_sc = (rel15_ul->bwp_start + rel15_ul->rb_start) * NR_NB_SC_PER_RB;
  const int grid_sc = frame_parms->N_RB_UL * NR_NB_SC_PER_RB;
  const int start_symbol = rel15_ul->start_symbol_index;
  const int end_symbol = start_symbol + rel15_ul->nr_of_symbols;
  const int stream_stride = frame_parms->symbols_per_slot * nb_re_pusch;

  for (int aatx = 0; aatx < nb_layer; aatx++) {
    for (int aarx = 0; aarx < nb_rx_ant; aarx++) {
      const int stream = aarx + aatx * desc->nb_rx;
      const struct complexd *channel = desc->ch[stream];
      if (channel == NULL)
        continue;
      c16_t *dst_stream = &dst_slot[(aatx * nb_rx_ant + aarx) * stream_stride];
      for (int symbol = start_symbol; symbol < end_symbol && symbol < frame_parms->symbols_per_slot; symbol++) {
        c16_t *dst_symbol = &dst_stream[symbol * nb_re_pusch];
        for (int k = 0; k < nb_re_pusch; k++) {
          const double freq_hz = ((double)(start_sc + k) - (double)grid_sc / 2.0) * scs_hz;
          const double theta_per_sample = two_pi * freq_hz / sample_rate_hz;
          double h_re = 0.0;
          double h_im = 0.0;
          for (int tap = 0; tap < desc->channel_length; tap++) {
            const double theta = theta_per_sample * (double)tap;
            const double c = cos(theta);
            const double s = sin(theta);
            h_re += channel[tap].r * c + channel[tap].i * s;
            h_im += channel[tap].i * c - channel[tap].r * s;
          }
          dst_symbol[k].r = clip_true_channel_sample(ce_unit_scale * h_re);
          dst_symbol[k].i = clip_true_channel_sample(ce_unit_scale * h_im);
        }
      }
    }
  }
}
#endif

#define OAI_AMMSE_CE_MAGIC 0x414d4d53u
#define OAI_AMMSE_CE_VERSION 1u

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t header_bytes;
  uint32_t request_id;
  uint32_t frame;
  uint32_t slot;
  uint32_t rb_size;
  uint32_t nr_symbols;
  uint32_t start_symbol;
  uint32_t nb_re;
  uint32_t grid_elems;
  uint32_t nr_layers;
  uint32_t nb_rx_ant;
  uint32_t nvar;
  uint32_t width_ppm;
  uint32_t depth_ppm;
} oai_ammse_ce_request_header_t;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t header_bytes;
  uint32_t request_id;
  uint32_t status;
  uint32_t grid_elems;
  uint32_t model_us;
  uint32_t reserved;
} oai_ammse_ce_response_header_t;

typedef struct {
  bool init_done;
  bool enabled;
  bool warned_connect;
  bool warned_unsupported;
  bool warned_subnet_file;
  int fd;
  uint32_t request_id;
  uint32_t width_ppm;
  uint32_t depth_ppm;
  uint64_t last_subnet_check_us;
  uint64_t subnet_poll_us;
  time_t subnet_file_mtime;
  long subnet_file_mtime_nsec;
  char socket_path[sizeof(((struct sockaddr_un *)0)->sun_path)];
  char subnet_file_path[512];
  FILE *metrics_fp;
} oai_ammse_ce_state_t;

static oai_ammse_ce_state_t oai_ammse_ce_state = {.fd = -1};
static bool oai_ammse_ce_applied_for_current_pusch = false;

static uint32_t oai_ammse_ce_value_to_ppm(double value);
static double oai_ammse_ce_ppm_to_value(uint32_t value_ppm);
static bool oai_ammse_ce_parse_width_depth(const char *text, uint32_t *width_ppm, uint32_t *depth_ppm);
static void oai_ammse_ce_load_subnet_from_file(oai_ammse_ce_state_t *state);

#if T_TRACER
#define OAI_CE_NMSE_QUEUE_DEPTH 4
#define OAI_CE_NMSE_MAX_RE 3300
#define OAI_CE_NMSE_MAX_SYMBOLS 14
#define OAI_CE_NMSE_MAX_GRID_ELEMS (OAI_CE_NMSE_MAX_RE * OAI_CE_NMSE_MAX_SYMBOLS)
#define OAI_CE_NMSE_MAX_TAPS 2048

typedef struct {
  bool valid;
  uint32_t frame;
  uint8_t slot;
  int rb_size;
  int nr_symbols;
  int start_symbol;
  int nb_re;
  int grid_elems;
  int start_sc;
  int n_rb_ul;
  int subcarrier_spacing;
  int channel_length;
  double sample_rate_hz;
  double path_loss_linear;
  char method_label[64];
  bool has_oai_inter_grid;
  c16_t applied_grid[OAI_CE_NMSE_MAX_GRID_ELEMS];
  c16_t oai_inter_grid[OAI_CE_NMSE_MAX_GRID_ELEMS];
  c16_t raw_dmrs_grid[OAI_CE_NMSE_MAX_GRID_ELEMS];
  struct complexd taps[OAI_CE_NMSE_MAX_TAPS];
} oai_ce_nmse_sample_t;

typedef struct {
  bool init_done;
  bool enabled;
  bool allow_time_varying;
  bool warned_unsupported;
  bool warned_channel;
  bool warned_time_varying;
  bool warned_thread;
  unsigned int period;
  unsigned int seen;
  unsigned int dropped;
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int head;
  int tail;
  int count;
  FILE *csv_fp;
  oai_ce_nmse_sample_t queue[OAI_CE_NMSE_QUEUE_DEPTH];
} oai_ce_nmse_state_t;

static oai_ce_nmse_state_t oai_ce_nmse_state = {
    .period = 1,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER,
};
#endif

static uint64_t monotonic_time_us(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

#if T_TRACER
static bool env_flag_enabled(const char *name)
{
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return false;
  return strcmp(value, "0") != 0 && strcasecmp(value, "false") != 0 && strcasecmp(value, "no") != 0
         && strcasecmp(value, "off") != 0;
}

static double ce_nmse_db(double err_power, double ref_power)
{
  if (ref_power <= 0.0 || err_power < 0.0)
    return NAN;
  return 10.0 * log10(err_power / ref_power);
}

static void ce_nmse_true_sample(const oai_ce_nmse_sample_t *sample, int k, double *true_re, double *true_im)
{
  const double scs_hz = 15000.0 * (double)(1U << sample->subcarrier_spacing);
  const int grid_sc = sample->n_rb_ul * NR_NB_SC_PER_RB;
  const double freq_hz = ((double)(sample->start_sc + k) - (double)grid_sc / 2.0) * scs_hz;
  const double theta_per_sample = 6.28318530717958647692 * freq_hz / sample->sample_rate_hz;
  double h_re = 0.0;
  double h_im = 0.0;
  for (int tap = 0; tap < sample->channel_length; tap++) {
    const double theta = theta_per_sample * (double)tap;
    const double c = cos(theta);
    const double s = sin(theta);
    h_re += sample->taps[tap].r * c + sample->taps[tap].i * s;
    h_im += sample->taps[tap].i * c - sample->taps[tap].r * s;
  }
  const double ce_unit_scale = 364.0 * sample->path_loss_linear;
  *true_re = (double)clip_true_channel_sample(ce_unit_scale * h_re);
  *true_im = (double)clip_true_channel_sample(ce_unit_scale * h_im);
}

static void ce_nmse_compute(const oai_ce_nmse_sample_t *sample,
                            double *applied_nmse_db,
                            double *oai_inter_nmse_db,
                            double *raw_dmrs_nmse_db,
                            int *raw_support)
{
  double applied_err = 0.0;
  double applied_ref = 0.0;
  double oai_inter_err = 0.0;
  double oai_inter_ref = 0.0;
  double raw_err = 0.0;
  double raw_ref = 0.0;
  int raw_count = 0;
  for (int rel_symbol = 0; rel_symbol < sample->nr_symbols; rel_symbol++) {
    for (int k = 0; k < sample->nb_re; k++) {
      double true_re = 0.0;
      double true_im = 0.0;
      ce_nmse_true_sample(sample, k, &true_re, &true_im);
      const int idx = rel_symbol * sample->nb_re + k;
      const c16_t applied = sample->applied_grid[idx];
      const double applied_dr = (double)applied.r - true_re;
      const double applied_di = (double)applied.i - true_im;
      applied_err += applied_dr * applied_dr + applied_di * applied_di;
      applied_ref += true_re * true_re + true_im * true_im;

      if (sample->has_oai_inter_grid) {
        const c16_t oai_inter = sample->oai_inter_grid[idx];
        const double oai_inter_dr = (double)oai_inter.r - true_re;
        const double oai_inter_di = (double)oai_inter.i - true_im;
        oai_inter_err += oai_inter_dr * oai_inter_dr + oai_inter_di * oai_inter_di;
        oai_inter_ref += true_re * true_re + true_im * true_im;
      }

      const c16_t raw = sample->raw_dmrs_grid[idx];
      if (raw.r != 0 || raw.i != 0) {
        const double raw_dr = (double)raw.r - true_re;
        const double raw_di = (double)raw.i - true_im;
        raw_err += raw_dr * raw_dr + raw_di * raw_di;
        raw_ref += true_re * true_re + true_im * true_im;
        raw_count++;
      }
    }
  }
  *applied_nmse_db = ce_nmse_db(applied_err, applied_ref);
  *oai_inter_nmse_db = sample->has_oai_inter_grid ? ce_nmse_db(oai_inter_err, oai_inter_ref) : NAN;
  *raw_dmrs_nmse_db = ce_nmse_db(raw_err, raw_ref);
  *raw_support = raw_count;
}

static void *ce_nmse_thread_main(void *arg)
{
  oai_ce_nmse_state_t *state = (oai_ce_nmse_state_t *)arg;
  while (true) {
    oai_ce_nmse_sample_t sample;
    memset(&sample, 0, sizeof(sample));
    pthread_mutex_lock(&state->mutex);
    while (state->count == 0)
      pthread_cond_wait(&state->cond, &state->mutex);
    sample = state->queue[state->tail];
    state->tail = (state->tail + 1) % OAI_CE_NMSE_QUEUE_DEPTH;
    state->count--;
    const unsigned int dropped = __atomic_load_n(&state->dropped, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&state->mutex);

    double applied_db = NAN;
    double oai_inter_db = NAN;
    double raw_db = NAN;
    int raw_support = 0;
    ce_nmse_compute(&sample, &applied_db, &oai_inter_db, &raw_db, &raw_support);
    const bool is_ammse = strncmp(sample.method_label, "ammse", strlen("ammse")) == 0;
    LOG_I(PHY,
          "CE_NMSE frame=%u slot=%u method=%s %s=%.3f dB oai_inter_all_re=%.3f dB raw_dmrs=%.3f dB raw_re=%d "
          "grid_re=%d dropped=%u\n",
          sample.frame,
          sample.slot,
          sample.method_label,
          is_ammse ? "ammse_all_re" : "oai_inter_all_re",
          applied_db,
          oai_inter_db,
          raw_db,
          raw_support,
          sample.grid_elems,
          dropped);
    if (state->csv_fp != NULL) {
      fprintf(state->csv_fp,
              "%u,%u,%s,%.6f,%.6f,%.6f,%d,%d,%u\n",
              sample.frame,
              sample.slot,
              sample.method_label,
              applied_db,
              oai_inter_db,
              raw_db,
              raw_support,
              sample.grid_elems,
              dropped);
      fflush(state->csv_fp);
    }
  }
  return NULL;
}

static void oai_ce_nmse_init(void)
{
  oai_ce_nmse_state_t *state = &oai_ce_nmse_state;
  if (state->init_done)
    return;
  state->init_done = true;
  state->enabled = env_flag_enabled("OAI_CE_NMSE_ENABLE") || env_flag_enabled("OAI_CE_NMSE");
  if (!state->enabled)
    return;
  state->allow_time_varying = env_flag_enabled("OAI_CE_NMSE_ALLOW_TIME_VARYING");

  const char *period = getenv("OAI_CE_NMSE_PERIOD");
  if (period != NULL && period[0] != '\0') {
    unsigned long parsed = strtoul(period, NULL, 10);
    state->period = parsed == 0 ? 1 : (unsigned int)parsed;
  }

  const char *csv_path = getenv("OAI_CE_NMSE_CSV");
  if (csv_path != NULL && csv_path[0] != '\0') {
    state->csv_fp = fopen(csv_path, "w");
    if (state->csv_fp != NULL) {
      fprintf(state->csv_fp,
              "frame,slot,method,applied_all_re_nmse_db,oai_inter_all_re_nmse_db,raw_dmrs_nmse_db,raw_dmrs_re,grid_re,"
              "dropped\n");
      fflush(state->csv_fp);
    } else {
      LOG_W(PHY, "failed to open OAI_CE_NMSE_CSV=%s: %s\n", csv_path, strerror(errno));
    }
  }

  int ret = pthread_create(&state->thread, NULL, ce_nmse_thread_main, state);
  if (ret != 0) {
    state->enabled = false;
    if (!state->warned_thread) {
      LOG_W(PHY, "failed to start CE NMSE logger thread: %s\n", strerror(ret));
      state->warned_thread = true;
    }
    return;
  }
  pthread_detach(state->thread);
  LOG_I(PHY, "online CE NMSE logger enabled, period=%u PUSCH event(s)\n", state->period);
}

bool oai_ce_nmse_enabled(void)
{
  oai_ce_nmse_init();
  return oai_ce_nmse_state.enabled;
}

static void oai_ce_nmse_maybe_enqueue(PHY_VARS_gNB *gNB,
                                      NR_gNB_PUSCH *pusch_vars,
                                      uint32_t frame,
                                      uint8_t slot,
                                      const nfapi_nr_pusch_pdu_t *rel15_ul,
                                      const c16_t *pusch_ch_est_dmrs_pos_slot_mem,
                                      const c16_t *oai_inter_grid)
{
  if (!oai_ce_nmse_enabled())
    return;

  oai_ce_nmse_state_t *state = &oai_ce_nmse_state;
  state->seen++;
  if (((state->seen - 1) % state->period) != 0)
    return;

  NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;
  const int nb_rx_ant = frame_parms->nb_antennas_rx;
  const int nb_layer = rel15_ul->nrOfLayers;
  const int nb_re = rel15_ul->rb_size * NR_NB_SC_PER_RB;
  const int grid_elems = nb_re * rel15_ul->nr_of_symbols;
  if (nb_rx_ant != 1 || nb_layer != 1 || rel15_ul->nr_of_symbols > OAI_CE_NMSE_MAX_SYMBOLS || nb_re > OAI_CE_NMSE_MAX_RE
      || grid_elems > OAI_CE_NMSE_MAX_GRID_ELEMS) {
    if (!state->warned_unsupported) {
      LOG_W(PHY,
            "CE NMSE logger supports 1 layer, 1 RX, <=%d RE/symbol, <=%d symbols; got layers=%d rx=%d re=%d symbols=%d\n",
            OAI_CE_NMSE_MAX_RE,
            OAI_CE_NMSE_MAX_SYMBOLS,
            nb_layer,
            nb_rx_ant,
            nb_re,
            rel15_ul->nr_of_symbols);
      state->warned_unsupported = true;
    }
    return;
  }

  channel_desc_t *desc = get_rfsim_uplink_channel_desc();
  if (desc == NULL || desc->ch == NULL || desc->channel_length <= 0 || desc->sampling_rate <= 0.0 || desc->nb_rx < 1
      || desc->nb_tx < 1 || desc->ch[0] == NULL) {
    if (!state->warned_channel) {
      LOG_W(PHY, "CE NMSE logger has no RFsim true channel; strict NMSE cannot be computed\n");
      state->warned_channel = true;
    }
    return;
  }
  if (desc->channel_length > OAI_CE_NMSE_MAX_TAPS) {
    if (!state->warned_channel) {
      LOG_W(PHY,
            "CE NMSE logger channel_length=%d exceeds max=%d; strict NMSE disabled for this run\n",
            desc->channel_length,
            OAI_CE_NMSE_MAX_TAPS);
      state->warned_channel = true;
    }
    return;
  }
  if (!state->allow_time_varying && desc->max_Doppler > 0.0 && desc->forgetting_factor < 1.0) {
    if (!state->warned_time_varying) {
      LOG_W(PHY,
            "CE NMSE logger skipped for time-varying RFsim channel %s: max_Doppler=%.3f Hz, forgetfact=%.6f. "
            "Strict NMSE needs a static RFsim channel snapshot; set OAI_CE_NMSE_ALLOW_TIME_VARYING=1 only for approximate "
            "debug values.\n",
            desc->model_name != NULL ? desc->model_name : "(unnamed)",
            desc->max_Doppler,
            desc->forgetting_factor);
      state->warned_time_varying = true;
    }
    return;
  }

  if (pthread_mutex_trylock(&state->mutex) != 0) {
    __sync_fetch_and_add(&state->dropped, 1);
    return;
  }
  if (state->count == OAI_CE_NMSE_QUEUE_DEPTH) {
    __sync_fetch_and_add(&state->dropped, 1);
    pthread_mutex_unlock(&state->mutex);
    return;
  }

  oai_ce_nmse_sample_t *sample = &state->queue[state->head];
  memset(sample, 0, sizeof(*sample));
  sample->valid = true;
  sample->frame = frame;
  sample->slot = slot;
  sample->rb_size = rel15_ul->rb_size;
  sample->nr_symbols = rel15_ul->nr_of_symbols;
  sample->start_symbol = rel15_ul->start_symbol_index;
  sample->nb_re = nb_re;
  sample->grid_elems = grid_elems;
  sample->start_sc = (rel15_ul->bwp_start + rel15_ul->rb_start) * NR_NB_SC_PER_RB;
  sample->n_rb_ul = frame_parms->N_RB_UL;
  sample->subcarrier_spacing = rel15_ul->subcarrier_spacing;
  sample->channel_length = desc->channel_length;
  sample->sample_rate_hz = desc->sampling_rate * 1e6;
  sample->path_loss_linear = pow(10.0, desc->path_loss_dB / 20.0);

  const char *label = getenv("OAI_CE_NMSE_LABEL");
  if ((label == NULL || label[0] == '\0') && oai_ammse_ce_applied_for_current_pusch)
    label = getenv("OAI_AMMSE_CE_LABEL");
  if (label == NULL || label[0] == '\0') {
    if (oai_ammse_ce_applied_for_current_pusch) {
      snprintf(sample->method_label,
               sizeof(sample->method_label),
               "ammse_w%.6g_d%.6g",
               oai_ammse_ce_ppm_to_value(oai_ammse_ce_state.width_ppm),
               oai_ammse_ce_ppm_to_value(oai_ammse_ce_state.depth_ppm));
    } else {
      snprintf(sample->method_label, sizeof(sample->method_label), "%s", "oai_interpolated");
    }
  } else {
    snprintf(sample->method_label, sizeof(sample->method_label), "%s", label);
  }

  const c16_t *raw_src = &pusch_ch_est_dmrs_pos_slot_mem[rel15_ul->start_symbol_index * nb_re];
  memcpy(sample->raw_dmrs_grid, raw_src, sizeof(c16_t) * grid_elems);
  if (oai_inter_grid != NULL) {
    sample->has_oai_inter_grid = true;
    memcpy(sample->oai_inter_grid, oai_inter_grid, sizeof(c16_t) * grid_elems);
  }

  const c16_t *applied_stream = (const c16_t *)pusch_vars->ul_ch_estimates[0];
  for (int rel_symbol = 0; rel_symbol < rel15_ul->nr_of_symbols; rel_symbol++) {
    const int symbol = rel15_ul->start_symbol_index + rel_symbol;
    const c16_t *src_symbol = &applied_stream[symbol * frame_parms->ofdm_symbol_size];
    c16_t *dst_symbol = &sample->applied_grid[rel_symbol * nb_re];
    memcpy(dst_symbol, src_symbol, sizeof(c16_t) * nb_re);
  }
  memcpy(sample->taps, desc->ch[0], sizeof(struct complexd) * desc->channel_length);

  state->head = (state->head + 1) % OAI_CE_NMSE_QUEUE_DEPTH;
  state->count++;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->mutex);
}
#else
bool oai_ce_nmse_enabled(void)
{
  return false;
}
#endif

static uint32_t oai_ammse_ce_value_to_ppm(double value)
{
  if (!isfinite(value) || value <= 0.0)
    return 0;
  if (value > 4.0)
    value = 4.0;
  return (uint32_t)llround(value * 1000000.0);
}

static double oai_ammse_ce_ppm_to_value(uint32_t value_ppm)
{
  return (double)value_ppm / 1000000.0;
}

static bool oai_ammse_ce_parse_width_depth(const char *text, uint32_t *width_ppm, uint32_t *depth_ppm)
{
  double width = 0.0;
  double depth = 0.0;
  const char *width_pos = strstr(text, "width=");
  const char *depth_pos = strstr(text, "depth=");
  if (width_pos != NULL && depth_pos != NULL) {
    width = strtod(width_pos + strlen("width="), NULL);
    depth = strtod(depth_pos + strlen("depth="), NULL);
  } else if (sscanf(text, "w=%lf d=%lf", &width, &depth) != 2 && sscanf(text, "%lf %lf", &width, &depth) != 2) {
    return false;
  }
  const uint32_t parsed_width = oai_ammse_ce_value_to_ppm(width);
  const uint32_t parsed_depth = oai_ammse_ce_value_to_ppm(depth);
  if (parsed_width == 0 || parsed_depth == 0)
    return false;
  *width_ppm = parsed_width;
  *depth_ppm = parsed_depth;
  return true;
}

static void oai_ammse_ce_load_subnet_from_file(oai_ammse_ce_state_t *state)
{
  if (state->subnet_file_path[0] == '\0')
    return;
  const uint64_t now_us = monotonic_time_us();
  if (state->last_subnet_check_us != 0 && now_us - state->last_subnet_check_us < state->subnet_poll_us)
    return;
  state->last_subnet_check_us = now_us;

  struct stat st;
  if (stat(state->subnet_file_path, &st) != 0)
    return;
  if (state->subnet_file_mtime == st.st_mtime && state->subnet_file_mtime_nsec == st.st_mtim.tv_nsec)
    return;

  FILE *fp = fopen(state->subnet_file_path, "r");
  if (fp == NULL) {
    if (!state->warned_subnet_file) {
      LOG_W(PHY, "failed to open OAI_AMMSE_CE_SUBNET_FILE=%s: %s\n", state->subnet_file_path, strerror(errno));
      state->warned_subnet_file = true;
    }
    return;
  }
  char line[128] = {0};
  char *got = fgets(line, sizeof(line), fp);
  fclose(fp);
  if (got == NULL)
    return;

  uint32_t width_ppm = 0;
  uint32_t depth_ppm = 0;
  if (!oai_ammse_ce_parse_width_depth(line, &width_ppm, &depth_ppm)) {
    if (!state->warned_subnet_file) {
      LOG_W(PHY,
            "failed to parse subnet file %s, expected 'width=0.25 depth=0.5' or '0.25 0.5'\n",
            state->subnet_file_path);
      state->warned_subnet_file = true;
    }
    return;
  }
  state->width_ppm = width_ppm;
  state->depth_ppm = depth_ppm;
  state->subnet_file_mtime = st.st_mtime;
  state->subnet_file_mtime_nsec = st.st_mtim.tv_nsec;
  state->warned_subnet_file = false;
  LOG_I(PHY,
        "A-MMSE CE subnet updated: width=%.6g depth=%.6g\n",
        oai_ammse_ce_ppm_to_value(state->width_ppm),
        oai_ammse_ce_ppm_to_value(state->depth_ppm));
}

static void oai_ammse_ce_init(void)
{
  oai_ammse_ce_state_t *state = &oai_ammse_ce_state;
  if (state->init_done)
    return;
  state->init_done = true;
  state->fd = -1;
  const char *socket_path = getenv("OAI_AMMSE_CE_SOCKET");
  if (socket_path == NULL || socket_path[0] == '\0')
    return;
  if (strlen(socket_path) >= sizeof(state->socket_path)) {
    LOG_E(PHY, "OAI_AMMSE_CE_SOCKET path too long, disabling A-MMSE CE hook\n");
    return;
  }
  snprintf(state->socket_path, sizeof(state->socket_path), "%s", socket_path);
  state->enabled = true;
  const char *width_env = getenv("OAI_AMMSE_CE_WIDTH");
  const char *depth_env = getenv("OAI_AMMSE_CE_DEPTH");
  state->width_ppm = oai_ammse_ce_value_to_ppm(width_env != NULL && width_env[0] != '\0' ? strtod(width_env, NULL) : 1.0);
  state->depth_ppm = oai_ammse_ce_value_to_ppm(depth_env != NULL && depth_env[0] != '\0' ? strtod(depth_env, NULL) : 1.0);
  if (state->width_ppm == 0)
    state->width_ppm = 1000000;
  if (state->depth_ppm == 0)
    state->depth_ppm = 1000000;
  state->subnet_poll_us = 100000;
  const char *poll_env = getenv("OAI_AMMSE_CE_SUBNET_POLL_US");
  if (poll_env != NULL && poll_env[0] != '\0') {
    uint64_t parsed_poll = strtoull(poll_env, NULL, 10);
    if (parsed_poll > 0)
      state->subnet_poll_us = parsed_poll;
  }
  const char *subnet_file = getenv("OAI_AMMSE_CE_SUBNET_FILE");
  if (subnet_file != NULL && subnet_file[0] != '\0') {
    if (strlen(subnet_file) >= sizeof(state->subnet_file_path)) {
      LOG_W(PHY, "OAI_AMMSE_CE_SUBNET_FILE path too long, ignoring dynamic subnet file\n");
    } else {
      snprintf(state->subnet_file_path, sizeof(state->subnet_file_path), "%s", subnet_file);
      oai_ammse_ce_load_subnet_from_file(state);
    }
  }
  const char *metrics_path = getenv("OAI_AMMSE_CE_METRICS");
  if (metrics_path != NULL && metrics_path[0] != '\0') {
    state->metrics_fp = fopen(metrics_path, "w");
    if (state->metrics_fp != NULL) {
      fprintf(state->metrics_fp,
              "request_id,frame,slot,rb_size,nr_symbols,grid_elems,width,depth,status,wall_us,model_us,max_ch\n");
      fflush(state->metrics_fp);
    } else {
      LOG_W(PHY, "failed to open OAI_AMMSE_CE_METRICS=%s: %s\n", metrics_path, strerror(errno));
    }
  }
  LOG_I(PHY,
        "A-MMSE CE hook enabled via Unix socket %s, width=%.6g depth=%.6g\n",
        state->socket_path,
        oai_ammse_ce_ppm_to_value(state->width_ppm),
        oai_ammse_ce_ppm_to_value(state->depth_ppm));
}

bool oai_ammse_ce_enabled(void)
{
  oai_ammse_ce_init();
  return oai_ammse_ce_state.enabled;
}

static bool oai_ammse_ce_connect(void)
{
  oai_ammse_ce_state_t *state = &oai_ammse_ce_state;
  if (!oai_ammse_ce_enabled())
    return false;
  if (state->fd >= 0)
    return true;
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0)
    return false;
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  size_t socket_path_len = strnlen(state->socket_path, sizeof(addr.sun_path) - 1);
  memcpy(addr.sun_path, state->socket_path, socket_path_len);
  if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    if (!state->warned_connect) {
      LOG_W(PHY, "failed to connect A-MMSE CE server at %s: %s\n", state->socket_path, strerror(errno));
      state->warned_connect = true;
    }
    close(fd);
    return false;
  }
  state->fd = fd;
  state->warned_connect = false;
  return true;
}

static bool send_all(int fd, const void *data, size_t bytes)
{
  const uint8_t *ptr = (const uint8_t *)data;
  while (bytes > 0) {
    ssize_t sent = send(fd, ptr, bytes, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (sent == 0)
      return false;
    ptr += sent;
    bytes -= (size_t)sent;
  }
  return true;
}

static bool recv_all(int fd, void *data, size_t bytes)
{
  uint8_t *ptr = (uint8_t *)data;
  while (bytes > 0) {
    ssize_t got = recv(fd, ptr, bytes, MSG_WAITALL);
    if (got < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (got == 0)
      return false;
    ptr += got;
    bytes -= (size_t)got;
  }
  return true;
}

static void oai_ammse_ce_close_socket(void)
{
  if (oai_ammse_ce_state.fd >= 0) {
    close(oai_ammse_ce_state.fd);
    oai_ammse_ce_state.fd = -1;
  }
}

static void apply_ammse_ce_grid(NR_gNB_PUSCH *pusch_vars,
                                const NR_DL_FRAME_PARMS *frame_parms,
                                const nfapi_nr_pusch_pdu_t *rel15_ul,
                                const c16_t *grid,
                                int *max_ch)
{
  const int nb_re = rel15_ul->rb_size * NR_NB_SC_PER_RB;
  c16_t *ul_ch = (c16_t *)pusch_vars->ul_ch_estimates[0];
  *max_ch = 0;
  for (int rel_symbol = 0; rel_symbol < rel15_ul->nr_of_symbols; rel_symbol++) {
    const int symbol = rel15_ul->start_symbol_index + rel_symbol;
    c16_t *dst_symbol = &ul_ch[symbol * frame_parms->ofdm_symbol_size];
    const c16_t *src_symbol = &grid[rel_symbol * nb_re];
    for (int k = 0; k < nb_re; k++) {
      dst_symbol[k] = src_symbol[k];
      int re_abs = abs(src_symbol[k].r);
      int im_abs = abs(src_symbol[k].i);
      if (re_abs > *max_ch)
        *max_ch = re_abs;
      if (im_abs > *max_ch)
        *max_ch = im_abs;
    }
  }
}

static bool oai_ammse_ce_try_replace(PHY_VARS_gNB *gNB,
                                     NR_gNB_PUSCH *pusch_vars,
                                     uint32_t frame,
                                     uint8_t slot,
                                     const nfapi_nr_pusch_pdu_t *rel15_ul,
                                     uint32_t nvar,
                                     c16_t *pusch_ch_est_dmrs_pos_slot_mem,
                                     int *max_ch)
{
  oai_ammse_ce_applied_for_current_pusch = false;
  if (!oai_ammse_ce_enabled())
    return false;
  NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;
  const int nb_rx_ant = frame_parms->nb_antennas_rx;
  const int nb_layer = rel15_ul->nrOfLayers;
  const int nb_re = rel15_ul->rb_size * NR_NB_SC_PER_RB;
  const int grid_elems = nb_re * rel15_ul->nr_of_symbols;
  if (nb_layer != 1 || nb_rx_ant != 1 || rel15_ul->rb_size != 50 || rel15_ul->nr_of_symbols != 13
      || rel15_ul->start_symbol_index != 0) {
    if (!oai_ammse_ce_state.warned_unsupported) {
      LOG_W(PHY,
            "A-MMSE CE hook supports only 1 layer, 1 RX, 50 RB, start_symbol=0, nr_symbols=13; got layers=%d rx=%d rb=%d "
            "start=%d symbols=%d. Falling back to OAI CE.\n",
            nb_layer,
            nb_rx_ant,
            rel15_ul->rb_size,
            rel15_ul->start_symbol_index,
            rel15_ul->nr_of_symbols);
      oai_ammse_ce_state.warned_unsupported = true;
    }
    return false;
  }
  if (!oai_ammse_ce_connect())
    return false;

  c16_t response_grid[grid_elems] __attribute__((aligned(64)));
  oai_ammse_ce_state_t *state = &oai_ammse_ce_state;
  oai_ammse_ce_load_subnet_from_file(state);
  oai_ammse_ce_request_header_t req = {
      .magic = OAI_AMMSE_CE_MAGIC,
      .version = OAI_AMMSE_CE_VERSION,
      .header_bytes = sizeof(oai_ammse_ce_request_header_t),
      .request_id = ++state->request_id,
      .frame = frame,
      .slot = slot,
      .rb_size = rel15_ul->rb_size,
      .nr_symbols = rel15_ul->nr_of_symbols,
      .start_symbol = rel15_ul->start_symbol_index,
      .nb_re = nb_re,
      .grid_elems = grid_elems,
      .nr_layers = nb_layer,
      .nb_rx_ant = nb_rx_ant,
      .nvar = nvar,
      .width_ppm = state->width_ppm,
      .depth_ppm = state->depth_ppm,
  };

  const uint64_t t0 = monotonic_time_us();
  const c16_t *raw_grid = &pusch_ch_est_dmrs_pos_slot_mem[rel15_ul->start_symbol_index * nb_re];
  bool ok = send_all(state->fd, &req, sizeof(req)) && send_all(state->fd, raw_grid, sizeof(c16_t) * grid_elems);
  oai_ammse_ce_response_header_t resp;
  memset(&resp, 0, sizeof(resp));
  if (ok)
    ok = recv_all(state->fd, &resp, sizeof(resp));
  if (ok) {
    ok = resp.magic == OAI_AMMSE_CE_MAGIC && resp.version == OAI_AMMSE_CE_VERSION && resp.request_id == req.request_id
         && resp.status == 0 && resp.grid_elems == (uint32_t)grid_elems;
  }
  if (ok)
    ok = recv_all(state->fd, response_grid, sizeof(c16_t) * grid_elems);
  const uint64_t t1 = monotonic_time_us();
  if (!ok) {
    LOG_W(PHY, "A-MMSE CE server request failed; falling back to OAI CE\n");
    oai_ammse_ce_close_socket();
    if (state->metrics_fp != NULL) {
      fprintf(state->metrics_fp,
              "%u,%u,%u,%u,%u,%u,%.6g,%.6g,%d,%llu,%u,%d\n",
              req.request_id,
              frame,
              slot,
              req.rb_size,
              req.nr_symbols,
              req.grid_elems,
              oai_ammse_ce_ppm_to_value(req.width_ppm),
              oai_ammse_ce_ppm_to_value(req.depth_ppm),
              -1,
              (unsigned long long)(t1 - t0),
              resp.model_us,
              *max_ch);
      fflush(state->metrics_fp);
    }
    return false;
  }

  apply_ammse_ce_grid(pusch_vars, frame_parms, rel15_ul, response_grid, max_ch);
  oai_ammse_ce_applied_for_current_pusch = true;
  if (state->metrics_fp != NULL) {
    fprintf(state->metrics_fp,
            "%u,%u,%u,%u,%u,%u,%.6g,%.6g,%u,%llu,%u,%d\n",
            req.request_id,
            frame,
            slot,
            req.rb_size,
            req.nr_symbols,
            req.grid_elems,
            oai_ammse_ce_ppm_to_value(req.width_ppm),
            oai_ammse_ce_ppm_to_value(req.depth_ppm),
            resp.status,
            (unsigned long long)(t1 - t0),
            resp.model_us,
            *max_ch);
    fflush(state->metrics_fp);
  }
  return true;
}

void nr_idft(int32_t *z, uint32_t Msc_PUSCH)
{

  simde__m128i idft_in128[1][3240], idft_out128[1][3240];
  simde__m128i norm128;
  int16_t *idft_in0 = (int16_t*)idft_in128[0], *idft_out0 = (int16_t*)idft_out128[0];

  int i, ip;

  LOG_T(PHY,"Doing nr_idft for Msc_PUSCH %d\n", Msc_PUSCH);

  if ((Msc_PUSCH % 1536) > 0) {
    // conjugate input
    for (i = 0; i < (Msc_PUSCH>>2); i++) {
      ((simde__m128i*)z)[i] = oai_mm_conj( ((simde__m128i*)z)[i] );
    }
    for (i = 0, ip = 0; i < Msc_PUSCH; i++, ip+=4)
      ((uint32_t*)idft_in0)[ip+0] = z[i];
  }
  dft_size_idx_t dftsize = get_dft(Msc_PUSCH);
  switch (Msc_PUSCH) {
    case 12:
      dft(dftsize, (int16_t *)idft_in0, (int16_t *)idft_out0, 0);

      norm128 = simde_mm_set1_epi16(9459);

      for (i = 0; i < 12; i++) {
        ((simde__m128i *)idft_out0)[i] = simde_mm_slli_epi16(simde_mm_mulhi_epi16(((simde__m128i *)idft_out0)[i], norm128), 1);
      }

      break;
    default:
      dft(dftsize, idft_in0, idft_out0, 1);
      break;
  }

  if ((Msc_PUSCH % 1536) > 0) {
    for (i = 0, ip = 0; i < Msc_PUSCH; i++, ip+=4)
      z[i] = ((uint32_t*)idft_out0)[ip];

    // conjugate output
    for (i = 0; i < (Msc_PUSCH>>2); i++) {
      ((simde__m128i*)z)[i] = oai_mm_conj(((simde__m128i*)z)[i]);
    }
  }
}

static void nr_ulsch_extract_rbs(c16_t* const rxdataF,
                                 c16_t* const chF,
                                 c16_t *rxFext,
                                 c16_t *chFext,
                                 int rxoffset,
                                 int choffset,
                                 int is_dmrs_symbol,
                                 const nfapi_nr_pusch_pdu_t *pusch_pdu,
                                 NR_DL_FRAME_PARMS *frame_parms)
{
  uint8_t delta = 0;
  int start_re = (frame_parms->first_carrier_offset + (pusch_pdu->rb_start + pusch_pdu->bwp_start) * NR_NB_SC_PER_RB)%frame_parms->ofdm_symbol_size;
  int nb_re_pusch = NR_NB_SC_PER_RB * pusch_pdu->rb_size;
  c16_t *rxF = &rxdataF[rxoffset];
  c16_t *rxF_ext = &rxFext[0];
  c16_t *ul_ch0 = &chF[choffset];
  c16_t *ul_ch0_ext = &chFext[0];

  if (is_dmrs_symbol == 0) {
    if (start_re + nb_re_pusch <= frame_parms->ofdm_symbol_size)
      memcpy(rxF_ext, &rxF[start_re], nb_re_pusch * sizeof(c16_t));
    else {
      int neg_length = frame_parms->ofdm_symbol_size - start_re;
      int pos_length = nb_re_pusch - neg_length;
      memcpy(rxF_ext, &rxF[start_re], neg_length * sizeof(c16_t));
      memcpy(&rxF_ext[neg_length], rxF, pos_length * sizeof(c16_t));
    }
    memcpy(ul_ch0_ext, ul_ch0, nb_re_pusch * sizeof(c16_t));
  }
  else if (pusch_pdu->dmrs_config_type == pusch_dmrs_type1) { // 6 REs / PRB
    AssertFatal(delta == 0 || delta == 1, "Illegal delta %d\n",delta);
    c16_t *rxF32 = &rxF[start_re];
    if (start_re + nb_re_pusch < frame_parms->ofdm_symbol_size) {
      for (int idx = 1 - delta; idx < nb_re_pusch; idx += 2) {
        *rxF_ext++ = rxF32[idx];
        *ul_ch0_ext++ = ul_ch0[idx];
      }
    }
    else { // handle the two pieces around DC
      int neg_length = frame_parms->ofdm_symbol_size - start_re;
      int pos_length = nb_re_pusch - neg_length;
      int idx, idx2;
      for (idx = 1 - delta; idx < neg_length; idx += 2) {
        *rxF_ext++ = rxF32[idx];
        *ul_ch0_ext++= ul_ch0[idx];
      }
      rxF32 = rxF;
      idx2 = idx;
      for (idx = 1 - delta; idx < pos_length; idx += 2, idx2 += 2) {
        *rxF_ext++ = rxF32[idx];
        *ul_ch0_ext++ = ul_ch0[idx2];
      }
    }
  }
  else if (pusch_pdu->dmrs_config_type == pusch_dmrs_type2) { // 8 REs / PRB
    AssertFatal(delta==0||delta==2||delta==4,"Illegal delta %d\n",delta);
    if (start_re + nb_re_pusch < frame_parms->ofdm_symbol_size) {
      for (int idx = 0; idx < nb_re_pusch; idx ++) {
        if (idx % 6 == 2 * delta || idx % 6 == 2 * delta + 1)
          continue;
        *rxF_ext++ = rxF[idx];
        *ul_ch0_ext++ = ul_ch0[idx];
      }
    }
    else {
      int neg_length = frame_parms->ofdm_symbol_size - start_re;
      int pos_length = nb_re_pusch - neg_length;
      c16_t *rxF64 = &rxF[start_re];
      int idx, idx2;
      for (idx = 0; idx < neg_length; idx ++) {
        if (idx % 6 == 2 * delta || idx % 6 == 2 * delta + 1)
          continue;
        *rxF_ext++ = rxF64[idx];
        *ul_ch0_ext++ = ul_ch0[idx];
      }
      rxF64 = rxF;
      idx2 = idx;
      for (idx = 0; idx < pos_length; idx++, idx2++) {
        if (idx % 6 == 2 * delta || idx % 6 == 2 * delta + 1)
          continue;
        *rxF_ext++ = rxF64[idx];
        *ul_ch0_ext++ = ul_ch0[idx2];
      }
    }
  }
}

static int get_nb_re_pusch (NR_DL_FRAME_PARMS *frame_parms, const nfapi_nr_pusch_pdu_t *rel15_ul, int symbol)
{
  uint8_t dmrs_symbol_flag = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01;
  if (dmrs_symbol_flag == 1) {
    if (rel15_ul->dmrs_config_type == 0) {
      // if no data in dmrs cdm group is 1 only even REs have no data
      // if no data in dmrs cdm group is 2 both odd and even REs have no data
      return(rel15_ul->rb_size *(12 - (rel15_ul->num_dmrs_cdm_grps_no_data*6)));
    }
    else return(rel15_ul->rb_size *(12 - (rel15_ul->num_dmrs_cdm_grps_no_data*4)));
  } else
    return (rel15_ul->rb_size * NR_NB_SC_PER_RB);
}

static void nr_ulsch_channel_compensation(uint32_t buffer_length,
                                          int nb_rx_ant,
                                          c16_t rxFext[][buffer_length],
                                          c16_t chFext[][nb_rx_ant][buffer_length],
                                          c16_t ul_ch_maga[][buffer_length],
                                          c16_t ul_ch_magb[][buffer_length],
                                          c16_t ul_ch_magc[][buffer_length],
                                          c16_t **rxComp,
                                          int nb_layers,
                                          c16_t rho[][nb_layers][buffer_length],
                                          const nfapi_nr_pusch_pdu_t *rel15_ul,
                                          uint32_t symbol,
                                          uint32_t output_shift)
{
  int mod_order  = rel15_ul->qam_mod_order;
  int nrOfLayers = rel15_ul->nrOfLayers;

  simde__m256i QAM_ampa_256 = simde_mm256_setzero_si256();
  simde__m256i QAM_ampb_256 = simde_mm256_setzero_si256();
  simde__m256i QAM_ampc_256 = simde_mm256_setzero_si256();

  if (mod_order == 4) {
    QAM_ampa_256 = simde_mm256_set1_epi16(QAM16_n1);
    QAM_ampb_256 = simde_mm256_setzero_si256();
    QAM_ampc_256 = simde_mm256_setzero_si256();
  }
  else if (mod_order == 6) {
    QAM_ampa_256 = simde_mm256_set1_epi16(QAM64_n1);
    QAM_ampb_256 = simde_mm256_set1_epi16(QAM64_n2);
    QAM_ampc_256 = simde_mm256_setzero_si256();
  }
  else if (mod_order == 8) {
    QAM_ampa_256 = simde_mm256_set1_epi16(QAM256_n1);
    QAM_ampb_256 = simde_mm256_set1_epi16(QAM256_n2);
    QAM_ampc_256 = simde_mm256_set1_epi16(QAM256_n3);
  }

  for (int aatx = 0; aatx < nrOfLayers; aatx++) {
    simde__m256i *rxComp_256 = (simde__m256i *)&rxComp[aatx * nb_rx_ant][symbol * buffer_length];
    simde__m256i *rxF_ch_maga_256 = (simde__m256i *)ul_ch_maga[aatx];
    simde__m256i *rxF_ch_magb_256 = (simde__m256i *)ul_ch_magb[aatx];
    simde__m256i *rxF_ch_magc_256 = (simde__m256i *)ul_ch_magc[aatx];
    for (int aarx = 0; aarx < nb_rx_ant; aarx++) {
      simde__m256i *rxF_256 = (simde__m256i *)rxFext[aarx];
      simde__m256i *chF_256 = (simde__m256i *)chFext[aatx][aarx];

      for (int i = 0; i < buffer_length >> 3; i++) 
      {
        // MRC        
        simde__m256i comp = oai_mm256_cpx_mult_conj(chF_256[i], rxF_256[i], output_shift);
        rxComp_256[i] = simde_mm256_add_epi16(rxComp_256[i], comp); 

        if (mod_order > 2) {
          simde__m256i mag = oai_mm256_smadd(chF_256[i], chF_256[i], output_shift); // |h|^2
          // pack and duplicate
          mag = simde_mm256_packs_epi32(mag, mag);
          mag = simde_mm256_unpacklo_epi16(mag, mag);

          rxF_ch_maga_256[i] = simde_mm256_add_epi16(rxF_ch_maga_256[i], simde_mm256_mulhrs_epi16(mag, QAM_ampa_256));

          if (mod_order > 4)
            rxF_ch_magb_256[i] = simde_mm256_add_epi16(rxF_ch_magb_256[i], simde_mm256_mulhrs_epi16(mag, QAM_ampb_256));

          if (mod_order > 6)
            rxF_ch_magc_256[i] = simde_mm256_add_epi16(rxF_ch_magc_256[i], simde_mm256_mulhrs_epi16(mag, QAM_ampc_256));
        }        
      }
      if (nb_layers > 1) {
        for (int atx = 0; atx < nrOfLayers; atx++) {
          simde__m256i *rho_256 = (simde__m256i *)rho[aatx][atx];
          simde__m256i *chF_256 = (simde__m256i *)chFext[aatx][aarx];
          simde__m256i *chF2_256 = (simde__m256i *)chFext[atx][aarx];
          for (int i = 0; i < buffer_length >> 3; i++) {
            rho_256[i] = simde_mm256_adds_epi16(rho_256[i], oai_mm256_cpx_mult_conj(chF_256[i], chF2_256[i], output_shift));
          }
        }
      }
    }
  }

}

// Zero Forcing Rx function: nr_det_HhH()
static void nr_ulsch_det_HhH(c16_t *after_mf_00, // a
                             c16_t *after_mf_01, // b
                             c16_t *after_mf_10, // c
                             c16_t *after_mf_11, // d
                             uint32_t *det_fin, // 1/ad-bc
                             unsigned short nb_rb)
{
  simde__m128i *after_mf_00_128,*after_mf_01_128, *after_mf_10_128, *after_mf_11_128, ad_re_128, bc_re_128; //ad_im_128, bc_im_128;
  simde__m128i *det_fin_128, det_re_128; //det_im_128, tmp_det0, tmp_det1;

  after_mf_00_128 = (simde__m128i *)after_mf_00;
  after_mf_01_128 = (simde__m128i *)after_mf_01;
  after_mf_10_128 = (simde__m128i *)after_mf_10;
  after_mf_11_128 = (simde__m128i *)after_mf_11;

  det_fin_128 = (simde__m128i *)det_fin;

  for (unsigned short rb=0; rb<3*nb_rb; rb++) {

    //complex multiplication (I_a+jQ_a)(I_d+jQ_d) = (I_aI_d - Q_aQ_d) + j(Q_aI_d + I_aQ_d)
    //The imag part is often zero, we compute only the real part
    ad_re_128 = simde_mm_madd_epi16(oai_mm_conj(after_mf_00_128[0]),after_mf_11_128[0]); //Re: I_a0*I_d0 - Q_a1*Q_d1
    //ad_im_128 = simde_mm_madd_epi16(oai_mm_swap(after_mf_00_128[0]),after_mf_11_128[0]);//Im: (Q_aI_d + I_aQ_d)

    //complex multiplication (I_b+jQ_b)(I_c+jQ_c) = (I_bI_c - Q_bQ_c) + j(Q_bI_c + I_bQ_c)
    //The imag part is often zero, we compute only the real part
    bc_re_128 = simde_mm_madd_epi16(oai_mm_conj(after_mf_01_128[0]),after_mf_10_128[0]); //Re: I_b0*I_c0 - Q_b1*Q_c1
    //bc_im_128 = simde_mm_madd_epi16(oai_mm_swap(after_mf_01_128[0]),after_mf_10_128[0]);//Im: (Q_bI_c + I_bQ_c)

    det_re_128 = simde_mm_sub_epi32(ad_re_128, bc_re_128);
    //det_im_128 = simde_mm_sub_epi32(ad_im_128, bc_im_128);

    //det in Q30 format
    det_fin_128[0] = simde_mm_abs_epi32(det_re_128);


#ifdef DEBUG_DLSCH_DEMOD
     printf("\n Computing det_HhH_inv \n");
     //print_ints("det_re_128:",(int32_t*)&det_re_128);
     //print_ints("det_im_128:",(int32_t*)&det_im_128);
     print_ints("det_fin_128:",(int32_t*)&det_fin_128[0]);
#endif
    det_fin_128+=1;
    after_mf_00_128+=1;
    after_mf_01_128+=1;
    after_mf_10_128+=1;
    after_mf_11_128+=1;
  }
}

/* Zero Forcing Rx function: nr_conjch0_mult_ch1()
 *
 *
 * */
// TODO: This function is just a wrapper, can be removed.
static void nr_ulsch_conjch0_mult_ch1(c16_t *ch0, c16_t *ch1, c16_t *ch0conj_ch1, unsigned short nb_rb, unsigned char output_shift0)
{
  //This function is used to compute multiplications in H_hermitian * H matrix
  mult_cpx_conj_vector(ch0, ch1, ch0conj_ch1, 12 * nb_rb, output_shift0);
}

static simde__m128i nr_ulsch_comp_muli_sum(simde__m128i input_x,
                                           simde__m128i input_y,
                                           simde__m128i input_w,
                                           simde__m128i input_z,
                                           simde__m128i det)
{

  // complex multiplication (x_re + jx_im)*(y_re + jy_im) = (x_re*y_re - x_im*y_im) + j(x_im*y_re + x_re*y_im)
  // complex multiplication (w_re + jw_im)*(z_re + jz_im) = (w_re*z_re - w_im*z_im) + j(w_im*z_re + w_re*z_im)
  // the real part
  simde__m128i xy_re_128 = simde_mm_madd_epi16(oai_mm_conj(input_x), input_y); //Re: (x_re*y_re - x_im*y_im)
  simde__m128i wz_re_128 = simde_mm_madd_epi16(oai_mm_conj(input_w), input_z); //Re: (w_re*z_re - w_im*z_im)
  xy_re_128 = simde_mm_sub_epi32(xy_re_128, wz_re_128);

  // the imag part
  simde__m128i xy_im_128 = simde_mm_madd_epi16(oai_mm_swap(input_x), input_y); //Im: (x_im*y_re + x_re*y_im)
  simde__m128i wz_im_128 = simde_mm_madd_epi16(oai_mm_swap(input_w), input_z); //Im: (w_im*z_re + w_re*z_im)
  xy_im_128 = simde_mm_sub_epi32(xy_im_128, wz_im_128);

  //print_ints("rx_re:",(int32_t*)&xy_re_128[0]);
  //print_ints("rx_Img:",(int32_t*)&xy_im_128[0]);
  //divide by matrix det and convert back to Q15 before packing
  uint64_t sum_det = 0;
  for (int k = 0; k < 4; k++) {
    sum_det += (((uint32_t *)&det)[k]);
  }
  // Add bias to reduce rounding error
  sum_det = (sum_det + 2) >> 2;

  int b = log2_approx(sum_det) - 8;
  if (b > 0) {
    xy_re_128 = simde_mm_srai_epi32(xy_re_128, b);
    xy_im_128 = simde_mm_srai_epi32(xy_im_128, b);
  } else {
    xy_re_128 = simde_mm_slli_epi32(xy_re_128, -b);
    xy_im_128 = simde_mm_slli_epi32(xy_im_128, -b);
  }

  simde__m128i output = oai_mm_pack(xy_re_128, xy_im_128);

  return(output);
}

/* Zero Forcing Rx function: nr_construct_HhH_elements()
 *
 *
 * */
static void nr_ulsch_construct_HhH_elements(c16_t *conjch00_ch00,
                                            c16_t *conjch01_ch01,
                                            c16_t *conjch11_ch11,
                                            c16_t *conjch10_ch10, //
                                            c16_t *conjch20_ch20,
                                            c16_t *conjch21_ch21,
                                            c16_t *conjch30_ch30,
                                            c16_t *conjch31_ch31,
                                            c16_t *conjch00_ch01, // 00_01
                                            c16_t *conjch01_ch00, // 01_00
                                            c16_t *conjch10_ch11, // 10_11
                                            c16_t *conjch11_ch10, // 11_10
                                            c16_t *conjch20_ch21,
                                            c16_t *conjch21_ch20,
                                            c16_t *conjch30_ch31,
                                            c16_t *conjch31_ch30,
                                            c16_t *after_mf_00,
                                            c16_t *after_mf_01,
                                            c16_t *after_mf_10,
                                            c16_t *after_mf_11,
                                            unsigned short nb_rb)
{
  //This function is used to construct the (H_hermitian * H matrix) matrix elements
  simde__m128i *conjch00_ch00_128 = (simde__m128i *)conjch00_ch00;
  simde__m128i *conjch01_ch01_128 = (simde__m128i *)conjch01_ch01;
  simde__m128i *conjch11_ch11_128 = (simde__m128i *)conjch11_ch11;
  simde__m128i *conjch10_ch10_128 = (simde__m128i *)conjch10_ch10;

  simde__m128i *conjch20_ch20_128 = (simde__m128i *)conjch20_ch20;
  simde__m128i *conjch21_ch21_128 = (simde__m128i *)conjch21_ch21;
  simde__m128i *conjch30_ch30_128 = (simde__m128i *)conjch30_ch30;
  simde__m128i *conjch31_ch31_128 = (simde__m128i *)conjch31_ch31;

  simde__m128i *conjch00_ch01_128 = (simde__m128i *)conjch00_ch01;
  simde__m128i *conjch01_ch00_128 = (simde__m128i *)conjch01_ch00;
  simde__m128i *conjch10_ch11_128 = (simde__m128i *)conjch10_ch11;
  simde__m128i *conjch11_ch10_128 = (simde__m128i *)conjch11_ch10;

  simde__m128i *conjch20_ch21_128 = (simde__m128i *)conjch20_ch21;
  simde__m128i *conjch21_ch20_128 = (simde__m128i *)conjch21_ch20;
  simde__m128i *conjch30_ch31_128 = (simde__m128i *)conjch30_ch31;
  simde__m128i *conjch31_ch30_128 = (simde__m128i *)conjch31_ch30;

  simde__m128i *after_mf_00_128 = (simde__m128i *)after_mf_00;
  simde__m128i *after_mf_01_128 = (simde__m128i *)after_mf_01;
  simde__m128i *after_mf_10_128 = (simde__m128i *)after_mf_10;
  simde__m128i *after_mf_11_128 = (simde__m128i *)after_mf_11;

  for (unsigned short rb=0; rb<3*nb_rb; rb++) {

    after_mf_00_128[0] = simde_mm_adds_epi16(conjch00_ch00_128[0], conjch10_ch10_128[0]); //00_00 + 10_10
    if (conjch20_ch20 != NULL) after_mf_00_128[0] = simde_mm_adds_epi16(after_mf_00_128[0], conjch20_ch20_128[0]);
    if (conjch30_ch30 != NULL) after_mf_00_128[0] = simde_mm_adds_epi16(after_mf_00_128[0], conjch30_ch30_128[0]);

    after_mf_11_128[0] = simde_mm_adds_epi16(conjch01_ch01_128[0], conjch11_ch11_128[0]); //01_01 + 11_11
    if (conjch21_ch21 != NULL) after_mf_11_128[0] = simde_mm_adds_epi16(after_mf_11_128[0], conjch21_ch21_128[0]);
    if (conjch31_ch31 != NULL) after_mf_11_128[0] = simde_mm_adds_epi16(after_mf_11_128[0], conjch31_ch31_128[0]);

    after_mf_01_128[0] = simde_mm_adds_epi16(conjch00_ch01_128[0], conjch10_ch11_128[0]); //00_01 + 10_11
    if (conjch20_ch21 != NULL) after_mf_01_128[0] = simde_mm_adds_epi16(after_mf_01_128[0], conjch20_ch21_128[0]);
    if (conjch30_ch31 != NULL) after_mf_01_128[0] = simde_mm_adds_epi16(after_mf_01_128[0], conjch30_ch31_128[0]);

    after_mf_10_128[0] = simde_mm_adds_epi16(conjch01_ch00_128[0], conjch11_ch10_128[0]); //01_00 + 11_10
    if (conjch21_ch20 != NULL) after_mf_10_128[0] = simde_mm_adds_epi16(after_mf_10_128[0], conjch21_ch20_128[0]);
    if (conjch31_ch30 != NULL) after_mf_10_128[0] = simde_mm_adds_epi16(after_mf_10_128[0], conjch31_ch30_128[0]);

#ifdef DEBUG_DLSCH_DEMOD
    if ((rb<=30))
    {
      printf(" \n construct_HhH_elements \n");
      print_shorts("after_mf_00_128:",(int16_t*)&after_mf_00_128[0]);
      print_shorts("after_mf_01_128:",(int16_t*)&after_mf_01_128[0]);
      print_shorts("after_mf_10_128:",(int16_t*)&after_mf_10_128[0]);
      print_shorts("after_mf_11_128:",(int16_t*)&after_mf_11_128[0]);
    }
#endif
    conjch00_ch00_128+=1;
    conjch10_ch10_128+=1;
    conjch01_ch01_128+=1;
    conjch11_ch11_128+=1;

    if (conjch20_ch20 != NULL) conjch20_ch20_128+=1;
    if (conjch21_ch21 != NULL) conjch21_ch21_128+=1;
    if (conjch30_ch30 != NULL) conjch30_ch30_128+=1;
    if (conjch31_ch31 != NULL) conjch31_ch31_128+=1;

    conjch00_ch01_128+=1;
    conjch01_ch00_128+=1;
    conjch10_ch11_128+=1;
    conjch11_ch10_128+=1;

    if (conjch20_ch21 != NULL) conjch20_ch21_128+=1;
    if (conjch21_ch20 != NULL) conjch21_ch20_128+=1;
    if (conjch30_ch31 != NULL) conjch30_ch31_128+=1;
    if (conjch31_ch30 != NULL) conjch31_ch30_128+=1;

    after_mf_00_128 += 1;
    after_mf_01_128 += 1;
    after_mf_10_128 += 1;
    after_mf_11_128 += 1;
  }
}

// MMSE Rx function: nr_ulsch_mmse_2layers()
static uint8_t nr_ulsch_mmse_2layers(c16_t **rxdataF_comp,
                                     uint32_t buffer_length,
                                     int nb_rx_ant,
                                     c16_t ul_ch_mag[][buffer_length],
                                     c16_t ul_ch_magb[][buffer_length],
                                     c16_t ul_ch_magc[][buffer_length],
                                     c16_t ul_ch_estimates_ext[][nb_rx_ant][buffer_length],
                                     unsigned short nb_rb,
                                     unsigned char mod_order,
                                     int shift,
                                     unsigned char symbol,
                                     int length,
                                     uint32_t noise_var)
{
  uint32_t nb_rb_0 = length/12 + ((length%12)?1:0);

  /* we need at least alignment to 16 bytes, let's put 32 to be sure
   * (maybe not necessary but doesn't hurt)
   */
  c16_t conjch00_ch01[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch01_ch00[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch10_ch11[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch11_ch10[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch00_ch00[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch01_ch01[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch10_ch10[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch11_ch11[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch20_ch20[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch21_ch21[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch30_ch30[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch31_ch31[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch20_ch21[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch30_ch31[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch21_ch20[12 * nb_rb] __attribute__((aligned(32)));
  c16_t conjch31_ch30[12 * nb_rb] __attribute__((aligned(32)));

  c16_t af_mf_00[12 * nb_rb] __attribute__((aligned(32)));
  c16_t af_mf_01[12 * nb_rb] __attribute__((aligned(32)));
  c16_t af_mf_10[12 * nb_rb] __attribute__((aligned(32)));
  c16_t af_mf_11[12 * nb_rb] __attribute__((aligned(32)));
  uint32_t determ_fin[12*nb_rb] __attribute__((aligned(32)));

  c16_t *ch00, *ch01, *ch10, *ch11;
  c16_t *ch20, *ch30, *ch21, *ch31;
  switch (nb_rx_ant) {
    case 2://
      ch00 = ul_ch_estimates_ext[0][0];
      ch01 = ul_ch_estimates_ext[1][0];
      ch10 = ul_ch_estimates_ext[0][1];
      ch11 = ul_ch_estimates_ext[1][1];
      ch20 = NULL;
      ch21 = NULL;
      ch30 = NULL;
      ch31 = NULL;
      break;

    case 4://
      ch00 = ul_ch_estimates_ext[0][0];
      ch01 = ul_ch_estimates_ext[1][0];
      ch10 = ul_ch_estimates_ext[0][1];
      ch11 = ul_ch_estimates_ext[1][1];
      ch20 = ul_ch_estimates_ext[0][2];
      ch21 = ul_ch_estimates_ext[1][2];
      ch30 = ul_ch_estimates_ext[0][3];
      ch31 = ul_ch_estimates_ext[1][3];
      break;

    default:
      return -1;
      break;
  }

  /* 1- Compute the rx channel matrix after compensation: (1/2^log2_max)x(H_herm x H)
   * for n_rx = 2
   * |conj_H_00       conj_H_10|    | H_00         H_01|   |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
   * |                         |  x |                  | = |                                                                 |
   * |conj_H_01       conj_H_11|    | H_10         H_11|   |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
   *
   */

  if (nb_rx_ant >= 2) {
    // (1/2^log2_maxh)*conj_H_00xH_00: (1/(64*2))conjH_00*H_00*2^15
    nr_ulsch_conjch0_mult_ch1(ch00,
                        ch00,
                        conjch00_ch00,
                        nb_rb_0,
                        shift);
    // (1/2^log2_maxh)*conj_H_10xH_10: (1/(64*2))conjH_10*H_10*2^15
    nr_ulsch_conjch0_mult_ch1(ch10,
                        ch10,
                        conjch10_ch10,
                        nb_rb_0,
                        shift);
    // conj_H_00xH_01
    nr_ulsch_conjch0_mult_ch1(ch00,
                        ch01,
                        conjch00_ch01,
                        nb_rb_0,
                        shift); // this shift is equal to the channel level log2_maxh
    // conj_H_10xH_11
    nr_ulsch_conjch0_mult_ch1(ch10,
                        ch11,
                        conjch10_ch11,
                        nb_rb_0,
                        shift);
    // conj_H_01xH_01
    nr_ulsch_conjch0_mult_ch1(ch01,
                        ch01,
                        conjch01_ch01,
                        nb_rb_0,
                        shift);
    // conj_H_11xH_11
    nr_ulsch_conjch0_mult_ch1(ch11,
                        ch11,
                        conjch11_ch11,
                        nb_rb_0,
                        shift);
    // conj_H_01xH_00
    nr_ulsch_conjch0_mult_ch1(ch01,
                        ch00,
                        conjch01_ch00,
                        nb_rb_0,
                        shift);
    // conj_H_11xH_10
    nr_ulsch_conjch0_mult_ch1(ch11,
                        ch10,
                        conjch11_ch10,
                        nb_rb_0,
                        shift);
  }
  if (nb_rx_ant == 4) {
    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2*16))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch20,
                        ch20,
                        conjch20_ch20,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_30xH_30: (1/(64*2*4))conjH_30*H_30*2^15
    nr_ulsch_conjch0_mult_ch1(ch30,
                        ch30,
                        conjch30_ch30,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch20,
                        ch21,
                        conjch20_ch21,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch30,
                        ch31,
                        conjch30_ch31,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch21,
                        ch21,
                        conjch21_ch21,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch31,
                        ch31,
                        conjch31_ch31,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch21,
                        ch20,
                        conjch21_ch20,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch31,
                        ch30,
                        conjch31_ch30,
                        nb_rb_0,
                        shift);

    nr_ulsch_construct_HhH_elements(conjch00_ch00,
                              conjch01_ch01,
                              conjch11_ch11,
                              conjch10_ch10,//
                              conjch20_ch20,
                              conjch21_ch21,
                              conjch30_ch30,
                              conjch31_ch31,
                              conjch00_ch01,
                              conjch01_ch00,
                              conjch10_ch11,
                              conjch11_ch10,//
                              conjch20_ch21,
                              conjch21_ch20,
                              conjch30_ch31,
                              conjch31_ch30,
                              af_mf_00,
                              af_mf_01,
                              af_mf_10,
                              af_mf_11,
                              nb_rb_0);
  }
  if (nb_rx_ant == 2) {
    nr_ulsch_construct_HhH_elements(conjch00_ch00,
                              conjch01_ch01,
                              conjch11_ch11,
                              conjch10_ch10,//
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              conjch00_ch01,
                              conjch01_ch00,
                              conjch10_ch11,
                              conjch11_ch10,//
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              af_mf_00,
                              af_mf_01,
                              af_mf_10,
                              af_mf_11,
                              nb_rb_0);
  }

  // Add noise_var such that: H^h * H + noise_var * I
  if (noise_var != 0) {
    simde__m128i nvar_128i = simde_mm_set1_epi32(noise_var >> shift);
    simde__m128i *af_mf_00_128i = (simde__m128i *)af_mf_00;
    simde__m128i *af_mf_11_128i = (simde__m128i *)af_mf_11;
    for (int k = 0; k < 3 * nb_rb_0; k++) {
      af_mf_00_128i[0] = simde_mm_add_epi32(af_mf_00_128i[0], nvar_128i);
      af_mf_11_128i[0] = simde_mm_add_epi32(af_mf_11_128i[0], nvar_128i);
      af_mf_00_128i++;
      af_mf_11_128i++;
    }
  }

  //det_HhH = ad -bc
  nr_ulsch_det_HhH(af_mf_00,//a
             af_mf_01,//b
             af_mf_10,//c
             af_mf_11,//d
             determ_fin,
             nb_rb_0);
  /* 2- Compute the channel matrix inversion **********************************
   *
     *    |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
     * A= |                                                                 |
     *    |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
     *
     *
     *
     *inv(A) =(1/det)*[d  -b
     *                 -c  a]
     *
     *
     **************************************************************************/
  simde__m128i *ul_ch_mag128_0 = NULL, *ul_ch_mag128b_0 = NULL, *ul_ch_mag128c_0 = NULL; // Layer 0
  simde__m128i *ul_ch_mag128_1 = NULL, *ul_ch_mag128b_1 = NULL, *ul_ch_mag128c_1 = NULL; // Layer 1
  simde__m128i mmtmpD0, mmtmpD1, mmtmpD2, mmtmpD3;
  simde__m128i QAM_amp128 = {0}, QAM_amp128b = {0}, QAM_amp128c = {0};

  simde__m128i *determ_fin_128 = (simde__m128i *)&determ_fin[0];

  simde__m128i *after_mf_a_128 = (simde__m128i *)af_mf_00;
  simde__m128i *after_mf_b_128 = (simde__m128i *)af_mf_01;
  simde__m128i *after_mf_c_128 = (simde__m128i *)af_mf_10;
  simde__m128i *after_mf_d_128 = (simde__m128i *)af_mf_11;
  
  simde__m128i *rxdataF_comp128_0 = (simde__m128i *)&rxdataF_comp[0][symbol * buffer_length];
  simde__m128i *rxdataF_comp128_1 = (simde__m128i *)&rxdataF_comp[nb_rx_ant][symbol * buffer_length];

  if (mod_order > 2) {
    if (mod_order == 4) {
      QAM_amp128 = simde_mm_set1_epi16(QAM16_n1); // 2/sqrt(10)
      QAM_amp128b = simde_mm_setzero_si128();
      QAM_amp128c = simde_mm_setzero_si128();
    } else if (mod_order == 6) {
      QAM_amp128 = simde_mm_set1_epi16(QAM64_n1); // 4/sqrt{42}
      QAM_amp128b = simde_mm_set1_epi16(QAM64_n2); // 2/sqrt{42}
      QAM_amp128c = simde_mm_setzero_si128();
    } else if (mod_order == 8) {
      QAM_amp128 =  simde_mm_set1_epi16(QAM256_n1);
      QAM_amp128b = simde_mm_set1_epi16(QAM256_n2);
      QAM_amp128c = simde_mm_set1_epi16(QAM256_n3);
    }
    ul_ch_mag128_0 = (simde__m128i *)&ul_ch_mag[0];
    ul_ch_mag128b_0 = (simde__m128i *)&ul_ch_magb[0];
    ul_ch_mag128c_0 = (simde__m128i *)&ul_ch_magc[0];
    ul_ch_mag128_1 = (simde__m128i *)&ul_ch_mag[1];
    ul_ch_mag128b_1 = (simde__m128i *)&ul_ch_magb[1];
    ul_ch_mag128c_1 = (simde__m128i *)&ul_ch_magc[1];
  }

  for (int rb = 0; rb < 3 * nb_rb_0; rb++) {

    // Magnitude computation
    if (mod_order > 2) {
      uint64_t sum_det = 0;
      for (int k = 0; k < 4; k++) {
        sum_det += (((uint32_t *)&determ_fin_128[0])[k]);
      }
      // Add bias to reduce rounding error
      sum_det = (sum_det + 2) >> 2;

      int b = log2_approx(sum_det) - 8;
      if (b > 0) {
        mmtmpD2 = simde_mm_srai_epi32(determ_fin_128[0], b);
      } else {
        mmtmpD2 = simde_mm_slli_epi32(determ_fin_128[0], -b);
      }
      mmtmpD3 = simde_mm_unpacklo_epi32(mmtmpD2, mmtmpD2);
      mmtmpD2 = simde_mm_unpackhi_epi32(mmtmpD2, mmtmpD2);
      mmtmpD2 = simde_mm_packs_epi32(mmtmpD3, mmtmpD2);

      // Layer 0
      ul_ch_mag128_0[0] = mmtmpD2;
      ul_ch_mag128b_0[0] = mmtmpD2;
      ul_ch_mag128c_0[0] = mmtmpD2;
      ul_ch_mag128_0[0] = simde_mm_mulhi_epi16(ul_ch_mag128_0[0], QAM_amp128);
      ul_ch_mag128_0[0] = simde_mm_slli_epi16(ul_ch_mag128_0[0], 1);
      ul_ch_mag128b_0[0] = simde_mm_mulhi_epi16(ul_ch_mag128b_0[0], QAM_amp128b);
      ul_ch_mag128b_0[0] = simde_mm_slli_epi16(ul_ch_mag128b_0[0], 1);
      ul_ch_mag128c_0[0] = simde_mm_mulhi_epi16(ul_ch_mag128c_0[0], QAM_amp128c);
      ul_ch_mag128c_0[0] = simde_mm_slli_epi16(ul_ch_mag128c_0[0], 1);

      // Layer 1
      ul_ch_mag128_1[0] = mmtmpD2;
      ul_ch_mag128b_1[0] = mmtmpD2;
      ul_ch_mag128c_1[0] = mmtmpD2;
      ul_ch_mag128_1[0] = simde_mm_mulhi_epi16(ul_ch_mag128_1[0], QAM_amp128);
      ul_ch_mag128_1[0] = simde_mm_slli_epi16(ul_ch_mag128_1[0], 1);
      ul_ch_mag128b_1[0] = simde_mm_mulhi_epi16(ul_ch_mag128b_1[0], QAM_amp128b);
      ul_ch_mag128b_1[0] = simde_mm_slli_epi16(ul_ch_mag128b_1[0], 1);
      ul_ch_mag128c_1[0] = simde_mm_mulhi_epi16(ul_ch_mag128c_1[0], QAM_amp128c);
      ul_ch_mag128c_1[0] = simde_mm_slli_epi16(ul_ch_mag128c_1[0], 1);
    }

    // multiply by channel Inv
    //rxdataF_zf128_0 = rxdataF_comp128_0*d - b*rxdataF_comp128_1
    //rxdataF_zf128_1 = rxdataF_comp128_1*a - c*rxdataF_comp128_0
    //printf("layer_1 \n");
    mmtmpD0 = nr_ulsch_comp_muli_sum(rxdataF_comp128_0[0],
                               after_mf_d_128[0],
                               rxdataF_comp128_1[0],
                               after_mf_b_128[0],
                               determ_fin_128[0]);

    //printf("layer_2 \n");
    mmtmpD1 = nr_ulsch_comp_muli_sum(rxdataF_comp128_1[0],
                               after_mf_a_128[0],
                               rxdataF_comp128_0[0],
                               after_mf_c_128[0],
                               determ_fin_128[0]);

    rxdataF_comp128_0[0] = mmtmpD0;
    rxdataF_comp128_1[0] = mmtmpD1;

#ifdef DEBUG_DLSCH_DEMOD
    printf("\n Rx signal after ZF l%d rb%d\n",symbol,rb);
    print_shorts(" Rx layer 1:",(int16_t*)&rxdataF_comp128_0[0]);
    print_shorts(" Rx layer 2:",(int16_t*)&rxdataF_comp128_1[0]);
#endif
    determ_fin_128 += 1;
    ul_ch_mag128_0 += 1;
    ul_ch_mag128_1 += 1;
    ul_ch_mag128b_0 += 1;
    ul_ch_mag128b_1 += 1;
    ul_ch_mag128c_0 += 1;
    ul_ch_mag128c_1 += 1;
    rxdataF_comp128_0 += 1;
    rxdataF_comp128_1 += 1;
    after_mf_a_128 += 1;
    after_mf_b_128 += 1;
    after_mf_c_128 += 1;
    after_mf_d_128 += 1;
  }
   return(0);
}

static void inner_rx(PHY_VARS_gNB *gNB,
                     int slot,
                     NR_DL_FRAME_PARMS *frame_parms,
                     NR_gNB_PUSCH *pusch_vars,
                     const nfapi_nr_pusch_pdu_t *rel15_ul,
                     c16_t **rxF,
                     int16_t **llr,
                     int soffset,
                     int symbol,
                     int output_shift,
                     uint32_t nvar,
                     c16_t *rxFext_slot,
                     c16_t *chFext_slot,
                     time_stats_t *pusch_extr,
                     time_stats_t *pusch_ch_comp,
                     time_stats_t *ulsch_llr)
{
  int nb_layer = rel15_ul->nrOfLayers;
  int nb_rx_ant = frame_parms->nb_antennas_rx;
  int dmrs_symbol_flag = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01;
  int buffer_length = ceil_mod(rel15_ul->rb_size * NR_NB_SC_PER_RB, 16);
  c16_t rxFext[nb_rx_ant][buffer_length] __attribute__((aligned(64)));
  c16_t chFext[nb_layer][nb_rx_ant][buffer_length] __attribute__((aligned(64)));

  memset(rxFext, 0, sizeof(rxFext));
  memset(chFext, 0, sizeof(chFext));
  int dmrs_symbol;
  if (gNB->chest_time == 0)
    dmrs_symbol = dmrs_symbol_flag ? symbol : get_valid_dmrs_idx_for_channel_est(rel15_ul->ul_dmrs_symb_pos, symbol);
  else { // average of channel estimates stored in first symbol
    int end_symbol = rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols;
    dmrs_symbol = get_next_dmrs_symbol_in_slot(rel15_ul->ul_dmrs_symb_pos, rel15_ul->start_symbol_index, end_symbol);
  }
  if (oai_ammse_ce_applied_for_current_pusch)
    dmrs_symbol = symbol;

  for (int aarx = 0; aarx < nb_rx_ant; aarx++) {
    for (int aatx = 0; aatx < nb_layer; aatx++) {
      start_meas(pusch_extr);
      nr_ulsch_extract_rbs(rxF[aarx],
                           (c16_t *)pusch_vars->ul_ch_estimates[aatx * nb_rx_ant + aarx],
                           rxFext[aarx],
                           chFext[aatx][aarx],
                           soffset+(symbol * frame_parms->ofdm_symbol_size),
                           dmrs_symbol * frame_parms->ofdm_symbol_size,
                           dmrs_symbol_flag, 
                           rel15_ul,
                           frame_parms);
      stop_meas(pusch_extr);
#if T_TRACER
      // Data Recording application supports only 1 layer and 1 Tx antenna, so only record the first layer and first Tx antenna
      if (aatx == 0 && aarx == 0) {
        int nb_re_pusch = NR_NB_SC_PER_RB * rel15_ul->rb_size;
        // Assume assume Tx and Rx = 1
        if (T_ACTIVE(T_GNB_PHY_UL_FD_PUSCH_IQ)) {
          copy_c16_data_to_slot_memory(rxFext[aarx], rxFext_slot, nb_re_pusch, symbol);
        }
        if (T_ACTIVE(T_GNB_PHY_UL_FD_CHAN_EST_DMRS_INTERPL)) {
          copy_c16_data_to_slot_memory(chFext[aatx][aarx], chFext_slot, nb_re_pusch, symbol);
        }
      }
#endif
    }
  }
  start_meas(pusch_ch_comp);
  c16_t rho[nb_layer][nb_layer][buffer_length] __attribute__((aligned(64)));
  c16_t rxF_ch_maga[nb_layer][buffer_length] __attribute__((aligned(64)));
  c16_t rxF_ch_magb[nb_layer][buffer_length] __attribute__((aligned(64)));
  c16_t rxF_ch_magc[nb_layer][buffer_length] __attribute__((aligned(64)));

  memset(rho, 0, sizeof(rho));
  memset(rxF_ch_maga, 0, sizeof(rxF_ch_maga));
  memset(rxF_ch_magb, 0, sizeof(rxF_ch_magb));
  memset(rxF_ch_magc, 0, sizeof(rxF_ch_magc));
  for (int i = 0; i < nb_layer; i++)
    memset(&pusch_vars->rxdataF_comp[i*nb_rx_ant][symbol * buffer_length], 0, sizeof(int32_t) * buffer_length);

  nr_ulsch_channel_compensation(buffer_length,
                                nb_rx_ant,
                                rxFext,
                                chFext,
                                rxF_ch_maga,
                                rxF_ch_magb,
                                rxF_ch_magc,
                                pusch_vars->rxdataF_comp,
                                nb_layer,
                                rho,
                                rel15_ul,
                                symbol,
                                output_shift);
  stop_meas(pusch_ch_comp);

  if (nb_layer == 1 && rel15_ul->transform_precoding == transformPrecoder_enabled && rel15_ul->qam_mod_order <= 6) {
    if (rel15_ul->qam_mod_order > 2)
      nr_freq_equalization(frame_parms,
                           &pusch_vars->rxdataF_comp[0][symbol * buffer_length],
                           rxF_ch_maga[0],
                           rxF_ch_magb[0],
                           symbol,
                           pusch_vars->ul_valid_re_per_slot[symbol],
                           rel15_ul->qam_mod_order);
    nr_idft((int32_t *)&pusch_vars->rxdataF_comp[0][symbol * buffer_length], pusch_vars->ul_valid_re_per_slot[symbol]);
  }
  if (rel15_ul->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {
    nr_pusch_ptrs_processing(gNB,
                             frame_parms,
                             rel15_ul,
                             pusch_vars,
                             slot,
                             symbol,
                             buffer_length);
    pusch_vars->ul_valid_re_per_slot[symbol] -= pusch_vars->ptrs_re_per_slot;
  }
  start_meas(ulsch_llr);
  if (nb_layer == 2) {
    if (rel15_ul->qam_mod_order <= 6) {
      nr_ulsch_compute_ML_llr((c16_t *)&pusch_vars->rxdataF_comp[0][symbol * buffer_length],
                              (c16_t *)&pusch_vars->rxdataF_comp[nb_rx_ant][symbol * buffer_length],
                              rxF_ch_maga[0],
                              rxF_ch_maga[1],
                              llr[0],
                              llr[1],
                              rho[0][1],
                              rho[1][0],
                              pusch_vars->ul_valid_re_per_slot[symbol],
                              rel15_ul->qam_mod_order);
    }
    else {
      nr_ulsch_mmse_2layers(pusch_vars->rxdataF_comp,
                            buffer_length,
                            nb_rx_ant,
                            rxF_ch_maga,
                            rxF_ch_magb,
                            rxF_ch_magc,
                            chFext,
                            rel15_ul->rb_size,
                            rel15_ul->qam_mod_order,
                            pusch_vars->log2_maxh,
                            symbol,
                            pusch_vars->ul_valid_re_per_slot[symbol],
                            nvar);
    }
  }
  if (nb_layer != 2 || rel15_ul->qam_mod_order > 6)
    for (int aatx = 0; aatx < nb_layer; aatx++)
      nr_ulsch_compute_llr(&pusch_vars->rxdataF_comp[aatx * nb_rx_ant][symbol * buffer_length],
                           rxF_ch_maga[aatx],
                           rxF_ch_magb[aatx],
                           rxF_ch_magc[aatx],
                           llr[aatx],
                           pusch_vars->ul_valid_re_per_slot[symbol],
                           symbol,
                           rel15_ul->qam_mod_order);
  stop_meas(ulsch_llr);
}

typedef struct puschSymbolProc_s {
  PHY_VARS_gNB *gNB;
  NR_DL_FRAME_PARMS *frame_parms;
  const nfapi_nr_pusch_pdu_t *rel15_ul;
  NR_gNB_PUSCH *pusch_vars;
  int slot;
  int startSymbol;
  int numSymbols;
  int16_t *llr;
  int16_t *scramblingSequence;
  uint32_t nvar;
  int beam_nb;
  time_stats_t pusch_extr;
  time_stats_t pusch_ch_comp;
  time_stats_t ulsch_llr;
  time_stats_t ul_demap;
  time_stats_t ul_unscram;
  task_ans_t *ans;
  c16_t *pusch_ch_est_dmrs_interpl_slot_mem;
  c16_t *rxFext_slot_mem;
} puschSymbolProc_t;

static void nr_pusch_symbol_processing(void *arg)
{
  puschSymbolProc_t *rdata=(puschSymbolProc_t*)arg;

  PHY_VARS_gNB *gNB = rdata->gNB;
  NR_DL_FRAME_PARMS *frame_parms = rdata->frame_parms;
  const nfapi_nr_pusch_pdu_t *rel15_ul = rdata->rel15_ul;
  int slot = rdata->slot;
  NR_gNB_PUSCH *pusch_vars = rdata->pusch_vars;
  for (int symbol = rdata->startSymbol; symbol < rdata->startSymbol + rdata->numSymbols; symbol++) {
    if (pusch_vars->ul_valid_re_per_slot[symbol] == 0)
      continue;
    int soffset = (slot % RU_RX_SLOT_DEPTH) * frame_parms->symbols_per_slot * frame_parms->ofdm_symbol_size;
    int buffer_length = ceil_mod(pusch_vars->ul_valid_re_per_slot[symbol] * NR_NB_SC_PER_RB, 16);
    int16_t llrs[rel15_ul->nrOfLayers][ceil_mod(buffer_length * rel15_ul->qam_mod_order, 64)];
    int16_t *llrss[rel15_ul->nrOfLayers];
    for (int l = 0; l < rel15_ul->nrOfLayers; l++)
      llrss[l] = llrs[l];

    inner_rx(gNB,
             slot,
             frame_parms,
             pusch_vars,
             rel15_ul,
             gNB->common_vars.rxdataF[rdata->beam_nb],
             llrss,
             soffset,
             symbol,
             pusch_vars->log2_maxh,
             rdata->nvar,
             rdata->rxFext_slot_mem,
             rdata->pusch_ch_est_dmrs_interpl_slot_mem,
             &rdata->pusch_extr,
             &rdata->pusch_ch_comp,
             &rdata->ulsch_llr);

    int nb_re_pusch = pusch_vars->ul_valid_re_per_slot[symbol];
    // layer de-mapping
    start_meas(&rdata->ul_demap);
    int16_t *llr_ptr = llrs[0];
    if (rel15_ul->nrOfLayers != 1) {
      llr_ptr = &rdata->llr[pusch_vars->llr_offset[symbol] * rel15_ul->nrOfLayers];
      for (int i = 0; i < (nb_re_pusch); i++)
        for (int l = 0; l < rel15_ul->nrOfLayers; l++)
          for (int m = 0; m < rel15_ul->qam_mod_order; m++)
            llr_ptr[i * rel15_ul->nrOfLayers * rel15_ul->qam_mod_order + l * rel15_ul->qam_mod_order + m] =
                llrss[l][i * rel15_ul->qam_mod_order + m];
    }
    stop_meas(&rdata->ul_demap);
    // unscrambling
    start_meas(&rdata->ul_unscram);
    int16_t *llr16 = (int16_t*)&rdata->llr[pusch_vars->llr_offset[symbol] * rel15_ul->nrOfLayers];
    int16_t *s = rdata->scramblingSequence + pusch_vars->llr_offset[symbol] * rel15_ul->nrOfLayers;
    const int end = nb_re_pusch * rel15_ul->qam_mod_order * rel15_ul->nrOfLayers;
    int i = 0;
    for (; (i + 8) <= end; i += 8) {
      simde__m128i llr128 = simde_mm_loadu_si128((simde__m128i *)&llr_ptr[i]);
      simde__m128i s128 = simde_mm_loadu_si128((simde__m128i *)&s[i]);
      simde_mm_storeu_si128(llr16 + i, simde_mm_mullo_epi16(llr128, s128));
    }
    for (; i < end; i++)
      llr16[i] = llr_ptr[i] * s[i];
    stop_meas(&rdata->ul_unscram);
  }

  // Task running in // completed
  completed_task_ans(rdata->ans);
}

static uint32_t average_u32(const uint32_t *x, uint16_t size)
{
  AssertFatal(size > 0 && x != NULL, "x is NULL or size is 0\n");

  uint64_t sum_x = 0;
  simde__m256i vec_sum = simde_mm256_setzero_si256();

  int i = 0;
  for (; i + 8 <= size; i += 8) {
    simde__m256i vec_data = simde_mm256_loadu_si256((simde__m256i *)&x[i]);
    vec_sum = simde_mm256_add_epi32(vec_sum, vec_data);
  }
  uint32_t *vec_sum32 = (uint32_t *)&vec_sum;
  for (int k = 0; k < 8; k++) {
    sum_x += vec_sum32[k];
  }
  for (; i < size; i++) {
    sum_x += x[i];
  }

  return (uint32_t)(sum_x / size);
}

int nr_rx_pusch_tp(PHY_VARS_gNB *gNB,
                   NR_gNB_PUSCH *pusch_vars,
                   const nfapi_nr_pusch_pdu_t *rel15_ul,
                   uint32_t *ret_unav_res,
                   uint32_t frame,
                   uint8_t slot,
                   int beam_nb)
{
  oai_ammse_ce_applied_for_current_pusch = false;
  NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;

  uint32_t bwp_start_subcarrier = ((rel15_ul->rb_start + rel15_ul->bwp_start) * NR_NB_SC_PER_RB + frame_parms->first_carrier_offset) % frame_parms->ofdm_symbol_size;
  LOG_D(PHY,"pusch %d.%d : bwp_start_subcarrier %d, rb_start %d, first_carrier_offset %d\n", frame,slot,bwp_start_subcarrier, rel15_ul->rb_start, frame_parms->first_carrier_offset);
  LOG_D(PHY,"pusch %d.%d : ul_dmrs_symb_pos %x\n",frame,slot,rel15_ul->ul_dmrs_symb_pos);

  // Memories to store data for data recording
  int buffer_length_slot = rel15_ul->rb_size * NR_NB_SC_PER_RB * 14; // 14 OFDM Symbols per slot
  // data recording application supports only a single layer.
  // nb_rx_ant (= frame_parms->nb_antennas_rx) is limited to 1 for data recording application.
  // int nb_layer (= rel15_ul->nrOfLayers) is limited to 1 for data recording application.

  // Initialize memory for DMRS signals
  c16_t pusch_dmrs_slot_mem[1 * buffer_length_slot] __attribute__((aligned(64)));
  // Initialize memory for channel estimates based on DMRS positions
  c16_t pusch_ch_est_dmrs_pos_slot_mem[buffer_length_slot * 1 * 1] __attribute__((aligned(64)));
  // memory to store slot grid with channel coefficients based on DMRS positions after interpolation
  c16_t pusch_ch_est_dmrs_interpl_slot_mem[buffer_length_slot * 1 * 1] __attribute__((aligned(64)));
  // memory to store extracted data including PUSCH + DMRS
  c16_t rxFext_slot_mem[1 * buffer_length_slot] __attribute__((aligned(64)));
#if T_TRACER
  // memory to store the RFsim true channel over the scheduled PUSCH grid
  c16_t rfsim_true_channel_slot_mem[buffer_length_slot * rel15_ul->nrOfLayers * frame_parms->nb_antennas_rx]
      __attribute__((aligned(64)));
  // memory to store the OAI interpolated CE grid before optional A-MMSE replacement
  c16_t oai_inter_ch_est_slot_mem[buffer_length_slot] __attribute__((aligned(64)));
#endif

#if T_TRACER
  // Initialize memory for DMRS signals
  if (T_ACTIVE(T_GNB_PHY_UL_FD_DMRS))
    memset(pusch_dmrs_slot_mem, 0, sizeof(c16_t) * 1 * buffer_length_slot);

  // Initialize memory for channel estimates based on DMRS positions
  if (T_ACTIVE(T_GNB_PHY_UL_FD_CHAN_EST_DMRS_POS))
    memset(pusch_ch_est_dmrs_pos_slot_mem, 0, sizeof(c16_t) * buffer_length_slot * 1 * 1);

  // memory to store slot grid with channel coefficients based on DMRS positions after interpolation
  if (T_ACTIVE(T_GNB_PHY_UL_FD_CHAN_EST_DMRS_INTERPL))
    memset(pusch_ch_est_dmrs_interpl_slot_mem, 0, sizeof(c16_t) * buffer_length_slot * 1 * 1);

  // memory to store extracted data including PUSCH + DMRS
  if (T_ACTIVE(T_GNB_PHY_UL_FD_PUSCH_IQ))
    memset(rxFext_slot_mem, 0, sizeof(c16_t) * buffer_length_slot * 1 * 1);

  // memory to store RFsim true channel coefficients
  if (T_ACTIVE(T_GNB_PHY_UL_FD_TRUE_CHANNEL))
    memset(rfsim_true_channel_slot_mem, 0, sizeof(rfsim_true_channel_slot_mem));
#endif
  if (oai_ammse_ce_enabled() || oai_ce_nmse_enabled())
    memset(pusch_ch_est_dmrs_pos_slot_mem, 0, sizeof(c16_t) * buffer_length_slot * 1 * 1);

  //----------------------------------------------------------
  //------------------- Channel estimation -------------------
  //----------------------------------------------------------
  start_meas(&gNB->ulsch_channel_estimation_stats);
  int max_ch = 0;
  uint32_t nvar = 0;
  int end_symbol = rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols;
  uint8_t dmrs_symb_idx = 0;
  for (uint8_t symbol = rel15_ul->start_symbol_index; symbol < end_symbol; symbol++) {
    uint8_t dmrs_symbol_flag = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01;
    LOG_D(PHY, "symbol %d, dmrs_symbol_flag :%d\n", symbol, dmrs_symbol_flag);

    if (dmrs_symbol_flag == 1) {
      for (int nl = 0; nl < rel15_ul->nrOfLayers; nl++) {
        uint32_t nvar_tmp = 0;
        nr_pusch_channel_estimation(gNB,
                                    slot,
                                    nl,
                                    get_dmrs_port(nl, rel15_ul->dmrs_ports),
                                    dmrs_symb_idx,
                                    symbol,
                                    pusch_vars,
                                    beam_nb,
                                    bwp_start_subcarrier,
                                    rel15_ul,
                                    &max_ch,
                                    &nvar_tmp,
                                    pusch_dmrs_slot_mem,
                                    pusch_ch_est_dmrs_pos_slot_mem);
        nvar += nvar_tmp;
      }
      dmrs_symb_idx++;
    }
  }

  if (dmrs_symb_idx > 0)
    nvar /= (dmrs_symb_idx * rel15_ul->nrOfLayers);

  allocCast2D(n0_subband_power,
              unsigned int,
              gNB->measurements.n0_subband_power,
              frame_parms->nb_antennas_rx,
              frame_parms->N_RB_UL,
              false);

  int start_sc = (rel15_ul->bwp_start + rel15_ul->rb_start) * NR_NB_SC_PER_RB;
  int middle_sc = frame_parms->ofdm_symbol_size - frame_parms->first_carrier_offset;
  int end_sc = (start_sc + rel15_ul->rb_size * NR_NB_SC_PER_RB - 1) % frame_parms->ofdm_symbol_size;
  for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    pusch_vars->ulsch_power[aarx] = 0;
    pusch_vars->ulsch_noise_power[aarx] = 0;
    int64_t symb_energy = 0;

    for (uint8_t symbol = rel15_ul->start_symbol_index; symbol < end_symbol; symbol++) {
      int offset0 = ((slot % RU_RX_SLOT_DEPTH) * frame_parms->symbols_per_slot + symbol) * frame_parms->ofdm_symbol_size;
      int offset = offset0 + (frame_parms->first_carrier_offset + start_sc) % frame_parms->ofdm_symbol_size;
      c16_t *ul_ch = &gNB->common_vars.rxdataF[beam_nb][aarx][offset];
      if (end_sc < start_sc) {
        int64_t symb_energy_aux = signal_energy_nodc(ul_ch, middle_sc - start_sc) * (middle_sc - start_sc);
        ul_ch = &gNB->common_vars.rxdataF[beam_nb][aarx][offset0];
        symb_energy_aux += (signal_energy_nodc(ul_ch, end_sc + 1) * (end_sc + 1));
        symb_energy += symb_energy_aux / (rel15_ul->rb_size * NR_NB_SC_PER_RB);
      } else {
        symb_energy += signal_energy_nodc(ul_ch, rel15_ul->rb_size * NR_NB_SC_PER_RB);
      }
    }
    pusch_vars->ulsch_power[aarx] += (symb_energy / rel15_ul->nr_of_symbols);

    pusch_vars->ulsch_noise_power[aarx] +=
        average_u32(&n0_subband_power[aarx][rel15_ul->bwp_start + rel15_ul->rb_start], rel15_ul->rb_size);

    LOG_D(PHY,
          "aa %d, bwp_start%d, rb_start %d, rb_size %d: ulsch_power %d, ulsch_noise_power %d\n",
          aarx,
          rel15_ul->bwp_start,
          rel15_ul->rb_start,
          rel15_ul->rb_size,
          pusch_vars->ulsch_power[aarx],
          pusch_vars->ulsch_noise_power[aarx]);
  }

  // averaging time domain channel estimates
  if (gNB->chest_time == 1)
    nr_chest_time_domain_avg(frame_parms,
                             pusch_vars->ul_ch_estimates,
                             rel15_ul->nr_of_symbols,
                             rel15_ul->start_symbol_index,
                             rel15_ul->ul_dmrs_symb_pos,
                             rel15_ul->rb_size,
                             rel15_ul->nrOfLayers);

#if T_TRACER
  c16_t *oai_inter_grid_for_nmse = NULL;
  if (oai_ce_nmse_enabled() && rel15_ul->nrOfLayers == 1 && frame_parms->nb_antennas_rx == 1) {
    const int nb_re_for_nmse = rel15_ul->rb_size * NR_NB_SC_PER_RB;
    const c16_t *oai_inter_stream = (const c16_t *)pusch_vars->ul_ch_estimates[0];
    for (int rel_symbol = 0; rel_symbol < rel15_ul->nr_of_symbols; rel_symbol++) {
      const int symbol = rel15_ul->start_symbol_index + rel_symbol;
      int dmrs_symbol = 0;
      if (gNB->chest_time == 0) {
        const int dmrs_symbol_flag = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01;
        dmrs_symbol = dmrs_symbol_flag ? symbol : get_valid_dmrs_idx_for_channel_est(rel15_ul->ul_dmrs_symb_pos, symbol);
      } else {
        dmrs_symbol = get_next_dmrs_symbol_in_slot(rel15_ul->ul_dmrs_symb_pos, rel15_ul->start_symbol_index, end_symbol);
      }
      const c16_t *src_symbol = &oai_inter_stream[dmrs_symbol * frame_parms->ofdm_symbol_size];
      c16_t *dst_symbol = &oai_inter_ch_est_slot_mem[rel_symbol * nb_re_for_nmse];
      memcpy(dst_symbol, src_symbol, sizeof(c16_t) * nb_re_for_nmse);
    }
    oai_inter_grid_for_nmse = oai_inter_ch_est_slot_mem;
  }
#endif

  oai_ammse_ce_try_replace(gNB, pusch_vars, frame, slot, rel15_ul, nvar, pusch_ch_est_dmrs_pos_slot_mem, &max_ch);

#if T_TRACER
  oai_ce_nmse_maybe_enqueue(gNB, pusch_vars, frame, slot, rel15_ul, pusch_ch_est_dmrs_pos_slot_mem, oai_inter_grid_for_nmse);
#endif

  stop_meas(&gNB->ulsch_channel_estimation_stats);

  start_meas(&gNB->rx_pusch_init_stats);

  // Scrambling initialization
  int number_dmrs_symbols = 0;
  for (int l = rel15_ul->start_symbol_index; l < end_symbol; l++)
    number_dmrs_symbols += ((rel15_ul->ul_dmrs_symb_pos)>>l) & 0x01;
  int nb_re_dmrs;
  if (rel15_ul->dmrs_config_type == pusch_dmrs_type1)
    nb_re_dmrs = 6*rel15_ul->num_dmrs_cdm_grps_no_data;
  else
    nb_re_dmrs = 4*rel15_ul->num_dmrs_cdm_grps_no_data;

  uint32_t unav_res = 0;
  if (rel15_ul->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {
    uint16_t ptrsSymbPos = 0;
    set_ptrs_symb_idx(&ptrsSymbPos,
                      rel15_ul->nr_of_symbols,
                      rel15_ul->start_symbol_index,
                      1 << rel15_ul->pusch_ptrs.ptrs_time_density,
                      rel15_ul->ul_dmrs_symb_pos);
    int ptrsSymbPerSlot = get_ptrs_symbols_in_slot(ptrsSymbPos, rel15_ul->start_symbol_index, rel15_ul->nr_of_symbols);
    int n_ptrs = (rel15_ul->rb_size + rel15_ul->pusch_ptrs.ptrs_freq_density - 1) / rel15_ul->pusch_ptrs.ptrs_freq_density;
    unav_res = n_ptrs * ptrsSymbPerSlot;
  }

  // get how many bit in a slot //
  int G = nr_get_G(rel15_ul->rb_size,
                   rel15_ul->nr_of_symbols,
                   nb_re_dmrs,
                   number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
                   unav_res,
                   rel15_ul->qam_mod_order,
                   rel15_ul->nrOfLayers);
  *ret_unav_res = unav_res;

  // initialize scrambling sequence //
  int16_t scramblingSequence[G + 96] __attribute__((aligned(64)));

  nr_codeword_unscrambling_init(scramblingSequence, G, 0, rel15_ul->data_scrambling_id, rel15_ul->rnti);

  // first the computation of channel levels

  int nb_re_pusch = 0, meas_symbol = -1;
  for(meas_symbol = rel15_ul->start_symbol_index; meas_symbol < end_symbol; meas_symbol++) 
    if ((nb_re_pusch = get_nb_re_pusch(frame_parms, rel15_ul, meas_symbol)) > 0)
      break;

  AssertFatal(nb_re_pusch > 0 && meas_symbol >= 0,
              "nb_re_pusch %d cannot be 0 or meas_symbol %d cannot be negative here\n",
              nb_re_pusch,
              meas_symbol);

  // extract the first dmrs for the channel level computation
  // extract the data in the OFDM frame, to the start of the array
  int soffset = (slot % RU_RX_SLOT_DEPTH) * frame_parms->symbols_per_slot * frame_parms->ofdm_symbol_size;

  nb_re_pusch = ceil_mod(nb_re_pusch, 16);
  int dmrs_symbol;
  if (gNB->chest_time == 0)
    dmrs_symbol = get_valid_dmrs_idx_for_channel_est(rel15_ul->ul_dmrs_symb_pos, meas_symbol);
  else // average of channel estimates stored in first symbol
    dmrs_symbol = get_next_dmrs_symbol_in_slot(rel15_ul->ul_dmrs_symb_pos, rel15_ul->start_symbol_index, end_symbol);
  if (oai_ammse_ce_applied_for_current_pusch)
    dmrs_symbol = meas_symbol;
  int size_est = nb_re_pusch * frame_parms->symbols_per_slot;
  __attribute__((aligned(64))) int ul_ch_estimates_ext[rel15_ul->nrOfLayers * frame_parms->nb_antennas_rx][size_est];
  memset(ul_ch_estimates_ext, 0, sizeof(ul_ch_estimates_ext));
  int buffer_length = rel15_ul->rb_size * NR_NB_SC_PER_RB;
  c16_t temp_rxFext[frame_parms->nb_antennas_rx][buffer_length] __attribute__((aligned(64)));
  for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++)
    for (int nl = 0; nl < rel15_ul->nrOfLayers; nl++) {
      start_meas(&gNB->pusch_extraction_stats);
      nr_ulsch_extract_rbs(gNB->common_vars.rxdataF[beam_nb][aarx],
                           (c16_t *)pusch_vars->ul_ch_estimates[nl * frame_parms->nb_antennas_rx + aarx],
                           temp_rxFext[aarx],
                           (c16_t*)&ul_ch_estimates_ext[nl * frame_parms->nb_antennas_rx + aarx][meas_symbol * nb_re_pusch],
                           soffset + meas_symbol * frame_parms->ofdm_symbol_size,
                           dmrs_symbol * frame_parms->ofdm_symbol_size,
                           (rel15_ul->ul_dmrs_symb_pos >> meas_symbol) & 0x01, 
                           rel15_ul,
                           frame_parms);
      stop_meas(&gNB->pusch_extraction_stats);
    }

  uint8_t shift_ch_ext = rel15_ul->nrOfLayers > 1 ? log2_approx(max_ch >> 11) : 0;

  //----------------------------------------------------------
  //--------------------- Channel Scaling --------------------
  //----------------------------------------------------------
  nr_scale_channel(size_est,
                   ul_ch_estimates_ext,
                   meas_symbol,
                   nb_re_pusch,
                   rel15_ul->nrOfLayers,
                   frame_parms->nb_antennas_rx,
                   shift_ch_ext);

  int avg[frame_parms->nb_antennas_rx*rel15_ul->nrOfLayers];
  nr_channel_level(meas_symbol,
                   size_est,
                   (c16_t (*)[size_est])ul_ch_estimates_ext,
                   frame_parms->nb_antennas_rx,
                   rel15_ul->nrOfLayers,
                   avg,
                   nb_re_pusch);

  int avgs = 0;
  for (int nl = 0; nl < rel15_ul->nrOfLayers; nl++)
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++)
      avgs = cmax(avgs, avg[nl * frame_parms->nb_antennas_rx + aarx]);

  if (rel15_ul->nrOfLayers == 2 && rel15_ul->qam_mod_order > 6)
    pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) - 3; // for MMSE
  else if (rel15_ul->nrOfLayers == 2)
    pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) - 2 + log2_approx(frame_parms->nb_antennas_rx >> 1);
  else 
    pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) + 1 + log2_approx(frame_parms->nb_antennas_rx >> 1);

  if (pusch_vars->log2_maxh < 0)
    pusch_vars->log2_maxh = 0;

  stop_meas(&gNB->rx_pusch_init_stats);

  start_meas(&gNB->rx_pusch_symbol_processing_stats);
  int numSymbols = gNB->num_pusch_symbols_per_thread;
  int total_res = 0;
  int const loop_iter = CEILIDIV(rel15_ul->nr_of_symbols, numSymbols);
  puschSymbolProc_t arr[loop_iter];
  task_ans_t ans;
  init_task_ans(&ans, loop_iter);

  int sz_arr = 0;
  for(uint8_t task_index = 0; task_index < loop_iter; task_index++) {
    int symbol = task_index * numSymbols + rel15_ul->start_symbol_index;
    int res_per_task = 0;
    for (int s = 0; s < numSymbols && s + symbol < end_symbol; s++) {
      pusch_vars->ul_valid_re_per_slot[symbol+s] = get_nb_re_pusch(frame_parms,rel15_ul,symbol+s);
      pusch_vars->llr_offset[symbol+s] = ((symbol+s) == rel15_ul->start_symbol_index) ? 
                                         0 : 
                                         pusch_vars->llr_offset[symbol+s-1] + pusch_vars->ul_valid_re_per_slot[symbol+s-1] * rel15_ul->qam_mod_order;
      res_per_task += pusch_vars->ul_valid_re_per_slot[symbol + s];
    }
    total_res += res_per_task;
    if (res_per_task > 0) {
      puschSymbolProc_t *rdata = &arr[sz_arr];
      rdata->ans = &ans;
      ++sz_arr;

      rdata->gNB = gNB;
      rdata->frame_parms = frame_parms;
      rdata->rel15_ul = rel15_ul;
      rdata->slot = slot;
      rdata->startSymbol = symbol;
      // Last task processes remainder symbols
      rdata->numSymbols = task_index == loop_iter - 1 ? rel15_ul->nr_of_symbols - (loop_iter - 1) * numSymbols : numSymbols;
      rdata->pusch_vars = pusch_vars;
      rdata->llr = pusch_vars->llr;
      rdata->scramblingSequence = scramblingSequence;
      rdata->nvar = nvar;
      rdata->beam_nb = beam_nb;
      rdata->rxFext_slot_mem = rxFext_slot_mem;
      rdata->pusch_ch_est_dmrs_interpl_slot_mem = pusch_ch_est_dmrs_interpl_slot_mem;
      reset_meas(&rdata->pusch_extr);
      reset_meas(&rdata->pusch_ch_comp);
      reset_meas(&rdata->ulsch_llr);
      reset_meas(&rdata->ul_demap);
      reset_meas(&rdata->ul_unscram);

      if (rel15_ul->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {
        nr_pusch_symbol_processing(rdata);
      } else {
        task_t t = {.func = &nr_pusch_symbol_processing, .args = rdata};
        pushTpool(&gNB->threadPool, t);
      }

      LOG_D(PHY, "%d.%d Added symbol %d to process, in pipe\n", frame, slot, symbol);
    } else {
      completed_task_ans(&ans);
    }
  } // symbol loop

#if T_TRACER
  int dmrs_port = get_dmrs_port(0, rel15_ul->dmrs_ports);

  log_ul_fd_dmrs(frame, slot, frame_parms, rel15_ul,
                 number_dmrs_symbols, dmrs_port,
                 (const c16_t *)(&(pusch_dmrs_slot_mem[0])),
                 rel15_ul->rb_size * NR_NB_SC_PER_RB * rel15_ul->nr_of_symbols * 4);

  log_ul_fd_chan_est_dmrs_pos(frame, slot, frame_parms, rel15_ul,
                              number_dmrs_symbols, dmrs_port,
                              (const c16_t *)(&(pusch_ch_est_dmrs_pos_slot_mem[0])),
                              rel15_ul->rb_size * NR_NB_SC_PER_RB * rel15_ul->nr_of_symbols * 4);

  log_ul_fd_pusch_iq(frame, slot, frame_parms, rel15_ul,
                     number_dmrs_symbols, dmrs_port,
                     (const c16_t *)(&(rxFext_slot_mem[0])),
                     rel15_ul->rb_size * NR_NB_SC_PER_RB * rel15_ul->nr_of_symbols * frame_parms->nb_antennas_rx * 4);

  log_ul_fd_chan_est_dmrs_interpl(frame, slot, frame_parms, rel15_ul,
                                  number_dmrs_symbols, dmrs_port,
                                  (const c16_t *)pusch_ch_est_dmrs_interpl_slot_mem,
                                  rel15_ul->rb_size * NR_NB_SC_PER_RB * rel15_ul->nr_of_symbols
                                      * frame_parms->nb_antennas_rx * rel15_ul->nrOfLayers * 4);

  if (T_ACTIVE(T_GNB_PHY_UL_FD_TRUE_CHANNEL)) {
    fill_rfsim_true_channel_slot(rfsim_true_channel_slot_mem, rel15_ul->rb_size * NR_NB_SC_PER_RB, frame_parms, rel15_ul);
    log_ul_fd_true_channel(frame, slot, frame_parms, rel15_ul,
                           number_dmrs_symbols, dmrs_port,
                           (const c16_t *)rfsim_true_channel_slot_mem,
                           rel15_ul->rb_size * NR_NB_SC_PER_RB * rel15_ul->nr_of_symbols
                               * frame_parms->nb_antennas_rx * rel15_ul->nrOfLayers * 4);
  }
#endif

  join_task_ans(&ans);
  for (int i = 0; i < sz_arr; ++i) {
    // retrieve measurements
    puschSymbolProc_t *rdata = &arr[i];
    merge_meas(&gNB->pusch_extraction_stats, &rdata->pusch_extr);
    merge_meas(&gNB->pusch_channel_compensation_stats, &rdata->pusch_ch_comp);
    merge_meas(&gNB->ulsch_llr_stats, &rdata->ulsch_llr);
    merge_meas(&gNB->ulsch_layer_demapping_stats, &rdata->ul_demap);
    merge_meas(&gNB->ulsch_unscrambling_stats, &rdata->ul_unscram);
  }
  stop_meas(&gNB->rx_pusch_symbol_processing_stats);

  // Copy the data to the scope. This cannot be performed in one call to gNBscopeCopy because the data is not contiguous in the
  // buffer due to reference symbol extraction and padding. The gNBscopeCopy call is broken up into steps: trylock, copy, unlock.
  metadata mt = {.slot = slot, .frame = frame};
  if (gNBTryLockScopeData(gNB, gNBPuschRxIq, sizeof(c16_t), 1, total_res, &mt)) {
    int buffer_length = ceil_mod(rel15_ul->rb_size * NR_NB_SC_PER_RB, 16);
    size_t offset = 0;
    for (uint8_t symbol = rel15_ul->start_symbol_index; symbol < (rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols);
         symbol++) {
      gNBscopeCopyUnsafe(gNB,
                         gNBPuschRxIq,
                         &pusch_vars->rxdataF_comp[0][symbol * buffer_length],
                         sizeof(c16_t) * pusch_vars->ul_valid_re_per_slot[symbol],
                         offset,
                         symbol - rel15_ul->start_symbol_index);
      offset += sizeof(c16_t) * pusch_vars->ul_valid_re_per_slot[symbol];
    }
    gNBunlockScopeData(gNB, gNBPuschRxIq)
  }
  uint32_t total_llrs = total_res * rel15_ul->qam_mod_order * rel15_ul->nrOfLayers;
  gNBscopeCopyWithMetadata(gNB, gNBPuschLlr, pusch_vars->llr, sizeof(c16_t), 1, total_llrs, 0, &mt);
  return 0;
}
