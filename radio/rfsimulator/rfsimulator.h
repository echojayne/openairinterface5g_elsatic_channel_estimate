/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef __RFSIMULATOR_H
#define __RFSIMULATOR_H
#include <stdbool.h>
#include <stdint.h>
#define RFSIMULATOR_CHANNEL_SNAPSHOT_SHM_MAGIC 0x52464353u
#define RFSIMULATOR_CHANNEL_SNAPSHOT_SHM_VERSION 1u
#define RFSIMULATOR_CHANNEL_SNAPSHOT_RING 512
#define RFSIMULATOR_CHANNEL_SNAPSHOT_MAX_TAPS 2048
typedef struct {
  bool valid;
  uint64_t start_timestamp;
  uint64_t end_timestamp;
  int channel_length;
  double sampling_rate;
  double path_loss_dB;
  double max_Doppler;
  double forgetting_factor;
  struct complexd taps[RFSIMULATOR_CHANNEL_SNAPSHOT_MAX_TAPS];
} rfsimulator_channel_snapshot_t;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint64_t write_index;
  rfsimulator_channel_snapshot_t snapshots[RFSIMULATOR_CHANNEL_SNAPSHOT_RING];
} rfsimulator_channel_snapshot_shm_t;

void rxAddInput(c16_t **input_sig, cf_t *after_channel_sig, int rxAnt, channel_desc_t *channelDesc, int nbSamples);
void update_channel_model(channel_desc_t *channelDesc, int nbSamples, uint64_t TS);
#if defined(__GNUC__)
__attribute__((weak))
#endif
bool rfsimulator_get_uplink_channel_snapshot(uint64_t timestamp, rfsimulator_channel_snapshot_t *snapshot);
#endif
