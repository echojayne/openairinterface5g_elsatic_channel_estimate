"""A-MMSE rank-adaptive model definition required for checkpoint inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AMMSERankAdaptiveConfig:
    num_subcarriers: int = 120
    num_symbols: int = 14
    pilot_vector_length: int = 80
    pilot_subcarrier_tokens: int = 40
    pilot_symbol_tokens: int = 2
    d_model: int = 64
    num_heads: int = 4
    frequency_layers: int = 2
    temporal_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    filter_rank: int = 8
    noise_embed_dim: int = 32

    @property
    def input_dim(self) -> int:
        return 2 * self.pilot_vector_length

    @property
    def output_dim(self) -> int:
        return 2 * self.num_subcarriers * self.num_symbols


class TwoStageAttentionEncoder(nn.Module):
    def __init__(self, config: AMMSERankAdaptiveConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if config.pilot_subcarrier_tokens * config.pilot_symbol_tokens != config.pilot_vector_length:
            raise ValueError("pilot token grid must match pilot_vector_length")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.config = config
        self.input_proj = nn.Linear(2, config.d_model)
        self.freq_pos = nn.Parameter(torch.zeros(1, config.pilot_subcarrier_tokens, config.d_model))
        self.time_pos = nn.Parameter(torch.zeros(1, config.pilot_symbol_tokens, config.d_model))
        self.frequency_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.frequency_layers)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=config.temporal_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.max_attention_groups = 4096

    def _encode_chunked(self, encoder: nn.TransformerEncoder, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[0] <= self.max_attention_groups:
            return encoder(tokens)
        outputs = []
        for start in range(0, tokens.shape[0], self.max_attention_groups):
            outputs.append(encoder(tokens[start : start + self.max_attention_groups]))
        return torch.cat(outputs, dim=0)

    def forward(self, pilot_vector: torch.Tensor) -> torch.Tensor:
        batch_size = pilot_vector.shape[0]
        tokens = pilot_vector.transpose(1, 2).contiguous()
        tokens = self.input_proj(tokens)
        tokens = tokens.view(
            batch_size,
            self.config.pilot_symbol_tokens,
            self.config.pilot_subcarrier_tokens,
            self.config.d_model,
        )

        freq_tokens = tokens + self.freq_pos.unsqueeze(1)
        freq_tokens = freq_tokens.view(
            batch_size * self.config.pilot_symbol_tokens,
            self.config.pilot_subcarrier_tokens,
            self.config.d_model,
        )
        freq_tokens = self._encode_chunked(self.frequency_encoder, freq_tokens)
        freq_tokens = freq_tokens.view(
            batch_size,
            self.config.pilot_symbol_tokens,
            self.config.pilot_subcarrier_tokens,
            self.config.d_model,
        )

        time_tokens = freq_tokens.permute(0, 2, 1, 3).contiguous()
        time_tokens = time_tokens + self.time_pos.unsqueeze(1)
        time_tokens = time_tokens.view(
            batch_size * self.config.pilot_subcarrier_tokens,
            self.config.pilot_symbol_tokens,
            self.config.d_model,
        )
        time_tokens = self._encode_chunked(self.temporal_encoder, time_tokens)
        time_tokens = time_tokens.view(
            batch_size,
            self.config.pilot_subcarrier_tokens,
            self.config.pilot_symbol_tokens,
            self.config.d_model,
        )
        time_tokens = time_tokens.permute(0, 2, 1, 3).contiguous()
        return self.output_norm(time_tokens)


class LowRankFilterLayer(nn.Module):
    def __init__(self, config: AMMSERankAdaptiveConfig) -> None:
        super().__init__()
        self.config = config
        self.base_filter = nn.Parameter(torch.empty(config.output_dim, config.input_dim))
        self.base_bias = nn.Parameter(torch.zeros(config.output_dim))
        self.context_to_u = nn.Linear(config.d_model, config.output_dim * config.filter_rank)
        self.context_to_v = nn.Linear(config.d_model, config.filter_rank * config.input_dim)
        nn.init.xavier_uniform_(self.base_filter)
        nn.init.zeros_(self.base_bias)

    def build_filter_factors(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        u = self.context_to_u(context).view(batch_size, self.config.output_dim, self.config.filter_rank)
        v = self.context_to_v(context).view(batch_size, self.config.filter_rank, self.config.input_dim)
        return u, v

    def apply_filter(
        self,
        pilot_vector: torch.Tensor,
        *,
        low_rank_u: torch.Tensor,
        low_rank_v: torch.Tensor,
    ) -> torch.Tensor:
        flat_input = pilot_vector.reshape(pilot_vector.shape[0], -1)
        base_output = F.linear(flat_input, self.base_filter, self.base_bias)
        low_rank_state = torch.bmm(low_rank_v, flat_input.unsqueeze(-1))
        low_rank_output = torch.bmm(low_rank_u, low_rank_state).squeeze(-1)
        return base_output + low_rank_output


class AMMSERankAdaptiveModel(nn.Module):
    """Two-stage attention encoder plus rank-adaptive linear filter synthesis."""

    def __init__(self, config: AMMSERankAdaptiveConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TwoStageAttentionEncoder(config)
        self.noise_embed = nn.Sequential(
            nn.Linear(1, config.noise_embed_dim),
            nn.GELU(),
            nn.Linear(config.noise_embed_dim, config.d_model),
        )
        self.context_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
        )
        self.filter_layer = LowRankFilterLayer(config)
        self.max_representation_dim = config.d_model

    def encode_context(
        self,
        pilot_vector: torch.Tensor,
        *,
        noise_var: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_tokens = self.encoder(pilot_vector)
        context = encoded_tokens.mean(dim=(1, 2))
        if noise_var is not None:
            safe_noise = torch.nan_to_num(noise_var, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            noise_features = self.noise_embed(torch.log1p(safe_noise).unsqueeze(-1))
            context = context + noise_features
        return self.context_head(context)

    def predict_filter(
        self,
        pilot_vector: torch.Tensor,
        *,
        noise_var: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.encode_context(pilot_vector, noise_var=noise_var)
        low_rank_u, low_rank_v = self.filter_layer.build_filter_factors(context)
        return {
            "context": context,
            "base_filter": self.filter_layer.base_filter,
            "base_bias": self.filter_layer.base_bias,
            "low_rank_u": low_rank_u,
            "low_rank_v": low_rank_v,
        }

    def apply_predicted_filter(
        self,
        pilot_vector: torch.Tensor,
        filter_state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        flat_output = self.filter_layer.apply_filter(
            pilot_vector,
            low_rank_u=filter_state["low_rank_u"],
            low_rank_v=filter_state["low_rank_v"],
        )
        return flat_output.view(
            pilot_vector.shape[0],
            2,
            self.config.num_subcarriers,
            self.config.num_symbols,
        )

    def forward(
        self,
        pilot_vector: torch.Tensor,
        *,
        noise_var: torch.Tensor | None = None,
        return_representation: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        filter_state = self.predict_filter(pilot_vector, noise_var=noise_var)
        prediction = self.apply_predicted_filter(pilot_vector, filter_state)
        if not return_representation:
            return prediction
        return prediction, filter_state["context"]
