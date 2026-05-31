"""Elastic Transformer runtime for A-MMSE checkpoint inference.

This is the small inference-only subset of the training-time elastic wrapper.
It keeps the checkpoint key layout compatible with the trained StruJEPA-style
elastic checkpoints while avoiding any dependency on an external source tree.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


STACK_PATHS = ("encoder.frequency_encoder", "encoder.temporal_encoder")


@dataclass(frozen=True)
class ElasticStackMetadata:
    family: str
    total_layers: int
    max_num_heads: int
    max_ffn_dim: int


@dataclass(frozen=True)
class StructureMaskDescriptor:
    width_multiplier: float
    depth_multiplier: float
    total_layers: int
    selected_layer_indices: tuple[int, ...]
    active_num_heads: int
    active_ffn_dim: int


@dataclass
class ForwardResult:
    model_output: Any
    encoder_state: torch.Tensor | None
    structure_mask: StructureMaskDescriptor
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElasticRuntimeState:
    width_multiplier: float
    depth_multiplier: float
    selected_layer_indices: tuple[int, ...]
    active_num_heads: int
    active_ffn_dim: int
    return_encoder_state: bool
    last_encoder_state: torch.Tensor | None = None
    trace_blocks: bool = False
    block_traces: list[dict[str, torch.Tensor | int]] = field(default_factory=list)
    completion_losses: list[torch.Tensor] = field(default_factory=list)


_CURRENT_RUNTIME: ContextVar[ElasticRuntimeState | None] = ContextVar(
    "oai_ammse_elastic_runtime",
    default=None,
)


def get_runtime_state() -> ElasticRuntimeState | None:
    return _CURRENT_RUNTIME.get()


@contextmanager
def elastic_runtime(state: ElasticRuntimeState):
    token = _CURRENT_RUNTIME.set(state)
    try:
        yield state
    finally:
        _CURRENT_RUNTIME.reset(token)


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def resolve_active_heads(*, max_heads: int, width_multiplier: float) -> int:
    return _clamp_int(int(round(max_heads * float(width_multiplier))), 1, max_heads)


def resolve_active_ffn(*, max_ffn_dim: int, width_multiplier: float) -> int:
    return _clamp_int(int(round(max_ffn_dim * float(width_multiplier))), 1, max_ffn_dim)


def resolve_active_layers(*, max_layers: int, depth_multiplier: float) -> int:
    return _clamp_int(int(round(max_layers * float(depth_multiplier))), 1, max_layers)


def select_depth_indices(*, total_layers: int, active_layers: int) -> list[int]:
    if active_layers >= total_layers:
        return list(range(total_layers))
    strategy = os.environ.get("OAI_AMMSE_CE_DEPTH_SELECTION", "").strip().lower()
    if strategy in {"prefix", "sequential", "first"}:
        return list(range(active_layers))
    if strategy not in {"", "uniform", "uniform_tail", "default"}:
        raise ValueError(
            "OAI_AMMSE_CE_DEPTH_SELECTION must be one of: prefix, sequential, first, "
            "uniform, uniform_tail, default"
        )
    return [
        max(0, math.floor((layer_idx + 1) * total_layers / active_layers) - 1)
        for layer_idx in range(active_layers)
    ]


def resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, (nn.ModuleList, nn.Sequential, list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"resolved object at path '{path}' is not an nn.Module")
    return current


def resolve_stack_sequence(target_module: nn.Module) -> nn.Module:
    if isinstance(target_module, (nn.ModuleList, nn.Sequential)):
        return target_module
    if hasattr(target_module, "layers") and isinstance(target_module.layers, (nn.ModuleList, nn.Sequential)):
        return target_module.layers
    if hasattr(target_module, "layer") and isinstance(target_module.layer, (nn.ModuleList, nn.Sequential)):
        return target_module.layer
    raise TypeError(f"unsupported stack container type: {type(target_module)!r}")


def replace_stack_blocks(target_module: nn.Module, blocks: list[nn.Module]) -> tuple[nn.Module, list[nn.Module]]:
    sequence = resolve_stack_sequence(target_module)
    if isinstance(sequence, nn.ModuleList):
        if len(sequence) != len(blocks):
            raise ValueError("replacement module count must match ModuleList length")
        for idx, module in enumerate(blocks):
            sequence[idx] = module
        return target_module, list(sequence)
    if isinstance(sequence, nn.Sequential):
        if len(sequence) != len(blocks):
            raise ValueError("replacement module count must match Sequential length")
        for idx, module in enumerate(blocks):
            sequence[idx] = module
        return target_module, list(sequence)
    raise TypeError(f"unsupported sequence container type: {type(sequence)!r}")


def get_target_module(model: nn.Module, stack_path: str) -> nn.Module:
    target = resolve_module_path(model, stack_path)
    if not isinstance(target, nn.Module):
        raise TypeError(f"target at path '{stack_path}' is not an nn.Module")
    return target


def ensure_batch_first(tokens: torch.Tensor, *, batch_first: bool) -> torch.Tensor:
    if batch_first:
        return tokens
    return tokens.transpose(0, 1).contiguous()


def build_additive_attention_mask(
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    active_heads: int,
    target_len: int,
    source_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    mask: torch.Tensor | None = None
    if attn_mask is not None:
        mask = attn_mask.to(device=device)
        if mask.dtype == torch.bool:
            float_mask = torch.zeros(mask.shape, device=device, dtype=dtype)
            float_mask.masked_fill_(mask, float("-inf"))
            mask = float_mask
        else:
            mask = mask.to(dtype=dtype)
        if mask.ndim == 2:
            mask = mask.view(1, 1, target_len, source_len)
        elif mask.ndim == 3:
            if mask.shape[0] == batch_size * active_heads:
                mask = mask.view(batch_size, active_heads, target_len, source_len)
            elif mask.shape[0] == batch_size:
                mask = mask.view(batch_size, 1, target_len, source_len)
            else:
                raise ValueError(f"unsupported attention mask shape {tuple(mask.shape)}")
        elif mask.ndim != 4:
            raise ValueError(f"unsupported attention mask rank {mask.ndim}")
    if key_padding_mask is not None:
        padding_mask = key_padding_mask.to(device=device)
        if padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(dtype=torch.bool)
        additive = torch.zeros((batch_size, 1, 1, source_len), device=device, dtype=dtype)
        additive.masked_fill_(padding_mask.view(batch_size, 1, 1, source_len), float("-inf"))
        mask = additive if mask is None else mask + additive
    return mask


def apply_activation(module_or_fn: nn.Module | Any, x: torch.Tensor) -> torch.Tensor:
    if isinstance(module_or_fn, nn.Module):
        return module_or_fn(x)
    return module_or_fn(x)


def elastic_torch_mha_forward(
    x: torch.Tensor,
    mha: nn.MultiheadAttention,
    *,
    active_heads: int,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    is_causal: bool,
    batch_first: bool,
) -> torch.Tensor:
    if not mha._qkv_same_embed_dim:
        raise NotImplementedError("separate q/k/v projection weights are not supported")
    if not batch_first:
        x = x.transpose(0, 1)
    batch_size, seq_len, embed_dim = x.shape
    head_dim = mha.head_dim
    active_dim = int(active_heads) * int(head_dim)
    q_weight, k_weight, v_weight = mha.in_proj_weight.split(embed_dim, dim=0)
    if mha.in_proj_bias is not None:
        q_bias, k_bias, v_bias = mha.in_proj_bias.split(embed_dim, dim=0)
    else:
        q_bias = k_bias = v_bias = None

    q = F.linear(x, q_weight[:active_dim, :], None if q_bias is None else q_bias[:active_dim])
    k = F.linear(x, k_weight[:active_dim, :], None if k_bias is None else k_bias[:active_dim])
    v = F.linear(x, v_weight[:active_dim, :], None if v_bias is None else v_bias[:active_dim])

    q = q.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, active_heads, head_dim).transpose(1, 2)

    additive_mask = build_additive_attention_mask(
        attn_mask,
        key_padding_mask,
        batch_size=batch_size,
        active_heads=active_heads,
        target_len=seq_len,
        source_len=seq_len,
        device=x.device,
        dtype=x.dtype,
    )
    context = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=additive_mask,
        dropout_p=mha.dropout if mha.training else 0.0,
        is_causal=is_causal and additive_mask is None,
    )
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, active_dim)
    output = F.linear(context, mha.out_proj.weight[:, :active_dim], mha.out_proj.bias)
    if not batch_first:
        output = output.transpose(0, 1).contiguous()
    return output


def elastic_ffn_forward(
    x: torch.Tensor,
    *,
    fc1: nn.Linear,
    fc2: nn.Linear,
    active_ffn_dim: int,
    activation: nn.Module | Any,
    dropout1: nn.Module | None = None,
) -> torch.Tensor:
    hidden = F.linear(x, fc1.weight[:active_ffn_dim, :], None if fc1.bias is None else fc1.bias[:active_ffn_dim])
    hidden = apply_activation(activation, hidden)
    if dropout1 is not None:
        hidden = dropout1(hidden)
    return F.linear(hidden, fc2.weight[:, :active_ffn_dim], fc2.bias)


class ElasticBlockBase(nn.Module):
    def __init__(self, *, layer_index: int, total_layers: int, max_num_heads: int, max_ffn_dim: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.total_layers = int(total_layers)
        self.max_num_heads = int(max_num_heads)
        self.max_ffn_dim = int(max_ffn_dim)

    def _is_active(self) -> tuple[bool, int, int]:
        runtime = get_runtime_state()
        if runtime is None:
            return True, self.max_num_heads, self.max_ffn_dim
        return (
            self.layer_index in runtime.selected_layer_indices,
            int(runtime.active_num_heads),
            int(runtime.active_ffn_dim),
        )

    def _record_encoder_state(self, tokens: torch.Tensor, *, batch_first: bool = True) -> None:
        runtime = get_runtime_state()
        if runtime is None or not runtime.return_encoder_state:
            return
        runtime.last_encoder_state = ensure_batch_first(tokens, batch_first=batch_first)

    def _record_block_trace(
        self,
        *,
        attention_residual: torch.Tensor,
        ffn_residual: torch.Tensor,
        output: torch.Tensor,
        batch_first: bool = True,
    ) -> None:
        runtime = get_runtime_state()
        if runtime is None or not runtime.trace_blocks:
            return
        runtime.block_traces.append(
            {
                "layer_index": int(self.layer_index),
                "attention_residual": ensure_batch_first(attention_residual, batch_first=batch_first),
                "ffn_residual": ensure_batch_first(ffn_residual, batch_first=batch_first),
                "output": ensure_batch_first(output, batch_first=batch_first),
            }
        )


class ElasticTorchEncoderLayer(ElasticBlockBase):
    """Elastic wrapper for a torch TransformerEncoderLayer."""

    def __init__(self, layer: nn.TransformerEncoderLayer, *, layer_index: int, total_layers: int) -> None:
        if not isinstance(layer.self_attn, nn.MultiheadAttention):
            raise TypeError("torch encoder adapter requires MultiheadAttention self_attn")
        super().__init__(
            layer_index=layer_index,
            total_layers=total_layers,
            max_num_heads=int(layer.self_attn.num_heads),
            max_ffn_dim=int(layer.linear1.out_features),
        )
        self.self_attn = layer.self_attn
        self.linear1 = layer.linear1
        self.dropout = layer.dropout
        self.linear2 = layer.linear2
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.dropout1 = layer.dropout1
        self.dropout2 = layer.dropout2
        self.activation = layer.activation
        self.norm_first = bool(layer.norm_first)
        self.batch_first = bool(layer.self_attn.batch_first)

    def _self_attention(
        self,
        src: torch.Tensor,
        *,
        active_heads: int,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        return elastic_torch_mha_forward(
            src,
            self.self_attn,
            active_heads=active_heads,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            batch_first=self.batch_first,
        )

    def _ffn(self, x: torch.Tensor, *, active_ffn_dim: int) -> torch.Tensor:
        return elastic_ffn_forward(
            x,
            fc1=self.linear1,
            fc2=self.linear2,
            active_ffn_dim=active_ffn_dim,
            activation=self.activation,
            dropout1=self.dropout,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        is_active, active_heads, active_ffn_dim = self._is_active()
        if not is_active:
            return src
        x = src
        if self.norm_first:
            attn_residual = self.dropout1(
                self._self_attention(
                    self.norm1(x),
                    active_heads=active_heads,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            x = x + attn_residual
            ffn_residual = self.dropout2(self._ffn(self.norm2(x), active_ffn_dim=active_ffn_dim))
            x = x + ffn_residual
        else:
            attn_residual = self.dropout1(
                self._self_attention(
                    x,
                    active_heads=active_heads,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            x = self.norm1(x + attn_residual)
            ffn_residual = self.dropout2(self._ffn(x, active_ffn_dim=active_ffn_dim))
            x = self.norm2(x + ffn_residual)
        self._record_block_trace(
            attention_residual=attn_residual,
            ffn_residual=ffn_residual,
            output=x,
            batch_first=self.batch_first,
        )
        self._record_encoder_state(x, batch_first=self.batch_first)
        return x


class MultiStackTorchEncoderWrapper(nn.Module):
    """Route one width/depth subnet descriptor through multiple encoder stacks."""

    def __init__(self, model: nn.Module, *, stack_paths: tuple[str, ...]) -> None:
        super().__init__()
        if not stack_paths:
            raise ValueError("stack_paths must not be empty")
        self.model = model
        self.stack_paths = tuple(str(path) for path in stack_paths)

        stack_layers: list[tuple[str, nn.TransformerEncoderLayer]] = []
        for stack_path in self.stack_paths:
            target = get_target_module(model, stack_path)
            sequence = resolve_stack_sequence(target)
            layers = list(sequence)
            if not layers:
                raise ValueError(f"stack at '{stack_path}' is empty")
            for layer in layers:
                if not isinstance(layer, nn.TransformerEncoderLayer):
                    raise TypeError("MultiStackTorchEncoderWrapper only supports nn.TransformerEncoderLayer stacks")
                stack_layers.append((stack_path, layer))

        total_layers = len(stack_layers)
        global_index = 0
        self.blocks: list[nn.Module] = []
        for stack_path in self.stack_paths:
            target = get_target_module(model, stack_path)
            sequence = resolve_stack_sequence(target)
            elastic_layers: list[nn.Module] = []
            for layer in list(sequence):
                elastic_layers.append(
                    ElasticTorchEncoderLayer(
                        layer,
                        layer_index=global_index,
                        total_layers=total_layers,
                    )
                )
                global_index += 1
            _, replaced = replace_stack_blocks(target, elastic_layers)
            self.blocks.extend(replaced)

        first_block = self.blocks[0]
        self.metadata = ElasticStackMetadata(
            family="torch_encoder_multi",
            total_layers=total_layers,
            max_num_heads=int(first_block.max_num_heads),
            max_ffn_dim=int(first_block.max_ffn_dim),
        )

    def _build_structure_mask(self, *, width_multiplier: float, depth_multiplier: float) -> StructureMaskDescriptor:
        active_heads = resolve_active_heads(
            max_heads=self.metadata.max_num_heads,
            width_multiplier=width_multiplier,
        )
        active_ffn = resolve_active_ffn(
            max_ffn_dim=self.metadata.max_ffn_dim,
            width_multiplier=width_multiplier,
        )
        active_layers = resolve_active_layers(
            max_layers=self.metadata.total_layers,
            depth_multiplier=depth_multiplier,
        )
        selected_layers = tuple(
            select_depth_indices(total_layers=self.metadata.total_layers, active_layers=active_layers)
        )
        return StructureMaskDescriptor(
            width_multiplier=float(width_multiplier),
            depth_multiplier=float(depth_multiplier),
            total_layers=int(self.metadata.total_layers),
            selected_layer_indices=selected_layers,
            active_num_heads=int(active_heads),
            active_ffn_dim=int(active_ffn),
        )

    def forward(
        self,
        *args: Any,
        width_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        return_encoder_state: bool = False,
        completion_module: nn.Module | None = None,
        trace_blocks: bool = False,
        **kwargs: Any,
    ) -> ForwardResult:
        if completion_module is not None:
            raise NotImplementedError("completion_module is not part of the deployment runtime")
        structure_mask = self._build_structure_mask(
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
        )
        runtime = ElasticRuntimeState(
            width_multiplier=float(width_multiplier),
            depth_multiplier=float(depth_multiplier),
            selected_layer_indices=structure_mask.selected_layer_indices,
            active_num_heads=structure_mask.active_num_heads,
            active_ffn_dim=structure_mask.active_ffn_dim,
            return_encoder_state=bool(return_encoder_state),
            trace_blocks=bool(trace_blocks),
        )
        with elastic_runtime(runtime):
            model_output = self.model(*args, **kwargs)
        aux = {"stack_paths": self.stack_paths, "block_family": self.metadata.family}
        aux.update(asdict(self.metadata))
        aux["block_traces"] = runtime.block_traces
        aux["completion_losses"] = runtime.completion_losses
        return ForwardResult(
            model_output=model_output,
            encoder_state=runtime.last_encoder_state if return_encoder_state else None,
            structure_mask=structure_mask,
            aux=aux,
        )


def prediction_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, ForwardResult):
        output = output.model_output
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"expected tensor output, got {type(output)!r}")
    return output


def call_model_for_spec(
    *,
    method: str,
    model: nn.Module,
    pilot_vector: torch.Tensor,
    noise_var: torch.Tensor,
    width: float,
    depth: float,
    model_config: dict[str, Any],
) -> torch.Tensor:
    del model_config
    if method in {"strujepa", "dynabert", "ofa"}:
        result = model(
            pilot_vector,
            width_multiplier=float(width),
            depth_multiplier=float(depth),
            return_encoder_state=False,
            noise_var=noise_var,
        )
        return prediction_from_output(result)
    if method == "static":
        return prediction_from_output(model(pilot_vector, noise_var=noise_var))
    raise ValueError(f"unsupported method {method!r}")
