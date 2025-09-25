"""Attention kernels for the flash-optimized LLaMA models."""
from __future__ import annotations

import math
from typing import Optional, Tuple

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
try:
    from transformers.utils import is_flash_attn_available, logging
except ImportError:  # pragma: no cover - fallback for older Transformers
    from transformers.utils import logging  # type: ignore

    def is_flash_attn_available():
        try:
            from transformers.utils import is_flash_attn_2_available  # type: ignore
        except ImportError:
            return False
        return is_flash_attn_2_available()

from normalization import LlamaRMSNorm
from rotary import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

logger = logging.get_logger(__name__)

@dataclass
class KeyValueCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class PackedSequenceInfo:
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


@dataclass
class PackedSequence:
    data: torch.Tensor
    info: PackedSequenceInfo


@dataclass
class AttentionProjections:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    cache: Optional[KeyValueCache]


@dataclass
class FlashAttentionVarlenData:
    query: PackedSequence
    key: PackedSequence
    value: PackedSequence


if is_flash_attn_available():  # pragma: no cover - optional dependency
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # type: ignore
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # type: ignore
else:  # pragma: no cover - optional dependency
    flash_attn_func = None  # type: ignore
    flash_attn_varlen_func = None  # type: ignore
    index_first_axis = None  # type: ignore
    pad_input = None  # type: ignore
    unpad_input = None  # type: ignore


def _get_unpad_data(padding_mask: torch.Tensor) -> PackedSequenceInfo:
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return PackedSequenceInfo(indices=indices, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the query head count."""

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need'."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self) -> None:
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[KeyValueCache],
        position_ids: Optional[torch.LongTensor],
        use_cache: bool,
    ) -> AttentionProjections:
        bsz, q_len, _ = hidden_states.size()

        if past_key_value is not None and not isinstance(past_key_value, KeyValueCache):
            past_key_value = KeyValueCache(*past_key_value)  # type: ignore[arg-type]

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value.key, key_states], dim=2)
            value_states = torch.cat([past_key_value.value, value_states], dim=2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        present_key_value = KeyValueCache(key=key_states, value=value_states) if use_cache else None

        return AttentionProjections(
            query=query_states,
            key=key_states,
            value=value_states,
            cache=present_key_value,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KeyValueCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KeyValueCache]]:
        bsz, q_len, _ = hidden_states.size()

        projections = self._project_qkv(hidden_states, past_key_value, position_ids, use_cache)
        query_states = projections.query
        key_states = projections.key
        value_states = projections.value
        present_key_value = projections.cache

        kv_seq_len = key_states.shape[-2]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


class LlamaCrossAttention(nn.Module):
    """Multi-headed cross attention from 'Attention Is All You Need'."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        input_hidden_size = getattr(config, "encoder_hidden_size", self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(input_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(input_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[KeyValueCache],
        position_ids: Optional[torch.LongTensor],
        use_cache: bool,
    ) -> AttentionProjections:
        bsz, q_len, _ = hidden_states.size()

        if past_key_value is not None and not isinstance(past_key_value, KeyValueCache):
            past_key_value = KeyValueCache(*past_key_value)  # type: ignore[arg-type]

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value.key, key_states], dim=2)
            value_states = torch.cat([past_key_value.value, value_states], dim=2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        present_key_value = KeyValueCache(key=key_states, value=value_states) if use_cache else None

        return AttentionProjections(
            query=query_states,
            key=key_states,
            value=value_states,
            cache=present_key_value,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KeyValueCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KeyValueCache]]:
        bsz, q_len, _ = hidden_states.size()
        bsz_enc, k_len, _ = encoder_hidden_states.size()
        assert bsz == bsz_enc

        hidden_states = self.layernorm(hidden_states)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(encoder_hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(encoder_hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[KeyValueCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KeyValueCache]]:
        if flash_attn_func is None or flash_attn_varlen_func is None:
            raise ImportError("flash_attn is required to use LlamaFlashAttention2")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        projections = self._project_qkv(hidden_states, past_key_value, position_ids, use_cache)
        query_states = projections.query.transpose(1, 2)
        key_states = projections.key.transpose(1, 2)
        value_states = projections.value.transpose(1, 2)
        present_key_value = projections.cache

        dropout_rate = 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, padding_mask, q_len, dropout=dropout_rate, is_causal=is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        is_causal=True,
    ):
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            packed = self._upad_input(query_states, key_states, value_states, padding_mask, query_length)

            attn_output_unpad = flash_attn_varlen_func(
                packed.query.data,
                packed.key.data,
                packed.value.data,
                cu_seqlens_q=packed.query.info.cu_seqlens,
                cu_seqlens_k=packed.key.info.cu_seqlens,
                max_seqlen_q=packed.query.info.max_seqlen,
                max_seqlen_k=packed.key.info.max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=is_causal,
            )

            attn_output = pad_input(attn_output_unpad, packed.query.info.indices, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=is_causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length) -> FlashAttentionVarlenData:
        key_info = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), key_info.indices)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), key_info.indices)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), key_info.indices
            )
            query_info = PackedSequenceInfo(
                indices=key_info.indices,
                cu_seqlens=key_info.cu_seqlens,
                max_seqlen=key_info.max_seqlen,
            )
        elif query_length == 1:
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
            query_info = PackedSequenceInfo(indices=indices_q, cu_seqlens=cu_seqlens_q, max_seqlen=1)
        else:
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)
            query_info = PackedSequenceInfo(
                indices=indices_q,
                cu_seqlens=cu_seqlens_q,
                max_seqlen=max_seqlen_in_batch_q,
            )

        return FlashAttentionVarlenData(
            query=PackedSequence(data=query_layer, info=query_info),
            key=PackedSequence(data=key_layer, info=key_info),
            value=PackedSequence(data=value_layer, info=key_info),
        )


class LlamaCrossFlashAttention2(LlamaCrossAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if flash_attn_func is None or flash_attn_varlen_func is None:
            raise ImportError("flash_attn is required to use LlamaCrossFlashAttention2")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        bsz_enc, k_len, _ = encoder_hidden_states.size()
        assert bsz == bsz_enc

        hidden_states = self.layernorm(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        present_key_value = KeyValueCache(key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, padding_mask, q_len, dropout=dropout_rate, is_causal=is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        padding_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        is_causal=True,
    ):
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            packed = self._upad_input(query_states, key_states, value_states, padding_mask, query_length)

            attn_output_unpad = flash_attn_varlen_func(
                packed.query.data,
                packed.key.data,
                packed.value.data,
                cu_seqlens_q=packed.query.info.cu_seqlens,
                cu_seqlens_k=packed.key.info.cu_seqlens,
                max_seqlen_q=packed.query.info.max_seqlen,
                max_seqlen_k=packed.key.info.max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=is_causal,
            )

            attn_output = pad_input(attn_output_unpad, packed.query.info.indices, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=is_causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length) -> FlashAttentionVarlenData:
        key_info = _get_unpad_data(padding_mask)
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), key_info.indices)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), key_info.indices)

        padding_mask = torch.ones((batch_size, query_length), device=query_layer.device)
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, padding_mask)
        query_info = PackedSequenceInfo(indices=indices_q, cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_in_batch_q)

        return FlashAttentionVarlenData(
            query=PackedSequence(data=query_layer, info=query_info),
            key=PackedSequence(data=key_layer, info=key_info),
            value=PackedSequence(data=value_layer, info=key_info),
        )
