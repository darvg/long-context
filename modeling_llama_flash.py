# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    LossKwargs,
    logging,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import (
    FlashAttentionKwargs,
    _flash_attention_forward,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    ...


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            start = 0
            if isinstance(past_key_value, Cache):
                start = past_key_value.get_seq_length(self.layer_idx)
            elif past_key_value is not None:
                start = past_key_value[0].shape[-2]
            position_ids = torch.arange(start, start + q_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        else:
            position_ids = position_ids.view(bsz, q_len).to(device=hidden_states.device)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        present_key_value: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]] = None
        if isinstance(past_key_value, Cache):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if use_cache:
                present_key_value = past_key_value
        else:
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            if use_cache:
                present_key_value = (key_states, value_states)

        # Prepare repeated heads for compute but keep original tensors for caching
        key_states_for_attn = repeat_kv(key_states, self.num_key_value_groups)
        value_states_for_attn = repeat_kv(value_states, self.num_key_value_groups)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, : key_states_for_attn.shape[-2]]

        dropout_p = self.config.attention_dropout if self.training else 0.0

        if output_attentions:
            attn_weights = torch.matmul(
                query_states, key_states_for_attn.transpose(2, 3)
            ) * (1.0 / math.sqrt(self.head_dim))
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states_for_attn)
        else:
            attn_output = nn.functional.scaled_dot_product_attention(
                query_states.contiguous(),
                key_states_for_attn.contiguous(),
                value_states_for_attn.contiguous(),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal if attn_mask is None else False,
            )
            attn_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, present_key_value


class LlamaCrossAttention(nn.Module):
    """Multi-headed cross attention from 'Attention Is All You Need' paper"""

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

        # layernorm could go into the decoder layer instead of here, but this is better for FSDP wrapping
        self.layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # the encoder hidden states shouldn't need the positional embedding here.

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

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
        bsz, q_len, _ = hidden_states.size()
        bsz_enc, k_len, _ = encoder_hidden_states.size()
        assert bsz == bsz_enc

        # apply layernorm first
        hidden_states = self.layernorm(hidden_states)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
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

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        flash_query_states = query_states.transpose(1, 2)
        flash_key_states = key_states.transpose(1, 2)
        flash_value_states = value_states.transpose(1, 2)

        dropout_rate = self.config.attention_dropout if self.training else 0.0

        input_dtype = flash_query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            flash_query_states = flash_query_states.to(torch.float16)
            flash_key_states = flash_key_states.to(torch.float16)
            flash_value_states = flash_value_states.to(torch.float16)
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        attn_output = self._flash_attention_forward(
            flash_query_states,
            flash_key_states,
            flash_value_states,
            attention_mask=padding_mask,
            query_length=q_len,
            dropout=dropout_rate,
            is_causal=is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_weights = None

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)

    # def _init_rope(self):
    #     if self.config.rope_scaling is None:
    #         scaling_factor = 1
    #     else:
    #         # we default to linear scaling for now
    #         # TODO: add DynamicNTKScaling?
    #         scaling_factor = self.config.rope_scaling["factor"]

    #     self.rotary_emb = FlashRotaryEmbedding(
    #         self.head_dim,
    #         interleaved=False,
    #         base=self.rope_theta,
    #         scaling_factor=scaling_factor,
    #     )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            start = 0
            if isinstance(past_key_value, Cache):
                start = past_key_value.get_seq_length(self.layer_idx)
            elif past_key_value is not None:
                start = past_key_value[0].shape[-2]
            position_ids = torch.arange(start, start + q_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        else:
            position_ids = position_ids.view(bsz, q_len).to(device=hidden_states.device)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        present_key_value: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]] = None
        if isinstance(past_key_value, Cache):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if use_cache:
                present_key_value = past_key_value
        else:
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            if use_cache:
                present_key_value = (key_states, value_states)

        flash_query_states = query_states.transpose(1, 2)
        flash_key_states = key_states.transpose(1, 2)
        flash_value_states = value_states.transpose(1, 2)

        dropout_rate = self.config.attention_dropout if self.training else 0.0
        input_dtype = flash_query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            flash_query_states = flash_query_states.to(torch.float16)
            flash_key_states = flash_key_states.to(torch.float16)
            flash_value_states = flash_value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            flash_query_states,
            flash_key_states,
            flash_value_states,
            attention_mask=padding_mask,
            query_length=q_len,
            dropout=dropout_rate,
            is_causal=is_causal,
            position_ids=position_ids,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        is_causal=True,
        position_ids: Optional[torch.LongTensor] = None,
        flash_attn_kwargs: Optional[FlashAttentionKwargs] = None,
    ):
        """Dispatch Flash Attention through the central Transformers wrapper."""

        flash_kwargs = flash_attn_kwargs or getattr(self, "flash_attn_kwargs", None) or {}

        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout=dropout,
            softmax_scale=softmax_scale,
            position_ids=position_ids,
            target_dtype=self.o_proj.weight.dtype,
            **flash_kwargs,
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
        # LlamaFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        bsz_enc, k_len, _ = encoder_hidden_states.size()
        assert bsz == bsz_enc

        # apply layernorm first
        hidden_states = self.layernorm(hidden_states)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(encoder_hidden_states).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(encoder_hidden_states).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        flash_query_states = query_states.transpose(1, 2)
        flash_key_states = key_states.transpose(1, 2)
        flash_value_states = value_states.transpose(1, 2)

        dropout_rate = self.config.attention_dropout if self.training else 0.0

        input_dtype = flash_query_states.dtype
        if input_dtype == torch.float32:
            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                " float16."
            )

            flash_query_states = flash_query_states.to(torch.float16)
            flash_key_states = flash_key_states.to(torch.float16)
            flash_value_states = flash_value_states.to(torch.float16)
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        attn_output = self._flash_attention_forward(
            flash_query_states,
            flash_key_states,
            flash_value_states,
            attention_mask=padding_mask,
            query_length=q_len,
            dropout=dropout_rate,
            is_causal=is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        is_causal=True,
        position_ids: Optional[torch.LongTensor] = None,
        flash_attn_kwargs: Optional[FlashAttentionKwargs] = None,
    ):
        """Dispatch Flash Attention through the central Transformers wrapper."""

        flash_kwargs = flash_attn_kwargs or getattr(self, "flash_attn_kwargs", None) or {}

        return _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal,
            dropout=dropout,
            softmax_scale=softmax_scale,
            position_ids=position_ids,
            target_dtype=self.o_proj.weight.dtype,
            **flash_kwargs,
        )

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention(config=config, layer_idx=layer_idx)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config, layer_idx=layer_idx)
        )

        self.do_cross_attention = getattr(config, "do_cross_attention", False)
        if self.do_cross_attention:
            self.cross_attn = (
                LlamaCrossAttention(config=config)
                if not getattr(config, "_flash_attn_2_enabled", False)
                else LlamaCrossFlashAttention2(config=config)
            )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[Tuple[torch.Tensor, torch.Tensor], Cache]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        is_causal: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            is_causal=is_causal,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Cross Attention
        if self.do_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states, cross_attn_weights, encoder_key_values = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=False,
                padding_mask=encoder_padding_mask,
                is_causal=False,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._pjrepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.do_cross_attention = getattr(config, "do_cross_attention", False)
        self.num_cross_attn_layers = getattr(config, "num_cross_attn_layers", 0)
        self.num_cross_attn_hidden_states = getattr(config, "num_cross_attn_hidden_states", 1)
        self.is_decoder = getattr(config, "is_decoder", True)
        self.attention_mask_converter = AttentionMaskConverter(
            is_causal=self.is_decoder,
            sliding_window=getattr(config, "sliding_window", None),
        )

        layer_list = []
        for i in range(config.num_hidden_layers):
            config.do_cross_attention = (i >= config.num_hidden_layers - self.num_cross_attn_layers) and self.do_cross_attention
            layer_list.append(LlamaDecoderLayer(config, layer_idx=i))
        config.do_cross_attention = self.do_cross_attention

        # self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList(layer_list)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        key_value_length = input_shape[-1] + past_key_values_length

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                combined_attention_mask = attention_mask.to(inputs_embeds.dtype)
            else:
                combined_attention_mask = self.attention_mask_converter.to_4d(
                    attention_mask,
                    input_shape[-1],
                    dtype=inputs_embeds.dtype,
                    key_value_length=key_value_length,
                ).to(inputs_embeds.device)
        else:
            combined_attention_mask = None

        if self.is_decoder and combined_attention_mask is None:
            combined_attention_mask = self.attention_mask_converter.to_causal_4d(
                batch_size=input_shape[0],
                query_length=input_shape[-1],
                key_value_length=key_value_length,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if isinstance(past_key_values, Cache):
            past_key_values_length = past_key_values.get_seq_length()
            seq_length_with_past = seq_length_with_past + past_key_values_length
        elif past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                dtype=torch.long,
                device=device,
            )
        else:
            cache_position = cache_position.to(device=device, dtype=torch.long)
            if cache_position.ndim == 2:
                if cache_position.size(0) == 1:
                    cache_position = cache_position.squeeze(0)
                elif cache_position.size(0) == batch_size:
                    if not torch.all(cache_position.eq(cache_position[0])):
                        raise ValueError("cache_position varies across batch; provide a shared position index")
                    cache_position = cache_position[0]
                else:
                    raise ValueError("cache_position has unexpected shape")
            elif cache_position.ndim > 2:
                cache_position = cache_position.view(-1)
            if cache_position.numel() != seq_length:
                raise ValueError("cache_position length does not match sequence length")

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        device = inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)
            padding_mask = None
        else:
            attention_mask = attention_mask.to(device)
            padding_mask = None
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    padding_mask_candidate = attention_mask
                else:
                    if torch.is_floating_point(attention_mask):
                        if torch.any(attention_mask < 0):
                            padding_mask_candidate = attention_mask >= 0
                        else:
                            padding_mask_candidate = attention_mask != 0
                    else:
                        padding_mask_candidate = attention_mask != 0
                padding_mask_candidate = padding_mask_candidate.to(torch.bool)
                if not bool(padding_mask_candidate.all()):
                    padding_mask = padding_mask_candidate
                attention_mask = padding_mask_candidate
            else:
                # Higher dimensional masks are assumed to already be in logits form
                padding_mask = None
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.to(torch.bool)

        encoder_padding_mask = None
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    (batch_size, encoder_hidden_states[0].shape[1]), dtype=torch.bool, device=device
                )
                encoder_padding_mask = None
            else:
                encoder_attention_mask = encoder_attention_mask.to(device)
                if encoder_attention_mask.dtype == torch.bool:
                    encoder_mask_bool = encoder_attention_mask
                else:
                    if torch.is_floating_point(encoder_attention_mask):
                        if torch.any(encoder_attention_mask < 0):
                            encoder_mask_bool = encoder_attention_mask >= 0
                        else:
                            encoder_mask_bool = encoder_attention_mask != 0
                    else:
                        encoder_mask_bool = encoder_attention_mask != 0
                encoder_mask_bool = encoder_mask_bool.to(torch.bool)
                if not bool(encoder_mask_bool.all()):
                    encoder_padding_mask = encoder_mask_bool
                encoder_attention_mask = encoder_mask_bool
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=seq_length).to(device)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        cache_position = cache_position if use_cache or cache_position is not None else None
        position_embeddings = None
        use_cache_object = isinstance(past_key_values, Cache)
        next_decoder_cache = () if use_cache and not use_cache_object else None

        if self.gradient_checkpointing and self.training:
            hidden_states.requires_grad_(True)
            if encoder_hidden_states is not None:
                encoder_hidden_states = [x.requires_grad_(True) for x in encoder_hidden_states]
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
                use_cache_object = False
                next_decoder_cache = None
                cache_position = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if use_cache_object and use_cache:
                past_key_value = past_key_values
            else:
                past_key_value = past_key_values[idx] if past_key_values is not None else None

            encoder_hidden_state = None
            if encoder_hidden_states is not None and decoder_layer.do_cross_attention:
                # if we are not using the last one, then we count from the back
                encoder_hidden_state = encoder_hidden_states[-1] if self.num_cross_attn_hidden_states == 1 else encoder_hidden_states[max(idx-len(self.layers), -len(self.layers))]

            if self.gradient_checkpointing and self.training:

                def custom_forward(*inputs):
                    return decoder_layer(
                        inputs[0],
                        attention_mask=inputs[1],
                        encoder_hidden_states=inputs[2],
                        encoder_attention_mask=inputs[3],
                        encoder_padding_mask=inputs[4],
                        position_ids=inputs[5],
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=False,
                        padding_mask=padding_mask,
                        is_causal=self.is_decoder,
                        cache_position=inputs[6],
                        position_embeddings=inputs[7],
                    )

                layer_outputs = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_state,
                    encoder_attention_mask,
                    encoder_padding_mask,
                    position_ids,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_state,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_padding_mask=encoder_padding_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                    is_causal=self.is_decoder,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache and not use_cache_object:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            next_cache = past_key_values if use_cache_object else next_decoder_cache
        else:
            next_cache = None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaEncoder(LlamaModel):
    pass

class LlamaForMLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        assert not config.is_decoder, "MLM model should not be decoder"

        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size

        # from RoBERTa https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/roberta/modeling_roberta.py#L1117
        self.lm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_act_fn = ACT2FN[config.hidden_act]
        self.lm_layer_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_head.bias = self.lm_bias

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_dense(hidden_states)
        logits = self.lm_act_fn(logits)
        logits = self.lm_layer_norm(logits)
        logits = self.lm_head(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes significant for long sequences or large vocabulary size.

        Returns:

        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states_for_logits = hidden_states[:, -num_logits_to_keep:, :] if num_logits_to_keep else hidden_states

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states_for_logits, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states_for_logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        context_scores=None,
        num_context=None,
        cache_position=None,
        **kwargs,
    ):
        model_inputs: Dict[str, Any] = {}

        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if (
                inputs_embeds is not None
                or cache_position is None
                or (not is_torchdynamo_compiling() and cache_position[-1] >= input_ids.shape[1])
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and (cache_position is None or cache_position[0] == 0):
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        position_ids = kwargs.get("position_ids", None)
        if (
            attention_mask is not None
            and position_ids is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids

        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        if isinstance(past_key_values, Cache) and attention_mask is not None and attention_mask.ndim == 2:
            base_model = getattr(self, self.base_model_prefix, None)
            causal_mask_creation_function = None
            if base_model is not None:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is not None and cache_position is not None:
                if model_inputs.get("inputs_embeds") is not None:
                    batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                    device = model_inputs["inputs_embeds"].device
                else:
                    batch_size, sequence_length = model_inputs[input_ids_key].shape
                    device = model_inputs[input_ids_key].device

                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape() if hasattr(past_key_values, "get_max_cache_shape") else None,
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if context_scores is not None:
            model_inputs["context_scores"] = context_scores
        if num_context is not None:
            model_inputs["num_context"] = num_context

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        model_inputs.pop("labels", None)
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if isinstance(past_key_values, Cache):
            return past_key_values.index_select(beam_idx)
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

@dataclass
class CausalLMOutputWithPastContext(CausalLMOutputWithPast):
    encoder_hidden_states: Optional[List[torch.FloatTensor]] = None

# based on LlamaForCausalLM
class LlamaForCausalContextLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, encoder=None):
        super().__init__(config)

        self.encoder_is_model = getattr(config, "encoder_is_model", False)
        if not config.is_decoder:
            logger.warning_once(
                "The LlamaForContextCausalLM model has config set to False, but setting it to True to get the expected behavior."
            )
            config.is_decoder = True
        if self.encoder_is_model:
            assert encoder is None
            self.encoder = None
        elif encoder is not None:
            self.encoder = encoder
        else:
            encoder_config = (
                LlamaConfig.from_dict(config.encoder_config)
                if isinstance(config.encoder_config, dict)
                else config.encoder_config
            )
            encoder_config._flash_attn_2_enabled = True
            self.encoder = LlamaEncoder(encoder_config)

        self.train_encoder = getattr(config, "train_encoder", False)
        self.lm_loss_cof = getattr(config, "lm_loss_cof", 1.0)
        self.kl_loss_cof = getattr(config, "kl_loss_cof", 1.0)
        self.kl_loss_mode = getattr(config, "kl_loss_mode", "smooth")
        self.encode_mode = getattr(config, "encode_mode", "context_only")
        self.train_batch_mode = getattr(config, "train_batch_mode", "none")
        self.offload_hidden_states = getattr(config, "offload_hidden_states", False)
        self.num_cross_attn_hidden_states = getattr(config, "num_cross_attn_hidden_states", 1)
        self.separate_forward = getattr(config, "separate_forward", False)
        self.lm_eval_mode = getattr(config, "lm_eval_mode", False)

        self.post_init()

    def set_encoder(self, encoder):
        if self.encoder_is_model:
            logger.warning("Encoder is tied to the base model; ignoring external encoder assignment.")
        else:
            self.encoder = encoder

    def get_encoder(self):
        return self.encoder

    def calculate_weighted_logits(self, logits, context_scores, bsz, n_ctx):
        vocab_size = logits.size(-1)
        logits = logits.view(bsz, n_ctx, -1, vocab_size)
        softmax_fn = nn.Softmax(dim=-1)
        temperature = getattr(self, "replug_passage_temperature", 1.0)
        context_scores = (context_scores / temperature).to(logits.device)
        score_prob = softmax_fn(context_scores).view(bsz, n_ctx, 1, 1)
        seq_prob = softmax_fn(logits) * score_prob
        weighted_prob = seq_prob.sum(dim=1, keepdim=True)
        log_prob = torch.log(weighted_prob)
        return log_prob.expand(-1, n_ctx, -1, -1).contiguous().view(bsz * n_ctx, -1, vocab_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        distill_prob: Optional[torch.Tensor] = None,
        distill_index: Optional[torch.Tensor] = None,
        distill_tokens: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        context_scores: Optional[torch.Tensor] = None,
        num_context: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_hidden_states is None and encoder_input_ids is not None:
            bsz, n_ctx, ctx_length = encoder_input_ids.size()

            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones_like(encoder_input_ids, device=encoder_input_ids.device)

            if "with_query" in self.encode_mode:
                query_length = int(self.encode_mode.split("-")[-1])
                assert query_length <= input_ids.size(-1), (
                    f"query length {query_length} is longer than input_ids {input_ids.size(-1)}"
                )
                encoder_input_ids = torch.concatenate(
                    [encoder_input_ids, input_ids[..., :query_length].unsqueeze(1).expand(-1, n_ctx, -1)], dim=2
                )
                if attention_mask is not None:
                    encoder_attention_mask = torch.concatenate(
                        [
                            encoder_attention_mask,
                            attention_mask[..., :query_length].unsqueeze(1).expand(-1, n_ctx, -1),
                        ],
                        dim=2,
                    )
                else:
                    encoder_attention_mask = torch.concatenate(
                        [
                            encoder_attention_mask,
                            torch.ones([bsz, n_ctx, query_length], device=encoder_attention_mask.device),
                        ],
                        dim=2,
                    )
                ctx_length += query_length

            encoder_input_ids = encoder_input_ids.view(-1, ctx_length)
            encoder_attention_mask = encoder_attention_mask.view(-1, ctx_length)

            train_encoder = self.training and self.encoder is not None and self.train_encoder
            with torch.set_grad_enabled(train_encoder):
                if self.encoder_is_model:
                    encoder_outputs = self.model(
                        input_ids=encoder_input_ids,
                        attention_mask=encoder_attention_mask,
                        use_cache=False,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True,
                    )
                    encoder_hidden_states = encoder_outputs.last_hidden_state
                else:
                    encoder_hidden_states = self.encoder(
                        input_ids=encoder_input_ids,
                        attention_mask=encoder_attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True,
                    ).last_hidden_state

            encoder_hidden_states = encoder_hidden_states.view(bsz, n_ctx, ctx_length, -1)

            if self.offload_hidden_states:
                encoder_hidden_states = encoder_hidden_states.cpu()

            if "with_query" in self.encode_mode and attention_mask is not None:
                encoder_attention_mask = attention_mask.view(bsz, n_ctx, -1)

        if encoder_hidden_states is not None and not self.train_encoder:
            encoder_hidden_states = encoder_hidden_states.detach()

        device = None
        if input_ids is not None:
            device = input_ids.device
            total_batch = input_ids.shape[0] if input_ids.dim() < 3 else input_ids.shape[0]
        elif inputs_embeds is not None:
            device = inputs_embeds.device
            total_batch = inputs_embeds.shape[0] if inputs_embeds.dim() < 4 else inputs_embeds.shape[0]
        else:
            total_batch = encoder_input_ids.shape[0] if encoder_input_ids is not None else None

        if context_scores is not None:
            bsz, n_ctx = context_scores.shape
            context_scores = context_scores.to(device) if device is not None else context_scores
        else:
            if num_context is not None:
                n_ctx = num_context
                if total_batch is None:
                    raise ValueError("Unable to infer batch size for default context_scores.")
                if total_batch % n_ctx != 0:
                    raise ValueError("num_context does not evenly divide batch size for causal context decoding.")
                bsz = total_batch // n_ctx
            elif input_ids is not None and input_ids.dim() == 3:
                bsz, n_ctx = input_ids.shape[:2]
            elif inputs_embeds is not None and inputs_embeds.dim() == 4:
                bsz, n_ctx = inputs_embeds.shape[:2]
            else:
                n_ctx = 1
                if total_batch is None:
                    raise ValueError("Unable to infer batch size for default context_scores.")
                bsz = total_batch

            ones_kwargs: Dict[str, Any] = {"dtype": torch.float32}
            if device is not None:
                ones_kwargs["device"] = device
            context_scores = torch.ones((bsz, n_ctx), **ones_kwargs)

        if input_ids is not None and input_ids.dim() == 3:
            if input_ids.shape[0] != bsz or input_ids.shape[1] != n_ctx:
                raise ValueError("input_ids shape does not match context scores")
            input_ids = input_ids.reshape(bsz * n_ctx, -1)
        elif input_ids is not None and input_ids.shape[0] != bsz * n_ctx:
            raise ValueError("input_ids shape does not align with number of contexts")

        if inputs_embeds is not None and inputs_embeds.dim() == 4:
            inputs_embeds = inputs_embeds.view(bsz * n_ctx, inputs_embeds.shape[2], inputs_embeds.shape[3])
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.view(bsz * n_ctx, -1)

        legacy_pkv = None
        if isinstance(past_key_values, Cache):
            legacy_pkv = past_key_values.to_legacy_cache()
        else:
            legacy_pkv = past_key_values

        encoder_hidden_states_out: Optional[torch.Tensor] = encoder_hidden_states

        if self.separate_forward or self.lm_eval_mode:
            all_last_hidden_states: List[torch.Tensor] = []
            all_past_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            all_hidden_states: List[torch.Tensor] = []
            all_attentions: List[torch.Tensor] = []
            all_logits: List[torch.Tensor] = []

            seq_count = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            for i in range(seq_count):
                pkvs = None
                if legacy_pkv is not None:
                    pkvs = tuple((k[i:i+1].to(input_ids.device), v[i:i+1].to(input_ids.device)) for k, v in legacy_pkv)

                outputs = self.model(
                    input_ids=input_ids[i:i+1] if input_ids is not None else None,
                    attention_mask=attention_mask[i:i+1] if attention_mask is not None else None,
                    position_ids=position_ids[i:i+1] if position_ids is not None else None,
                    past_key_values=pkvs,
                    inputs_embeds=inputs_embeds[i:i+1] if inputs_embeds is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    cache_position=cache_position,
                    **kwargs,
                )
                all_last_hidden_states.append(outputs.last_hidden_state)
                if not self.lm_eval_mode:
                    pkv_out = outputs.past_key_values
                    if isinstance(pkv_out, Cache):
                        pkv_out = pkv_out.to_legacy_cache()
                    if pkv_out is not None and len(pkv_out) > 0:
                        all_past_kvs.append(tuple((kv[0].cpu(), kv[1].cpu()) for kv in pkv_out))
                    if output_hidden_states:
                        all_hidden_states.append(outputs.hidden_states.cpu())
                    if output_attentions:
                        all_attentions.append(outputs.attentions.cpu())

                hidden_states = outputs.last_hidden_state
                if self.config.pretraining_tp > 1:
                    lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                    logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                    logits = torch.cat(logits, dim=-1)
                else:
                    logits = self.lm_head(hidden_states)

                all_logits.append(logits.float().cpu())

            if not self.lm_eval_mode:
                logits_cpu = torch.cat(all_logits, dim=0)
                weighted_logits = []
                for i in range(bsz):
                    logits_slice = torch.cat(all_logits[i * n_ctx : (i + 1) * n_ctx], dim=0)
                    logits_slice = self.calculate_weighted_logits(logits_slice, context_scores[i:i+1], 1, n_ctx)
                    weighted_logits.append(logits_slice)
                logits = torch.cat(weighted_logits, dim=0).to(input_ids.device if input_ids is not None else inputs_embeds.device)

                last_hidden_state = torch.cat(all_last_hidden_states, dim=0)

                past_key_values = None
                if all_past_kvs:
                    merged_layers = []
                    num_layers = len(all_past_kvs[0])
                    for layer_idx in range(num_layers):
                        k = torch.cat([kv[layer_idx][0] for kv in all_past_kvs], dim=0)
                        v = torch.cat([kv[layer_idx][1] for kv in all_past_kvs], dim=0)
                        merged_layers.append((k, v))
                    past_key_values = DynamicCache.from_legacy_cache(tuple(merged_layers))

                hidden_states_out = (
                    torch.cat(all_hidden_states, dim=0).to(last_hidden_state.device) if output_hidden_states else None
                )
                attentions_out = (
                    torch.cat(all_attentions, dim=0).to(last_hidden_state.device) if output_attentions else None
                )

                model_outputs = BaseModelOutputWithPast(
                    last_hidden_state=last_hidden_state,
                    past_key_values=past_key_values,
                    hidden_states=hidden_states_out,
                    attentions=attentions_out,
                )
                encoder_hidden_states_out = encoder_hidden_states
            else:
                logits = torch.cat(all_logits, dim=0).to(input_ids.device if input_ids is not None else inputs_embeds.device)
                model_outputs = BaseModelOutputWithPast(
                    last_hidden_state=None,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                )
                encoder_hidden_states_out = encoder_hidden_states
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()
            model_outputs = outputs
            encoder_hidden_states_out = getattr(outputs, "encoder_hidden_states", encoder_hidden_states)

        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:, :]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
            if self.training:
                loss *= self.lm_loss_cof

        kl_loss = None
        if distill_prob is not None and distill_index is not None:
            kl_fct = nn.KLDivLoss(reduction="batchmean")
            prob = F.softmax(logits[..., -distill_prob.size(1) :, :].contiguous(), dim=-1)
            top_prob = torch.gather(prob, 2, distill_index)
            other_prob = 1 - top_prob.sum(-1)
            distill_prob = distill_prob.view(-1, distill_prob.size(2))
            together_prob = torch.cat([top_prob, other_prob.unsqueeze(2)], dim=2).view(-1, distill_prob.size(1))

            if "smooth" in self.kl_loss_mode:
                delta = float(self.kl_loss_mode.split("_")[-1])
                distill_prob = (distill_prob + delta) / (1 + delta * distill_prob.size(1))
                together_prob = (together_prob + delta) / (1 + delta * together_prob.size(1))
            elif self.kl_loss_mode == "drop":
                distill_prob = distill_prob[..., :-1]
                together_prob = together_prob[..., :-1]

            log_prob = torch.log(together_prob)
            kl_loss = self.kl_loss_cof * kl_fct(log_prob, distill_prob)
            loss = loss + kl_loss if loss is not None else kl_loss
            kl_loss = kl_loss.item()

        if not return_dict:
            output = (logits,) + model_outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastContext(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            encoder_hidden_states=encoder_hidden_states_out,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        encoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_hidden_states=None,
        cache_position=None,
        context_scores=None,
        num_context=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            context_scores=context_scores,
            num_context=num_context,
            **kwargs,
        )

        if encoder_input_ids is not None:
            model_inputs["encoder_input_ids"] = encoder_input_ids
        if encoder_attention_mask is not None:
            model_inputs["encoder_attention_mask"] = encoder_attention_mask
        if encoder_hidden_states is not None:
            model_inputs["encoder_hidden_states"] = encoder_hidden_states

        return model_inputs
