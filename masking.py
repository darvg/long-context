"""Shared mask preparation helpers for flash-optimized LLaMA attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class AttentionMaskBundle:
    dense: Optional[torch.Tensor]
    padding: Optional[torch.Tensor]


@dataclass
class DecoderMaskBundle(AttentionMaskBundle):
    raw: torch.Tensor


def _to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    return mask != 0


def make_causal_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    tgt_len: int,
    device: torch.device,
) -> AttentionMaskBundle:
    if attention_mask is None:
        return AttentionMaskBundle(dense=None, padding=None)

    bool_mask = _to_bool_mask(attention_mask.to(device))
    padding = None if bool_mask.all() else bool_mask
    dense = expand_mask(bool_mask, dtype, tgt_len=tgt_len).to(device)
    return AttentionMaskBundle(dense=dense, padding=padding)


def prepare_decoder_masks(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    is_decoder: bool,
) -> DecoderMaskBundle:
    batch_size, seq_len = input_shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    seq_length_with_past = seq_len + past_key_values_length

    if attention_mask is None:
        raw_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=device)
        padding_mask = None
    else:
        raw_mask = _to_bool_mask(attention_mask.to(device))
        padding_mask = None if raw_mask.all() else raw_mask

    dense_mask: Optional[torch.Tensor] = None
    if is_decoder and seq_len > 1:
        dense_mask = make_causal_mask(input_shape, dtype, device, past_key_values_length)

    if raw_mask is not None:
        expanded = expand_mask(raw_mask, dtype, tgt_len=seq_len).to(device)
        dense_mask = expanded if dense_mask is None else expanded + dense_mask

    return DecoderMaskBundle(dense=dense_mask, padding=padding_mask, raw=raw_mask)
