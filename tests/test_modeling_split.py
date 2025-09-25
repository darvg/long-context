import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import modeling_llama_flash as modeling
from masking import DecoderMaskBundle, prepare_decoder_masks
from replug import LlamaForReplugCausalLM
from modeling_llama_flash import (
    KeyValueCache,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.llama.configuration_llama import LlamaConfig


def _tiny_config():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        hidden_act="relu",
    )
    config.use_cache = False
    config._flash_attn_2_enabled = False
    return config


def test_reexports_available():
    exported = {
        "LlamaRMSNorm",
        "LlamaRotaryEmbedding",
        "LlamaAttention",
        "LlamaForCausalLM",
        "CausalLMOutputWithPastContext",
    }
    missing = [name for name in exported if not hasattr(modeling, name)]
    assert not missing, f"Missing exports: {missing}"


def test_llama_rms_norm_matches_manual_computation():
    norm = LlamaRMSNorm(4, eps=1e-6)
    x = torch.randn(3, 4)
    out = norm(x)
    denom = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = x / denom
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_rotary_embedding_shapes():
    rotary = LlamaRotaryEmbedding(8, max_position_embeddings=16)
    dummy = torch.zeros(2, 4, 6, 8)
    cos, sin = rotary(dummy, seq_len=6)
    assert cos.shape == (1, 1, 6, 8)
    assert sin.shape == (1, 1, 6, 8)


def test_llama_model_forward_cpu():
    config = _tiny_config()
    model = LlamaModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.last_hidden_state.shape == (2, 5, config.hidden_size)


def test_llama_for_causal_lm_loss_shapes():
    config = _tiny_config()
    config.is_decoder = True
    model = LlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.logits.shape == (2, 5, config.vocab_size)
    assert outputs.loss is not None

def test_past_key_values_use_dataclass():
    config = _tiny_config()
    config.use_cache = True
    model = LlamaModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    cache_layer = outputs.past_key_values[0]
    assert isinstance(cache_layer, KeyValueCache)
    assert cache_layer.key.shape[-2] == input_ids.shape[1]

    extended_attention_mask = torch.ones((1, input_ids.shape[1] + 1), dtype=torch.long, device=input_ids.device)
    next_outputs = model(
        input_ids=input_ids[:, -1:],
        attention_mask=extended_attention_mask,
        past_key_values=outputs.past_key_values,
        use_cache=True,
    )
    assert isinstance(next_outputs.past_key_values[0], KeyValueCache)

def test_prepare_decoder_masks_bundle_shapes():
    hidden_states = torch.randn(1, 3, 16)
    attention_mask = torch.tensor([[1, 1, 0]])
    bundle = prepare_decoder_masks(
        attention_mask=attention_mask,
        input_shape=(1, 3),
        inputs_embeds=hidden_states,
        past_key_values_length=0,
        is_decoder=True,
    )
    assert isinstance(bundle, DecoderMaskBundle)
    assert bundle.dense is not None and bundle.dense.shape == (1, 1, 3, 3)
    assert bundle.padding is not None and bundle.padding.shape == (1, 3)
    assert bundle.raw.dtype == torch.bool

def test_replug_head_exposes_retrieval_docstring():
    doc = LlamaForReplugCausalLM.__doc__ or ""
    assert "retrieval" in doc.lower()
    assert hasattr(modeling, "LlamaForReplugCausalLM")

