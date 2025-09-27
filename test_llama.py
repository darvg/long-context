import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_FORCE_DISABLE_CUDA", "1")

import torch
import pytest
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from modeling_llama_flash import LlamaForCausalContextLM


class DummyContextDecoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided for the dummy decoder")
        batch, seq_length = input_ids.shape
        device = input_ids.device
        hidden_states = torch.arange(
            batch * seq_length * self.hidden_size,
            dtype=torch.float32,
            device=device,
        ).view(batch, seq_length, self.hidden_size)
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states)
        output.encoder_hidden_states = kwargs.get("encoder_hidden_states")
        return output


def build_tiny_context_model():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    config.encoder_is_model = True
    model = LlamaForCausalContextLM(config).eval()
    model.separate_forward = False
    model.lm_eval_mode = False
    model.model = DummyContextDecoder(config.hidden_size)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    model.generation_config.pad_token_id = config.pad_token_id
    model.generation_config.bos_token_id = config.bos_token_id
    model.generation_config.eos_token_id = config.eos_token_id
    return model


@pytest.mark.cpu
def test_multi_context_forward_pass_runs_on_cpu():
    model = build_tiny_context_model()
    device = torch.device("cpu")
    model.to(device)

    input_ids = torch.tensor(
        [
            [
                [1, 3, 4],
                [1, 5, 6],
            ]
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.size(-1), device=device).view(1, 1, -1).expand_as(input_ids)
    encoder_hidden_states = torch.randn(
        1, 2, 3, model.config.hidden_size, device=device
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=False,
    )

    # The dummy decoder emits deterministic hidden states, so logits should be finite and shaped for contexts.
    assert outputs.logits.shape == (input_ids.shape[0] * input_ids.shape[1], input_ids.shape[2], model.config.vocab_size)
    assert torch.isfinite(outputs.logits).all()
