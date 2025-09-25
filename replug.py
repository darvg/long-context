"""RePlug retrieval head for LLaMA.

This module isolates the retrieval-augmented language modelling head so its
behaviour and knobs are easier to understand.  The RePlug pipeline blends base
LM logits with evidence from retrieved passages: the passages are scored,
renormalised by ``config.replug_passage_temperature`` and optionally combined in
an auxiliary forward pass when ``config.replug_separate_forward`` is enabled.
Keeping the head in its own module documents those assumptions explicitly and
prepares the code for further retrieval-specific extensions (e.g. different
fusion strategies or passage weighting heuristics).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

from attention import KeyValueCache
from heads import LlamaForCausalLM
from modeling_llama_flash import (
    LLAMA_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
)


class LlamaForReplugCausalLM(LlamaForCausalLM):
    """Causal LM head that optionally mixes retrieved passage evidence.

    The head keeps the standard decoder architecture but exposes two
    retrieval-specific configuration knobs:

    * ``config.replug_passage_temperature`` rescales passage scores before they
      are blended with base logits.
    * ``config.replug_separate_forward`` toggles whether the model performs a
      dedicated forward pass for retrieved passages prior to fusion.
    """

    def __init__(self, config, return_logits: bool = False):
        super().__init__(config)
        self.return_logits = return_logits
        self.replug_passage_temperature = getattr(config, "replug_passage_temperature", 1.0)
        self.replug_separate_forward = getattr(config, "replug_separate_forward", False)

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[KeyValueCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        context_inputs: Optional[Tuple[torch.LongTensor, torch.LongTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""Run the base decoder and expose logits for downstream retrieval weighting.

        Args:
            input_ids: Token ids for the base decoder pass.
            past_key_values: Optional [`KeyValueCache`] entries from a previous step.

        Returns:
            A tuple mirroring :meth:`LlamaForCausalLM.forward`, with logits optionally
            included so calling code can mix in retrieval scores.
        """

        decoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        loss = decoder_outputs.loss
        logits = decoder_outputs.logits if self.return_logits else None

        if return_dict:
            return ModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=decoder_outputs.hidden_states,
                attentions=decoder_outputs.attentions,
                past_key_values=decoder_outputs.past_key_values,
            )

        output = (logits,) + decoder_outputs[1:]
        return (loss,) + output if loss is not None else output


__all__ = ["LlamaForReplugCausalLM"]
