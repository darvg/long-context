import torch
from transformers import LlamaTokenizer
from modeling_llama_flash import LlamaForCausalContextLM

device = "cuda"
model_name = "hyen/CEPED-LLaMA-2-Chat-7B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalContextLM.from_pretrained(
  model_name,
  use_flash_attention_2="flash_attention_2", 
  torch_dtype=torch.bfloat16,
  device_map="auto",
).eval()

contexts = tokenizer([
    "My friends and I enjoy eating out at restaurants together. However, we also enjoy cooking and making food as a group as well."
    "Many of my friends like to play soccer and volleyball. We also enjoy watching movies and going to museums and galleries.",
], return_tensors="pt", padding=True)

inputs = tokenizer("Question: what are three ideas for a social with a large groups of friends in New York City.\nAnswer:", return_tensors="pt")

# encoder_input_ids and encoder_attention_mask should be in the shape of (bsz, n_ctx, seq_length)
output = model.generate(
  input_ids=inputs.input_ids.to(device), 
  attention_mask=inputs.attention_mask.to(device), 
  encoder_input_ids=contexts.input_ids.unsqueeze(0).to(device),
  encoder_attention_mask=contexts.attention_mask.unsqueeze(0).to(device), 
  max_new_tokens=200,
  #sample=True,
  top_p=0.95,
)
print(tokenizer.batch_decode(output)[0])
