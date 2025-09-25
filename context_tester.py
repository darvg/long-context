#!/usr/bin/env python3
# context_tester.py — RAG-style RoPE packing + rich diagnostics
# Works with DynamicCache (Transformers >= 4.43). Tested on RTX 3090.
# pip install torch transformers accelerate bitsandbytes

import argparse, math, torch, torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.cache_utils import DynamicCache

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DTYPE = torch.float16
DEVICE = "cuda"

# ------------------------- Utilities -------------------------


def make_pos_ids(start: int, length: int, device=DEVICE):
    return torch.arange(
        start, start + length, device=device, dtype=torch.long
    ).unsqueeze(0)


def load_model(model_id=MODEL_ID, no_quant=False):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if no_quant:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DTYPE, device_map="auto", low_cpu_mem_usage=True
        )
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            quantization_config=bnb_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.eval()
    return tok, model


# ------------------------- Build doc cache (block-causal) -------------------------


def prefill_docs_block_causal(model, tok, docs):
    """
    Feed each doc sequentially; inside a doc tokens only see that doc's past.
    position_ids reset to 0.. for each doc.
    Returns (cache, spans): DynamicCache and list of (start,end) kv indices per doc.
    """
    cache = DynamicCache()
    spans = []
    with torch.no_grad():
        for text in docs:
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
                DEVICE
            )
            L = ids.size(1)
            doc_start = cache.get_seq_length()
            for t in range(L):
                token = ids[:, t : t + 1]
                past_len = cache.get_seq_length()
                # Allow only this doc's past + self
                attn = torch.zeros(1, past_len + 1, dtype=torch.bool, device=DEVICE)
                attn[:, doc_start : past_len + 1] = True
                pos = make_pos_ids(t, 1)
                out = model(
                    input_ids=token,
                    attention_mask=attn,
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
            spans.append((doc_start, cache.get_seq_length()))
    return cache, spans


# ------------------------- Query prefill; inspect 1st-step logits & attentions -------------------------


@torch.no_grad()
def first_step_logits_and_attn(model, tok, docs, query, O=512):
    """
    Build doc cache, prefill the query, then compute:
      - next-token logits (for the first decode step after query)
      - last-layer attention over KV for that step (avg over heads)
      - doc spans (for aggregating attention mass by doc)
    """
    cache, spans = prefill_docs_block_causal(model, tok, docs)
    q_ids = tok(query, add_special_tokens=False, return_tensors="pt").input_ids.to(
        DEVICE
    )
    cur_cache = cache
    # Prefill query tokens with offset O
    for r in range(q_ids.size(1)):
        token = q_ids[:, r : r + 1]
        past_len = cur_cache.get_seq_length()
        attn = torch.ones(1, past_len + 1, dtype=torch.bool, device=DEVICE)
        pos = make_pos_ids(O + r, 1)
        out = model(
            input_ids=token,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=cur_cache,
            use_cache=True,
        )
        cur_cache = out.past_key_values

    # Next step: request attentions
    past_len = cur_cache.get_seq_length()
    attn = torch.ones(1, past_len + 1, dtype=torch.bool, device=DEVICE)
    pos = make_pos_ids(O + q_ids.size(1), 1)
    out = model(
        input_ids=q_ids[:, -1:],  # standard one-step decode input
        attention_mask=attn,
        position_ids=pos,
        past_key_values=cur_cache,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )
    logits = out.logits[:, -1, :].detach()  # [1, vocab]
    # Average last-layer attentions over heads; shape [kv_len]
    last_att = out.attentions[-1]  # [1, n_heads, q_len=1, kv_len]
    att_vec = (
        last_att.mean(dim=1).squeeze(0).squeeze(0).detach().float().cpu()
    )  # [kv_len]
    return logits, att_vec, spans


# ------------------------- Diagnostics -------------------------


def topk_logit_diffs(tok, logits_a, logits_b, k=10):
    diff = (logits_a - logits_b).abs().squeeze(0)
    vals, idx = torch.topk(diff, k)
    tokens = tok.convert_ids_to_tokens(idx.tolist())
    return list(
        zip(
            tokens,
            idx.tolist(),
            vals.tolist(),
            logits_a.squeeze(0)[idx].tolist(),
            logits_b.squeeze(0)[idx].tolist(),
        )
    )


def more_metrics(logits_a, logits_b):
    a = logits_a.squeeze(0)
    b = logits_b.squeeze(0)
    diff = a - b
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    l2 = diff.norm().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    # Softmax distributions (temperature 1)
    pa = F.softmax(a, dim=0)
    pb = F.softmax(b, dim=0)
    # Add small eps to avoid log(0)
    log_pa = torch.log_softmax(a, dim=0)
    log_pb = torch.log_softmax(b, dim=0)
    pa = torch.exp(log_pa)
    pb = torch.exp(log_pb)
    kl_ab = torch.sum(pa * (log_pa - log_pb)).item()
    kl_ba = torch.sum(pb * (log_pb - log_pa)).item()
    js = 0.5 * (kl_ab + kl_ba)
    # Top-1 token check
    top_a = torch.argmax(a).item()
    top_b = torch.argmax(b).item()
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "l2": l2,
        "cosine": cos,
        "kl(a||b)": kl_ab,
        "kl(b||a)": kl_ba,
        "JS": js,
        "argmax_equal": (top_a == top_b),
        "top_a": top_a,
        "top_b": top_b,
    }


def summarize_attn_by_doc(att_vec, spans):
    """
    Sum attention mass per document from a kv attention vector.
    att_vec: [kv_len] over all cached tokens (docs + query prefill)
    spans: list of (start,end) per doc
    Returns list of per-doc sums and a residual (non-doc) sum (should be ~query tokens only).
    """
    masses = []
    for s, e in spans:
        masses.append(att_vec[s:e].sum().item())
    # Residual (query tokens attended during the step)
    q_start = spans[-1][1] if len(spans) > 0 else 0
    residual = att_vec[q_start:].sum().item()
    return masses, residual


# ------------------------- End-to-end test & report -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o", type=int, default=512, help="RoPE offset O for query")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--no-quant", action="store_true", help="Load fp16 weights (no 4-bit quant)"
    )
    args = ap.parse_args()

    set_seed(args.seed)
    print("Loading model…")
    tok, model = load_model(no_quant=args.no_quant)

    # Example inputs
    docs = [
        "Doc A: The Eiffel Tower is in Paris. It was completed in 1889.",
        "Doc B: Mount Everest is the highest mountain on Earth, at 8,849 meters.",
        "Doc C: The Python language emphasizes readability with significant indentation.",
    ]
    query = (
        "Use the context above.\n"
        "Q: Which city has the Eiffel Tower and when was it completed?\n"
        "A:"
    )

    print(
        f"\n— Building caches & collecting diagnostics (O={args.o}, {'fp16' if args.no_quant else '4-bit'} weights)…"
    )

    # Normal order
    logits_a, att_a, spans_a = first_step_logits_and_attn(
        model, tok, docs, query, O=args.o
    )
    # Reversed order
    logits_b, att_b, spans_b = first_step_logits_and_attn(
        model, tok, list(reversed(docs)), query, O=args.o
    )

    # Logit metrics
    m = more_metrics(logits_a, logits_b)
    print("\nLogit deltas (normal vs reversed):")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:>12}: {v:.6e}")
        else:
            print(f"  {k:>12}: {v}")

    # Top-10 changed logits
    print("\nTop-10 tokens by |Δ logit|:")
    for tok_str, tok_id, dval, la, lb in topk_logit_diffs(
        tok, logits_a, logits_b, k=10
    ):
        print(
            f"  id={tok_id:6d}  Δ={dval: .3e}   a={la: .3e}  b={lb: .3e}  token={tok_str!r}"
        )

    # Attention mass per document
    masses_a, resid_a = summarize_attn_by_doc(att_a, spans_a)
    masses_b, resid_b = summarize_attn_by_doc(att_b, spans_b)
    print(
        "\nAttention mass by document (avg over heads, last layer) — first decode step"
    )
    print(
        "  Normal order: ",
        ["{:.4f}".format(x) for x in masses_a],
        f" | residual(query)={resid_a:.4f}",
    )
    print(
        "  Reversed order:",
        ["{:.4f}".format(x) for x in masses_b],
        f"| residual(query)={resid_b:.4f}",
    )
    print("  (These should be very similar up to numerical noise.)")

    def align_attn_to_original_order(att_rev, spans_rev, spans_orig):
        """
        Given attention over KV for the reversed run (att_rev) and its spans,
        sum per doc, then place those sums in the ORIGINAL doc index slots.
        This lets us compare apples-to-apples (content, not order).
        """
        # Sum mass per doc for the reversed run
        rev_masses = [att_rev[s:e].sum().item() for (s, e) in spans_rev]
        # Map: reversed docs [N-1, ..., 0] -> original indices [0, ..., N-1]
        N = len(spans_orig)
        aligned = [0.0] * N
        for k in range(N):  # k-th doc in reversed order corresponds to orig index N-1-k
            aligned[N - 1 - k] = rev_masses[k]
        return aligned

    # After you get att_a, spans_a (normal) and att_b, spans_b (reversed):
    aligned_b = align_attn_to_original_order(att_b, spans_b, spans_a)
    orig_masses = [att_a[s:e].sum().item() for (s, e) in spans_a]
    l1 = sum(abs(x - y) for x, y in zip(orig_masses, aligned_b))
    linf = max(abs(x - y) for x, y in zip(orig_masses, aligned_b))
    print("\nAttention mass per doc (aligned):")
    print("  original order:", ["{:.6f}".format(x) for x in orig_masses])
    print("  aligned(rev)  :", ["{:.6f}".format(x) for x in aligned_b])
    print(f"  L1 diff: {l1:.6e}   Linf diff: {linf:.6e}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
