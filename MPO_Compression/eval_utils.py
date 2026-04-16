#!/usr/bin/env python3
"""
Evaluation utilities for MPO compression experiments.
Contains functions for perplexity evaluation and text generation.
"""

import torch


def eval_ppl(model, tokenizer, max_len=2048, stride=512, max_tokens=50000):
    """Evaluate perplexity on Wikitext-2 test."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    effective_max = min(max_len, max_ctx)
    seq_len = min(input_ids.size(1), max_tokens)
    input_ids = input_ids[:, :seq_len]

    nlls = []
    for begin in range(0, seq_len, stride):
        end = min(begin + effective_max, seq_len)
        chunk = input_ids[:, begin:end]
        if chunk.size(1) <= 1:
            continue
        with torch.no_grad():
            out = model(chunk)
            lm_logits = out.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
        nlls.append(loss.item())
        if end >= seq_len:
            break

    if not nlls:
        return float("nan")
    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text from prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)
