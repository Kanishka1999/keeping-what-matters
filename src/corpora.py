"""Streaming text sources for the perturbation experiment.

Both verified to load. v1 crashed 100 minutes in on a script-based dataset that
current `datasets` refuses to execute -- a risk flagged in writing beforehand
and then not checked. Anything added here gets loaded once before use.
"""
from __future__ import annotations

import torch
from datasets import load_dataset

SOURCES = {
    # burn-in: instruction/chat text, close to the model's post-training distribution
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages"),
    # out-of-domain: Python source. A far larger shift from chat than prose would
    # be, so the detector gets an unambiguous signal on its first real test, and
    # coding is the one capability Thomson measurably lost.
    "code": ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
}


def stream_text(kind: str):
    name, cfg, split, field = SOURCES[kind]
    ds = load_dataset(name, cfg, split=split, streaming=True)
    for row in ds:
        val = row.get(field)
        if field == "messages":
            val = val[0]["content"] if val else None
        if val and len(val) > 200:
            yield val


def blocks(tok, kind: str, seq_len: int, micro_batch: int):
    """Packed fixed-length blocks, identical packing for both sources."""
    buf, eos = [], tok.eos_token_id
    for text in stream_text(kind):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= seq_len * micro_batch:
            take = buf[: seq_len * micro_batch]
            buf = buf[seq_len * micro_batch:]
            yield torch.tensor(take).view(micro_batch, seq_len)


def verify() -> dict:
    """Load one block from each source. Run before any training starts."""
    out = {}
    for kind in SOURCES:
        try:
            txt = next(stream_text(kind))
            out[kind] = {"ok": True, "sample": txt[:60].replace("\n", " ")}
        except Exception as e:
            out[kind] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:90]}"}
    return out
