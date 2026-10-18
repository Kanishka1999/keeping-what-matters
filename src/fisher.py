"""Diagonal Fisher information in a component's OUTPUT space.

The quantity Eq. 4 routes against. Two details that are easy to get wrong and
that change the answer:

1. OUTPUT space, not weight space. F_i is the sensitivity of the predictive
   distribution to coordinate i of the component's output activation. A
   weight-space Fisher is a different object and routing against it does not do
   what the optimality argument claims.

2. One position per sequence. Eq. 3 omits token indices, but under causal
   attention z at position t influences every prediction from t onward, so
   backpropagating a sum over positions and squaring the accumulated gradient
   silently includes cross terms. Sampling a single position keeps each
   gradient a genuine per-example gradient.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def cache_key(model_label: str, module_names: list[str], texts: list[str], seed: int) -> str:
    blob = json.dumps({"m": model_label, "mods": sorted(module_names),
                       "n": len(texts), "t": texts[:3], "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def estimate(model, tok, texts, modules: dict, batch_size: int = 8,
             seq_len: int = 192, seed: int = 0, model_label: str = "",
             use_cache: bool = True, progress=print) -> dict:
    """{module_label: [d_out]}. Cached -- F does not depend on the behaviour edited."""
    key = cache_key(model_label, list(modules), texts, seed)
    path = CACHE / f"fisher_{key}.pt"
    if use_cache and path.exists():
        progress(f"    fisher: cache hit {key}")
        return torch.load(path, map_location="cpu")

    g = torch.Generator().manual_seed(seed)
    acc = {k: torch.zeros(m.out_features, dtype=torch.float64) for k, m in modules.items()}
    seen = 0
    captured, handles = {}, []

    def mk(label):
        def hook(mod, inp, out):
            out.retain_grad()
            captured[label] = out
            return out
        return hook

    for label, mod in modules.items():
        handles.append(mod.register_forward_hook(mk(label)))

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prev, tok.padding_side = tok.padding_side, "right"
    model.eval()
    try:
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=seq_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            mask = enc["attention_mask"].bool()
            model.zero_grad(set_to_none=True)
            captured.clear()
            logits = model(**enc, use_cache=False).logits.float()

            pos = []
            for b in range(logits.shape[0]):
                valid = mask[b].nonzero().flatten()
                valid = valid[valid < logits.shape[1] - 1]
                pos.append(valid[torch.randint(len(valid), (1,), generator=g)].item()
                           if len(valid) else 0)
            pos_t = torch.tensor(pos, device=logits.device)
            rows = torch.arange(logits.shape[0], device=logits.device)

            lp = torch.log_softmax(logits[rows, pos_t], dim=-1)
            sampled = torch.multinomial(lp.exp(), 1).squeeze(-1)   # a ~ p_theta
            lp[rows, sampled].sum().backward()

            for label, t in captured.items():
                if t.grad is not None:
                    # .cpu() before .double(): MPS has no float64 at all
                    acc[label] += t.grad[rows, pos_t].detach().cpu().double().pow(2).sum(0)
            seen += logits.shape[0]
            if seen % (batch_size * 4) < batch_size:
                progress(f"    fisher {seen}/{len(texts)}")
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = prev
        model.zero_grad(set_to_none=True)

    F = {k: (v / max(seen, 1)).float() for k, v in acc.items()}
    if use_cache:
        torch.save(F, path)
    return F
