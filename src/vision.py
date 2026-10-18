"""Vision-language binding probe.

Measures whether text-only continual training degrades the binding between
the vision encoder and the language model.

Design note (why three conditions, not one):
  ScienceQA image items are not uniformly image-dependent -- many are
  answerable from the text alone. Selecting items the base model gets right
  only-with-image would fix that, but it selects on the outcome and pins
  baseline lift at 1.0 by construction, so any noise reads as erosion.
  Instead we take a FIXED item set (no outcome-based selection) and score it
  three ways:

    correct    - the item's own image
    mismatched - another item's image  (permutation control)
    absent     - no image at all

  binding_lift = acc(correct) - acc(mismatched)

  Chance-level binding means all three coincide. Erosion of the encoder->LM
  binding shows up as binding_lift shrinking while acc(absent) holds, i.e.
  the model reverting to a text-only reader. A drop in all three together is
  general capability loss, not a binding effect -- the control separates them.
"""
from __future__ import annotations
import hashlib, random
import torch

# patch 16, merge 2 -> one visual token per 32x32 px block.
MAX_PIXELS = 262_144          # -> <=256 visual tokens per image
LETTERS = "ABCDE"


def cap_resolution(processor, max_pixels: int = MAX_PIXELS):
    """Bound visual tokens per image. Default processor allows 16.7M px (~16k tokens)."""
    s = processor.image_processor.size
    s["longest_edge"] = max_pixels
    if s.get("shortest_edge") and s["shortest_edge"] > max_pixels:
        s["shortest_edge"] = max_pixels
    return processor


def build_items(n: int = 240, seed: int = 0, split: str = "validation"):
    """Fixed, hashed item set. No selection on model behaviour."""
    from datasets import load_dataset
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    pool = [r for r in ds if r["image"] is not None and 2 <= len(r["choices"]) <= 5]
    rng = random.Random(seed)
    rng.shuffle(pool)
    items = []
    for r in pool[:n]:
        items.append({
            "question": r["question"],
            "hint": r["hint"] or "",
            "choices": r["choices"],
            "answer": r["answer"],
            "image": r["image"].convert("RGB"),
            "subject": r["subject"],
        })
    # permutation control: each item paired with a different item's image
    perm = list(range(len(items)))
    rng.shuffle(perm)
    for i, j in enumerate(perm):
        if j == i:                                   # derangement fixup
            j = (i + 1) % len(items)
        items[i]["mismatch_idx"] = j
    return items


def items_hash(items) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it["question"].encode())
        h.update("|".join(it["choices"]).encode())
        h.update(str(it["answer"]).encode())
    return h.hexdigest()[:16]


def _prompt(it) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
    hint = f"{it['hint']}\n" if it["hint"] else ""
    return (f"{hint}{it['question']}\n{opts}\n"
            f"Answer with the letter only.")


def _letter_ids(processor):
    tok = processor.tokenizer
    ids = []
    for L in LETTERS:
        e = tok.encode(L, add_special_tokens=False)
        assert len(e) == 1, f"letter {L} is not single-token: {e}"
        ids.append(e[0])
    return torch.tensor(ids)


@torch.no_grad()
def score(model, processor, items, condition: str, batch_size: int = 4):
    """DETERMINISM: in bf16 roughly 6% of items are exact top-2 ties, and batched
    matmul reduction order decides them (verified: bs1 vs bs8 disagree 1/16 in
    bf16, 0/16 in fp32, and the flipped items had margin 0.0000). The tie is
    stable for a fixed batch composition, so it cancels under paired
    differencing across checkpoints -- but ONLY if item order and batch_size
    never change within a run. Do not vary either mid-trajectory."""
    # condition in {'correct','mismatched','absent'}. Generation-free: reads
    # renormalised log-odds over the answer letters at the final position.
    assert condition in ("correct", "mismatched", "absent")
    dev = next(model.parameters()).device
    lids = _letter_ids(processor).to(dev)
    n_correct, per_item = 0, []

    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        texts, imgs = [], []
        for i, it in enumerate(chunk):
            content = []
            if condition != "absent":
                content.append({"type": "image"})
                src = it if condition == "correct" else items[it["mismatch_idx"]]
                imgs.append(src["image"])
            content.append({"type": "text", "text": _prompt(it)})
            texts.append(processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False))
        processor.tokenizer.padding_side = "left"   # required by logits_to_keep=1
        kw = dict(text=texts, return_tensors="pt", padding=True)
        if imgs:
            kw["images"] = imgs
        enc = processor(**kw).to(dev)

        # Vocab is 248,320. Full-sequence logits are ~3 GiB at bs8 x seq400 in
        # fp32; we only ever read the final position. 400x less memory.
        tail = model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1]

        for i, it in enumerate(chunk):
            k = len(it["choices"])
            lp = torch.log_softmax(tail[i, lids[:k]].float(), dim=-1)
            pred = int(lp.argmax())
            ok = pred == it["answer"]
            n_correct += ok
            per_item.append({"ok": bool(ok), "pred": pred,
                             "margin": float(lp[it["answer"]] - lp.max())})
    return {"acc": n_correct / len(items), "n": len(items), "per_item": per_item}


def read_all(model, processor, items, batch_size: int = 4):
    out = {c: score(model, processor, items, c, batch_size)
           for c in ("correct", "mismatched", "absent")}
    return {
        "acc_correct":    out["correct"]["acc"],
        "acc_mismatched": out["mismatched"]["acc"],
        "acc_absent":     out["absent"]["acc"],
        "binding_lift":   out["correct"]["acc"] - out["mismatched"]["acc"],
        "text_floor":     out["absent"]["acc"],
        "detail":         {c: out[c]["per_item"] for c in out},
    }


def freeze_vision(model):
    """Text-only training must not move the encoder -- otherwise a change in
    binding is ambiguous between 'the LM drifted' and 'the encoder drifted'.
    Thomson's CPT is text; the vision side is untouched. Mirror that.
    Also removes the tower's grads/momentum from memory."""
    frozen = trainable = 0
    for n, p in model.named_parameters():
        if n.startswith("model.visual.") or ".visual." in n or n.startswith("visual."):
            p.requires_grad_(False); frozen += p.numel()
        else:
            trainable += p.numel()
    assert frozen > 0, "no vision tower found -- check parameter naming"
    return {"frozen": frozen, "trainable": trainable}
