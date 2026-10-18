"""Model registry with architecture-aware module resolution.

Written because v1 hard-coded `self_attn.o_proj` as a residual writer. That
module does not exist on Qwen3.5, which uses gated delta-net linear attention
with `linear_attn.out_proj` instead. The edit would have silently targeted half
as many modules as intended, on two of the three models, and nothing would have
raised an error -- `target_modules` would simply have failed to find it.

Everything here was read off the loaded models, not taken from documentation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# A residual writer is the LAST linear of its sub-block. Selecting by
# `out_features == hidden_size` instead also matches self_attn.q_proj and
# linear_attn.in_proj_z, which are INPUT projections -- both false positives
# were observed on real models. Editing an input projection is a different
# intervention from the one under study, so the match is on name.
WRITER_SUFFIXES = ("down_proj", "o_proj", "out_proj")


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    label: str
    params_b: float
    layers: int
    hidden: int
    attention: str          # "softmax" | "gated_delta_net"
    generation: str         # "3" | "3.5"
    notes: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "qwen35_2b": ModelSpec(
        "Qwen/Qwen3.5-2B", "Qwen3.5-2B", 2.0, 24, 2048,
        "gated_delta_net", "3.5",
        "primary: same generation and attention mechanism as Thomson's 397B-A17B base",
    ),
    "qwen35_08b": ModelSpec(
        "Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B", 0.75, 24, 1024,
        "gated_delta_net", "3.5",
        "scale arm, and the only model cheap enough for the training experiment",
    ),
    "qwen3_17b": ModelSpec(
        "Qwen/Qwen3-1.7B", "Qwen3-1.7B", 1.7, 28, 2048,
        "softmax", "3",
        "prior generation AND prior attention mechanism -- the two are confounded",
    ),
}

INFERENCE_SET = ("qwen35_2b", "qwen35_08b", "qwen3_17b")
TRAINING_MODEL = "qwen35_08b"
PRIMARY = "qwen35_2b"


def decoder_layers(model) -> nn.ModuleList:
    """The decoder block list, whatever the wrapper calls it.

    Qwen3.5 configs declare ForConditionalGeneration and carry a vision_config,
    but AutoModelForCausalLM returns a text-only Qwen3_5ForCausalLM whose blocks
    still live at model.model.layers. Probing rather than assuming costs one
    attribute lookup and removes a whole class of silent failure.
    """
    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 1:
            return obj
    raise RuntimeError(f"cannot locate decoder layers on {type(model).__name__}")


def residual_writers(model, layer_idx: int) -> dict[str, nn.Linear]:
    """{qualified_name: module} for every linear writing into the residual stream."""
    block = decoder_layers(model)[layer_idx]
    out: dict[str, nn.Linear] = {}
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in WRITER_SUFFIXES:
            out[f"{layer_idx}.{name}"] = mod
    if not out:
        raise RuntimeError(f"no residual writers found in layer {layer_idx}")
    return out


def target_modules(model, layers) -> dict[str, nn.Linear]:
    out: dict[str, nn.Linear] = {}
    for l in layers:
        out.update(residual_writers(model, l))
    return out


def edit_band(spec: ModelSpec, frac_lo: float = 0.55, frac_hi: float = 0.85) -> list[int]:
    """Mid-to-late band, as a fraction of depth.

    Absolute indices do not transfer: v1's band of 12-20 is mid-stack on a
    28-layer model and late-stack on a 24-layer one. Fractions keep the band
    comparable across models of different depth, which the cross-model
    comparison requires.
    """
    lo, hi = int(spec.layers * frac_lo), int(spec.layers * frac_hi)
    return list(range(lo, max(hi, lo + 1)))


# Measured on this machine, 2B on MPS: bf16 is 1.71x faster than fp32 with no
# measurable cost to the geometry (d_oos 7.32 -> 7.27, split-half identical at
# 9.3 deg). Safe because every precision-sensitive step -- the SVD, the Fisher
# routing, the v^T b constraint check -- already runs on CPU in float64
# regardless of the model's dtype. Only forward passes change.
INFERENCE_DTYPE = "bfloat16"
EVAL_BATCH = 16          # benchmarked: 11.6 prompts/s vs 9.7 at 8, worse at 32+


def provenance(spec: ModelSpec, model=None) -> dict:
    """Recorded in every output. Two of three v1 runs did not record their model."""
    import transformers
    p = {
        "hf_id": spec.hf_id, "label": spec.label, "generation": spec.generation,
        "attention": spec.attention, "declared_layers": spec.layers,
        "declared_hidden": spec.hidden, "notes": spec.notes,
        "transformers": transformers.__version__, "torch": torch.__version__,
    }
    if model is not None:
        p["measured_params"] = sum(x.numel() for x in model.parameters())
        p["measured_layers"] = len(decoder_layers(model))
        p["writers_per_layer"] = sorted(
            n.split(".", 1)[1] for n in residual_writers(model, 0)
        )
    return p
