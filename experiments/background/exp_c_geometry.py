"""Experiment C -- are contested values the same object as refusal?

Cheapest of the three, run first, so a structural problem surfaces early.
Inference only. Also carries the power probe that sizes every other probe set.
"""
from __future__ import annotations

import sys

import torch

import controls, gates, geometry as G, harness, models, probes
from harness import Run, Sensitivity

LAYER_FRACS = [i / 20 for i in range(6, 19)]
POWER_NS = (50, 100, 200, 340)


def scan_layers(model, tok, pos, neg, spec):
    layers = sorted({max(1, int(spec.layers * f)) for f in LAYER_FRACS})
    A = G.final_token_states(model, tok, pos, layers)
    B = G.final_token_states(model, tok, neg, layers)
    best = max(layers, key=lambda l: abs(G.cohens_d_oos(A[l], B[l])))
    return best, A[best], B[best]


def run(model_key: str, seed: int = 0, n_refusal: int = 200, n_value: int = 340,
        dry: bool = False) -> str:
    spec = models.REGISTRY[model_key]
    if dry:
        n_refusal, n_value = 24, 24

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=torch.bfloat16).to(
        "mps" if torch.backends.mps.is_available() else "cpu").eval()

    rp = probes.refusal_pairs(n_refusal, seed)
    vp = probes.value_pairs(n_value, seed)
    cfg = {"n_refusal": len(rp), "n_value": len(vp), "dry": dry,
           "refusal_hash": harness.content_hash([p.prompt_pos for p in rp]),
           "value_hash": harness.content_hash([p.prompt_pos for p in vp])}
    r = Run("C_geometry", spec.label, seed, cfg)
    r.provenance(model=models.provenance(spec, model),
                 chat_suffix=probes.chat(tok, "x")[-40:],
                 frozen=controls.freeze_analysis(
                     ["geometry.py", "measure.py", "gates.py", "controls.py"]))

    out = {}
    for kind, pairs in (("refusal", rp), ("value", vp)):
        pos, neg = probes.rendered(tok, pairs)
        layer, A, B = scan_layers(model, tok, pos, neg, spec)
        floor = G.random_floor_deg(A.shape[1], seed=seed)
        sh = G.split_half_angle(A, B, seed)
        d = G.cohens_d_oos(A, B, seed)

        r.sensitivity(Sensitivity(f"{kind}_split_half", sh,
                                  f"n={len(pairs)} at layer {layer}, floor {floor:.1f} deg"))
        r.control(f"{kind}_permutation", controls.permutation_control(A, B, seed))
        r.control(f"{kind}_estimators", controls.estimator_agreement(A, B))

        g1 = r.gate(gates.reproducible_direction(sh, floor))
        g2 = r.gate(gates.separable(d))

        # power probe: the item count this behaviour actually needs
        power = [{"n": n, "split_half_deg": G.split_half_angle(A[:n], B[:n], seed)}
                 for n in POWER_NS if n <= len(A)]
        U, S = G.contrast_subspace(A, B, k=min(12, len(A) - 1))
        out[kind] = {"layer": layer, "v": G.diff_of_means(A, B), "A": A, "B": B,
                     "split_half_deg": sh, "cohens_d_oos": d, "floor": floor,
                     "power": power, "spectrum": G.spectrum_stats(S),
                     "gate_repro": g1, "gate_sep": g2}
        r.cell(kind=kind, layer=layer, split_half_deg=sh, cohens_d_oos=d,
               random_floor_deg=floor, power_curve=power,
               spectrum=out[kind]["spectrum"])

    # the comparison only means something if both directions are real
    both_ok = all(out[k]["gate_repro"] == gates.PASS for k in out)
    if both_ok:
        ang = G.angle_deg(out["refusal"]["v"], out["value"]["v"])
        r.cell(comparison="refusal_vs_value", angle_deg=ang,
               cos=G.cosine(out["refusal"]["v"], out["value"]["v"]),
               interpretation=("independent" if ang > 60 else
                               "overlapping" if ang > 30 else "same object"))
        r.finish("complete", f"refusal vs value = {ang:.1f} deg")
    else:
        fail = [k for k in out if out[k]["gate_repro"] != gates.PASS]
        r.finish("gated", f"not reproducible: {fail}; power curve reports required n")
    return r.doc["status"]


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else models.PRIMARY
    run(key, dry="--dry" in sys.argv)
