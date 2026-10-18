"""Experiment A -- what does each additional stripped direction cost an attacker?

Tests the assumption every ablation-resistance defence rests on: that spreading
a behaviour across more directions raises the price of removing it.

Targets REFUSAL, not sycophancy. v1 measured an emergent behaviour when the
hypothesis concerns deliberately installed safety training, and then reported
the mismatch as a negative result.
"""
from __future__ import annotations

import sys

import torch

import controls, edits, fisher, gates, geometry as G, harness, measure as M, models, probes
from harness import Run, Sensitivity

K_MAX = 12
EVAL_BS = 16
LAMBDA = 1.0
N_FISHER = 48
N_KL = 96


def _load(spec):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=torch.bfloat16).to(dev).eval()
    return model, tok


def run(model_key: str, seed: int = 0, n_refusal: int = 200, dry: bool = False) -> str:
    spec = models.REGISTRY[model_key]
    k_max, n_fisher = (4, 8) if dry else (K_MAX, N_FISHER)
    if dry:
        n_refusal = 24

    model, tok = _load(spec)
    pairs = probes.refusal_pairs(n_refusal, seed)
    pos, neg = probes.rendered(tok, pairs)
    band = models.edit_band(spec)
    cfg = {"n_refusal": len(pairs), "k_max": k_max, "lambda": LAMBDA, "band": band,
           "dry": dry, "refusal_hash": harness.content_hash([p.prompt_pos for p in pairs])}
    r = Run("A_rank_cost", spec.label, seed, cfg)
    r.provenance(model=models.provenance(spec, model),
                 chat_suffix=probes.chat(tok, "x")[-40:],
                 frozen=controls.freeze_analysis(
                     ["geometry.py", "measure.py", "gates.py", "edits.py", "controls.py"]))

    # --- direction, and the universal reproducibility gate -----------------
    layers = sorted({int(spec.layers * f) for f in (0.4, 0.5, 0.6, 0.7, 0.8)})
    A_all = G.final_token_states(model, tok, pos, layers)
    B_all = G.final_token_states(model, tok, neg, layers)
    layer = max(layers, key=lambda l: abs(G.cohens_d_oos(A_all[l], B_all[l], seed)))
    A, B = A_all[layer], B_all[layer]

    floor = G.random_floor_deg(A.shape[1], seed=seed)
    sh, d_oos = G.split_half_angle(A, B, seed), G.cohens_d_oos(A, B, seed)
    r.control("permutation", controls.permutation_control(A, B, seed))
    r.control("estimators", controls.estimator_agreement(A, B))
    r.sensitivity(Sensitivity("split_half_deg", sh, f"n={len(pairs)}, floor {floor:.1f} deg"))

    if r.gate(gates.reproducible_direction(sh, floor)) != gates.PASS:
        r.finish("gated", "refusal direction not reproducible -- subspace would be noise")
        return "gated"
    if r.gate(gates.separable(d_oos)) != gates.PASS:
        r.finish("gated", "insufficient separability")
        return "gated"

    # --- can the capability ruler resolve anything? ------------------------
    kl_texts = [probes.chat(tok, p.prompt_neg) for p in pairs[:N_KL]]
    ref_lp = M.final_logprobs(model, tok, kl_texts, EVAL_BS)

    rt, ct = M.refusal_tokens(tok)
    base = M.bootstrap(M.refusal_margin(model, tok, pos, rt, ct, EVAL_BS))
    r.cell(stage="baseline", refusal_margin=base)

    # trial edit at k=1 gates both the ruler and the intervention strength
    v0 = G.diff_of_means(A, B)
    mods = models.target_modules(model, band)
    snap = edits.snapshot(mods)
    edits.install(mods, v0, LAMBDA, "standard")
    trial = M.bootstrap(M.refusal_margin(model, tok, pos, rt, ct, EVAL_BS))
    trial = trial  # keep name
    res = M.kl_resolvable(ref_lp, M.final_logprobs(model, tok, kl_texts, EVAL_BS), seed)
    edits.restore(mods, snap)
    r.sensitivity(Sensitivity("min_resolvable_KL", 3 * res["half_width"],
                              "3x the bootstrap half-width of the mean-KL estimator"))

    # Qwen3-1.7B gated here at both seeds: KL genuinely cannot resolve the edit on
    # that model. Rather than abandon it, fall back to a behavioural axis that does
    # not depend on KL headroom -- the refusal margin itself, which is measured
    # per item and so carries a bootstrap interval.
    kl_usable = r.gate(gates.ruler_headroom(res["mean"], res["half_width"], factor=3.0)) == gates.PASS
    r.cell(stage="capability_axis", kl_usable=kl_usable,
           axis="kl" if kl_usable else "refusal_margin_ci",
           note=("KL resolves the edit" if kl_usable else
                 "KL blind on this model; using the behavioural axis with bootstrap CIs"))
    if r.gate(gates.behaviour_moved(trial["mean"] - base["mean"], base["mean"])) != gates.PASS:
        r.finish("gated", "edit too weak at lambda=1; widen the band")
        return "gated"

    # --- Fisher, cached ----------------------------------------------------
    ref_texts = [probes.chat(tok, p.prompt_neg) for p in pairs[:n_fisher]]
    F = fisher.estimate(model, tok, ref_texts, mods, seed=seed, model_label=spec.label)

    # --- subspace and the sweep -------------------------------------------
    U, S = G.contrast_subspace(A, B, k=k_max)
    r.cell(stage="subspace", spectrum=G.spectrum_stats(S),
           cos_u_v0=[G.cosine(U[:, j], v0) for j in range(U.shape[1])])

    for mode in ("standard", "fisher"):
        edits.restore(mods, snap)
        for k in range(1, k_max + 1):
            u = U[:, k - 1]
            edits.install(mods, u, LAMBDA, mode, fisher=F)
            marg = M.refusal_margin(model, tok, pos, rt, ct, EVAL_BS)
            ci = M.bootstrap(marg)
            kr = M.kl_resolvable(ref_lp, M.final_logprobs(model, tok, kl_texts, EVAL_BS), seed)
            kl = kr["mean"]
            removed = base["mean"] - ci["mean"]
            r.cell(stage="sweep", mode=mode, k=k, kl=kl, kl_ci=[kr["lo"], kr["hi"]],
                   margin=ci, removed=removed, resolvable=kr["resolvable"],
                   removed_ci=M.bootstrap([base["mean"] - x for x in marg]))
            print(f"  {mode:8s} k={k:<2} KL={kl:.4f} removed={removed:+.3f} "
                  f"[{ci['lo']:+.2f},{ci['hi']:+.2f}] {'' if kr['resolvable'] else '(unresolved)'}",
                  flush=True)
        edits.restore(mods, snap)

    r.finish("complete", f"swept k=1..{k_max} on both writebacks")
    return "complete"


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else models.PRIMARY
    run(key, dry="--dry" in sys.argv)
