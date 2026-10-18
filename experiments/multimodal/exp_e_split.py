"""Split test: where does the image-derived value direction's capability damage live?

Removing the image-derived value direction v_img cost 15 ARC points; the text-derived one
cost 2-3 (E_crossmodal_value). v_img = shared part (along the text direction) + image-specific
part (orthogonal to it). Removing the shared part IS removing the text direction, already
measured. Here: remove the image-specific part alone, and both parts together; reproduce
v_img itself as a pipeline check (same items, same batches, deterministic -> must match).

Also tests one cheap explanation: that the image-specific part is just "this input is an
image" (cosine with the image-minus-text modality offset), not anything about values.

Run with --behaviour value (the imported module reads it).
"""
import json, os, time, torch
import exp_e_crossmodal as E
import probes, geometry as G

OUT = f"results/E_split_value__{E.LABEL}__seed{E.SEED}{'.dry' if E.DRY else ''}.json"
PRIOR_F = f"results/E_crossmodal_value__{E.LABEL}__seed{E.SEED}.json"


class AblateK:
    """Project a k-dim subspace (orthonormal columns of U) out of every decoder-layer output."""
    def __init__(self, model, U):
        self.U = U.to(next(model.parameters()).device, torch.float32)
        self.hooks = [m.register_forward_hook(self.hook) for m in E.lm_layers(model)]

    def hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h32 = h.float()
        h2 = (h32 - (h32 @ self.U) @ self.U.T).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    def close(self):
        for x in self.hooks: x.remove()


def main():
    assert E.BEHAVIOUR == "value", "run with --behaviour value"
    t0 = time.time(); torch.manual_seed(E.SEED)
    proc = E.vision.cap_resolution(E.AutoProcessor.from_pretrained(E.MID))
    model = E.AutoModelForImageTextToText.from_pretrained(E.MID, dtype=torch.bfloat16).to("mps").eval()
    tok = proc.tokenizer
    lid = [tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"]
    keep = lid[:2]                                   # margin = logit(A) - logit(B), as in E_crossmodal
    marg = lambda L: (L[:, 0] - L[:, 1]).tolist()
    nL = model.config.text_config.num_hidden_layers
    layers = sorted({int(nL * f) for f in (0.4, 0.5, 0.6, 0.7, 0.8)})
    pairs = probes.value_pairs(E.N_PAIRS, E.SEED)
    cond = {
        "text_plain": ([(p.prompt_pos, None) for p in pairs], [(p.prompt_neg, None) for p in pairs], E.BS_TXT),
        "text_wrap": ([(E.WRAP_TXT.format(q=p.prompt_pos), None) for p in pairs],
                      [(E.WRAP_TXT.format(q=p.prompt_neg), None) for p in pairs], E.BS_TXT),
        "image_wrap": ([(E.WRAP_IMG, E.render(p.prompt_pos)) for p in pairs],
                       [(E.WRAP_IMG, E.render(p.prompt_neg)) for p in pairs], E.BS_IMG),
    }
    arc = E.arc_items(E.N_ARC)
    arc_t = [(b + "\nAnswer with the letter only.", None) for b, _ in arc]
    arc_i = [("Answer the multiple-choice question in the image with the letter only.", E.render(b)) for b, _ in arc]
    acc = lambda L: E.mean([int(L[i].argmax()) == a for i, (_, a) in enumerate(arc)])
    R = {"experiment": "E_split_value", "model": E.MID, "seed": E.SEED, "dry": E.DRY, "n_pairs": len(pairs), "status": "running"}

    base, H = {}, {}
    for name, (pos, neg, bs) in cond.items():
        Hp, Lp = E.run(model, proc, pos, layers, keep, bs); Hn, Ln = E.run(model, proc, neg, layers, keep, bs)
        base[name] = E.mean(marg(Lp)) - E.mean(marg(Ln)); H[name] = (Hp, Hn)
    arc_t0 = acc(E.run(model, proc, arc_t, [], lid, E.BS_TXT)[1])
    arc_i0 = acc(E.run(model, proc, arc_i, [], lid, E.BS_IMG)[1])
    layer = max(layers, key=lambda l: abs(G.cohens_d_oos(H["text_plain"][0][l], H["text_plain"][1][l], E.SEED)))
    A = lambda n: H[n][0][layer]; B = lambda n: H[n][1][layer]
    unit = lambda v: v / v.norm()

    v_txt, v_img = G.diff_of_means(A("text_plain"), B("text_plain")), G.diff_of_means(A("image_wrap"), B("image_wrap"))
    u_t, u_img = unit(v_txt), unit(v_img)
    v_perp = v_img - (v_img @ u_t) * u_t; u_p = unit(v_perp)
    offset = torch.cat([A("image_wrap"), B("image_wrap")]).mean(0) - torch.cat([A("text_wrap"), B("text_wrap")]).mean(0)
    R["baseline"] = {"gap": base, "arc_text": arc_t0, "arc_image": arc_i0, "layer": layer}
    R["decomposition"] = {
        "shared_fraction_of_v_img": float((v_img @ u_t) / v_img.norm()),
        "specific_fraction_of_v_img": float(v_perp.norm() / v_img.norm()),
        "cos_with_modality_offset": {"text_dir": G.cosine(u_t, offset), "image_dir": G.cosine(u_img, offset),
                                     "image_specific": G.cosine(u_p, offset)},
        "separation_d": {"image_inputs_along_shared": G.cohens_d(A("image_wrap"), B("image_wrap"), u_t),
                         "image_inputs_along_specific": G.cohens_d(A("image_wrap"), B("image_wrap"), u_p),
                         "text_inputs_along_specific": G.cohens_d(A("text_plain"), B("text_plain"), u_p)},
    }
    json.dump(R, open(OUT, "w"), indent=1); print(json.dumps(R["decomposition"], default=float), flush=True)

    rnd = unit(torch.randn(u_t.shape, generator=torch.Generator().manual_seed(E.SEED)))
    subspaces = {"image_dir_repro": u_img[:, None], "image_specific": u_p[:, None],
                 "both_parts": torch.stack([u_t, u_p], 1), "random": rnd[:, None]}
    R["ablation"] = {}
    for an, U in subspaces.items():
        hk = AblateK(model, U); res = {}
        try:
            for name in ("text_plain", "image_wrap"):
                pos, neg, bs = cond[name]
                gap = E.mean(marg(E.run(model, proc, pos, [], keep, bs)[1])) - E.mean(marg(E.run(model, proc, neg, [], keep, bs)[1]))
                res[f"kept_{name}"] = gap / base[name]
            res["arc_text"] = acc(E.run(model, proc, arc_t, [], lid, E.BS_TXT)[1])
            res["arc_image"] = acc(E.run(model, proc, arc_i, [], lid, E.BS_IMG)[1])
        finally:
            hk.close()
        R["ablation"][an] = res; json.dump(R, open(OUT, "w"), indent=1)
        print(f"remove {an:<16} value kept text {res['kept_text_plain']:.2f} image {res['kept_image_wrap']:.2f}  "
              f"ARC t/i {res['arc_text']:.2f}/{res['arc_image']:.2f}  (baseline {arc_t0:.2f}/{arc_i0:.2f})", flush=True)

    checks = {"random_inert": all(abs(1 - R["ablation"]["random"][k]) < 0.10 for k in ("kept_text_plain", "kept_image_wrap"))}
    if not E.DRY and os.path.exists(PRIOR_F):
        P = json.load(open(PRIOR_F)); pi = P["ablation"]["image_dir"]; me = R["ablation"]["image_dir_repro"]
        checks["layer_matches_prior"] = layer == P["geometry"]["layer"]
        checks["repro_matches_prior"] = (abs(me["kept_image_wrap"] - pi["image_wrap"]["gap_retained"]) < 0.03
                                         and abs(me["arc_text"] - pi["arc_text"]) <= 0.02 and abs(me["arc_image"] - pi["arc_image"]) <= 0.02)
    R["checks"], R["status"], R["elapsed_s"] = checks, "complete", time.time() - t0
    json.dump(R, open(OUT, "w"), indent=1)
    for k, v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"done in {R['elapsed_s']/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
