"""Experiment B -- can a cheap sensor detect real degradation before a benchmark?

The v1 design was fatally flawed and the flaw is fixed structurally here rather
than by guessing a longer burn-in: the run MEASURES whether its own baseline has
gone stationary and refuses to plant the change until it has. v1 planted the
change after 9 reads while every channel was still drifting 2.4-3.5x its own SD,
so CUSUM accumulated the drift and alarmed on the control arm by construction.

Single model by design (see SPEC 7.8) -- training-bound, so three models is
three times the most expensive thing. It makes no generation or scale claim.
"""
from __future__ import annotations

import sys

import torch

import controls, corpora, cusum, edits, gates, geometry as G, harness, measure as M, models, probes
from harness import Run, Sensitivity

SEQ_LEN, MICRO_BATCH = 128, 4
TOK_PER_STEP = SEQ_LEN * MICRO_BATCH          # 512
SENSOR_EVERY = 1                               # steps -> a read every 512 tokens
BENCH_EVERY = 8
# Calibrated, not guessed. SGD's step is LR x gradient where Adam's is ~LR, so
# carrying 5e-5 over from the AdamW version produced -0.2% damage in 12 steps and
# B spent two hours measuring an undamaged model. Measured: 2e-3 -> 8.5%,
# 8e-3 -> 59%. 3e-3 over 20 steps lands in the 15-20% range: clears the 10% gate
# with margin while staying gradual enough that early detection is non-trivial.
LR = 3e-3
MIN_BASELINE_READS = 24
PLANNED_BURN_STEPS = 28
MAX_BURN_MULT = 3.0
CHANGE_STEPS = 24
N_SENSOR_ITEMS, N_BENCH_ITEMS = 16, 48

# orientation: +1 where degradation makes the channel RISE
SIGNS = {"refusal_margin": -1.0, "task_acc": -1.0, "kl_chat": +1.0, "nll_heldout": +1.0}


def _load(spec):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # bf16 here, unlike the inference experiments: fp32 weights + grads +
    # momentum at 0.8B exceeds 16GB shared memory and MPS responds by returning
    # garbage from failed command buffers rather than raising.
    return AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype=torch.bfloat16).to(dev), tok


def sensor_read(model, tok, probe, ref_lp) -> dict:
    """Cheap, generation-free, capability-resolved. Same quantities as the
    benchmark, at a smaller sample -- so the two are directly comparable."""
    model.eval()
    with torch.no_grad():
        rm = M.refusal_margin(model, tok, probe["refusal_pos"], *probe["rt_ct"])
        lp = M.final_logprobs(model, tok, probe["kl_prompts"])
        acc = M.bootstrap(M.value_margin(model, tok, probe["task_prompts"]))["mean"]
        enc = tok(probe["nll_text"], return_tensors="pt", truncation=True,
                  max_length=256).to(model.device)
        nll = model(**enc, labels=enc["input_ids"], use_cache=False).loss.item()
    model.train()
    return {"refusal_margin": sum(rm) / len(rm), "task_acc": acc,
            "kl_chat": M.mean_kl(ref_lp, lp), "nll_heldout": nll}


def bench_read(model, tok, probe) -> dict:
    """The expensive measurement the sensor stands in for."""
    model.eval()
    with torch.no_grad():
        rm = M.refusal_margin(model, tok, probe["bench_refusal"], *probe["rt_ct"])
    model.train()
    return M.bootstrap(rm)


def run_arm(arm: str, spec, r: Run, seed: int, dry: bool) -> dict:
    model, tok = _load(spec)
    pairs = probes.refusal_pairs(N_BENCH_ITEMS + N_SENSOR_ITEMS, seed)
    s_pairs, b_pairs = pairs[:N_SENSOR_ITEMS], pairs[N_SENSOR_ITEMS:]
    vpairs = probes.value_pairs(N_SENSOR_ITEMS, seed)
    probe = {
        "refusal_pos": [probes.chat(tok, p.prompt_pos) for p in s_pairs],
        "bench_refusal": [probes.chat(tok, p.prompt_pos) for p in b_pairs],
        "task_prompts": [probes.chat(tok, p.prompt_pos) for p in vpairs],
        "kl_prompts": [probes.chat(tok, p.prompt_neg) for p in s_pairs],
        "nll_text": next(corpora.stream_text("chat"))[:2000],
        "rt_ct": M.refusal_tokens(tok),
    }
    with torch.no_grad():
        ref_lp = M.final_logprobs(model, tok, probe["kl_prompts"])

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)  # half the state of Adam
    burn_cap = int(PLANNED_BURN_STEPS * MAX_BURN_MULT)
    chat_blocks = corpora.blocks(tok, "chat", SEQ_LEN, MICRO_BATCH)
    ood_blocks = corpora.blocks(tok, "code" if arm == "ood" else "chat", SEQ_LEN, MICRO_BATCH)

    series: dict[str, list[float]] = {c: [] for c in SIGNS}
    toks, bench, step, change_step = [], [], 0, None
    model.train()

    def train_step(block):
        ids = block.to(model.device)
        try:
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        except RuntimeError as e:
            raise RuntimeError(
                "MPS allocation failed during training. MPS keeps executing after a "
                "failed command buffer and returns garbage, so this must abort rather "
                "than continue: every downstream number would be silently corrupt."
            ) from e
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        return loss.item()

    total = burn_cap + CHANGE_STEPS
    while step < total:
        pre_change = change_step is None
        if pre_change and step >= PLANNED_BURN_STEPS:
            # has the baseline actually settled? measured, not assumed
            worst = max((cusum.stationary_stats(series[c])[0] /
                         cusum.stationary_stats(series[c])[1]) for c in series)
            enough = len(series["kl_chat"]) >= (6 if dry else MIN_BASELINE_READS)
            if (worst < 1.0 and enough) or step >= burn_cap:
                change_step = step
                r.cell(stage="change_planted", step=step, tokens=step * TOK_PER_STEP,
                       worst_drift_ratio=worst, baseline_reads=len(series["kl_chat"]),
                       forced=step >= burn_cap)
                print(f"  change planted at step {step}, drift/SD={worst:.2f}", flush=True)
            else:
                if step % 4 == 0:
                    print(f"  extending burn-in: drift/SD={worst:.2f} reads={len(series['kl_chat'])}",
                          flush=True)
        blocks_src = chat_blocks if change_step is None else ood_blocks
        loss = train_step(next(blocks_src))
        step += 1

        if step % SENSOR_EVERY == 0:
            rd = sensor_read(model, tok, probe, ref_lp)
            for c, v in rd.items():
                series[c].append(v)
            toks.append(step * TOK_PER_STEP)
            r.cell(stage="sensor", arm=arm, step=step, tokens=step * TOK_PER_STEP,
                   loss=loss, **rd)
        if step % BENCH_EVERY == 0:
            b = bench_read(model, tok, probe)
            bench.append({"tokens": step * TOK_PER_STEP, **b})
            r.cell(stage="bench", arm=arm, tokens=step * TOK_PER_STEP, **b)
            print(f"    [bench] refusal={b['mean']:+.3f} [{b['lo']:+.2f},{b['hi']:+.2f}]", flush=True)
        if change_step is not None and step - change_step >= CHANGE_STEPS:
            break

    del model, opt
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"series": series, "tokens": toks, "bench": bench,
            "change_step": change_step, "change_tokens": (change_step or 0) * TOK_PER_STEP,
            "baseline_reads": (change_step or 0) // SENSOR_EVERY}


def run(model_key: str = models.TRAINING_MODEL, seed: int = 0, dry: bool = False) -> str:
    spec = models.REGISTRY[model_key]
    global PLANNED_BURN_STEPS, CHANGE_STEPS, MAX_BURN_MULT, BENCH_EVERY
    if dry:
        PLANNED_BURN_STEPS, CHANGE_STEPS, MAX_BURN_MULT, BENCH_EVERY = 8, 6, 1.5, 4
    cfg = {"seq_len": SEQ_LEN, "micro_batch": MICRO_BATCH, "lr": LR, "dry": dry,
           "planned_burn_steps": PLANNED_BURN_STEPS, "change_steps": CHANGE_STEPS,
           "min_baseline_reads": MIN_BASELINE_READS, "optimiser": "SGD+momentum"}
    r = Run("B_sensor", spec.label, seed, cfg)

    cv = corpora.verify()                       # before any training, not after 100 min
    r.control("corpora", cv)
    if not all(v["ok"] for v in cv.values()):
        r.finish("error", f"corpus unavailable: {cv}")
        return "error"
    r.provenance(model={"hf_id": spec.hf_id, "label": spec.label},
                 frozen=controls.freeze_analysis(["cusum.py", "measure.py", "gates.py"]))

    arms = {}
    for arm in ("ood", "benign"):
        print(f"\n=== arm {arm} ===", flush=True)
        arms[arm] = run_arm(arm, spec, r, seed, dry)

    ood, ben = arms["ood"], arms["benign"]
    base_n = ood["baseline_reads"]

    worst = max(cusum.stationary_stats(ood["series"][c][:base_n])[0] /
                cusum.stationary_stats(ood["series"][c][:base_n])[1] for c in ood["series"])
    st = r.gate(gates.stationary(worst, 1.0))
    rd = r.gate(gates.enough_baseline(base_n, 6 if dry else MIN_BASELINE_READS))
    r.sensitivity(Sensitivity("detectable_shift_sd", 3.0 / max(base_n ** 0.5, 1),
                              f"from {base_n} in-control reads"))

    drop_pp = 0.0
    if len(ood["bench"]) >= 2:
        drop_pp = 100 * (ood["bench"][0]["mean"] - ood["bench"][-1]["mean"]) / max(abs(ood["bench"][0]["mean"]), 1e-9)
    dg = r.gate(gates.degradation_present(abs(drop_pp), 10.0 if not dry else 0.0))

    oc = cusum.operating_characteristic(ood["series"], ben["series"], ood["tokens"],
                                        ood["change_tokens"], base_n, SIGNS)
    oc_dt = cusum.operating_characteristic_detrended(
        ood["series"], ben["series"], ood["tokens"], ood["change_tokens"], base_n, SIGNS)
    r.cell(stage="operating_characteristic", detector="raw", rows=oc)
    r.cell(stage="operating_characteristic", detector="detrended", rows=oc_dt)
    clean = [row for row in oc if row["false_alarms_benign"] == 0 and row["alarm_tokens"]]
    clean_dt = [row for row in oc_dt if row["false_alarms_benign"] == 0 and row["alarm_tokens"]]
    for name, rows in (("raw", oc), ("detrended", oc_dt)):
        print(f"  --- {name} detector ---", flush=True)
        for row in rows:
            print(f"  h={row['h']:<4} alarm={row['alarm_tokens']}  "
                  f"false_on_benign={row['false_alarms_benign']}/{row['n_channels']}", flush=True)

    if gates.ABORT in (st, rd, dg):
        r.finish("gated", "preconditions not met; see gates")
    else:
        r.finish("complete",
                 f"raw: {len(clean)} clean threshold(s); detrended: {len(clean_dt)}")
    return r.doc["status"]


if __name__ == "__main__":
    run(dry="--dry" in sys.argv)
