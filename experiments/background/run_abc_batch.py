"""Sequential batch runner with a dry mode.

The dry run exercises every gate, control and analysis path at trivial scale.
It is the thing that would have caught v1's dataset crash and its invalid CUSUM
in minutes rather than after the compute was spent. Nothing expensive starts
until it passes.
"""
from __future__ import annotations

import argparse
import time
import traceback

import models

PLAN = [
    # (experiment, model_key, note)
    ("C", "qwen3_17b",  "positive control: the architecture the refusal literature was built on"),
    ("C", "qwen35_2b",  "primary: Thomson's generation and attention mechanism"),
    ("C", "qwen35_08b", "scale arm"),
    ("A", "qwen3_17b",  "positive control before trusting A on a new architecture"),
    ("A", "qwen35_2b",  "primary"),
    ("B", "qwen35_08b", "sensor: single model by design, training-bound"),
]


MODULES = {"C": "exp_c_geometry", "A": "exp_a_rank", "B": "exp_b_sensor"}


def run_one(exp: str, key: str, seed: int, dry: bool) -> tuple[str, str]:
    """Each cell in its OWN PROCESS.

    In-process sequential runs leaked memory across models and MPS under
    pressure produced silent numerical corruption -- the dry run reported a nan
    split-half angle that vanished when the same cell was run alone. A fresh
    process per cell guarantees clean memory, and an OOM or crash in one cell
    can no longer take down the batch.
    """
    import subprocess, sys
    t0 = time.time()
    mod = MODULES.get(exp)
    if mod is None:
        return "skipped", "unknown experiment"
    cmd = [sys.executable, "-u", "-c",
           f"import {mod} as m; import sys; "
           f"sys.exit(0 if m.run({key!r}, seed={seed}, dry={dry}) in "
           f"('complete','gated') else 3)"]
    try:
        proc = subprocess.run(cmd, timeout=None)
        took = f"{time.time() - t0:.0f}s"
        return ("ok" if proc.returncode == 0 else f"exit{proc.returncode}"), took
    except Exception as e:
        return "error", f"{type(e).__name__}: {str(e)[:120]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="trivial scale, exercises every path")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                    help="two seeds minimum; a result that does not replicate is not a result")
    ap.add_argument("--only", nargs="*", help="e.g. C:qwen35_2b")
    args = ap.parse_args()

    seeds = [0] if args.dry else args.seeds
    plan = PLAN
    if args.only:
        want = {tuple(o.split(":")) for o in args.only}
        plan = [p for p in plan if (p[0], p[1]) in want]

    print(f"{'DRY RUN' if args.dry else 'FULL RUN'}  {len(plan)} cells x {len(seeds)} seed(s)\n")
    results = []
    for exp, key, note in plan:
        for seed in seeds:
            label = models.REGISTRY[key].label
            print(f"\n{'='*70}\n{exp} / {label} / seed {seed}\n{note}\n{'='*70}", flush=True)
            status, detail = run_one(exp, key, seed, args.dry)
            results.append((exp, label, seed, status, detail))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for exp, label, seed, status, detail in results:
        print(f"  {exp}  {label:14s} seed {seed}  {status:9s} {detail}")

    bad = [r for r in results if r[3] == "error"]
    if bad:
        print(f"\n{len(bad)} ERRORED -- do not start the full run")
        return 1
    gated = [r for r in results if r[3] == "gated"]
    if gated:
        print(f"\n{len(gated)} gated (a design fact, not a crash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
