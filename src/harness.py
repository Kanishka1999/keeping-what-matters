"""Provenance, incremental saving, and sensitivity reporting.

Three v1 failures this exists to prevent:
  * two of three result files did not record which model produced them
  * the sensor script saved once at the end, so an interruption cost 100 minutes
  * "sensitivity" was promised in the plan and never defined, so nobody knew
    whether a null meant "no effect" or "could not have seen one"
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def content_hash(obj) -> str:
    """Stable hash of probe content. Written into every output.

    Cross-model comparison is only meaningful if every model saw identical
    items in identical order. Mismatched hashes invalidate the comparison, and
    without recording them the mismatch is silent.
    """
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def environment() -> dict:
    import torch, transformers
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=ROOT, timeout=5)
        rev = git.stdout.strip() or None
    except Exception:
        rev = None
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_rev": rev,
    }


@dataclass
class Sensitivity:
    """What this design can resolve, stated BEFORE the expensive phase.

    Without it a null is ambiguous between "no effect" and "no power", which is
    exactly the ambiguity that made v1's negatives unusable.
    """
    quantity: str
    resolvable: float
    basis: str

    def line(self) -> str:
        return f"  SENSITIVITY  {self.quantity}: can resolve >= {self.resolvable:.4g}  ({self.basis})"


class Run:
    """One experiment on one model. Saves after every cell."""

    def __init__(self, experiment: str, model_label: str, seed: int, config: dict):
        # Dry runs MUST NOT share filenames with real runs. A dry pass silently
        # overwrote a completed 200-item result with its 24-item output, and the
        # loss was only visible from file mtimes -- the JSON looked valid and the
        # numbers looked like a seed disagreement.
        suffix = ".dry" if config.get("dry") else ""
        self.path = OUT / f"{experiment}__{model_label}__seed{seed}{suffix}.json"
        self.doc = {
            "experiment": experiment,
            "model_label": model_label,
            "seed": seed,
            "config": config,
            "environment": environment(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "gates": [],
            "sensitivity": [],
            "cells": [],
            "controls": {},
        }
        self._t0 = time.time()
        self.flush()

    def flush(self) -> None:
        self.doc["elapsed_s"] = round(time.time() - self._t0, 1)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=2, default=float))
        tmp.replace(self.path)          # atomic: a crash mid-write cannot corrupt

    def provenance(self, **kw) -> None:
        self.doc.setdefault("provenance", {}).update(kw)
        self.flush()

    def gate(self, verdict) -> str:
        self.doc["gates"].append(verdict.to_dict())
        print(verdict.line(), flush=True)
        self.flush()
        return verdict.status

    def sensitivity(self, s: Sensitivity) -> None:
        self.doc["sensitivity"].append({"quantity": s.quantity,
                                        "resolvable": s.resolvable, "basis": s.basis})
        print(s.line(), flush=True)
        self.flush()

    def cell(self, **kw) -> None:
        """One measurement point. Saved immediately -- a crash costs this cell only."""
        self.doc["cells"].append(kw)
        self.flush()

    def control(self, name: str, payload: dict) -> None:
        self.doc["controls"][name] = payload
        self.flush()

    def finish(self, status: str, note: str = "") -> None:
        self.doc["status"] = status
        self.doc["note"] = note
        self.doc["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.flush()
        print(f"\n>>> {self.doc['experiment']} / {self.doc['model_label']} "
              f"-> {status}  {note}  ({self.doc['elapsed_s']:.0f}s)", flush=True)
