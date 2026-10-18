"""Controls: confirm the apparatus works, rather than detect that it is broken.

Gates catch a broken run. Controls catch a run that looks fine and is measuring
an artefact. The first draft of the spec had gates and no controls, which was
its largest hole -- nothing in it would have noticed a pipeline that
manufactures signal from noise.

  permutation      shuffle the labels; everything must collapse to the floor
  synthetic        plant a known direction; extraction must recover it
  base model       a model with no refusal training must show no refusal direction
  frozen analysis  hash the analysis code before the data exists
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import torch

import geometry as G


def permute(A: torch.Tensor, B: torch.Tensor, seed: int = 0):
    """Reassign rows between the two populations at random, sizes preserved.

    Under a shuffle there is no real contrast, so every downstream statistic
    must collapse: split-half angle to the random floor, out-of-sample d to
    zero. If it does not, the pipeline is manufacturing signal and every result
    it has produced is void.
    """
    pool = torch.cat([A, B])
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pool), generator=g)
    return pool[idx[:len(A)]], pool[idx[len(A):]]


def permutation_control(A: torch.Tensor, B: torch.Tensor, seed: int = 0,
                        n_rep: int = 5) -> dict:
    """Run the real statistics on shuffled labels. Returns the null distribution."""
    angles, ds = [], []
    for r in range(n_rep):
        Ap, Bp = permute(A, B, seed + r)
        angles.append(G.split_half_angle(Ap, Bp, seed + r))
        ds.append(abs(G.cohens_d_oos(Ap, Bp, seed + r)))
    floor = G.random_floor_deg(A.shape[1], seed=seed)
    return {
        "shuffled_split_half_deg": sum(angles) / len(angles),
        "shuffled_cohens_d_oos": sum(ds) / len(ds),
        "random_floor_deg": floor,
        "n_rep": n_rep,
    }


def synthetic_recovery(dim: int, n: int, sep: float, seed: int = 0) -> dict:
    """Plant a known direction and check extraction recovers it.

    The only test here with exact ground truth: we know the answer because we
    put it there. Validates difference-of-means, the logistic estimator, and
    the agreement between them, independent of any model.
    """
    g = torch.Generator().manual_seed(seed)
    truth = torch.randn(dim, generator=g)
    truth = truth / truth.norm()
    A = torch.randn(n, dim, generator=g) + sep * truth
    B = torch.randn(n, dim, generator=g)
    v_dom = G.diff_of_means(A, B)
    v_log = G.logistic_direction(A, B)
    return {
        "recovery_angle_dom_deg": G.angle_deg(truth, v_dom),
        "recovery_angle_logistic_deg": G.angle_deg(truth, v_log),
        "estimator_agreement_deg": G.angle_deg(v_dom, v_log),
        "planted_separation_sd": sep, "dim": dim, "n_per_side": n,
    }


def estimator_agreement(A: torch.Tensor, B: torch.Tensor) -> dict:
    """Two independent estimators of one quantity must agree.

    Disagreement is an implementation bug, and this is the cheapest way to see
    one. It says nothing about whether the direction is real -- the permutation
    control covers that.
    """
    v1, v2 = G.diff_of_means(A, B), G.logistic_direction(A, B)
    return {"agreement_deg": G.angle_deg(v1, v2), "cosine": G.cosine(v1, v2)}


def freeze_analysis(paths: list[str | Path]) -> dict:
    """Hash the analysis code before any data exists.

    Recorded in every output. If a hash changes between the pre-registration
    and the result, the analysis was edited after seeing data. In v1 I revised
    the interpretation four times during a single sweep; this makes that
    visible rather than deniable.
    """
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def summarise(name: str, payload: dict) -> str:
    body = "  ".join(f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in payload.items())
    return f"  CONTROL {name:22s} {body}"
