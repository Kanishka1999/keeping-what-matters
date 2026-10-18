"""Direction extraction, subspaces, and the geometry measurements.

Deliberately separate from the experiments so all three use one implementation.
In v1 each agent wrote its own, and they diverged in ways that made results
non-comparable.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def final_token_states(model, tok, prompts, layers, batch_size=8):
    """Residual stream at the final prompt token -- the instant before generation.

    Left padding is what puts that token at index -1 for every row regardless of
    length. Getting this wrong reads padding instead of content, silently.
    """
    prev, tok.padding_side = tok.padding_side, "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    acc = {l: [] for l in layers}
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        for l in layers:
            acc[l].append(hs[l][:, -1, :].float().cpu())
    tok.padding_side = prev
    return {l: torch.cat(v) for l, v in acc.items()}


def diff_of_means(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    v = A.mean(0) - B.mean(0)
    return v / v.norm()


def logistic_direction(A: torch.Tensor, B: torch.Tensor, steps: int = 200) -> torch.Tensor:
    """Second, independent estimator. Must agree with diff-of-means.

    Disagreement between two estimators of the same quantity is an
    implementation bug, and this is the cheapest way to find one.
    """
    X = torch.cat([A, B]).double()
    y = torch.cat([torch.ones(len(A)), torch.zeros(len(B))]).double()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y)
        loss = loss + 1e-3 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    v = w.detach().float()
    return v / v.norm()


def cohens_d(A: torch.Tensor, B: torch.Tensor, v: torch.Tensor) -> float:
    """In-sample separation along a GIVEN direction.

    Only valid when v came from data other than A and B. Passing a direction
    fitted on the same A and B makes this circular and wildly optimistic: on
    pure noise with a fitted direction it returns d ~ 3, which would sail
    through any separability gate. Use cohens_d_oos for gating.
    """
    a, b = A @ v, B @ v
    pooled = torch.sqrt(0.5 * (a.var(unbiased=True) + b.var(unbiased=True)))
    return ((a.mean() - b.mean()) / pooled.clamp_min(1e-9)).item()


def cohens_d_oos(A: torch.Tensor, B: torch.Tensor, seed: int = 0) -> float:
    """Out-of-sample separation: fit the direction on one half, score the other.

    This is what the separability gate must use. The in-sample version reports
    how well a direction fits the data it was fitted on, which is a statement
    about overfitting rather than about the behaviour.
    """
    g = torch.Generator().manual_seed(seed)
    ia, ib = torch.randperm(len(A), generator=g), torch.randperm(len(B), generator=g)
    ha, hb = len(A) // 2, len(B) // 2
    v = diff_of_means(A[ia[:ha]], B[ib[:hb]])
    return cohens_d(A[ia[ha:]], B[ib[hb:]], v)


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    return torch.clamp(torch.dot(u / u.norm(), v / v.norm()), -1, 1).item()


def angle_deg(u: torch.Tensor, v: torch.Tensor) -> float:
    return math.degrees(math.acos(cosine(u, v)))


def split_half_angle(A: torch.Tensor, B: torch.Tensor, seed: int = 0) -> float:
    """The universal gate's statistic: derive the direction twice, independently."""
    g = torch.Generator().manual_seed(seed)
    ia, ib = torch.randperm(len(A), generator=g), torch.randperm(len(B), generator=g)
    ha, hb = len(A) // 2, len(B) // 2
    v1 = diff_of_means(A[ia[:ha]], B[ib[:hb]])
    v2 = diff_of_means(A[ia[ha:]], B[ib[hb:]])
    return angle_deg(v1, v2)


def random_floor_deg(dim: int, n: int = 200, seed: int = 0) -> float:
    """MEASURED random-vector angle, not assumed to be 90.

    In finite dimensions with finite samples the expected angle between random
    vectors is close to but not exactly 90 degrees, and an observed angle only
    means something against the right floor.
    """
    g = torch.Generator().manual_seed(seed)
    angs = [angle_deg(torch.randn(dim, generator=g), torch.randn(dim, generator=g))
            for _ in range(n)]
    return sum(angs) / len(angs)


def contrast_subspace(A: torch.Tensor, B: torch.Tensor, k: int = 12):
    """Ordered directions of the paired contrast, by energy.

    Uncentred on purpose: centring removes the mean difference, which is exactly
    the direction the attack aims at. Returns (U, singular values).
    """
    D = (A - B).double()
    U, S, _ = torch.linalg.svd(D.T @ D)
    return U[:, :k].float(), S[:k].float().sqrt()


def spectrum_stats(sv: torch.Tensor) -> dict:
    e = sv.pow(2)
    p = e / e.sum()
    return {
        "stable_rank": (e.sum() / e.max()).item(),
        "participation_ratio": (1.0 / p.pow(2).sum()).item(),
        "top1_frac": p[0].item(),
        "top2_frac": p[:2].sum().item(),
    }
