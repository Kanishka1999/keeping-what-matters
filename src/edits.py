"""Rank-one directional ablation: the two writebacks, and the constraint.

Thomson Eq. 2:  dW = b (v^T W),  v^T b = -lambda
                => v^T (W + dW) = (1 - lambda) v^T W

The constraint pins one dimension of b and leaves d-1 free. Standard ablation
zeroes them (b = -lambda v). Fisher routing (Eq. 4) spends them on minimising
predicted collateral. Both remove the SAME fraction of the target projection,
so the comparison is fair -- they differ only in where the correction lands.
"""
from __future__ import annotations

import torch


def standard_writeback(v: torch.Tensor, lam: float) -> torch.Tensor:
    return -lam * v


def fisher_writeback(v: torch.Tensor, F: torch.Tensor, lam: float,
                     rho: float = 0.05) -> torch.Tensor:
    """Eq. 4. Computed on CPU in float64 regardless of device.

    MPS has no float64 at all, and the routing arithmetic is a handful of
    d-dimensional ops, so doing it on CPU costs nothing and removes a class of
    silent precision loss. v1 hit the MPS float64 failure here.
    """
    target = v.device
    v64 = v.detach().cpu().double()
    F64 = F.detach().cpu().double().clamp_min(0.0)
    h = F64 + rho * F64.mean()
    g = v64 / h
    s = torch.dot(v64, g)
    if s.abs() < 1e-12:
        raise ValueError("degenerate routing: v^T g vanished")
    return (-(lam / s) * g).float().to(target)


def check_constraint(v: torch.Tensor, b: torch.Tensor, lam: float, tol: float = 1e-3) -> float:
    """Assert v^T b = -lambda. Cheap, and catches sign and scale slips."""
    v64, b64 = v.detach().cpu().double(), b.detach().cpu().double()
    got = torch.dot(v64 / v64.norm(), b64).item()
    if abs(got + lam) > tol:
        raise AssertionError(f"v^T b = {got:.6f}, expected {-lam:.6f}")
    return got


@torch.no_grad()
def apply_(W: torch.Tensor, v: torch.Tensor, b: torch.Tensor) -> None:
    """In place: W <- W + b (v^T W). v and b live in the output (residual) space."""
    assert W.shape[0] == v.numel() == b.numel(), (W.shape, v.shape, b.shape)
    Wf = W.float()
    W.copy_((Wf + torch.outer(b.float(), v.float() @ Wf)).to(W.dtype))


@torch.no_grad()
def install(modules: dict, v: torch.Tensor, lam: float, mode: str,
            fisher: dict | None = None, rho: float = 0.05) -> dict:
    """Edit every target module. Returns realised v^T b per module."""
    realised = {}
    for label, mod in modules.items():
        vd = v.to(mod.weight.device, dtype=torch.float32)
        if mode == "fisher":
            if not fisher or label not in fisher:
                raise ValueError(f"no Fisher estimate for {label}")
            b = fisher_writeback(vd, fisher[label].to(vd.device), lam, rho)
        elif mode == "standard":
            b = standard_writeback(vd, lam)
        else:
            raise ValueError(mode)
        realised[label] = check_constraint(vd, b, lam)
        apply_(mod.weight.data, vd, b)
    return realised


def snapshot(modules: dict) -> dict:
    """Weights of the target modules only, for restore without a disk reload.

    v1 reloaded the whole model from disk per arm. Snapshotting the handful of
    edited matrices is far cheaper and exactly equivalent, since nothing else
    is touched.
    """
    return {k: m.weight.detach().clone() for k, m in modules.items()}


@torch.no_grad()
def restore(modules: dict, snap: dict) -> None:
    for k, m in modules.items():
        m.weight.data.copy_(snap[k])


def ablation_hook(v: torch.Tensor, strength: float = 1.0):
    """Inference-time projection, for causal tests that must not mutate weights."""
    vv = (v / v.norm()).float()

    def hook(module, inputs, output):
        tup = isinstance(output, tuple)
        h = output[0] if tup else output
        u = vv.to(h.device, dtype=h.dtype)
        h = h - strength * (h @ u).unsqueeze(-1) * u
        return (h,) + output[1:] if tup else h

    return hook
