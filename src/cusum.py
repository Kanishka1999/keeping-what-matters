"""One-sided CUSUM, and the operating characteristic it produces.

A detector quoted without its false-alarm rate is a heuristic. v1 reported a
lead time at thresholds that also fired on the control arm, which is not
detection -- it is noise crossing a line. The deliverable here is the trade-off
curve, not a single number.
"""
from __future__ import annotations

import math


def standardise(series: list[float], baseline_n: int, sign: float) -> list[float]:
    """Express each reading in baseline SD units, oriented so degradation is +."""
    base = series[:baseline_n]
    mu = sum(base) / len(base)
    var = sum((x - mu) ** 2 for x in base) / max(len(base) - 1, 1)
    sd = math.sqrt(var) or 1e-9
    return [sign * (x - mu) / sd for x in series]


def run_cusum(z: list[float], k: float = 0.5, h: float = 5.0) -> int | None:
    """Index of the first alarm, or None. Accumulates evidence, so a small
    persistent shift is caught where no single reading would cross a threshold."""
    s = 0.0
    for i, x in enumerate(z):
        s = max(0.0, s + x - k)
        if s > h:
            return i
    return None


def stationary_stats(series: list[float]) -> tuple[float, float]:
    """Drift across the window, and the window's own SD.

    CUSUM assumes the in-control window is stationary. v1's channels drifted
    2.4-3.5x their own SD, so alarms on the control arm were guaranteed.
    """
    if len(series) < 3:
        return float("inf"), 1e-9
    mu = sum(series) / len(series)
    var = sum((x - mu) ** 2 for x in series) / (len(series) - 1)
    return abs(series[-1] - series[0]), math.sqrt(var) or 1e-9


def operating_characteristic(ood: dict, benign: dict, tokens: list[int],
                             change_tok: int, baseline_n: int, signs: dict,
                             h_grid=(1, 2, 3, 4, 5, 6, 8, 10, 14, 20)) -> list[dict]:
    """For each threshold: when it fired on the real change, and how often it
    fired on the control where by construction there was nothing to find."""
    rows = []
    for h in h_grid:
        fired_ood, false_alarms, premature = {}, 0, 0
        for ch, series in ood.items():
            z = standardise(series, baseline_n, signs.get(ch, 1.0))
            i = run_cusum(z, h=h)
            if i is not None:
                fired_ood[ch] = tokens[i]
                if tokens[i] < change_tok:
                    premature += 1
        for ch, series in benign.items():
            z = standardise(series, baseline_n, signs.get(ch, 1.0))
            if run_cusum(z, h=h) is not None:
                false_alarms += 1
        valid = [t for t in fired_ood.values() if t >= change_tok]
        rows.append({
            "h": h,
            "alarm_tokens": min(valid) if valid else None,
            "channels_fired": len(fired_ood),
            "false_alarms_benign": false_alarms,
            "n_channels": len(benign),
            "premature": premature,
        })
    return rows


def detrend(series: list[float], baseline_n: int) -> list[float]:
    """Remove the baseline's linear trend, then express residuals in SD units.

    Added because the dry run showed the baseline never becomes stationary: a
    model under continuous training keeps moving, so drift/SD oscillated between
    1.4 and 3.0 across 64 reads with no downward trend. Demanding stationarity
    is therefore not merely impatient, it may be unsatisfiable during training.

    The alternative is to model the drift rather than wait it out: fit the
    baseline's slope, extrapolate it forward, and detect departures from the
    EXTRAPOLATION. A real change shows up as a break in the trend even when the
    trend itself never flattens.
    """
    n = min(baseline_n, len(series))
    if n < 3:
        return [0.0] * len(series)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(series[:n]) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1e-9
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, series[:n])) / denom
    resid = [series[i] - (my + slope * (i - mx)) for i in range(len(series))]
    base_r = resid[:n]
    var = sum(r * r for r in base_r) / max(n - 1, 1)
    sd = math.sqrt(var) or 1e-9
    return [r / sd for r in resid]


def operating_characteristic_detrended(ood: dict, benign: dict, tokens: list[int],
                                       change_tok: int, baseline_n: int, signs: dict,
                                       h_grid=(1, 2, 3, 4, 5, 6, 8, 10, 14, 20)) -> list[dict]:
    """Same trade-off curve, on detrended residuals.

    Reported ALONGSIDE the raw version, never instead of it. Which of the two
    behaves better is itself the finding: if raw fires on the control and
    detrended does not, the alarms were tracking training drift rather than
    capability loss, and that is a statement about what the sensor measures.
    """
    rows = []
    for h in h_grid:
        fired, false_alarms, premature = {}, 0, 0
        for ch, series in ood.items():
            z = [signs.get(ch, 1.0) * r for r in detrend(series, baseline_n)]
            i = run_cusum(z, h=h)
            if i is not None:
                fired[ch] = tokens[i]
                if tokens[i] < change_tok:
                    premature += 1
        for ch, series in benign.items():
            z = [signs.get(ch, 1.0) * r for r in detrend(series, baseline_n)]
            if run_cusum(z, h=h) is not None:
                false_alarms += 1
        valid = [t for t in fired.values() if t >= change_tok]
        rows.append({"h": h, "alarm_tokens": min(valid) if valid else None,
                     "channels_fired": len(fired), "false_alarms_benign": false_alarms,
                     "n_channels": len(benign), "premature": premature})
    return rows
