"""Preconditions that halt, rather than diagnostics that get read past.

v1 printed warnings and carried on, and its two negative results were both
artefacts the warnings had described. The one experiment that *stopped* when its
precondition failed produced the only trustworthy output. This module
generalises that: a gate returns a verdict, and the runner refuses to spend the
expensive phase on a failed one.

Three verdicts, because "stop" is not always right:
    PASS     proceed
    ADAPT    the design can fix itself (extend a burn-in) -- retry, bounded
    ABORT    no setting of the knobs rescues this run; say why and stop
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

PASS, ADAPT, ABORT = "PASS", "ADAPT", "ABORT"


@dataclass
class Verdict:
    name: str
    status: str
    value: float | None
    threshold: float | None
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PASS

    def line(self) -> str:
        v = "n/a" if self.value is None else f"{self.value:.4g}"
        t = "" if self.threshold is None else f" (need {self.threshold:.4g})"
        return f"  [{self.status:5s}] {self.name:28s} {v}{t}  {self.detail}"

    def to_dict(self) -> dict:
        return asdict(self)


def _nan(name: str, what: str) -> Verdict:
    """A nan is a numerical failure, not a failed hypothesis.

    In the dry run a nan reached this gate and was reported as "direction not
    reproducible", which reads as a result about the model. It was memory
    corruption. Distinguish them or the run lies about what happened.
    """
    return Verdict(name, ABORT, None, None,
                   f"NUMERICALLY INVALID: {what} was nan/inf -- suspect memory "
                   f"pressure or precision loss, not a property of the model")


def reproducible_direction(angle_deg: float, random_floor_deg: float,
                           threshold: float = 45.0) -> Verdict:
    """Universal gate: does the direction survive a split of its own probe set?

    Applies to every experiment that extracts a direction, not just the geometry
    one. v1's rank-cost experiment built an entire subspace on an unvalidated
    direction; had this run there, it would have stopped in minutes.

    The random floor is MEASURED, not assumed to be 90 degrees -- in finite
    dimensions with finite samples it is not exactly 90, and quoting an angle
    against the wrong floor is how 85.7 degrees looked like a number rather
    than like noise.
    """
    import math
    if not math.isfinite(angle_deg) or not math.isfinite(random_floor_deg):
        return _nan("direction_reproducible", "split-half angle")
    margin = random_floor_deg - angle_deg
    status = PASS if angle_deg < threshold else ABORT
    return Verdict(
        "direction_reproducible", status, angle_deg, threshold,
        f"split-half angle; measured random floor {random_floor_deg:.1f} deg, margin {margin:.1f}",
        {"random_floor_deg": random_floor_deg},
    )


def separable(cohens_d: float, threshold: float = 0.8) -> Verdict:
    import math
    if not math.isfinite(cohens_d):
        return _nan("separability", "out-of-sample Cohen's d")
    status = PASS if abs(cohens_d) >= threshold else ABORT
    return Verdict("separability", status, abs(cohens_d), threshold,
                   "contrast effect size along the extracted direction")


def ruler_headroom(effect: float, noise_floor: float, factor: float = 5.0) -> Verdict:
    """Can the capability metric resolve the effect being asked of it?

    The floor is the metric's own reading when NOTHING has changed -- the model
    against itself on disjoint prompt halves. v1's KL spanned 0.0018-0.029 and
    wandered non-monotonically, which is what a metric operating inside its own
    noise looks like. Nobody checked.
    """
    need = noise_floor * factor
    status = PASS if effect >= need else ABORT
    return Verdict("ruler_headroom", status, effect, need,
                   f"effect vs {factor:g}x the model-vs-itself floor ({noise_floor:.3g})",
                   {"noise_floor": noise_floor})


def behaviour_moved(delta: float, baseline: float, frac: float = 0.20) -> Verdict:
    need = abs(baseline) * frac
    status = PASS if abs(delta) >= need else ABORT
    return Verdict("behaviour_moved", status, abs(delta), need,
                   f"intervention must shift behaviour >= {frac:.0%} of baseline")


def stationary(drift: float, sd: float, threshold: float = 1.0) -> Verdict:
    """CUSUM's precondition, and the reason v1's sensor result was void.

    CUSUM estimates an in-control mean and variance from the pre-change window,
    then accumulates departures. If that window is itself trending, it
    accumulates the trend and alarms whether or not anything changed. v1's
    channels drifted 2.4-3.5x their own SD across a 9-read baseline, so alarms
    on the control arm were guaranteed by construction.

    ADAPT rather than ABORT: a longer burn-in genuinely fixes this.
    """
    ratio = drift / max(sd, 1e-12)
    status = PASS if ratio < threshold else ADAPT
    return Verdict("baseline_stationary", status, ratio, threshold,
                   "burn-in drift in units of its own SD; ADAPT extends the burn-in",
                   {"drift": drift, "sd": sd})


def enough_baseline(n_reads: int, minimum: int = 24) -> Verdict:
    status = PASS if n_reads >= minimum else ADAPT
    return Verdict("baseline_reads", status, float(n_reads), float(minimum),
                   "in-control observations available to set control limits")


def degradation_present(drop_pp: float, minimum_pp: float = 10.0) -> Verdict:
    status = PASS if drop_pp >= minimum_pp else ABORT
    return Verdict("degradation_present", status, drop_pp, minimum_pp,
                   "there must be real damage before detecting it means anything")


def null_is_null(observed: float, floor: float, tol: float = 0.15) -> Verdict:
    """Control gate: a run that SHOULD find nothing must find nothing.

    Used by the permutation test and the base-model control. A pipeline that
    reports a finding on shuffled labels is manufacturing signal, and every
    result it has ever produced is void. This is the check whose absence was
    the largest hole in the first draft of the spec.
    """
    slack = abs(floor) * tol
    status = PASS if abs(observed - floor) <= max(slack, 1e-9) else ABORT
    return Verdict("null_control", status, observed, floor,
                   f"control must land within {tol:.0%} of its null floor")


def report(verdicts: list[Verdict]) -> str:
    worst = ABORT if any(v.status == ABORT for v in verdicts) else (
        ADAPT if any(v.status == ADAPT for v in verdicts) else PASS)
    body = "\n".join(v.line() for v in verdicts)
    return f"GATES -> {worst}\n{body}"


def overall(verdicts: list[Verdict]) -> str:
    if any(v.status == ABORT for v in verdicts):
        return ABORT
    if any(v.status == ADAPT for v in verdicts):
        return ADAPT
    return PASS
