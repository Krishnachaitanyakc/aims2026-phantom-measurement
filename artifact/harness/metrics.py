"""Determinism metrics and uncertainty machinery (pure stdlib).

D = max-vote-share (plurality fraction) over N chain replications,
grouped by numeric answer equivalence — Definition consistent with the
corrected (2026-06-15) implementation of the original pipeline.
"""
from __future__ import annotations

import itertools
import random
from statistics import NormalDist, mean

from .parsing import answers_match

_ND = NormalDist()


def determinism_max_vote(answers: list[str]) -> float:
    """Max-vote-share over equivalence classes. n<2 -> 1.0 by convention.

    Order-independent (Amendment A10): answers are grouped in a canonical
    sorted order, and each answer joins the first canonical group it
    matches, so input permutation cannot change the result.
    """
    n = len(answers)
    if n < 2:
        return 1.0
    groups: list[list[str]] = []
    for ans in sorted(answers, key=lambda a: (a is None, str(a))):
        for grp in groups:
            if answers_match(ans, grp[0]):
                grp.append(ans)
                break
        else:
            groups.append([ans])
    return max(len(g) for g in groups) / n


def design_permutation_p(gaps: list[float], scale: int = 60) -> float:
    """Legacy alias for the exact sign-flip computation on a fixed panel.

    Inferential validity requires sign symmetry or exchangeability under
    the null. Interleaving and order balancing do not establish that
    assumption by themselves. Exact enumeration describes the computation,
    not an assumption-free guarantee for hosted model calls.
    """
    return paired_permutation_p(gaps, scale=scale)


def tara(answers: list[str]) -> float:
    """Unanimous parsed-answer agreement (Atil et al.'s TARa), 0/1 per problem."""
    if len(answers) < 2:
        return 1.0
    return 1.0 if all(answers_match(a, answers[0]) for a in answers[1:]) else 0.0


def bca_ci(values: list[float], b: int = 10_000, alpha: float = 0.05,
           seed: int = 20260728) -> tuple[float, float]:
    """BCa bootstrap CI for the mean, resampling at the cluster (problem) level."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or len(set(values)) == 1:
        # Point-mass distribution: the resampling distribution is a point
        # mass, so the interval is honestly zero-width; callers flag
        # degeneracy explicitly (paired_gap emits *_ci_degenerate) and the
        # TOST rule treats it as descriptive-only (Amendments A6/A10).
        return values[0], values[0]
    rng = random.Random(seed)
    obs = mean(values)
    boots = sorted(mean(rng.choices(values, k=n)) for _ in range(b))
    prop = sum(1 for x in boots if x < obs) / b
    prop = min(max(prop, 1.0 / (2 * b)), 1.0 - 1.0 / (2 * b))
    z0 = _ND.inv_cdf(prop)
    jack = [mean(values[:i] + values[i + 1:]) for i in range(n)]
    jm = mean(jack)
    num = sum((jm - j) ** 3 for j in jack)
    den = 6.0 * (sum((jm - j) ** 2 for j in jack)) ** 1.5
    a = num / den if den > 1e-12 else 0.0

    def q(level: float) -> float:
        z = _ND.inv_cdf(level)
        adj = _ND.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        idx = min(b - 1, max(0, int(adj * b)))
        return boots[idx]

    return q(alpha / 2), q(1 - alpha / 2)


def paired_permutation_p(gaps: list[float], scale: int = 60) -> float:
    """EXACT two-sided sign-flip permutation p-value for mean(gaps) = 0.

    Interpretation requires a sign-symmetry or exchangeability null.
    Computed by lattice dynamic programming over the distribution of
    sum(±g_i): per-problem gaps in this project live on a 1/`scale` lattice
    (D and accuracy are counts over N runs), so each gap is first snapped to
    round(g*scale) — exact for lattice-valued inputs; for arbitrary inputs
    the snap introduces at most 1/(2*scale) rounding, and `scale` may be
    raised. Never Monte Carlo, never returns 0 for nonempty input (minimum
    is 2/2^n, attained when the observed |sum| is strictly the maximum).
    """
    n = len(gaps)
    if n == 0:
        return float("nan")
    ints = [round(g * scale) for g in gaps]
    obs = abs(sum(ints))
    # distribution over sum of ±ints as {sum: count}
    dist: dict[int, int] = {0: 1}
    for v in ints:
        nxt: dict[int, int] = {}
        for s, c in dist.items():
            for t in (s + v, s - v):
                nxt[t] = nxt.get(t, 0) + c
        dist = nxt
    total = 2 ** n
    count = sum(c for s, c in dist.items() if abs(s) >= obs)
    return count / total


def holm_correct(pvals: dict[str, float],
                 m_total: int | None = None) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values keyed like the input.

    `m_total` fixes the family size a priori (Amendment A6): when arms are
    missing, adjustment still uses the full registered family size rather
    than shrinking to the available tests.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    if m_total is not None and len(items) > m_total:
        raise ValueError("more tests than the registered family size")
    m = m_total if m_total is not None else len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    return adjusted
