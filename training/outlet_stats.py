"""Outlet-level statistics for PRD 8.3, in numpy only so the tests run in serving CI.

Every interval resamples outlets (units), never articles: an outlet's articles
share its rating, so resampling articles would make the interval look narrower
than the evidence is.
"""

from collections.abc import Callable, Sequence

import numpy as np

Interval = tuple[float, float, float, int]


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties given the average of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho with average ranks for ties; nan when either side is constant."""
    rx = _ranks(np.asarray(x, dtype=np.float64))
    ry = _ranks(np.asarray(y, dtype=np.float64))
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denominator = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    if denominator == 0:
        return float("nan")
    return float(np.sum(dx * dy) / denominator)


def _bootstrap(stat: Callable[[np.ndarray], float], n: int, n_boot: int, seed: int) -> Interval:
    """Point estimate and 95% percentile interval of `stat` over resampled unit indices.

    Draws where the statistic is undefined (nan) are skipped and counted.
    """
    point = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    draws = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    kept = [value for value in draws if not np.isnan(value)]
    skipped = n_boot - len(kept)
    if not kept:
        return point, float("nan"), float("nan"), skipped
    lo, hi = np.percentile(kept, [2.5, 97.5])
    return point, float(lo), float(hi), skipped


def bootstrap_spearman(
    x: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> Interval:
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return _bootstrap(lambda idx: spearman(xa[idx], ya[idx]), len(xa), n_boot, seed)


def paired_spearman_diff(
    x: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Interval:
    """Spearman(x, a) - Spearman(x, b), both computed on the same resampled units."""
    xa = np.asarray(x, dtype=np.float64)
    aa, ba = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return _bootstrap(
        lambda idx: spearman(xa[idx], aa[idx]) - spearman(xa[idx], ba[idx]),
        len(xa),
        n_boot,
        seed,
    )


def median_gap(
    scores: Sequence[float], partisan: Sequence[bool], n_boot: int = 2000, seed: int = 0
) -> Interval:
    """Median score of partisan units minus median of centre units."""
    sa = np.asarray(scores, dtype=np.float64)
    pa = np.asarray(partisan, dtype=bool)

    def gap(idx: np.ndarray) -> float:
        s, p = sa[idx], pa[idx]
        if p.all() or not p.any():
            return float("nan")
        return float(np.median(s[p]) - np.median(s[~p]))

    return _bootstrap(gap, len(sa), n_boot, seed)
