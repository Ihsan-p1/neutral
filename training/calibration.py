"""Temperature scaling and calibration metrics for M4b, in numpy only.

CI installs `requirements-dev.txt` alone (no scipy, sklearn or torch), so
everything here that a test imports must stay on numpy and the stdlib.

`diff` is logit[:, 1] - logit[:, 0]; for a binary softmax, P(biased) is the
sigmoid of diff / T. Dividing by a positive T never changes the sign of diff,
so labels at threshold 0.5 do not depend on T.
"""

from collections.abc import Callable, Sequence

import numpy as np

N_BINS = 15
N_BOOT = 2000


def logit_diff(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return logits[:, 1] - logits[:, 0]


def _check_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")


def prob_biased(diff: np.ndarray, temperature: float) -> np.ndarray:
    """Sigmoid of diff / T, written so that neither branch overflows."""
    _check_temperature(temperature)
    x = np.asarray(diff, dtype=np.float64) / temperature
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def nll(diff: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log likelihood, via logaddexp so saturated logits stay finite."""
    _check_temperature(temperature)
    x = np.asarray(diff, dtype=np.float64) / temperature
    y = np.asarray(labels)
    losses = np.where(y == 1, np.logaddexp(0.0, -x), np.logaddexp(0.0, x))
    return float(np.mean(losses))


def brier(p: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(labels)) ** 2))


def reliability_bins(
    p: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS
) -> list[tuple[float, float, int]]:
    """Equal-mass bins over top-label confidence: (mean confidence, accuracy, count)."""
    p = np.asarray(p, dtype=np.float64)
    labels = np.asarray(labels)
    confidence = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(int) == labels).astype(np.float64)
    order = np.argsort(confidence, kind="mergesort")
    bins = []
    for chunk in np.array_split(order, min(n_bins, len(p))):
        bins.append((float(confidence[chunk].mean()), float(correct[chunk].mean()), len(chunk)))
    return bins


def ece(p: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = reliability_bins(p, labels, n_bins)
    total = sum(count for _, _, count in bins)
    return float(sum(count / total * abs(acc - conf) for conf, acc, count in bins))


def fit_temperature(
    diff: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0
) -> float:
    """T minimising NLL, by golden-section search on log T (NLL is unimodal there)."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)

    def loss(log_t: float) -> float:
        return nll(diff, labels, float(np.exp(log_t)))

    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = loss(c), loss(d)
    while b - a > 1e-5:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = loss(d)
    return float(np.exp((a + b) / 2.0))


def group_bootstrap(
    stat: Callable[[np.ndarray], float],
    groups: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and 95% interval of `stat`, resampling whole groups.

    `stat` gets row indices; each draw passes the rows of the drawn groups,
    so anything `stat` computes from them (bin edges included) is recomputed
    per draw.
    """
    rows: dict[str, list[int]] = {}
    for index, key in enumerate(groups):
        rows.setdefault(key, []).append(index)
    members = [np.array(rows[key]) for key in sorted(rows)]
    point = stat(np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, len(members), len(members))
        draws[b] = stat(np.concatenate([members[k] for k in drawn]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def calibration_verdict(lo: float, hi: float) -> str:
    """Verdict on the CI of ECE(after) - ECE(before), fixed before the run (spec 2)."""
    if hi < 0:
        return "lowers"
    if lo > 0:
        return "raises"
    return "already_close"
