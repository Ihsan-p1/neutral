"""Statistics for the M8 report, in numpy only so the tests run in serving CI.

`f1_macro` matches sklearn's `f1_score(average="macro", zero_division=0)` for
binary labels 0/1.
"""

import numpy as np

Split = tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]


def f1_macro(gold: np.ndarray, pred: np.ndarray) -> float:
    """Mean F1 over classes 0 and 1; a class with no gold and no predictions scores 0."""
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    scores = []
    for label in (0, 1):
        tp = int(np.sum((gold == label) & (pred == label)))
        fp = int(np.sum((gold != label) & (pred == label)))
        fn = int(np.sum((gold == label) & (pred != label)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def pooled_paired_bootstrap(
    splits: list[Split], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% interval of F1(A) - F1(B), resampling whole articles.

    Each draw picks one split at random, then resamples that split's articles
    with replacement, so the interval carries both the split-to-split spread and
    the sampling noise inside one test half.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for gold, pred_a, pred_b, groups in splits:
        keys = sorted(set(groups))
        rows = {key: [] for key in keys}
        for index, key in enumerate(groups):
            rows[key].append(index)
        prepared.append((np.asarray(gold), np.asarray(pred_a), np.asarray(pred_b), keys, rows))

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        gold, pred_a, pred_b, keys, rows = prepared[rng.integers(len(prepared))]
        drawn = rng.choice(len(keys), size=len(keys), replace=True)
        index = np.concatenate([rows[keys[k]] for k in drawn])
        diffs[b] = f1_macro(gold[index], pred_a[index]) - f1_macro(gold[index], pred_b[index])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def equivalence_verdict(lo: float, hi: float, margin: float = 0.03) -> str:
    """"equivalent" if the interval sits inside +/-margin, "different" if it excludes 0."""
    if -margin <= lo and hi <= margin:
        return "equivalent"
    if lo > 0 or hi < 0:
        return "different"
    return "inconclusive"


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
