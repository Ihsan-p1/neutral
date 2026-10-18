"""Split conformal prediction for the binary bias classifier (M8).

Labels are 0 = neutral and 1 = biased. Every function takes P(biased) per
sentence, the same number `Classifier.predict_proba` returns.

Imports only stdlib, numpy and `training.split`, so the tests run in the serving
CI venv, which has no torch or sklearn.
"""

import math

import numpy as np

LABELS = (0, 1)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the split-conformal threshold q-hat for nonconformity scores.

    q-hat is the k-th smallest score with k = ceil((n + 1)(1 - alpha)). When
    k > n there are too few calibration rows for the requested coverage, and
    the answer is infinity: every label enters every set.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(np.sort(np.asarray(scores, dtype=np.float64))[k - 1])


def _label_probs(p_biased: np.ndarray) -> np.ndarray:
    p = np.asarray(p_biased, dtype=np.float64)
    return np.stack([1 - p, p], axis=1)


def marginal_qhat(p_biased: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    """q-hat from scores 1 - p(true label) over all calibration rows."""
    probs = _label_probs(p_biased)
    labels = np.asarray(labels, dtype=np.int64)
    scores = 1 - probs[np.arange(len(labels)), labels]
    return conformal_quantile(scores, alpha)


def prediction_sets(p_biased: np.ndarray, qhat: float) -> list[frozenset[int]]:
    """Label y is in the set when p(y) >= 1 - q-hat."""
    return _sets(p_biased, {label: qhat for label in LABELS})


def _sets(p_biased: np.ndarray, qhats: dict[int, float]) -> list[frozenset[int]]:
    probs = _label_probs(p_biased)
    return [
        frozenset(label for label in LABELS if row[label] >= 1 - qhats[label])
        for row in probs
    ]


def forward_mask(sets: list[frozenset[int]]) -> np.ndarray:
    """True where the router sends the sentence to the LLM: anything but one label."""
    return np.array([len(s) != 1 for s in sets], dtype=bool)


def equivalent_tau(qhat: float) -> float:
    """The max-probability threshold whose routing equals the marginal sets.

    With two classes, forwarding happens exactly when max(p) < max(q-hat, 1 - q-hat),
    ties aside. Spec section 4.4 has the derivation.
    """
    return max(qhat, 1 - qhat)


def coverage(sets: list[frozenset[int]], labels: np.ndarray) -> float:
    """Share of rows whose true label is in their set."""
    hits = [int(label) in s for s, label in zip(sets, labels, strict=True)]
    return float(np.mean(hits))


def coverage_by_class(
    sets: list[frozenset[int]], labels: np.ndarray
) -> dict[int, tuple[float, int]]:
    """Coverage and row count per true class."""
    labels = np.asarray(labels)
    result: dict[int, tuple[float, int]] = {}
    for label in LABELS:
        rows = [s for s, y in zip(sets, labels, strict=True) if y == label]
        if rows:
            result[label] = (coverage(rows, np.full(len(rows), label)), len(rows))
    return result
