import math

import numpy as np

from neutral.classifier import softmax_with_temperature
from training.conformal import (
    calibration_split,
    conformal_quantile,
    coverage,
    coverage_by_class,
    equivalent_tau,
    forward_mask,
    marginal_qhat,
    mondrian_qhats,
    mondrian_sets,
    prediction_sets,
)


def test_quantile_hand_computed():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # n = 9, alpha = 0.2 -> k = ceil(10 * 0.8) = 8 -> eighth smallest score.
    assert conformal_quantile(scores, 0.2) == 0.8


def test_quantile_small_n_is_inf():
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), 0.05) == math.inf
    assert prediction_sets(np.array([0.5]), math.inf) == [frozenset({0, 1})]


def test_sets_single_empty_double():
    assert prediction_sets(np.array([0.92, 0.60]), 0.25) == [frozenset({1}), frozenset()]
    assert prediction_sets(np.array([0.55]), 0.7) == [frozenset({0, 1})]


def test_boundary_included():
    assert prediction_sets(np.array([0.75]), 0.25) == [frozenset({1})]


def test_forward_mask():
    sets = [frozenset({1}), frozenset(), frozenset({0, 1}), frozenset({0})]
    assert forward_mask(sets).tolist() == [False, True, True, False]


def test_marginal_equals_tau_router():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    labels = rng.integers(0, 2, size=2000)
    qhat = marginal_qhat(p[:1000], labels[:1000], 0.1)
    test = p[1000:]
    by_sets = forward_mask(prediction_sets(test, qhat))
    by_tau = np.maximum(test, 1 - test) < equivalent_tau(qhat)
    assert by_sets.tolist() == by_tau.tolist()


def test_temperature_invariance():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=(2000, 2)).astype(np.float32)
    labels = rng.integers(0, 2, size=2000)
    results = []
    for temperature in (1.0, 2.5):
        p = softmax_with_temperature(logits, temperature)[:, 1].astype(np.float64)
        qhat = marginal_qhat(p[:1000], labels[:1000], 0.1)
        results.append(prediction_sets(p[1000:], qhat))
    assert results[0] == results[1]


def test_coverage_by_class():
    sets = [frozenset({1}), frozenset({0}), frozenset({0, 1}), frozenset()]
    labels = np.array([1, 1, 0, 0])
    assert coverage(sets, labels) == 0.5
    assert coverage_by_class(sets, labels) == {0: (0.5, 2), 1: (0.5, 2)}


def test_mondrian_uses_own_class():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.2, 0.1, 0.3, 0.05, 0.6])
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    qhats = mondrian_qhats(p, labels, 0.2)
    assert qhats[1] == conformal_quantile(1 - p[labels == 1], 0.2)
    assert qhats[0] == conformal_quantile(p[labels == 0], 0.2)
    assert qhats[0] != qhats[1]


def test_mondrian_small_class_is_inf():
    p = np.array([0.9, 0.8] + [0.1] * 50)
    labels = np.array([1, 1] + [0] * 50)
    qhats = mondrian_qhats(p, labels, 0.05)
    assert qhats[1] == math.inf
    assert all(1 in s for s in mondrian_sets(np.array([0.01, 0.5, 0.99]), qhats))


def test_mondrian_differs_from_marginal():
    rng = np.random.default_rng(0)
    labels = (rng.uniform(size=4000) < 0.05).astype(int)
    p = np.clip(rng.normal(0.35 + 0.3 * labels, 0.2), 0.001, 0.999)
    cal, test = slice(0, 2000), slice(2000, 4000)
    marginal = forward_mask(prediction_sets(p[test], marginal_qhat(p[cal], labels[cal], 0.1)))
    mondrian = forward_mask(mondrian_sets(p[test], mondrian_qhats(p[cal], labels[cal], 0.1)))
    assert (marginal != mondrian).any()


def test_calibration_split_group_clean():
    # 20 distinct articles, so no seed in 0-9 puts every key on one side.
    keys = [f"article-{i % 20}" for i in range(80)]
    for seed in range(10):
        cal, test = calibration_split(keys, seed)
        assert cal and test
        assert sorted(cal + test) == list(range(len(keys)))
        assert not {keys[i] for i in cal} & {keys[i] for i in test}


def test_calibration_split_deterministic():
    keys = [f"article-{i // 3}" for i in range(300)]
    assert calibration_split(keys, 0) == calibration_split(keys, 0)
    assert calibration_split(keys, 0) != calibration_split(keys, 1)
