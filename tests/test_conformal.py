import math

import numpy as np

from neutral.classifier import softmax_with_temperature
from training.conformal import (
    conformal_quantile,
    coverage,
    coverage_by_class,
    equivalent_tau,
    forward_mask,
    marginal_qhat,
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
