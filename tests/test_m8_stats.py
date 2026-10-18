import numpy as np
import pytest

from training.m8_stats import (
    equivalence_verdict,
    f1_macro,
    percentile,
    pooled_paired_bootstrap,
)


def test_f1_macro_hand_computed():
    gold = np.array([1, 1, 0, 0, 1])
    pred = np.array([1, 0, 0, 1, 1])
    # Class 1: tp 2, fp 1, fn 1 -> 4/6. Class 0: tp 1, fp 1, fn 1 -> 2/4.
    assert f1_macro(gold, pred) == pytest.approx(7 / 12)


def test_f1_macro_missing_class():
    zeros = np.zeros(4, dtype=int)
    assert f1_macro(zeros, zeros) == 0.5


def test_bootstrap_identical_predictions():
    gold = np.array([1, 0, 1, 0])
    pred = np.array([1, 1, 0, 0])
    split = (gold, pred, pred, ["a", "a", "b", "b"])
    assert pooled_paired_bootstrap([split], n_boot=200) == (0.0, 0.0, 0.0)


def test_bootstrap_resamples_groups():
    gold = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    pred_b = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    pred_a = gold.copy()  # A wins only inside g1.
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
    split = (gold, pred_a, pred_b, groups)
    first = pooled_paired_bootstrap([split], n_boot=500, seed=0)
    _, lo, hi = first
    assert lo < hi
    assert pooled_paired_bootstrap([split], n_boot=500, seed=0) == first


def test_verdict():
    assert equivalence_verdict(-0.02, 0.025) == "equivalent"
    assert equivalence_verdict(0.04, 0.09) == "different"
    assert equivalence_verdict(-0.05, 0.01) == "inconclusive"
    assert equivalence_verdict(-0.031, 0.0) == "inconclusive"


def test_percentile():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 95) == np.percentile(values, 95)
