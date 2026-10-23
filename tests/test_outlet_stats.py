import math

import pytest

from training.outlet_stats import (
    bootstrap_spearman,
    median_gap,
    paired_spearman_diff,
    spearman,
)


def test_spearman_monotone():
    assert spearman([0, 1, 2], [0.1, 0.2, 0.9]) == pytest.approx(1.0)
    assert spearman([0, 1, 2], [0.9, 0.2, 0.1]) == pytest.approx(-1.0)


def test_spearman_ties_hand_computed():
    # Ranks of x are [1.5, 1.5, 3, 4]; covariance 4.5, variances 4.5 and 5.
    assert spearman([0, 0, 1, 2], [1, 2, 3, 4]) == pytest.approx(0.948683, abs=1e-6)


def test_spearman_constant_is_nan():
    assert math.isnan(spearman([1, 1, 1], [0.1, 0.2, 0.3]))


def test_bootstrap_point_equals_spearman_and_is_seeded():
    x = [0, 1, 2, 0, 1, 2, 2, 0]
    y = [0.1, 0.3, 0.5, 0.2, 0.2, 0.6, 0.4, 0.1]
    first = bootstrap_spearman(x, y, n_boot=300, seed=0)
    assert first == bootstrap_spearman(x, y, n_boot=300, seed=0)
    assert first[0] == pytest.approx(spearman(x, y))
    assert first[1] <= first[0] <= first[2]


def test_bootstrap_skips_constant_draws():
    # With two units, many draws pick the same unit twice, so x is constant there.
    point, lo, hi, skipped = bootstrap_spearman([0, 2], [0.1, 0.9], n_boot=200, seed=0)
    assert 0 < skipped < 200
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_paired_diff_identical_scores_is_zero():
    x, a = [0, 1, 2, 2], [0.1, 0.4, 0.5, 0.7]
    assert paired_spearman_diff(x, a, a, n_boot=200)[:3] == (0.0, 0.0, 0.0)


def test_median_gap_hand_computed():
    point, lo, hi, _ = median_gap(
        [0.1, 0.2, 0.5, 0.7], [False, False, True, True], n_boot=200
    )
    assert point == pytest.approx(0.6 - 0.15)
    assert lo <= point <= hi
