import json

import numpy as np
import pytest

from training.calibration import (
    apply_calibration,
    brier,
    calibration_verdict,
    ece,
    file_sha256,
    fit_temperature,
    group_bootstrap,
    load_calibration,
    logit_diff,
    nll,
    prob_biased,
    reliability_bins,
)


def test_logit_diff():
    assert list(logit_diff(np.array([[1.0, 3.0], [2.0, -1.0]]))) == [2.0, -3.0]


def test_labels_unchanged_by_temperature():
    diff = np.array([-3.0, -0.2, 0.0, 0.1, 4.0])
    base = prob_biased(diff, 1.0) >= 0.5
    for t in (0.1, 0.7, 1.5, 20.0):
        assert np.array_equal(prob_biased(diff, t) >= 0.5, base)


def test_saturated_logits_stay_finite():
    diff = np.array([-900.0, 900.0, 800.0])
    labels = np.array([1, 0, 1])
    assert np.isfinite(nll(diff, labels, 1.0))
    assert np.all(np.isfinite(prob_biased(diff, 0.05)))
    assert np.isfinite(fit_temperature(diff, labels))


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        prob_biased(np.array([1.0]), 0.0)


def test_fit_recovers_inflated_logits():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 2, 5000)
    labels = (rng.random(5000) < 1 / (1 + np.exp(-true))).astype(int)
    assert fit_temperature(true * 3.0, labels) == pytest.approx(3.0, rel=0.1)


def test_ece_near_zero_when_calibrated_and_larger_when_overconfident():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.99, 20000)
    labels = (rng.random(20000) < p).astype(int)
    assert ece(p, labels) < 0.03
    sharp = np.where(p >= 0.5, 0.99, 0.01)
    assert ece(sharp, labels) > ece(p, labels) + 0.1


def test_ece_hand_computed_two_bins():
    p = np.array([0.9, 0.9, 0.2, 0.2])  # confidences 0.9, 0.9, 0.8, 0.8
    labels = np.array([1, 0, 0, 0])  # correct: yes, no, yes, yes
    # bin A conf 0.8 acc 1.0, bin B conf 0.9 acc 0.5 -> 0.5 * 0.2 + 0.5 * 0.4
    assert ece(p, labels, n_bins=2) == pytest.approx(0.3)


def test_ece_fewer_rows_than_bins_and_constant_confidence():
    assert np.isfinite(ece(np.array([0.7, 0.6, 0.9]), np.array([1, 0, 1])))
    constant = np.full(10, 0.8)
    labels = np.array([1] * 8 + [0] * 2)
    assert np.isfinite(ece(constant, labels))
    assert ece(constant, labels, n_bins=1) == pytest.approx(0.0)
    assert len(reliability_bins(np.array([0.7, 0.6]), np.array([1, 0]))) == 2


def test_brier_hand_computed():
    expected = (0 + 1 + 0.25) / 3
    assert brier(np.array([1.0, 0.0, 0.5]), np.array([1, 1, 0])) == pytest.approx(expected)


def test_group_bootstrap_resamples_whole_groups():
    values = np.array([0.0, 0.0, 1.0, 1.0])
    groups = ["a", "a", "b", "b"]
    point, lo, hi = group_bootstrap(lambda idx: float(values[idx].mean()), groups, n_boot=500)
    assert point == 0.5 and lo == 0.0 and hi == 1.0


def test_group_bootstrap_single_group_is_degenerate():
    point, lo, hi = group_bootstrap(lambda idx: float(len(idx)), ["g"] * 3, n_boot=50)
    assert point == lo == hi == 3.0


def test_calibration_verdict():
    assert calibration_verdict(-0.05, -0.01) == "lowers"
    assert calibration_verdict(-0.02, 0.01) == "already_close"
    assert calibration_verdict(0.01, 0.03) == "raises"


BASE = {"version": "deberta-babe-f0s3", "temperature": 1.0, "calibrated": False,
        "max_length": 128, "opset": 17, "babe_fold0_f1_macro": 0.8765, "note": "old"}
CAL = {"temperature": 1.7, "fitted_on": "babe_fold1", "n": 883,
       "model_version": "deberta-babe-f0s3", "checkpoint_sha256": "abc",
       "ece_babe_after": {"point": 0.03, "lo": 0.02, "hi": 0.05},
       "note": "new", "date": "2026-10-24", "commit": "deadbee"}


def test_apply_calibration_matching_hash():
    meta, applied = apply_calibration(BASE, CAL, "abc")
    assert applied and meta["temperature"] == 1.7 and meta["calibrated"] is True
    assert meta["note"] == "new" and meta["ece_babe_after"]["point"] == 0.03
    assert meta["calibration"] == {"fitted_on": "babe_fold1", "n": 883,
                                   "date": "2026-10-24", "commit": "deadbee"}
    for key in ("version", "max_length", "opset", "babe_fold0_f1_macro"):
        assert meta[key] == BASE[key]
    assert BASE["temperature"] == 1.0  # base not mutated


def test_apply_calibration_other_checkpoint_falls_back():
    stale = {**BASE, "temperature": 1.7, "calibrated": True, "ece_babe_after": {"point": 0.03}}
    meta, applied = apply_calibration(stale, CAL, "other")
    assert not applied and meta["temperature"] == 1.0 and meta["calibrated"] is False
    assert meta["note"] == "old" and "ece_babe_after" not in meta


def test_apply_calibration_missing_file_falls_back():
    meta, applied = apply_calibration(BASE, None, "abc")
    assert not applied and meta["temperature"] == 1.0


def test_load_calibration_bad_content(tmp_path):
    assert load_calibration(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert load_calibration(tmp_path / "bad.json") is None
    (tmp_path / "partial.json").write_text('{"temperature": 1.5}', encoding="utf-8")
    assert load_calibration(tmp_path / "partial.json") is None
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert load_calibration(tmp_path / "list.json") is None


def test_load_calibration_round_trip(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(CAL), encoding="utf-8")
    assert load_calibration(path) == CAL


def test_file_sha256(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
