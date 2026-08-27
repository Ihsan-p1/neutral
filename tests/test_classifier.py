import numpy as np
import pytest

from neutral.classifier import StubClassifier, softmax_with_temperature


def test_softmax_sums_to_one():
    out = softmax_with_temperature(np.array([[2.0, 1.0]]), temperature=1.0)
    assert abs(out[0].sum() - 1.0) < 1e-6


def test_temperature_flattens_distribution():
    logits = np.array([[4.0, 0.0]])
    sharp = softmax_with_temperature(logits, temperature=1.0)
    flat = softmax_with_temperature(logits, temperature=5.0)
    # Higher temperature must yield lower confidence. If this passes with the
    # division removed, calibration is silently gone.
    assert flat[0].max() < sharp[0].max()


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        softmax_with_temperature(np.array([[1.0, 0.0]]), temperature=0.0)


def test_softmax_is_stable_on_large_logits():
    out = softmax_with_temperature(np.array([[1000.0, 999.0]]), temperature=1.0)
    assert np.isfinite(out).all()
    assert abs(out[0].sum() - 1.0) < 1e-6


def test_stub_is_deterministic():
    clf = StubClassifier(temperature=1.0)
    a = clf.predict_proba(["The senator lashed out at critics."])
    b = clf.predict_proba(["The senator lashed out at critics."])
    assert a == b


def test_stub_returns_probability_per_sentence():
    clf = StubClassifier(temperature=1.0)
    out = clf.predict_proba(["One.", "Two.", "Three."])
    assert len(out) == 3
    assert all(0.0 <= p <= 1.0 for p in out)


def test_stub_applies_temperature():
    sentences = ["The senator lashed out at critics.", "The bill passed."]
    sharp = StubClassifier(temperature=1.0).predict_proba(sentences)
    flat = StubClassifier(temperature=10.0).predict_proba(sentences)
    # A high temperature must pull every probability toward 0.5.
    for s, f in zip(sharp, flat, strict=True):
        assert abs(f - 0.5) < abs(s - 0.5)


def test_stub_handles_empty_input():
    assert StubClassifier(temperature=1.0).predict_proba([]) == []
