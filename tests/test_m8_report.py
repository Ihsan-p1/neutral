import numpy as np

from training.m8_report import mode_metrics, router_predictions


def test_router_predictions_uses_llm_only_when_forwarded():
    deberta = np.array([1, 0, 1, 0])
    llm = np.array([0, 1, -1, 1])  # -1 marks a failed Qwen3 answer
    forward = np.array([False, True, True, True])
    pred, fallbacks = router_predictions(deberta, llm, forward)
    assert pred.tolist() == [1, 1, 1, 1]
    assert fallbacks.tolist() == [False, False, True, False]


def test_mode_metrics_singleton_error_counts_only_kept_rows():
    gold = np.array([1, 0, 1, 0])
    pred = np.array([1, 1, 1, 0])
    forward = np.array([False, False, True, True])
    metrics = mode_metrics(gold, pred, forward)
    assert metrics["forwarded"] == 0.5
    assert metrics["singleton_error"] == 0.5  # rows 0-1 kept, row 1 wrong
    assert 0 <= metrics["f1"] <= 1


def test_mode_metrics_without_routing():
    gold = np.array([1, 0])
    metrics = mode_metrics(gold, gold, None)
    assert metrics["f1"] == 1.0
    assert metrics["forwarded"] is None
    assert metrics["singleton_error"] is None
