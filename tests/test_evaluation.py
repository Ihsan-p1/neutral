from neutral.evaluation import (
    EVALUATION_PATH,
    EVALUATIONS,
    load_evaluations,
    select_evaluation,
)

SERVED = "deberta-babe-f0s3"


def _triple(d):
    return (d.macro_f1, d.n_sentences, d.constant_baseline_macro_f1)


def test_committed_file_holds_served_numbers():
    ev = load_evaluations(EVALUATION_PATH)[SERVED]
    assert ev.threshold == 0.5
    assert [d.name for d in ev.domains] == ["babe", "basil", "sentra"]
    by = {d.name: d for d in ev.domains}
    assert _triple(by["babe"]) == (0.8765, 751, 0.3598)
    assert _triple(by["basil"]) == (0.5656, 5780, 0.4868)
    assert _triple(by["sentra"]) == (None, None, None)


def test_selects_block_for_served_onnx_model():
    assert select_evaluation(EVALUATIONS, SERVED, "onnx", 0.5) is not None


def test_threshold_written_differently_still_matches():
    assert select_evaluation(EVALUATIONS, SERVED, "onnx", 0.50000000001) is not None


def test_stub_backend_never_gets_block_even_with_real_version():
    assert select_evaluation(EVALUATIONS, SERVED, "stub", 0.5) is None


def test_other_threshold_gives_none():
    assert select_evaluation(EVALUATIONS, SERVED, "onnx", 0.6) is None


def test_unknown_version_gives_none():
    assert select_evaluation(EVALUATIONS, "other-model", "onnx", 0.5) is None


def test_missing_file_gives_empty(tmp_path):
    assert load_evaluations(tmp_path / "nope.json") == {}


def test_invalid_json_gives_empty(tmp_path):
    p = tmp_path / "e.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_evaluations(p) == {}


def test_entry_with_wrong_type_gives_empty(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(
        '{"m": {"model_version": "m", "metric": "macro_f1", '
        '"threshold": "high", "domains": []}}',
        encoding="utf-8",
    )
    assert load_evaluations(p) == {}


def test_selected_block_is_not_shared():
    a = select_evaluation(EVALUATIONS, SERVED, "onnx", 0.5)
    a.domains[0].macro_f1 = 0.0
    b = select_evaluation(EVALUATIONS, SERVED, "onnx", 0.5)
    assert b.domains[0].macro_f1 == 0.8765
