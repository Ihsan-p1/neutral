import pytest

from neutral.classifier import StubClassifier
from neutral.config import Settings
from neutral.detect import TextTooLargeError
from neutral.pipeline import analyze

CLF = StubClassifier(temperature=1.0, version="stub-test")

THREE = "The bill passed. The senator objected. Critics called it reckless."


def _settings(**kw) -> Settings:
    return Settings(**kw)


def test_counts_sentences():
    r = analyze(THREE, CLF, _settings())
    assert r.n_sentences == 3
    assert len(r.sentences) == 3


def test_article_score_is_proportion_not_mean():
    r = analyze(THREE, CLF, _settings())
    # Scores are rounded to 4 decimals for the response payload.
    assert r.article_score == pytest.approx(r.n_biased / r.n_sentences, abs=1e-4)


def test_sentence_offsets_map_back_to_input():
    r = analyze(THREE, CLF, _settings())
    for s in r.sentences:
        assert THREE[s.start : s.end] == s.text


def test_threshold_controls_labels():
    strict = analyze(THREE, CLF, _settings(decision_threshold=0.99))
    loose = analyze(THREE, CLF, _settings(decision_threshold=0.01))
    assert strict.n_biased <= loose.n_biased
    assert loose.n_biased == 3


def test_short_article_gets_warning():
    r = analyze("The bill passed.", CLF, _settings())
    assert "too_short" in r.warnings


def test_non_english_gets_warning_but_is_still_analyzed():
    text = (
        "Senator itu menyampaikan kritik keras terhadap rancangan anggaran "
        "pada hari Selasa kemarin di gedung parlemen."
    )
    r = analyze(text, CLF, _settings())
    assert "non_english" in r.warnings
    assert r.n_sentences >= 1


def test_truncates_and_warns_beyond_max_sentences():
    text = " ".join(f"Sentence number {i} appears here." for i in range(20))
    r = analyze(text, CLF, _settings(max_sentences=5))
    assert r.n_sentences == 5
    assert "truncated" in r.warnings


def test_rejects_oversized_text():
    with pytest.raises(TextTooLargeError):
        analyze("x" * 200, CLF, _settings(max_chars=100))


def test_empty_text_returns_zero_score():
    r = analyze("   ", CLF, _settings())
    assert r.n_sentences == 0
    assert r.article_score == 0.0
    assert r.confidence_mean == 0.0


def test_return_sentences_false_omits_detail_but_keeps_counts():
    r = analyze(THREE, CLF, _settings(), return_sentences=False)
    assert r.sentences == []
    assert r.n_sentences == 3


def test_model_version_is_reported():
    assert analyze(THREE, CLF, _settings()).model_version == "stub-test"


SERVED = StubClassifier(temperature=1.0, version="deberta-babe-f0s3")


def test_evaluation_present_for_served_model_on_onnx_backend():
    # analyze() never builds a classifier, so a stub with the served version
    # exercises the onnx branch of the selection without loading ONNX.
    r = analyze(THREE, SERVED, _settings(model_backend="onnx"))
    assert r.evaluation is not None
    assert r.evaluation.model_version == "deberta-babe-f0s3"


def test_evaluation_absent_on_stub_backend():
    assert analyze(THREE, SERVED, _settings(model_backend="stub")).evaluation is None


def test_evaluation_present_on_empty_input():
    r = analyze("   ", SERVED, _settings(model_backend="onnx"))
    assert r.n_sentences == 0
    assert r.evaluation is not None
