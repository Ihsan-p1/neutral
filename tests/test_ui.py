from neutral.classifier import StubClassifier
from neutral.config import Settings
from neutral.ui import make_analyze_fn

CLF = StubClassifier(temperature=1.0, version="stub-test")
THREE = "The bill passed. The senator objected. Critics called it reckless."


def fn(**settings_kw):
    return make_analyze_fn(Settings(**settings_kw), lambda: CLF)


def test_returns_summary_and_highlights():
    summary, highlights = fn()(THREE)
    assert "3" in summary
    assert len(highlights) == 3
    for text, label in highlights:
        assert isinstance(text, str)
        assert label in {"biased", None}


def test_empty_input_is_handled():
    summary, highlights = fn()("")
    assert highlights == []
    assert "0 sentences" in summary


def test_oversized_input_returns_message_not_exception():
    summary, highlights = fn(max_chars=100)("x" * 200)
    assert "too long" in summary.lower()
    assert highlights == []


def test_warnings_are_surfaced_to_the_reader():
    summary, _ = fn()("The bill passed.")
    assert "short" in summary.lower()


def test_summary_states_both_count_and_total():
    # A proportion alone is misleading; the reader must see how many
    # sentences it was computed from.
    summary, _ = fn()(THREE)
    assert "of 3 sentences" in summary


def test_ui_makes_no_network_call_at_startup():
    # Gradio phones api.gradio.app unless analytics are disabled. That breaks
    # the "no network call at boot" constraint and slows every cold start.
    from neutral.ui import build_ui

    demo = build_ui(Settings(), lambda: CLF)
    assert demo.analytics_enabled is False


def test_gradio_telemetry_is_disabled_at_import():
    # Blocks(analytics_enabled=False) does not cover Gradio's messaging call;
    # only the env var does, and it must be set before gradio is imported.
    import os

    import neutral  # noqa: F401

    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
