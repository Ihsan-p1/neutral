import json
import logging

from neutral.logging_setup import JsonFormatter


def _format(record_extra: dict) -> dict:
    record = logging.LogRecord(
        name="neutral",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request",
        args=(),
        exc_info=None,
    )
    for k, v in record_extra.items():
        setattr(record, k, v)
    return json.loads(JsonFormatter().format(record))


def test_emits_valid_json_with_core_fields():
    out = _format({"path": "/analyze", "latency_ms": 12.5, "status": 200})
    assert out["message"] == "request"
    assert out["path"] == "/analyze"
    assert out["status"] == 200
    assert out["level"] == "INFO"


def test_never_emits_article_text():
    # User text must never reach the log. If such a field slips into the
    # record, the formatter drops it.
    out = _format({"text": "user secret", "n_chars": 11})
    assert "user secret" not in json.dumps(out)
    assert out["n_chars"] == 11


def test_drops_every_known_text_bearing_field():
    out = _format(
        {
            "text": "a",
            "article": "b",
            "body": "c",
            "content": "d",
            "sentences": ["e"],
            "n_sentences": 5,
        }
    )
    dumped = json.dumps(out)
    for leaked in ("a", "b", "c", "d", "e"):
        assert f'"{leaked}"' not in dumped
    assert out["n_sentences"] == 5
