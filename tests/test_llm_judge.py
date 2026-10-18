import json
from urllib.error import URLError

import pytest

from training import llm_judge
from training.llm_judge import (
    LABEL_SCHEMA,
    MODEL_TAG,
    JudgeCache,
    JudgeResult,
    check_ollama,
    judge,
    parse_label,
    route,
    trace_record,
)

SENTENCE = "The senator's reckless scheme gutted the program."


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append((url, body, timeout))
        if self.error:
            raise self.error
        return self.response


def chat_response(content):
    return {"message": {"content": content}, "prompt_eval_count": 120, "eval_count": 7}


def test_check_ollama_missing_tag():
    transport = FakeTransport({"models": [{"name": "qwen3:4b"}]})
    with pytest.raises(RuntimeError, match="qwen3:4b-instruct-2507-q4_K_M"):
        check_ollama(transport)


def test_check_ollama_unreachable():
    with pytest.raises(RuntimeError):
        check_ollama(FakeTransport(error=URLError("refused")))


def test_judge_request_body():
    transport = FakeTransport(chat_response('{"label": "biased"}'))
    result = judge(SENTENCE, transport)
    url, body, _ = transport.calls[0]
    assert url.endswith("/api/chat")
    assert body["model"] == MODEL_TAG
    assert body["format"] == LABEL_SCHEMA
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0, "seed": 0}
    assert body["messages"][-1] == {"role": "user", "content": SENTENCE}
    assert result.label == 1
    assert (result.prompt_tokens, result.completion_tokens) == (120, 7)


def test_parse_label():
    assert parse_label('{"label": "biased"}') == 1
    assert parse_label('{"label": "neutral"}') == 0
    assert parse_label('{"label": "maybe"}') is None
    assert parse_label("not json") is None
    assert parse_label("{}") is None


def test_judge_error_is_result_not_exception():
    result = judge(SENTENCE, FakeTransport(error=TimeoutError("slow")))
    assert result.label is None
    assert result.error


def test_cache_roundtrip_has_no_text(tmp_path):
    path = tmp_path / "cache.jsonl"
    result = JudgeResult(label=1, latency_s=0.4, prompt_tokens=120, completion_tokens=7, error=None)
    JudgeCache(path).put(SENTENCE, result)
    assert JudgeCache(path).get(SENTENCE) == result
    raw = path.read_text(encoding="utf-8")
    assert SENTENCE not in raw
    assert "reckless" not in raw


def test_cache_ignores_other_prompt_version(tmp_path, monkeypatch):
    path = tmp_path / "cache.jsonl"
    result = JudgeResult(label=0, latency_s=0.3, prompt_tokens=1, completion_tokens=1, error=None)
    monkeypatch.setattr(llm_judge, "PROMPT_VERSION", "v0")
    JudgeCache(path).put(SENTENCE, result)
    monkeypatch.setattr(llm_judge, "PROMPT_VERSION", "v1")
    assert JudgeCache(path).get(SENTENCE) is None


def test_cache_skips_truncated_last_line(tmp_path):
    path = tmp_path / "cache.jsonl"
    result = JudgeResult(label=1, latency_s=0.4, prompt_tokens=1, completion_tokens=1, error=None)
    JudgeCache(path).put(SENTENCE, result)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "abc", "label"')
    assert JudgeCache(path).get(SENTENCE) == result


def test_route():
    yes = JudgeResult(label=1, latency_s=0.5, prompt_tokens=1, completion_tokens=1, error=None)
    broken = JudgeResult(label=None, latency_s=0.5, prompt_tokens=0, completion_tokens=0, error="x")
    assert route(1, 0, None) == (0, "deberta")
    assert route(2, 0, yes) == (1, "qwen3")
    assert route(0, 1, broken) == (1, "deberta_fallback")


def test_trace_has_no_text():
    llm = JudgeResult(label=1, latency_s=0.5, prompt_tokens=120, completion_tokens=7, error=None)
    record = trace_record(SENTENCE, 2, "qwen3", 0.05, llm)
    assert set(record) == {
        "sentence_sha256", "n_chars", "set_size", "source", "deberta_latency_s",
        "llm_latency_s", "prompt_tokens", "completion_tokens", "model", "prompt_version",
    }
    assert SENTENCE not in json.dumps(record)
    assert record["n_chars"] == len(SENTENCE)


def test_ollama_url_avoids_localhost():
    # On Windows "localhost" tries ::1 first and waits about 2 s per call before
    # falling back to IPv4, which Ollama listens on (measured 2026-10-19).
    assert llm_judge.OLLAMA_URL == "http://127.0.0.1:11434"


def test_check_ollama_unreachable_names_tag():
    with pytest.raises(RuntimeError, match="qwen3:4b-instruct-2507-q4_K_M"):
        check_ollama(FakeTransport(error=URLError("refused")))
