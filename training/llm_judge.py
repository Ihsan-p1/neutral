"""Qwen3 as the second opinion for sentences DeBERTa is unsure about (M8).

Talks to a local Ollama server over its REST API with stdlib urllib, so no new
dependency enters either venv. Every answer is cached as one JSON line keyed by
a hash of model, prompt version and sentence. The cache and the traces never
hold sentence text, only its hash and length.
"""

import dataclasses
import hashlib
import json
import pathlib
import time
import urllib.request
from collections.abc import Callable

MODEL_TAG = "qwen3:4b-instruct-2507-q4_K_M"
PROMPT_VERSION = "v1"
# Not "localhost": on Windows it resolves to ::1 first and each call waits about
# 2 s before falling back to IPv4, where Ollama listens.
OLLAMA_URL = "http://127.0.0.1:11434"
TIMEOUT_S = 60
# Transport errors in a row before a warm run gives up: the server is gone.
MAX_TRANSPORT_ERRORS = 3

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["biased", "neutral"]}},
    "required": ["label"],
}
_LABELS = {"biased": 1, "neutral": 0}

SYSTEM_PROMPT = (
    "You label one sentence from an English news article for lexical bias, following\n"
    'the BABE annotation guidelines. "biased" means the wording itself is one-sided:\n'
    "loaded, emotional, or opinionated words that take a side. \"neutral\" means the\n"
    "wording is factual and even-handed, even if the topic is controversial or the\n"
    "facts are unpleasant. Judge the words, not whether the claim is true.\n"
    'Answer with JSON only: {"label": "biased"} or {"label": "neutral"}.'
)

Transport = Callable[[str, dict | None, float], dict]


def urllib_transport(url: str, body: dict | None, timeout: float) -> dict:
    """GET when body is None, otherwise POST JSON. Returns the parsed response."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_ollama(
    transport: Transport = urllib_transport, url: str = OLLAMA_URL, model: str = MODEL_TAG
) -> None:
    """Fail before the first sentence if the server or the exact model tag is missing."""
    try:
        tags = transport(f"{url}/api/tags", None, 10)
    except Exception as exc:
        raise RuntimeError(f"Ollama is not reachable at {url} (model {model}): {exc}") from exc
    names = {entry.get("name") for entry in tags.get("models", [])}
    if model not in names:
        raise RuntimeError(f"model {model} is not installed; run: ollama pull {model}")


def build_messages(sentence: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]


def parse_label(content: str) -> int | None:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _LABELS.get(payload.get("label"))


@dataclasses.dataclass(frozen=True)
class JudgeResult:
    label: int | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None


def judge(
    sentence: str, transport: Transport = urllib_transport, url: str = OLLAMA_URL
) -> JudgeResult:
    """Ask Qwen3 for one label. A failure comes back as a result, never as an exception."""
    body = {
        "model": MODEL_TAG,
        "messages": build_messages(sentence),
        "format": LABEL_SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }
    start = time.perf_counter()
    try:
        response = transport(f"{url}/api/chat", body, TIMEOUT_S)
    except Exception as exc:
        return JudgeResult(None, time.perf_counter() - start, 0, 0, type(exc).__name__)
    latency = time.perf_counter() - start
    label = parse_label(response.get("message", {}).get("content", ""))
    return JudgeResult(
        label=label,
        latency_s=latency,
        prompt_tokens=int(response.get("prompt_eval_count", 0)),
        completion_tokens=int(response.get("eval_count", 0)),
        error=None if label is not None else "unparseable",
    )


def cache_key(sentence: str) -> str:
    return hashlib.sha256(f"{MODEL_TAG}|{PROMPT_VERSION}|{sentence}".encode()).hexdigest()


class JudgeCache:
    """Append-only JSONL of judge results. A truncated last line is skipped."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self._entries: dict[str, JudgeResult] | None = None

    def _load(self) -> dict[str, JudgeResult]:
        if self._entries is None:
            self._entries = {}
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(line)
                        key = record.pop("key")
                        self._entries[key] = JudgeResult(**record)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
        return self._entries

    def get(self, sentence: str) -> JudgeResult | None:
        return self._load().get(cache_key(sentence))

    def _ends_mid_line(self) -> bool:
        """True when a killed write left the file without its final newline."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as handle:
            handle.seek(-1, 2)
            return handle.read(1) != b"\n"

    def put(self, sentence: str, result: JudgeResult) -> None:
        key = cache_key(sentence)
        self._load()[key] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"key": key, **dataclasses.asdict(result)}) + "\n"
        if self._ends_mid_line():
            line = "\n" + line
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def judge_pending(
    texts: list[str], cache: JudgeCache, judge_fn: Callable[[str], JudgeResult] = judge
) -> int:
    """Judge and cache each text; returns how many came back without a label.

    Raises once MAX_TRANSPORT_ERRORS calls in a row fail to reach the server, so a
    dead Ollama stops the run instead of failing every remaining sentence.
    """
    failures = 0
    in_a_row = 0
    for done, text in enumerate(texts, start=1):
        result = judge_fn(text)
        # A transport error is not an answer: leave it uncached so the next run retries.
        if result.error in (None, "unparseable"):
            cache.put(text, result)
            in_a_row = 0
        else:
            in_a_row += 1
            if in_a_row >= MAX_TRANSPORT_ERRORS:
                raise RuntimeError(
                    f"{in_a_row} calls in a row failed ({result.error}) after {done} of "
                    f"{len(texts)}; check Ollama, then rerun --warm to resume"
                )
        if result.label is None:
            failures += 1
        if done % 100 == 0:
            print(f"warm progress      : {done} of {len(texts)}, failures {failures}")
    return failures


def route(set_size: int, deberta_label: int, llm: JudgeResult | None) -> tuple[int, str]:
    """Final label and its source for one sentence."""
    if set_size == 1:
        return deberta_label, "deberta"
    if llm is None or llm.label is None:
        return deberta_label, "deberta_fallback"
    return llm.label, "qwen3"


def trace_record(
    sentence: str,
    set_size: int,
    source: str,
    deberta_latency_s: float,
    llm: JudgeResult | None,
) -> dict:
    """One routing decision, without the sentence."""
    return {
        "sentence_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(),
        "n_chars": len(sentence),
        "set_size": set_size,
        "source": source,
        "deberta_latency_s": deberta_latency_s,
        "llm_latency_s": llm.latency_s if llm else None,
        "prompt_tokens": llm.prompt_tokens if llm else 0,
        "completion_tokens": llm.completion_tokens if llm else 0,
        "model": MODEL_TAG,
        "prompt_version": PROMPT_VERSION,
    }
