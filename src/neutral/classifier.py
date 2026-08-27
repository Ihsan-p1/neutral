import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from neutral.config import Settings

_MAX_LEN = 128


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling, then softmax.

    Temperature is the only surviving artifact of calibration. If this division
    is dropped, the model still returns plausible-looking probabilities that are
    simply uncalibrated, with no error anywhere. That is why it has its own test.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


class Classifier(Protocol):
    version: str

    def predict_proba(self, sentences: list[str]) -> list[float]:
        """Return the probability of the `biased` class for each sentence."""
        ...


class StubClassifier:
    """Deterministic classifier used before trained weights exist.

    Not random: the same sentence always yields the same score, so tests of the
    pipeline and API can assert exact values. It has no predictive value.
    """

    def __init__(self, temperature: float = 1.0, version: str = "stub-dev"):
        self.temperature = temperature
        self.version = version

    def predict_proba(self, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        logits = []
        for s in sentences:
            digest = hashlib.sha256(s.encode("utf-8")).digest()
            # Map the first two bytes onto the logit range [-3, 3].
            neutral = (digest[0] / 255.0) * 6.0 - 3.0
            biased = (digest[1] / 255.0) * 6.0 - 3.0
            logits.append([neutral, biased])
        probs = softmax_with_temperature(np.array(logits, dtype=np.float32), self.temperature)
        return [float(p[1]) for p in probs]


class OnnxClassifier:
    """Production classifier backed by a quantized ONNX model."""

    def __init__(self, model_dir: str, temperature: float, version: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(model_dir)
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=_MAX_LEN)
        self.tokenizer.enable_padding()
        self.temperature = temperature
        self.version = version

    def predict_proba(self, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        encoded = self.tokenizer.encode_batch(sentences)
        feeds = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        logits = self.session.run(None, feeds)[0]
        probs = softmax_with_temperature(logits, self.temperature)
        return [float(p[1]) for p in probs]


def load_classifier(settings: Settings) -> Classifier:
    """Select a backend from configuration.

    Temperature and model version are read from metadata stored beside the
    weights when present, so calibration travels with the model rather than
    living in environment variables that can drift apart from it.
    """
    if settings.model_backend == "stub":
        return StubClassifier(temperature=settings.temperature, version=settings.model_version)

    meta_path = Path(settings.model_dir) / "metadata.json"
    temperature = settings.temperature
    version = settings.model_version
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        temperature = float(meta.get("temperature", temperature))
        version = str(meta.get("version", version))

    return OnnxClassifier(model_dir=settings.model_dir, temperature=temperature, version=version)
