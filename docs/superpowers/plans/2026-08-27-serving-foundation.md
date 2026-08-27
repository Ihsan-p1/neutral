# Serving Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun seluruh lapisan serving Media Neutrality Classifier — API, segmentasi, agregasi, UI, container, CI — sampai siap deploy, tanpa memerlukan GPU atau bobot model terlatih.

**Architecture:** Satu container berisi FastAPI yang melayani API sekaligus antarmuka Gradio yang di-mount ke aplikasi yang sama. Classifier berada di balik satu interface kecil dengan dua implementasi: `StubClassifier` (deterministik, dipakai selama pengembangan dan di CI) dan `OnnxClassifier` (dipakai setelah bobot tersedia). Seluruh pipeline dapat dikembangkan dan diuji penuh sebelum model pertama dilatih.

**Tech Stack:** Python 3.13, FastAPI, Gradio, onnxruntime (CPU), pysbd, py3langid, slowapi, pydantic-settings, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md`

## Global Constraints

- **Python 3.13** untuk lokal dan container. Base image `python:3.13-slim`, bukan alpine.
- **PyTorch tidak boleh masuk image produksi.** Dockerfile hanya menyalin `src/`. Direktori `training/` tidak pernah disalin.
- **Port dibaca dari `$PORT`**, default 8000. Tidak boleh di-hardcode.
- **Handler inferensi ditulis `def`, bukan `async def`.** FastAPI menjalankannya di threadpool; `async def` yang memanggil ONNX secara sinkron membekukan seluruh server.
- **Teks artikel tidak pernah masuk log.** Hanya panjangnya yang dicatat.
- **Batas:** 50.000 karakter per request, 300 kalimat per request, 20 request/menit dan 300/jam per IP.
- **Commit hanya atas nama `ihsan-p1`.** Tanpa trailer `Co-Authored-By`, tanpa footer generator.
- **Test memakai pytest dengan assert biasa.** Tanpa fixture berlapis, tanpa mock untuk onnxruntime.
- **Nilai `article_score` adalah proporsi kalimat biased**, bukan rata-rata probabilitas.
- **Temperature scaling selalu diterapkan** pada logits sebelum softmax, termasuk oleh stub.

---

## File Structure

| File | Tanggung jawab |
|---|---|
| `src/neutral/config.py` | Seluruh konfigurasi runtime dari environment variable |
| `src/neutral/schemas.py` | Kontrak request dan response |
| `src/neutral/segment.py` | Memecah artikel menjadi kalimat beserta offset |
| `src/neutral/detect.py` | Deteksi bahasa dan validasi ukuran input |
| `src/neutral/classifier.py` | Interface classifier, implementasi stub dan ONNX, temperature |
| `src/neutral/pipeline.py` | Merangkai segmentasi, klasifikasi, dan agregasi menjadi satu hasil |
| `src/neutral/logging_setup.py` | Formatter JSON dan konfigurasi logger |
| `src/neutral/api.py` | Rute FastAPI dan rate limiting |
| `src/neutral/ui.py` | Antarmuka Gradio |
| `src/neutral/app.py` | Menggabungkan API dan UI menjadi satu aplikasi |
| `Dockerfile` | Image produksi dua tahap |
| `.github/workflows/ci.yml` | Lint, test, build image, smoke test |

Pemisahan `pipeline.py` dari `api.py` disengaja: seluruh logika analisis dapat diuji tanpa menyentuh HTTP, dan antarmuka Gradio memanggil fungsi yang sama dengan yang dipanggil endpoint.

---

## Task 1: Bootstrap repositori

**Files:**
- Create: `.gitignore`, `README.md`, `requirements.txt`, `pyproject.toml`, `CLAUDE.md`
- Create: `src/neutral/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Consumes: tidak ada
- Produces: paket `neutral` dapat diimpor; `pytest` dapat dijalankan

- [ ] **Step 1: Inisialisasi git dan hubungkan remote**

```bash
cd "c:/Users/MIS/Documents/Ihsan_Presuniv/Project/neutral"
git init -b main
git remote add origin https://github.com/Ihsan-p1/neutral.git
git config user.name "Ihsan-p1"
git config user.email "Ihsan-p1@users.noreply.github.com"
```

- [ ] **Step 2: Tulis `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ruff_cache/
*.egg-info/
models/
*.onnx
.env
.DS_Store
```

`models/` diabaikan karena bobot model diambil dari HF Hub saat build, bukan disimpan di git.

- [ ] **Step 3: Tulis `requirements.txt`**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pydantic-settings==2.7.0
pysbd==0.3.4
py3langid==0.3.0
slowapi==0.1.9
gradio==5.9.1
onnxruntime==1.20.1
tokenizers==0.21.0
numpy==2.2.1
```

Dan `requirements-dev.txt`:

```
-r requirements.txt
pytest==8.3.4
httpx==0.28.1
ruff==0.8.6
```

- [ ] **Step 4: Tulis `pyproject.toml`**

```toml
[project]
name = "neutral"
version = "0.1.0"
requires-python = ">=3.13"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

`pythonpath = ["src"]` membuat pytest menemukan paket tanpa perlu instalasi editable.

- [ ] **Step 5: Buat paket kosong dan virtual environment**

```bash
mkdir -p src/neutral tests
touch src/neutral/__init__.py tests/__init__.py
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

- [ ] **Step 6: Verifikasi seluruh dependency benar-benar terpasang di Python 3.13**

```bash
.venv/Scripts/python.exe -c "import fastapi, pysbd, py3langid, slowapi, gradio, onnxruntime, tokenizers; print('all imports ok')"
```

Expected: `all imports ok`

Bila `pysbd` gagal diimpor atau error saat runtime di Python 3.13, hentikan dan lapor — fallback yang dipakai adalah `blingfire`, dan Task 3 harus disesuaikan. Jangan menulis segmenter regex sendiri tanpa membicarakannya lebih dulu.

- [ ] **Step 7: Tulis `README.md` awal**

```markdown
# Media Neutrality Classifier

Deteksi bias leksikal pada level kalimat berita berbahasa Inggris.

**Status:** dalam pengembangan — lapisan serving dibangun lebih dulu, model menyusul.

- PRD: `PRD_Media_Neutrality_Classifier.md`
- Desain deployment: `docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md`
- Rencana implementasi: `docs/superpowers/plans/2026-08-27-serving-foundation.md`

## Menjalankan secara lokal

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn neutral.app:app --reload
```

Tanpa bobot model, aplikasi berjalan dengan `NEUTRAL_MODEL_BACKEND=stub`.
```

- [ ] **Step 8: Commit**

```bash
git add .gitignore README.md requirements.txt requirements-dev.txt pyproject.toml src tests docs PRD_Media_Neutrality_Classifier.md
git commit -m "chore: bootstrap repo structure and dependencies"
```

---

## Task 2: Konfigurasi dan skema

**Files:**
- Create: `src/neutral/config.py`, `src/neutral/schemas.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: tidak ada
- Produces:
  - `Settings` dengan field `model_backend: str`, `model_dir: str`, `decision_threshold: float`, `temperature: float`, `max_chars: int`, `max_sentences: int`, `min_sentences_warn: int`, `rate_limit_minute: str`, `rate_limit_hour: str`, `log_level: str`, `model_version: str`
  - `get_settings() -> Settings` (di-cache)
  - `AnalyzeRequest`, `SentenceResult`, `AnalyzeResponse`, `HealthResponse`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_config.py
from neutral.config import Settings


def test_defaults_are_safe():
    s = Settings()
    assert s.model_backend == "stub"
    assert s.max_chars == 50_000
    assert s.max_sentences == 300
    assert 0.0 < s.decision_threshold < 1.0
    assert s.temperature > 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("NEUTRAL_MAX_CHARS", "1234")
    monkeypatch.setenv("NEUTRAL_MODEL_BACKEND", "onnx")
    s = Settings()
    assert s.max_chars == 1234
    assert s.model_backend == "onnx"


def test_rejects_invalid_backend():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(model_backend="nonsense")
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.config'`

- [ ] **Step 3: Tulis `src/neutral/config.py`**

```python
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Seluruh konfigurasi runtime. Tidak ada nilai yang di-hardcode di tempat lain."""

    model_config = SettingsConfigDict(env_prefix="NEUTRAL_", extra="ignore")

    # Backend model. "stub" dipakai sebelum bobot tersedia dan di CI.
    model_backend: Literal["stub", "onnx"] = "stub"
    model_dir: str = "models/current"
    model_version: str = "stub-dev"

    # Ambang keputusan biner dan temperature hasil kalibrasi.
    decision_threshold: float = 0.5
    temperature: float = 1.0

    # Batas input.
    max_chars: int = 50_000
    max_sentences: int = 300
    min_sentences_warn: int = 5

    # Rate limiting.
    rate_limit_minute: str = "20/minute"
    rate_limit_hour: str = "300/hour"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Tulis `src/neutral/schemas.py`**

```python
from typing import Literal

from pydantic import BaseModel, Field

Warning_ = Literal["non_english", "too_short", "truncated"]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    return_sentences: bool = True


class SentenceResult(BaseModel):
    text: str
    label: Literal["biased", "neutral"]
    prob: float
    start: int
    end: int


class AnalyzeResponse(BaseModel):
    article_score: float
    n_sentences: int
    n_biased: int
    confidence_mean: float
    sentences: list[SentenceResult]
    model_version: str
    warnings: list[Warning_]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    backend: str
```

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/neutral/config.py src/neutral/schemas.py tests/test_config.py
git commit -m "feat: add settings and API schemas"
```

---

## Task 3: Segmentasi kalimat

**Files:**
- Create: `src/neutral/segment.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: tidak ada
- Produces: `split_sentences(text: str) -> list[Sentence]`, dengan `Sentence` sebagai dataclass ber-field `text: str`, `start: int`, `end: int`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_segment.py
from neutral.segment import split_sentences


def test_splits_simple_sentences():
    out = split_sentences("The bill passed. The senator objected.")
    assert len(out) == 2
    assert out[0].text == "The bill passed."


def test_abbreviations_do_not_split():
    # Ini alasan utama memakai pysbd, bukan split pada titik.
    text = "The U.S. economy grew. Mr. Smith of Acme Inc. disagreed."
    out = split_sentences(text)
    assert len(out) == 2
    assert "U.S." in out[0].text
    assert "Inc." in out[1].text


def test_offsets_map_back_to_source():
    text = "The bill passed. The senator objected."
    for s in split_sentences(text):
        assert text[s.start : s.end] == s.text


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_segment.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.segment'`

- [ ] **Step 3: Tulis `src/neutral/segment.py`**

```python
from dataclasses import dataclass

import pysbd


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


_segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)


def split_sentences(text: str) -> list[Sentence]:
    """Pecah artikel menjadi kalimat beserta offset ke teks asli.

    Offset dipertahankan agar antarmuka dapat menyorot kalimat pada teks
    yang dikirim pengguna tanpa mencocokkan string ulang.
    """
    if not text or not text.strip():
        return []

    out: list[Sentence] = []
    for span in _segmenter.segment(text):
        raw = span.sent
        stripped = raw.strip()
        if not stripped:
            continue
        # Sesuaikan offset setelah strip agar tetap memetakan ke teks asli.
        lead = len(raw) - len(raw.lstrip())
        start = span.start + lead
        out.append(Sentence(text=stripped, start=start, end=start + len(stripped)))
    return out
```

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_segment.py -v`
Expected: 4 passed

Bila `test_abbreviations_do_not_split` gagal, jangan menambal dengan daftar singkatan buatan sendiri. Laporkan hasilnya — itu sinyal `pysbd` bermasalah di Python 3.13 dan fallback `blingfire` harus dipertimbangkan.

- [ ] **Step 5: Commit**

```bash
git add src/neutral/segment.py tests/test_segment.py
git commit -m "feat: add sentence segmentation with source offsets"
```

---

## Task 4: Deteksi bahasa dan validasi input

**Files:**
- Create: `src/neutral/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `neutral.config.get_settings`
- Produces: `is_english(text: str) -> bool`, `TextTooLargeError`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_detect.py
from neutral.detect import is_english


def test_detects_english():
    assert is_english(
        "The senator delivered a scathing rebuke of the proposed budget on Tuesday."
    )


def test_detects_non_english():
    assert not is_english(
        "Senator itu menyampaikan kritik keras terhadap rancangan anggaran pada hari Selasa."
    )


def test_short_text_defaults_to_english():
    # Teks sangat pendek tidak dapat dideteksi andal. Jangan menandai
    # pengguna secara keliru; anggap dalam cakupan dan biarkan lolos.
    assert is_english("Budget cuts.")
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detect.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.detect'`

- [ ] **Step 3: Tulis `src/neutral/detect.py`**

```python
import py3langid as langid

# Batas di bawahnya deteksi bahasa tidak dapat diandalkan.
_MIN_CHARS_FOR_DETECTION = 40


class TextTooLargeError(ValueError):
    """Teks melebihi batas yang dikonfigurasi."""


def is_english(text: str) -> bool:
    """Kembalikan True bila teks tampak berbahasa Inggris.

    Teks yang terlalu pendek dianggap Inggris: menandai input yang sah
    sebagai di luar cakupan lebih merugikan daripada melewatkan satu
    input non-Inggris yang pendek.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS_FOR_DETECTION:
        return True
    lang, _ = langid.classify(stripped)
    return lang == "en"
```

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detect.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/neutral/detect.py tests/test_detect.py
git commit -m "feat: add language detection for out-of-scope input"
```

---

## Task 5: Classifier — interface, stub, dan temperature

**Files:**
- Create: `src/neutral/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `neutral.config.Settings`
- Produces:
  - `softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray`
  - `Classifier` (Protocol) dengan `predict_proba(sentences: list[str]) -> list[float]` dan properti `version: str`
  - `StubClassifier`, `OnnxClassifier`
  - `load_classifier(settings: Settings) -> Classifier`

Probabilitas yang dikembalikan selalu **probabilitas kelas `biased`**.

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_classifier.py
import numpy as np

from neutral.classifier import StubClassifier, softmax_with_temperature


def test_softmax_sums_to_one():
    out = softmax_with_temperature(np.array([[2.0, 1.0]]), temperature=1.0)
    assert abs(out[0].sum() - 1.0) < 1e-6


def test_temperature_flattens_distribution():
    logits = np.array([[4.0, 0.0]])
    sharp = softmax_with_temperature(logits, temperature=1.0)
    flat = softmax_with_temperature(logits, temperature=5.0)
    # Temperature lebih tinggi menghasilkan keyakinan lebih rendah.
    assert flat[0].max() < sharp[0].max()


def test_temperature_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        softmax_with_temperature(np.array([[1.0, 0.0]]), temperature=0.0)


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


def test_stub_handles_empty_input():
    assert StubClassifier(temperature=1.0).predict_proba([]) == []
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_classifier.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.classifier'`

- [ ] **Step 3: Tulis `src/neutral/classifier.py`**

```python
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from neutral.config import Settings

_MAX_LEN = 128


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Terapkan temperature scaling lalu softmax.

    Temperature adalah satu-satunya sisa dari proses kalibrasi. Bila
    pembagian ini hilang, model tetap mengembalikan angka yang tampak
    wajar tetapi tidak terkalibrasi, tanpa error apa pun. Karena itu
    langkah ini punya test tersendiri.
    """
    if temperature <= 0:
        raise ValueError(f"temperature harus positif, diterima {temperature}")
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


class Classifier(Protocol):
    version: str

    def predict_proba(self, sentences: list[str]) -> list[float]:
        """Kembalikan probabilitas kelas `biased` untuk tiap kalimat."""
        ...


class StubClassifier:
    """Classifier deterministik untuk pengembangan sebelum bobot tersedia.

    Bukan sekadar nilai acak: hasilnya stabil untuk kalimat yang sama,
    sehingga test terhadap pipeline dan API dapat menyatakan nilai pasti.
    Tidak memiliki nilai prediktif apa pun.
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
            # Dua byte pertama dipetakan ke rentang logit [-3, 3].
            biased = (digest[0] / 255.0) * 6.0 - 3.0
            neutral = (digest[1] / 255.0) * 6.0 - 3.0
            logits.append([neutral, biased])
        probs = softmax_with_temperature(np.array(logits, dtype=np.float32), self.temperature)
        return [float(p[1]) for p in probs]


class OnnxClassifier:
    """Classifier produksi yang memuat model ONNX terkuantisasi."""

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
    """Pilih backend berdasarkan konfigurasi.

    Temperature dan versi model dibaca dari metadata di samping bobot bila
    tersedia, sehingga hasil kalibrasi ikut berpindah bersama model.
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

    return OnnxClassifier(
        model_dir=settings.model_dir, temperature=temperature, version=version
    )
```

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_classifier.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/neutral/classifier.py tests/test_classifier.py
git commit -m "feat: add classifier interface with stub backend and temperature scaling"
```

---

## Task 6: Pipeline analisis dan agregasi

**Files:**
- Create: `src/neutral/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `split_sentences`, `is_english`, `Classifier`, `Settings`, skema dari Task 2
- Produces: `analyze(text: str, clf: Classifier, settings: Settings, return_sentences: bool = True) -> AnalyzeResponse`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_pipeline.py
import pytest

from neutral.classifier import StubClassifier
from neutral.config import Settings
from neutral.detect import TextTooLargeError
from neutral.pipeline import analyze

CLF = StubClassifier(temperature=1.0, version="stub-test")


def _settings(**kw) -> Settings:
    return Settings(**kw)


def test_counts_sentences():
    text = "The bill passed. The senator objected. Critics called it reckless."
    r = analyze(text, CLF, _settings())
    assert r.n_sentences == 3
    assert len(r.sentences) == 3


def test_article_score_is_proportion_not_mean():
    text = "The bill passed. The senator objected. Critics called it reckless."
    r = analyze(text, CLF, _settings())
    # Proporsi harus tepat n_biased / n_sentences.
    assert r.article_score == pytest.approx(r.n_biased / r.n_sentences)


def test_short_article_gets_warning():
    r = analyze("The bill passed.", CLF, _settings())
    assert "too_short" in r.warnings


def test_non_english_gets_warning_but_still_analyzed():
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


def test_return_sentences_false_omits_detail():
    text = "The bill passed. The senator objected."
    r = analyze(text, CLF, _settings(), return_sentences=False)
    assert r.sentences == []
    assert r.n_sentences == 2
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.pipeline'`

- [ ] **Step 3: Tulis `src/neutral/pipeline.py`**

```python
from neutral.classifier import Classifier
from neutral.config import Settings
from neutral.detect import TextTooLargeError, is_english
from neutral.schemas import AnalyzeResponse, SentenceResult, Warning_
from neutral.segment import split_sentences


def analyze(
    text: str,
    clf: Classifier,
    settings: Settings,
    return_sentences: bool = True,
) -> AnalyzeResponse:
    """Jalankan pipeline lengkap: validasi, segmentasi, klasifikasi, agregasi."""
    if len(text) > settings.max_chars:
        raise TextTooLargeError(
            f"teks {len(text)} karakter melebihi batas {settings.max_chars}"
        )

    warnings: list[Warning_] = []
    if text.strip() and not is_english(text):
        warnings.append("non_english")

    sentences = split_sentences(text)
    if len(sentences) > settings.max_sentences:
        sentences = sentences[: settings.max_sentences]
        warnings.append("truncated")

    if not sentences:
        return AnalyzeResponse(
            article_score=0.0,
            n_sentences=0,
            n_biased=0,
            confidence_mean=0.0,
            sentences=[],
            model_version=clf.version,
            warnings=warnings,
        )

    if len(sentences) < settings.min_sentences_warn:
        warnings.append("too_short")

    probs = clf.predict_proba([s.text for s in sentences])

    results = [
        SentenceResult(
            text=s.text,
            label="biased" if p >= settings.decision_threshold else "neutral",
            prob=round(p, 4),
            start=s.start,
            end=s.end,
        )
        for s, p in zip(sentences, probs, strict=True)
    ]

    n_biased = sum(1 for r in results if r.label == "biased")
    # Skor artikel adalah proporsi, bukan rata-rata probabilitas: artikel
    # pendek dan panjang harus dapat dibandingkan secara langsung.
    article_score = n_biased / len(results)
    # Keyakinan dihitung sebagai jarak dari titik ragu, bukan rata-rata
    # probabilitas biased — nilai 0.5 berarti model paling tidak yakin.
    confidence_mean = sum(abs(p - 0.5) * 2 for p in probs) / len(probs)

    return AnalyzeResponse(
        article_score=round(article_score, 4),
        n_sentences=len(results),
        n_biased=n_biased,
        confidence_mean=round(confidence_mean, 4),
        sentences=results if return_sentences else [],
        model_version=clf.version,
        warnings=warnings,
    )
```

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/neutral/pipeline.py tests/test_pipeline.py
git commit -m "feat: add analysis pipeline with proportion-based article scoring"
```

---

## Task 7: Logging JSON

**Files:**
- Create: `src/neutral/logging_setup.py`
- Test: `tests/test_logging_setup.py`

**Interfaces:**
- Consumes: tidak ada
- Produces: `JsonFormatter`, `configure_logging(level: str) -> None`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_logging_setup.py
import json
import logging

from neutral.logging_setup import JsonFormatter


def _format(record_extra: dict) -> dict:
    record = logging.LogRecord(
        name="neutral", level=logging.INFO, pathname=__file__,
        lineno=1, msg="request", args=(), exc_info=None,
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
    # Teks pengguna tidak boleh pernah tertulis di log. Bila field ini
    # menyelinap ke record, formatter harus membuangnya.
    out = _format({"text": "rahasia pengguna", "n_chars": 16})
    assert "rahasia pengguna" not in json.dumps(out)
    assert out["n_chars"] == 16
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logging_setup.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.logging_setup'`

- [ ] **Step 3: Tulis `src/neutral/logging_setup.py`**

```python
import json
import logging
import sys

_STANDARD = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}

# Field yang tidak boleh pernah keluar ke log meski dikirim pemanggil.
_FORBIDDEN = {"text", "article", "body", "content", "sentences"}


class JsonFormatter(logging.Formatter):
    """Formatter satu baris JSON ke stdout.

    Membuang field yang mungkin memuat teks pengguna. Ini pengaman
    terakhir: pemanggil tetap wajib tidak mengirimkannya.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD or key in _FORBIDDEN or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
```

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_logging_setup.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/neutral/logging_setup.py tests/test_logging_setup.py
git commit -m "feat: add JSON logging that strips user text"
```

---

## Task 8: API FastAPI dengan rate limiting

**Files:**
- Create: `src/neutral/api.py`, `src/neutral/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: seluruh modul sebelumnya
- Produces: `create_api() -> FastAPI` di `api.py`; `app` di `app.py`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_api.py
from fastapi.testclient import TestClient

from neutral.api import create_api


def client() -> TestClient:
    return TestClient(create_api())


def test_health_reports_backend():
    r = client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"


def test_analyze_returns_expected_shape():
    r = client().post(
        "/analyze",
        json={"text": "The bill passed. The senator objected. Critics called it reckless."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_sentences"] == 3
    assert len(body["sentences"]) == 3
    assert 0.0 <= body["article_score"] <= 1.0
    assert body["sentences"][0]["label"] in {"biased", "neutral"}


def test_analyze_rejects_empty_text():
    r = client().post("/analyze", json={"text": ""})
    assert r.status_code == 422


def test_analyze_rejects_oversized_text():
    r = client().post("/analyze", json={"text": "x" * 60_000})
    assert r.status_code == 413


def test_version_endpoint():
    r = client().get("/version")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_analyze_handler_is_sync():
    # Handler CPU-bound wajib sinkron agar FastAPI menjalankannya di
    # threadpool. `async def` akan membekukan event loop selama inferensi.
    import inspect

    from neutral import api

    assert not inspect.iscoroutinefunction(api.analyze_endpoint)
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.api'`

- [ ] **Step 3: Tulis `src/neutral/api.py`**

```python
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from neutral.classifier import load_classifier
from neutral.config import get_settings
from neutral.detect import TextTooLargeError
from neutral.logging_setup import configure_logging
from neutral.pipeline import analyze
from neutral.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

log = logging.getLogger("neutral.api")

settings = get_settings()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_minute, settings.rate_limit_hour],
)

_classifier = None


def get_classifier():
    """Muat classifier sekali, saat pertama dibutuhkan."""
    global _classifier
    if _classifier is None:
        _classifier = load_classifier(settings)
    return _classifier


# Handler sinkron: FastAPI menjalankannya di threadpool sehingga inferensi
# yang CPU-bound tidak memblokir event loop.
def analyze_endpoint(request: Request, payload: AnalyzeRequest) -> AnalyzeResponse:
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    try:
        result = analyze(
            payload.text,
            get_classifier(),
            settings,
            return_sentences=payload.return_sentences,
        )
    except TextTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    # Hanya metadata yang dicatat. Teks artikel tidak pernah masuk log.
    log.info(
        "analyze",
        extra={
            "request_id": request_id,
            "path": "/analyze",
            "n_chars": len(payload.text),
            "n_sentences": result.n_sentences,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": 200,
        },
    )
    return result


def health_endpoint() -> HealthResponse:
    try:
        clf = get_classifier()
        clf.predict_proba(["Health check sentence."])
        status = "ok"
    except Exception:
        log.exception("health check failed")
        return HealthResponse(
            status="degraded", model_version="unknown", backend=settings.model_backend
        )
    return HealthResponse(
        status=status, model_version=clf.version, backend=settings.model_backend
    )


def version_endpoint() -> dict:
    return {"model_version": get_classifier().version, "backend": settings.model_backend}


def create_api() -> FastAPI:
    configure_logging(settings.log_level)
    api = FastAPI(title="Media Neutrality Classifier", version="0.1.0")
    api.state.limiter = limiter
    api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    api.post("/analyze", response_model=AnalyzeResponse)(analyze_endpoint)
    api.get("/health", response_model=HealthResponse)(health_endpoint)
    api.get("/version")(version_endpoint)

    @api.exception_handler(TextTooLargeError)
    def _too_large(_request: Request, exc: TextTooLargeError) -> JSONResponse:
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    return api
```

- [ ] **Step 4: Tulis `src/neutral/app.py`**

```python
from neutral.api import create_api

# Titik masuk produksi. UI Gradio di-mount di Task 9.
app = create_api()
```

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: 6 passed

- [ ] **Step 6: Jalankan seluruh test dan lint**

```bash
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/python.exe -m ruff check src tests
```

Expected: seluruh test lulus, ruff bersih.

- [ ] **Step 7: Commit**

```bash
git add src/neutral/api.py src/neutral/app.py tests/test_api.py
git commit -m "feat: add FastAPI endpoints with rate limiting and request logging"
```

---

## Task 9: Antarmuka Gradio

**Files:**
- Create: `src/neutral/ui.py`
- Modify: `src/neutral/app.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `analyze`, `get_classifier`, `get_settings`
- Produces: `analyze_for_ui(text: str) -> tuple[str, list[tuple[str, str | None]]]`, `build_ui() -> gr.Blocks`

- [ ] **Step 1: Tulis test yang gagal**

```python
# tests/test_ui.py
from neutral.ui import analyze_for_ui


def test_returns_summary_and_highlights():
    summary, highlights = analyze_for_ui(
        "The bill passed. The senator objected. Critics called it reckless."
    )
    assert "3" in summary
    assert len(highlights) == 3
    for text, label in highlights:
        assert isinstance(text, str)
        assert label in {"biased", None}


def test_empty_input_is_handled():
    summary, highlights = analyze_for_ui("")
    assert highlights == []
    assert "0" in summary


def test_oversized_input_returns_message_not_exception():
    summary, highlights = analyze_for_ui("x" * 60_000)
    assert "terlalu panjang" in summary.lower()
    assert highlights == []
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ui.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'neutral.ui'`

- [ ] **Step 3: Tulis `src/neutral/ui.py`**

```python
import gradio as gr

from neutral.api import get_classifier
from neutral.config import get_settings
from neutral.detect import TextTooLargeError
from neutral.pipeline import analyze

_WARNING_TEXT = {
    "non_english": "Teks terdeteksi bukan bahasa Inggris — model dilatih pada berita berbahasa Inggris.",
    "too_short": "Artikel sangat pendek, sehingga skor kurang dapat diandalkan.",
    "truncated": "Artikel dipotong karena melebihi batas jumlah kalimat.",
}


def analyze_for_ui(text: str) -> tuple[str, list[tuple[str, str | None]]]:
    """Bungkus pipeline untuk Gradio. Mengembalikan ringkasan dan sorotan."""
    settings = get_settings()
    try:
        result = analyze(text, get_classifier(), settings)
    except TextTooLargeError:
        return (
            f"Teks terlalu panjang. Batas {settings.max_chars:,} karakter.",
            [],
        )

    if result.n_sentences == 0:
        return "0 kalimat terdeteksi. Tempelkan teks artikel untuk dianalisis.", []

    lines = [
        f"**{result.n_biased} dari {result.n_sentences} kalimat** ditandai biased "
        f"({result.article_score:.0%}).",
        f"Keyakinan rata-rata: {result.confidence_mean:.0%} · model `{result.model_version}`",
    ]
    lines += [f"> {_WARNING_TEXT[w]}" for w in result.warnings if w in _WARNING_TEXT]

    highlights = [(s.text + " ", s.label if s.label == "biased" else None) for s in result.sentences]
    return "\n\n".join(lines), highlights


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Media Neutrality Classifier") as demo:
        gr.Markdown(
            "# Media Neutrality Classifier\n"
            "Tempelkan teks artikel berbahasa Inggris. Model menandai kalimat yang "
            "menunjukkan bias leksikal.\n\n"
            "Skor artikel adalah **proporsi kalimat yang ditandai**, sehingga artikel "
            "pendek dan panjang dapat dibandingkan. Jumlah kalimat selalu ditampilkan "
            "karena proporsi dari sedikit kalimat kurang dapat diandalkan."
        )
        inp = gr.Textbox(lines=12, label="Teks artikel", placeholder="Tempelkan teks di sini...")
        btn = gr.Button("Analisis", variant="primary")
        summary = gr.Markdown()
        highlighted = gr.HighlightedText(
            label="Per kalimat", color_map={"biased": "red"}, show_legend=True
        )
        btn.click(analyze_for_ui, inputs=inp, outputs=[summary, highlighted])
    return demo
```

- [ ] **Step 4: Ubah `src/neutral/app.py` agar me-mount UI**

```python
import gradio as gr

from neutral.api import create_api
from neutral.ui import build_ui

app = create_api()
# UI dan API berbagi satu proses dan satu port: satu container, satu hal
# yang di-deploy.
app = gr.mount_gradio_app(app, build_ui(), path="/")
```

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ui.py -v`
Expected: 3 passed

- [ ] **Step 6: Verifikasi aplikasi benar-benar berjalan**

```bash
.venv/Scripts/python.exe -m uvicorn neutral.app:app --port 8000
```

Buka `http://localhost:8000` untuk UI dan `http://localhost:8000/health` untuk API. Hentikan dengan Ctrl+C setelah keduanya terkonfirmasi.

- [ ] **Step 7: Commit**

```bash
git add src/neutral/ui.py src/neutral/app.py tests/test_ui.py
git commit -m "feat: add Gradio UI mounted on the same FastAPI app"
```

---

## Task 10: Dockerfile dan CI

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: seluruh paket
- Produces: image yang dapat dijalankan; pipeline CI yang memverifikasinya

Docker tidak tersedia di mesin ini, sehingga verifikasi image dilakukan di GitHub Actions. Runner Linux justru merupakan target yang sebenarnya.

- [ ] **Step 1: Tulis `.dockerignore`**

```
.venv/
venv/
.git/
.github/
__pycache__/
.pytest_cache/
.ruff_cache/
tests/
training/
docs/
*.md
```

`training/` dan `tests/` sengaja dikecualikan: itulah yang menjaga PyTorch tidak pernah masuk image.

- [ ] **Step 2: Tulis `Dockerfile`**

```dockerfile
FROM python:3.13-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim

# Pengguna non-root.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NEUTRAL_MODEL_BACKEND=stub

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
ENV PYTHONPATH=/app/src

USER appuser
EXPOSE 8000

# Port dibaca dari environment: HF Spaces memakai 7860, Cloud Run dan Fly
# memakai 8080. Nilai yang di-hardcode mengunci image ke satu platform.
CMD ["sh", "-c", "uvicorn neutral.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

- [ ] **Step 3: Tulis `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -r requirements-dev.txt
      - run: ruff check src tests
      - run: pytest -v

  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t neutral:ci .

      - name: Verify torch is absent from the image
        run: |
          if docker run --rm neutral:ci python -c "import torch" 2>/dev/null; then
            echo "PyTorch ada di image produksi — Dockerfile bocor"
            exit 1
          fi
          echo "torch tidak ada, sesuai harapan"

      - name: Smoke test
        run: |
          docker run -d --name neutral-ci -p 8000:8000 neutral:ci
          for i in $(seq 1 30); do
            if curl -sf http://localhost:8000/health > /dev/null; then break; fi
            sleep 1
          done
          curl -sf http://localhost:8000/health | tee health.json
          grep -q '"status":"ok"' health.json
          curl -sf -X POST http://localhost:8000/analyze \
            -H 'Content-Type: application/json' \
            -d '{"text":"The bill passed. The senator objected."}' | tee analyze.json
          grep -q '"n_sentences":2' analyze.json
          docker logs neutral-ci
          docker rm -f neutral-ci
```

Langkah "Verify torch is absent" adalah penjaga otomatis untuk batasan paling penting di spec. Tanpa itu, satu impor yang salah letak dapat menggandakan ukuran image tanpa disadari.

- [ ] **Step 4: Commit dan push, lalu tunggu CI**

```bash
git add Dockerfile .dockerignore .github/workflows/ci.yml
git commit -m "ci: add Dockerfile and GitHub Actions pipeline with smoke test"
git push -u origin main
```

- [ ] **Step 5: Verifikasi CI hijau**

Buka `https://github.com/Ihsan-p1/neutral/actions`. Kedua job harus lulus. Bila job `docker` gagal pada smoke test, baca `docker logs` di output — penyebab paling mungkin adalah startup melebihi 30 detik, yang berarti ada network call saat boot dan itu melanggar batasan portabilitas.

---

## Task 11: Dokumentasi penutup

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Lengkapi README**

Tambahkan bagian: ringkasan problem, status (serving selesai, model menyusul), cara menjalankan lokal, cara menjalankan lewat Docker, daftar environment variable dari `Settings`, dan bagian limitation yang menyatakan bahwa backend saat ini adalah stub tanpa nilai prediktif.

Bagian limitation harus menyebut eksplisit: selama `NEUTRAL_MODEL_BACKEND=stub`, seluruh angka yang ditampilkan tidak bermakna dan hanya untuk memvalidasi pipeline.

- [ ] **Step 2: Commit dan push**

```bash
git add README.md
git commit -m "docs: document setup, configuration, and current limitations"
git push
```

---

## Ditunda sampai tersedia mesin ber-GPU

Tugas berikut tidak dapat dikerjakan di mesin ini dan akan menjadi rencana
tersendiri:

- `training/split.py` dan `training/data.py` — loading BABE, group split per
  artikel sumber lewat `news_link`, test anti-kebocoran. Sudah dikerjakan; lihat
  `docs/superpowers/plans/2026-08-27-training-m1-data-foundation.md`
- `training/train.py` — fine-tune DeBERTa, sweep learning rate, 5 seed
- `training/calibrate.py` — temperature scaling, ECE dengan bootstrap CI
- `training/evaluate.py` — cross-dataset BASIL, korelasi outlet, error analysis
- `training/export_onnx.py` — ekspor, kuantisasi INT8, uji paritas

Ketika bobot siap, satu-satunya perubahan pada sisi serving adalah menyetel
`NEUTRAL_MODEL_BACKEND=onnx` dan menempatkan model di `NEUTRAL_MODEL_DIR`.
Itulah tujuan dari `StubClassifier`.

---

## Self-Review

**Spec coverage.** Bagian 3.1 pemisahan training/serving → Task 10 (`.dockerignore` dan pemeriksaan torch di CI). Bagian 3.2 struktur repo → Task 1. Bagian 3.3 alur permintaan → Task 6. Bagian 4.1 kontrak → Task 2 dan 8. Bagian 4.2 `article_score` → Task 6. Bagian 4.3 health check sungguhan → Task 8. Bagian 5 pipeline model → ditunda, dengan interface sudah disiapkan di Task 5. Bagian 6.1 Dockerfile → Task 10. Bagian 6.2 checklist portabilitas → Task 10. Bagian 7.1 handler sinkron → Task 8, diuji eksplisit. Bagian 7.2 rate limit → Task 8. Bagian 7.3 logging → Task 7. Bagian 8 CI → Task 10.

**Deviasi terhadap spec yang disengaja.** Python 3.13 menggantikan 3.11 agar lokal sama dengan container. `venv` dan `pip` menggantikan `uv`. `py3langid` menggantikan `fasttext` karena fasttext memerlukan kompilasi C++ di Windows. Ketiganya perlu dicatat ke spec setelah Task 1 terkonfirmasi berjalan.

**Konsistensi tipe.** `predict_proba(list[str]) -> list[float]` dipakai identik di Task 5, 6, dan 8. `analyze(text, clf, settings, return_sentences)` dipakai identik di Task 6, 8, dan 9. `Sentence` ber-field `text/start/end` konsisten antara Task 3 dan 6. `Warning_` didefinisikan sekali di Task 2 dan dipakai di Task 6.
