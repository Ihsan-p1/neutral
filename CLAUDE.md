# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sentence-level lexical bias detection for English news text. Fine-tuned encoder
classifier (binary: `biased` / `neutral`), served as a single container with a
FastAPI API and a mounted Gradio UI.

Project documents are written in Indonesian; code, comments, and commit
messages are in English.

- `PRD_Media_Neutrality_Classifier.md` — scope, datasets, metrics, milestones
- `docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md` — serving/deployment design
- `docs/superpowers/plans/2026-08-27-serving-foundation.md` — task-by-task implementation plan

Read the spec before changing anything under `src/neutral/`. Several
constraints below exist for reasons documented there and are not obvious from
the code alone.

## Commands

This is a Windows machine; the venv interpreter lives under `Scripts/`, not `bin/`.

```bash
.venv/Scripts/python.exe -m pytest -v                      # all tests
.venv/Scripts/python.exe -m pytest tests/test_api.py -v    # one file
.venv/Scripts/python.exe -m pytest -k test_name -v         # one test
.venv/Scripts/python.exe -m ruff check src tests           # lint
.venv/Scripts/python.exe -m ruff check --fix src tests     # lint + autofix
.venv/Scripts/python.exe -m uvicorn neutral.app:app --reload
```

Dependencies are pinned in `requirements.txt` (runtime) and
`requirements-dev.txt` (adds pytest, httpx, ruff). `pythonpath = ["src"]` in
`pyproject.toml` means pytest resolves the package without an editable install.

## Environment limits

`docker`, `gh`, and `uv` are **not installed**, and this machine has **no GPU**.

Consequences that shape the work:

- The Docker image cannot be built or smoke-tested locally. GitHub Actions is
  the verification surface — a Linux runner, which is the real target anyway.
- Training cannot run here. Everything under `training/` is deferred to a
  GPU machine; only the serving layer is built on this one.
- Use `git` directly, not `gh`, for remote operations.

## Architecture

Three boundaries carry most of the design:

**Training is separated from serving so that PyTorch never enters the
production image.** `training/` uses torch and transformers; `src/neutral/`
uses only onnxruntime and tokenizers. `.dockerignore` excludes `training/` and
`tests/`, and CI asserts that `import torch` fails inside the built image. Torch
plus CUDA wheels are ~2.5 GB against ~40 MB for onnxruntime CPU — this is the
difference between an image that starts in under 10 seconds and one that does
not.

**The classifier sits behind a small interface with two implementations.**
`StubClassifier` is deterministic (SHA-256 of the sentence mapped to logits) and
lets the entire pipeline, API, UI, and CI run before any weights exist.
`OnnxClassifier` is the production path. `load_classifier()` selects between
them via `NEUTRAL_MODEL_BACKEND`. When trained weights arrive, the only serving
change is setting that variable and pointing `NEUTRAL_MODEL_DIR` at the model.

**`pipeline.py` holds the analysis logic; `api.py` only handles HTTP.** The
Gradio UI calls the same `analyze()` function the endpoint calls, so analysis
behavior is tested without HTTP and the two surfaces cannot drift apart.

## Invariants

These are correctness constraints, not style preferences. Each one is a silent
failure if broken — nothing raises.

- **Inference handlers are `def`, never `async def`.** FastAPI runs sync
  handlers in a threadpool; an `async def` handler calling ONNX synchronously
  blocks the event loop, so one request freezes every other. `test_api.py`
  asserts this.
- **Temperature scaling is always applied** to logits before softmax, including
  by the stub. It is the only surviving artifact of calibration; if the division
  is dropped, the model still returns plausible-looking probabilities that are
  simply uncalibrated.
- **Article text is never logged.** Log lengths and counts only. `JsonFormatter`
  drops known text-bearing fields as a last line of defense, but callers must
  not pass them in the first place.
- **`article_score` is the proportion of biased sentences**, not the mean
  probability. It is length-normalized so short and long articles compare
  directly. Always surface `n_sentences` alongside it — a proportion over few
  sentences is unreliable, and the API emits a `too_short` warning below 5.
- **The port is read from `$PORT`.** HF Spaces uses 7860; Cloud Run and Fly use
  8080. A hardcoded port locks the image to one platform.
- **Non-English input is flagged, never rejected.** Text under 40 characters is
  treated as English because detection is unreliable at that length.
- All configuration goes through `Settings` in `config.py`, prefix `NEUTRAL_`.
  Do not read `os.environ` elsewhere.

## Deviations from the spec

Taken deliberately during implementation; the spec still names the original choice.

| Spec | Actual | Reason |
|---|---|---|
| Python 3.11 | 3.13 | Matches the local interpreter, so local equals container |
| `uv` + `uv.lock` | `venv` + pinned `requirements.txt` | ~10 direct dependencies do not justify installing a new resolver |
| `fasttext` lid.176 | `py3langid` | fasttext needs a C++ toolchain on Windows; py3langid is pure Python and removes a 917 KB model file from the image |

## Commits

**Only `ihsan-p1` may appear as author or committer.** Do not add
`Co-Authored-By` trailers or generator footers to commits or PR descriptions.
Repo-local git identity is already configured.

Conventional-commit prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `ci:`, `test:`.

Work proceeds one plan task at a time: implement, run the tests named in the
task, then commit. Do not batch several tasks into one commit.
