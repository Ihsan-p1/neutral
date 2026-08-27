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

## Current state

The serving layer is finished and runs end to end on the stub backend: 49 tests
pass, ruff is clean, and the container definition and CI pipeline are written.
No model has been trained yet, so every number the running app reports is
meaningless — see "Switching to the trained model" below.

`training/` and `models/` do not exist yet. Both are already named in
`.dockerignore` and in the architecture notes below in anticipation of the next
phase, which runs on a GPU machine and is the work that remains.

The one step never verified from this machine is the `docker` job in CI, which
needs a Linux runner. Check the Actions tab before trusting the image.

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

On the Windows machine where the serving layer was built, `docker`, `gh`, and
`uv` are **not installed** and there is **no GPU**. Verify what is actually
available before assuming these limits still apply — the training phase is meant
to run somewhere else.

Consequences that shaped the code as it stands:

- The Docker image was never built or smoke-tested locally. GitHub Actions is
  the verification surface — a Linux runner, which is the real target anyway.
- Training cannot run there. Everything under `training/` is deferred to a
  GPU machine; only the serving layer was built on that one.
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
them via `NEUTRAL_MODEL_BACKEND`. No serving logic changes when the weights
arrive, but the packaging does — see the next section.

**`pipeline.py` holds the analysis logic; `api.py` only handles HTTP.** The
Gradio UI calls the same `analyze()` function the endpoint calls, so analysis
behavior is tested without HTTP and the two surfaces cannot drift apart.

## Switching to the trained model

Setting `NEUTRAL_MODEL_BACKEND=onnx` is not sufficient. Three things currently
assume the stub, and two of them fail silently:

1. **`Dockerfile`** pins `NEUTRAL_MODEL_BACKEND=stub` and has no `COPY models/`.
   Section 2.3 of the spec bakes the ONNX file into the image at build time
   rather than fetching it at startup: a thin container that downloads from HF
   Hub turns a cold start into 30-60 seconds and makes the Hub a boot-time point
   of failure.
2. **`.dockerignore`** excludes `models/`. Adding the `COPY` without removing
   that line copies nothing, and the build still reports success.
3. **`metadata.json`** must sit beside the weights. `load_classifier()` reads
   the calibrated temperature and the model version from it; if it is missing,
   the temperature falls back to `1.0` and every probability the API returns is
   uncalibrated while still looking entirely plausible.

The README's limitations section also states in bold that the reported numbers
have no predictive value. That claim stops being true at the same moment and
must be rewritten with the real evaluation figures.

## Data handling

**Never commit the training data.** `data/` is in `.gitignore`. BABE's
HuggingFace card states no license at all, and the paper's GitHub repository —
which contains the same labels — carries AGPL-3.0 at the repo level. AGPL is
copyleft that triggers on network use, which is exactly this project's
deployment model. The risk is closed by not redistributing rather than by
switching datasets: download at training time, never check it in, never copy it
into the image. Section 5.0 of the PRD records the licenses that were actually
verified.

The evaluation uses a small hand-labelled held-out set of recent news alongside
BABE's test split. It is **test-only** — never trained on, never used for model
selection or threshold tuning. Labelling follows BABE's own annotation
guidelines so the two sets define `biased` the same way; otherwise the numbers
cannot be compared. PRD 5.3 has the full constraints.

## Scope questions already answered

Two extensions were evaluated on 2026-08-27 and deliberately left out of v1.
PRD 4.5 records both in full. Neither is a rejected idea — both are gated on a
condition that does not hold yet, and neither belongs in the training phase.

- **Three-class lexical vs informational.** BASIL already carries that taxonomy,
  but BASIL is the cross-dataset test set. Moving it to the training side
  deletes the only measurement that separates generalization from memorizing
  BABE's distribution. Hand-labelling a training set instead needs a *second*
  annotator: a test set only measures, so one annotator with a reported kappa
  holds up, but a training set teaches the model one person's reading of a
  boundary the literature still argues about. Gate: a second annotator exists.
- **Search-engine lookup and misinformation detection.** A different task, not
  an extension — it needs external evidence rather than the sentence itself. It
  would put a third-party call on the request path, send the user's pasted text
  to that third party, and make a claim with no ground truth anywhere in this
  project, so its accuracy could never be reported. The defensible form is
  corroboration ("this claim appears in N outlets"), which is a separate
  service. Gate: v1 has real evaluation numbers and the new claim has a defined
  metric.

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

The serving-foundation plan is fully executed. The training phase has no plan
document yet — write one before starting it, rather than working from the
deferred-task list at the end of the old plan.
