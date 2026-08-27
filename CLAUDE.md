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

The serving layer is finished and runs end to end on the stub backend: 60 tests
pass, ruff is clean, and the container definition and CI pipeline are written.
No model has been trained yet, so every number the running app reports is
meaningless — see "Switching to the trained model" below.

M1 of the training phase is done. `training/` holds `split.py`, `data.py`,
`baseline_tfidf.py`, and `run_m1.py`; the plan is
`docs/superpowers/plans/2026-08-27-training-m1-data-foundation.md` and the design
is `docs/superpowers/specs/2026-08-27-training-phase-design.md`. `models/` still
does not exist.

Three M1 results carry into everything after it:

- **BABE's own train/test split leaks.** 384 of 2,151 group keys appear on both
  sides. Fold assignment is built from the pooled data instead, so this project's
  numbers are not directly comparable to figures computed on the official split.
- **TF-IDF baseline is F1 macro 0.7420 +/- 0.0194** over 5 folds. PRD 8.1 targets
  0.78 for the fine-tuned model, so the margin M2 has to clear is about 0.04.
- **3,874 of 4,121 rows survive filtering**: 32 dropped for a missing
  `news_link`, 215 for low annotator agreement, 0 for an unexpected label.

M1 was committed on branch `training-m1` as six commits, one per plan task, and
PR #1 is open against `main` with both CI jobs green. Merge it as a merge commit
or a rebase, not a squash: the one-commit-per-task mapping is what the plan is
checked against. That Linux CI run is also the only evidence `tests/test_split.py`
passes off Windows.

M2 (fine-tuning DeBERTa) has a design spec but no plan document yet:
`docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md`. Work happens
on `training-m2`, branched from `training-m1` because `main` still has no
`training/` directory until PR #1 merges.

Four M2 decisions are already closed there and are not re-openable by reading the
code:

- **28 runs, not 75.** The LR sweep (1e-5/2e-5/3e-5) runs on fold 0 only; the
  chosen rate then runs 5 folds x 5 seeds. LR is selected on validation loss, so
  no test fold ever influences a choice.
- **Validation set is the neighbouring fold** (`held_out=i` uses fold `i+1`), so
  the M1 group-clean folds carry over with no new split code.
- **One checkpoint survives**, from fold 0 at the seed with the best validation
  loss. Seed averaging is a reporting device, not the shipped model: serving loads
  a single ONNX file.
- **The M2 stop criterion is a number.** DeBERTa must win on at least 4 of 5
  folds and beat 0.7420 by 0.02. Below that, PRD 9 applies: stop and check data
  quality, do not keep tuning.

## Commands

This is a Windows machine; the venv interpreter lives under `Scripts/`, not `bin/`.

The venv is not in the repository. Create it first — `uv` fetches 3.13 itself,
which is why it is used here even though `requirements.txt` remains the pin:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
```

The training side has its own venv, so torch never reaches the serving one:

```bash
uv venv --python 3.13 .venv-train
uv pip install --python .venv-train/Scripts/python.exe -r training/requirements-train.txt
.venv-train/Scripts/python.exe -m training.run_m1          # M1 report
uv pip freeze --python .venv-train/Scripts/python.exe     # re-pin; uv venv has no pip
```

```bash
.venv/Scripts/python.exe -m pytest -v                      # all tests
.venv/Scripts/python.exe -m pytest tests/test_api.py -v    # one file
.venv/Scripts/python.exe -m pytest -k test_name -v         # one test
.venv/Scripts/python.exe -m ruff check src tests           # lint
.venv/Scripts/python.exe -m ruff check --fix src tests     # lint + autofix
.venv/Scripts/python.exe -m uvicorn neutral.app:app --reload
```

Dependencies are pinned in `requirements.txt` (runtime) and
`requirements-dev.txt` (adds pytest, httpx, ruff). `pythonpath = ["src", "."]`
in `pyproject.toml` means pytest resolves both `neutral` and `training` without an
editable install; the repo root is on the list because `tests/test_split.py`
imports `training.split`.

## Environment limits

Verified on this machine on 2026-08-27: `docker` 29.7.2, `gh`, and `uv` 0.11.29
are all installed, and the GPU is an RTX 3050 6GB Laptop on driver 581.86. The
earlier machine had none of these, which is why parts of the code were written
without them.

- **6 GB of VRAM is the binding constraint on training.** Fine-tuning a
  base-size encoder fits, but only at a small batch size; expect to reach the
  target effective batch through gradient accumulation and mixed precision
  rather than raw batch size.
- The Docker image has still never been built here, only made buildable. CI on a
  Linux runner remains the verification surface — it is the real target — but the
  `docker` job can now be reproduced locally when it fails.
- `gh` is available for remote operations.

## Architecture

Three boundaries carry most of the design:

**Training is separated from serving so that PyTorch never enters the
production image.** `training/` uses torch and transformers; `src/neutral/`
uses only onnxruntime and tokenizers. `.dockerignore` excludes `training/` and
`tests/`, and CI asserts that `import torch` fails inside the built image. Torch
plus CUDA wheels are ~2.5 GB against ~40 MB for onnxruntime CPU — this is the
difference between an image that starts in under 10 seconds and one that does
not.

The two venvs carry the same separation locally: serving lives in `.venv/` and
training in `.venv-train/`, so `import torch` fails in the serving venv rather
than only failing in CI.

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
- **`training/split.py` imports only stdlib.** `tests/test_split.py` imports it,
  and those tests run in CI with `requirements-dev.txt` alone. Adding a
  `datasets`, `pandas`, `sklearn`, `torch`, or `numpy` import there breaks CI —
  a test asserts the import set.
- **Fold assignment is grouped by article URL, never by row.** BABE has no
  `article_id`; the key is a normalised `news_link`. A random split puts
  neighbouring sentences from one article in train and test at once, which
  inflates F1 and measures nothing. `run_m1.py` asserts this rather than warning.
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
| `uv` + `uv.lock` | `uv venv` + pinned `requirements.txt` | ~10 direct dependencies do not justify a lockfile; `uv` is used to create the venv because it installs Python 3.13 on its own |
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
