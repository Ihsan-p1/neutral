# Media Neutrality Classifier

Sentence-level lexical bias detection for English news. It measures how neutrally a
sentence is worded, not whether its content is factually true.

**It can say when it is unsure, and the claim is measured.** Conformal prediction gives each
sentence a set of possible labels. On BABE the true label is in that set 95.2% of the time
against a 95% target, over 100 splits by article. On BASIL (other outlets, other annotators)
the guarantee breaks for biased sentences: 71.0% instead of 95%.

**Passing the unsure sentences to a local LLM does not help.** A router that sends them to
Qwen3-4B scores 0.850 macro F1 against 0.877 for the fine-tuned model alone on BABE, and
0.536 against 0.566 on BASIL. On BABE that gap is within noise at this sample size. Details in
[M8](#m8-the-model-knows-when-it-is-unsure-an-llm-does-not-fix-it).

Status: in development. The serving layer was built first against a stub classifier. The
DeBERTa model has been trained, evaluated (see Evaluation results below), and exported to
fp32 ONNX by `training/export_onnx.py`, so the `onnx` backend now runs on the real weights.

The weights and the export are gitignored and do not travel with a clone. The default is
still `stub`; to run the real model, export first (see Running locally) and set
`NEUTRAL_MODEL_BACKEND=onnx`. The shipped temperature is 1.0 — calibration is M4b and has
not run, so the probabilities are valid as a ranking but are not calibrated.

## Documents

| Document | Contents |
|---|---|
| [PRD](PRD_Media_Neutrality_Classifier.md) | Problem statement, scope, dataset, metrics, milestones |

## Running locally

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn neutral.app:app --reload
```

The UI is at `http://localhost:8000` and the API at `http://localhost:8000/analyze`.

Tests:

```bash
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/python.exe -m ruff check src tests
```

Running the `onnx` backend on the real weights. The export runs in `.venv-train/`
because it needs torch, and wants `onnx`, `onnxscript` and `onnxruntime`, which are
deliberately absent from every requirements file — the serving image may carry
onnxruntime and nothing else:

```bash
uv pip install --python .venv-train/Scripts/python.exe onnx onnxscript onnxruntime==1.20.1
.venv-train/Scripts/python.exe -m training.export_onnx
NEUTRAL_MODEL_BACKEND=onnx .venv/Scripts/python.exe -m uvicorn neutral.app:app
```

The export writes `models/current/` and refuses to finish if the serving tokenizer
disagrees with the training one, or if the ONNX logits drift more than 1e-3 from
PyTorch. Both are failures that raise nothing on their own.

## Running in Docker

```bash
docker build -t neutral .
docker run --rm -p 8000:8000 neutral
```

The default build bakes `models/current/` into the image, so run the ONNX export first.
Without the weights, `docker build --target stub -t neutral .` builds the stub image that CI
tests.

The port comes from `$PORT`, so the same image runs on HF Spaces (7860), Cloud Run and Fly
(8080), or any VPS without a change:

```bash
docker run --rm -e PORT=7860 -p 7860:7860 neutral
```

## Architecture

Three boundaries carry most of the design.

Training is separate from serving. `training/` uses PyTorch; `src/neutral/` uses only
onnxruntime. `.dockerignore` excludes `training/`, and CI asserts that `import torch` fails
inside the image. The difference is roughly 2.5 GB against 40 MB, which is the difference
between an image that is ready to serve in under 10 seconds and one that is not.

The classifier sits behind one interface with two implementations. `StubClassifier` is
deterministic and lets the pipeline, API, UI, and CI all run before any weights exist.
`OnnxClassifier` is the production path. When the weights are ready, the only change on the
serving side is setting `NEUTRAL_MODEL_BACKEND=onnx`.

`pipeline.py` holds the analysis logic and `api.py` only handles HTTP. The Gradio interface
calls the same `analyze()` function the endpoint calls, so the two cannot drift apart in
behaviour.

## Configuration

Every setting is read from an environment variable prefixed `NEUTRAL_`.

| Variable | Default | Meaning |
|---|---|---|
| `NEUTRAL_MODEL_BACKEND` | `stub` | `stub` or `onnx` |
| `NEUTRAL_MODEL_DIR` | `models/current` | Location of `model.onnx`, `model.onnx.data`, `tokenizer.json`, `metadata.json` |
| `NEUTRAL_DECISION_THRESHOLD` | `0.5` | Probability threshold for the `biased` label |
| `NEUTRAL_TEMPERATURE` | `1.0` | Calibrated temperature |
| `NEUTRAL_MAX_CHARS` | `50000` | Text size limit per request |
| `NEUTRAL_MAX_SENTENCES` | `300` | Sentence count limit per request |
| `NEUTRAL_LOG_LEVEL` | `INFO` | Logging level |
| `PORT` | `8000` | HTTP port, read by the container |

## Evaluation results

### M2: fine-tuned DeBERTa

| | Macro F1 |
|---|---|
| TF-IDF + logistic regression | 0.7420 +/- 0.0194 |
| DeBERTa-v3-base, 5 folds x 5 seeds | 0.8477 +/- 0.0277 |

Learning rate 3e-5, chosen by validation loss on a single fold. Each per-fold number is the
mean over 5 seeds, and the fold-to-fold and seed-to-seed deviations are reported separately
because they mean different things: 0.0195 across folds and 0.0187 across seeds. They are
almost the same size, so one number from one fold at one seed would hide most of the
spread.

Folds are built per article (`news_link`) rather than per row, so these numbers cannot be
compared directly against anything computed on the official BABE split. That official split
leaks: 384 of its 2,151 group keys appear in both train and test. The 0.78 target in the
literature was computed on it.

Only macro F1 on BABE folds was measured. Cross-dataset generalisation, calibration, and a
held-out set of recent news were not.

### M3: NELA baseline

Explicit linguistic features, measured on the same folds.

| | Macro F1 | Delta vs TF-IDF |
|---|---|---|
| TF-IDF + logistic regression | 0.7420 +/- 0.0194 | +0.0000 |
| NELA + logistic regression | 0.6837 +/- 0.0116 | -0.0583 |
| TF-IDF + NELA + logistic regression | 0.7294 +/- 0.0130 | -0.0126 |
| DeBERTa-v3-base | 0.8477 +/- 0.0195 | +0.1058 |

Per feature group, all with the same logistic regression:

| Group | Features | Surviving | Macro F1 |
|---|---|---|---|
| style | 50 | 40 | 0.6599 +/- 0.0089 |
| affect | 9 | 9 | 0.6194 +/- 0.0211 |
| bias | 8 | 8 | 0.5884 +/- 0.0240 |
| complexity | 7 | 7 | 0.5382 +/- 0.0375 |
| moral | 11 | 11 | 0.5028 +/- 0.0310 |
| event | 2 | 1 | 0.4401 +/- 0.0175 |

NELA lands 0.0583 below TF-IDF and 0.1640 below DeBERTa. Concatenating it onto TF-IDF drops
the score by 0.0126 instead of raising it, so these are not strong features that happen to
overlap with bag-of-words. They are genuinely weak on text as short as a sentence.
DeBERTa's advantage therefore comes from context, not from counting factive verbs and
hedges.

What signal there is sits in the style group: its 40 columns alone reach 0.6599 of the
0.6837 that all 87 features reach. The `event` group gives 0.4401, about 0.09 above a
constant predictor (0.3534 if it always guesses `biased`), while consuming 92% of the
extraction time.

Those four standard deviations do not measure the same thing. DeBERTa's mixes fold-to-fold
and seed-to-seed spread over 25 runs; the three logistic regressions report fold-to-fold
spread over 5 runs only, because lbfgs on fixed data is deterministic. The comparable
quantity is the per-fold mean, and the deltas in the table are read from those.

`nela-features==3.0.1` provides 87 features, not the 141 written in the first version of the
PRD. 141 is the figure from Horne et al. 2018.

### M4a: cross-dataset evaluation on BASIL

Trained on BABE, tested on BASIL with no adaptation. BASIL holds 300 articles from NYT, Fox
News, and HuffPost, 2010 to 2019. Labels are computed from the official `launchnlp/BASIL`
release under the rule that a sentence counts as biased if it contains at least one span of
type lexical. Sentence segmentation is the dataset's own, not an extra segmenter. The
HuggingFace re-upload was only compared against it (99.3% match, 0.6% differing labels) and
never used as the label source.

Measured on 5,780 sentences from 73 events; the remaining 27 events were used to pick the
threshold. The positive prior there is 5.1%, against 54.6% biased in BABE.

| Model | F1 as-is | F1 after recalibration | AP |
|---|---|---|---|
| DeBERTa-v3-base | 0.5656 | 0.5854 | 0.1523 |
| TF-IDF + logistic regression | 0.3912 | 0.4963 | 0.0524 |

A constant "everything is neutral" predictor scores 0.4868 macro F1 on the same test set.
Read that first: DeBERTa is only 0.079 above it, and TF-IDF as-is sits below it.

Three things are reported alongside those numbers:

- Recalibrating the threshold does not rescue the drop. DeBERTa gains 0.0198, so against
  0.8477 on BABE it is still 0.26 down. PRD 8.2 predicted that most of the drop would turn
  out to be a threshold artefact and label mismatch. That prediction holds for only a small
  part of it.
- The signal is still there, thinly. DeBERTa's AP of 0.1523 at a prior of 0.051 is 2.99
  times a random predictor, so the ranking has not broken. TF-IDF reaches 1.03 times
  random, which means it moves nothing.
- The numbers come from one checkpoint, not a mean over seeds. M2 measured 0.0187 of
  seed-to-seed spread on BABE; the spread on BASIL was not measured.

### M8: the model knows when it is unsure; an LLM does not fix it

```
sentence -> DeBERTa (ONNX, CPU) -> prediction set
                                     one label        -> answer
                                     none or both     -> Qwen3-4B (Ollama, GPU) -> answer
```

Split conformal prediction on the served checkpoint, α = 0.05. The threshold is calibrated
on half of BABE fold 0 and tested on the other half, split by article, 100 times. Qwen3 is
`qwen3:4b-instruct-2507-q4_K_M` with one fixed prompt (v1), checked on 50 fold-1 sentences
before fold 0 was opened. The run is offline; the API is unchanged.

BABE fold 0, mean ± sd over 100 splits:

| Mode | Macro F1 | Coverage | Sent to Qwen3 |
|---|---|---|---|
| DeBERTa only | 0.8767 ± 0.0154 | – | – |
| Qwen3 only | 0.7925 ± 0.0186 | – | 100% |
| Router, conformal | 0.8497 ± 0.0157 | 0.9523 ± 0.0159 | 25.0% |
| Router, per-class conformal | 0.8482 ± 0.0189 | 0.9539 ± 0.0164 | 25.9% |
| Router, confidence < 0.9 (fixed) | 0.8472 ± 0.0175 | 0.9635 ± 0.0064 | 29.2% |
| Router, confidence < 0.59 (tuned on fold 1) | 0.8732 ± 0.0160 | 0.8998 ± 0.0123 | 5.4% |

Coverage for a confidence threshold is the coverage of the sets it implies (a label is kept
when its probability is at least 1 − τ).

BASIL test (5,780 sentences, 5.1% biased), threshold from BABE unless stated:

| Mode | Macro F1 | Coverage | Coverage, biased | Sent to Qwen3 |
|---|---|---|---|---|
| DeBERTa only | 0.5656 | – | – | – |
| Qwen3 only | 0.4738 | – | – | 100% |
| Router, confidence < 0.9 (fixed) | 0.5279 | 0.9303 | 0.7677 | 36.4% |
| Router, confidence < 0.59 (tuned on fold 1) | 0.5596 | 0.8420 | 0.5522 | 4.7% |
| Router, conformal | 0.5356 | 0.9189 | 0.7104 | 30.6% |
| Router, per-class conformal | 0.5342 | 0.9192 | 0.7172 | 31.1% |
| Router, conformal, recalibrated on BASIL dev | 0.5088 | 0.9528 | 0.8418 | 51.1% |
| Router, per-class, recalibrated on BASIL dev | 0.4840 | 0.9486 | 0.9057 | 69.0% |

What this shows:

- The coverage guarantee holds where its assumption holds. On BABE it is 95.2% overall and
  95.2% for each class. Under a shift to BASIL it drops to 91.9%, and to 71.0% for biased
  sentences. Recalibrating on BASIL's dev events (30% of events, 2,197 sentences) brings biased coverage back to 84.2%, or 90.6%
  per class, at the price of sending half or more of the sentences to the LLM.
- Qwen3-4B is the weaker model here: 0.79 against 0.88 on BABE and 0.47 against 0.57 on
  BASIL (BASIL figures are single values, no interval). Every router row on BASIL scores
  below DeBERTa alone; the closest is τ = 0.59, which forwards 4.7%. The router still beats Qwen3 alone
  (paired bootstrap, F1 difference +0.057, 95% CI +0.021 to +0.098). Against DeBERTa alone
  the difference is −0.028 with a CI of −0.073 to +0.016, too wide to call either way.
- With two classes, a conformal router routes exactly like "send it on if confidence is
  below τ" for one τ. Conformal adds a way to choose τ with a stated coverage, not a
  different routing rule. What the choice buys shows in coverage. A τ tuned for F1 on fold 1
  sends only 5.4% of sentences and stays close to DeBERTa (0.873), but covers 90.0% on BABE
  and 55.2% of biased BASIL sentences. A hand-picked τ = 0.9 over-covers on BABE (96.4%)
  and forwards more. Conformal lands on the target it was given.
- The share sent to Qwen3 does not shrink steadily as α grows (25.0%, 6.0%, 4.7%, 12.3%
  for α 0.05 to 0.20). Above a threshold of 0.5 the forwarded sentences are the two-label
  sets; below it they are the empty ones.
- Cost per 1,000 sentences on this laptop (i5-13450HX, RTX 3050 6 GB): DeBERTa 38 s,
  router 138 s, Qwen3 alone 401 s. p95 latency is 55 ms, 466 ms and 462 ms.

Limits: one prompt version, checked on 50 sentences; one 4B model at 4-bit; one DeBERTa
checkpoint; BASIL figures have no confidence intervals. A stronger LLM or a better prompt could change the second finding, not the first.

Reproduce (needs Ollama with the model pulled, about 50 minutes for the Qwen3 pass):

```bash
PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m8 --pilot
PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m8 --warm
PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m8 --report
```

`.venv-train` also needs `pydantic==2.10.4 pydantic-settings==2.7.0`, the serving pins,
because the runner loads the serving classifier.

## Limitations

While `NEUTRAL_MODEL_BACKEND=stub`, every number the app displays has no predictive value
at all. The stub is deterministic on the sentence text, which makes the pipeline and API
fully testable, but its output is not bias analysis. That is still the default, so a fresh
clone shows meaningless numbers until the export is run.

On the `onnx` backend the numbers mean what the evaluation measured, which is narrower than
it looks. 0.8477 macro-F1 is the BABE figure. On BASIL, a second corpus, the same checkpoint
reaches 0.5854 against a 0.4868 all-neutral floor — 0.079 above predicting nothing. The
BABE number alone overstates what the served model does, so the two belong together
wherever either is quoted. The temperature is also uncalibrated at 1.0 until M4b runs, so
the probabilities rank correctly but their absolute values do not mean what they appear to.

Limitations that still apply once the model is in place:

- It detects lexical bias only, not selection or omission bias, and not political stance.
- English only. Non-English input is flagged rather than rejected.
- `article_score` is the proportion of flagged sentences. A proportion over few sentences is
  less reliable, which is why the sentence count is always shown next to the score.

## Data

The model is trained on BABE (Bias Annotations By Experts), roughly 3,700 expert-annotated
news sentences from the Media Bias Group.

The dataset is not included in this repository or in the Docker image. It is downloaded at
training time and never committed. The BABE HuggingFace card states no licence, while the
paper's GitHub repository is AGPL-3.0 at the repository level. By not redistributing it,
neither source's redistribution clause is ever triggered. Details in PRD section 5.0.

Evaluation uses three sets: the BABE test split (grouped per article), BASIL as the
cross-dataset check, and a small held-out set of recent news labelled by hand following the
BABE annotation guidelines. That last set is only ever used for testing, never for training
or model selection.

If you use the dataset, cite the paper:

```bibtex
@inproceedings{spinde2021neural,
    title     = {Neural Media Bias Detection Using Distant Supervision With BABE -- Bias Annotations By Experts},
    author    = {Spinde, Timo and Plank, Manuel and Krieger, Jan-David and Ruas, Terry and Gipp, Bela and Aizawa, Akiko},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
    year      = {2021},
    publisher = {Association for Computational Linguistics}
}
```

## Licence

The code is MIT; see [LICENSE](LICENSE). It does not cover BABE or BASIL, which this
repository does not redistribute, or model weights trained on them.
