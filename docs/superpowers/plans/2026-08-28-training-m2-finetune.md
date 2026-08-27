# M2 Fine-Tune DeBERTa Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `microsoft/deberta-v3-base` on the M1 group-clean folds and decide, against a criterion fixed in advance, whether it beats the TF-IDF baseline of 0.7420.

**Architecture:** `training/train.py` runs exactly one `(fold, seed, learning rate)` combination through a HuggingFace `Trainer` and returns one `RunResult`. `training/run_m2.py` orchestrates: 3 sweep runs on fold 0, then 25 final runs, appending one JSON line per run so a crash resumes instead of restarting. The fold-triple function lives in `training/split.py` so its test runs in the serving CI job, which has no torch.

**Tech Stack:** torch 2.13.0+cu126, transformers 5.16.1, datasets 5.0.1, scikit-learn 1.9.0 (all already pinned in `training/requirements-train.txt`), plus `sentencepiece` and `protobuf` added in Task 2.

**Spec:** `docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md`

## Global Constraints

- **Two interpreters.** Anything importing torch or datasets runs under `.venv-train/Scripts/python.exe`. Tests and lint run under `.venv/Scripts/python.exe`. This is a Windows machine: `Scripts/`, not `bin/`.
- **`training/split.py` imports only stdlib.** `tests/test_split.py` asserts the import set is a subset of `{"hashlib", "urllib"}`. Adding `datasets`, `pandas`, `sklearn`, `torch`, or `numpy` there breaks CI.
- **`tests/` must never import torch.** The serving CI job installs `requirements-dev.txt` only. Checks that need torch live as `__main__` self-checks inside `training/`.
- **Never commit data, weights, or run logs.** `data/` and `models/` are already in `.gitignore`; Task 3 adds `runs/`.
- **Fold assignment is `group_split(keys, n_folds=5, seed=0)`, identical to M1.** Changing any argument invalidates the comparison with the 0.7420 baseline.
- **Fixed hyperparameters, spec 3.4:** `max_length=128` with dynamic padding, batch size 16 (fallback 8 with `gradient_accumulation_steps=2`), fp16, `num_train_epochs=5`, early stopping patience 2 on `eval_loss`, LLRD decay 0.95, warmup ratio 0.06, weight decay 0.01. Learning rate is the only swept value: 1e-5, 2e-5, 3e-5.
- **Stop criterion, spec 6:** pass requires DeBERTa winning on at least 4 of 5 folds AND `mean(DeBERTa) - mean(baseline) >= 0.02`. On failure, stop and check data quality per PRD 9; do not add sweep points.
- **Comments and docstrings in Indonesian**, matching `split.py`, `data.py`, and `baseline_tfidf.py`. Commit messages in English, conventional prefixes.
- **Only `Ihsan-p1` as author and committer.** No `Co-Authored-By`, no generator footer.
- **No sentence text in printed output.** M1 prints counts and distributions only; keep it that way.
- **One commit per task.** Do not batch tasks.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `training/split.py` | add `train_val_test_folds()`; still stdlib only | 1 |
| `tests/test_split.py` | add four tests for the fold triple; runs in CI | 1 |
| `training/requirements-train.txt` | add `sentencepiece`, `protobuf` | 2 |
| `training/train.py` | create: tokenization, LLRD optimizer groups, `Trainer` wiring, `run_one()`, self-check | 2 |
| `training/run_m2.py` | create: data load, three-way leakage assert, token percentiles, JSONL resume, sweep, final runs, report | 3, 4, 5, 6 |
| `.gitignore` | add `runs/` | 3 |
| `README.md` | M2 numbers section | 7 |
| `CLAUDE.md` | M2 results carried forward | 7 |
| `docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md` | add measured-results section | 7 |

`data.py`, `baseline_tfidf.py`, and `run_m1.py` are not modified. `run_m2.py` consumes `load_babe()`, `group_split()`, and `run_baseline()` as they are.

---

### Task 1: Fold triple selection in `split.py`

**Files:**
- Modify: `training/split.py` (append after `group_split`)
- Test: `tests/test_split.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `train_val_test_folds(held_out: int, n_folds: int = 5) -> tuple[list[int], int, int]`, returning `(train_folds, val_fold, test_fold)`. Task 2 and Task 3 both call it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split.py`. Also extend the existing import line at the top of the file to `from training.split import fold_of, group_key_from_url, group_split, train_val_test_folds`.

```python
def test_fold_triple_parts_are_disjoint():
    # Spec 3.2: val tidak boleh sama dengan test, dan keduanya tidak boleh ikut
    # train. Bila ini rusak, model melihat fold test saat memilih epoch.
    for held_out in range(5):
        train, val, test = train_val_test_folds(held_out, n_folds=5)
        assert val != test
        assert val not in train
        assert test not in train
        assert len(set(train)) == len(train)


def test_fold_triple_covers_every_fold():
    for held_out in range(5):
        train, val, test = train_val_test_folds(held_out, n_folds=5)
        assert sorted([*train, val, test]) == [0, 1, 2, 3, 4]


def test_fold_triple_test_is_the_held_out_fold():
    # Tabel spec 3.2: held_out=i memakai fold i sebagai test dan fold i+1 sebagai val.
    for held_out in range(5):
        train, val, test = train_val_test_folds(held_out, n_folds=5)
        assert test == held_out
        assert val == (held_out + 1) % 5


def test_fold_triple_rejects_bad_arguments():
    with pytest.raises(ValueError):
        train_val_test_folds(5, n_folds=5)
    with pytest.raises(ValueError):
        train_val_test_folds(-1, n_folds=5)
    with pytest.raises(ValueError):
        # Dua fold tidak menyisakan apa pun untuk train.
        train_val_test_folds(0, n_folds=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_split.py -k fold_triple -v`
Expected: FAIL, `ImportError: cannot import name 'train_val_test_folds'`.

- [ ] **Step 3: Write the implementation**

Append to `training/split.py`:

```python
def train_val_test_folds(
    held_out: int, n_folds: int = 5
) -> tuple[list[int], int, int]:
    """Petakan satu fold yang di-hold-out menjadi (train, val, test).

    Fold test adalah `held_out`, fold val adalah tetangganya, dan sisanya train.
    Karena fold sudah group-clean terhadap `news_link` (lihat `group_split`),
    ketiganya otomatis tidak berbagi artikel tanpa kode split tambahan.

    Fungsi ini tinggal di sini, bukan di `train.py`, agar dapat di-test di job CI
    serving yang tidak memasang torch. Lihat catatan di kepala modul.
    """
    if n_folds < 3:
        raise ValueError(
            f"n_folds minimal 3 agar train tidak kosong, diterima {n_folds}"
        )
    if not 0 <= held_out < n_folds:
        raise ValueError(f"held_out di luar rentang 0..{n_folds - 1}: {held_out}")
    test = held_out
    val = (held_out + 1) % n_folds
    train = [fold for fold in range(n_folds) if fold not in (test, val)]
    return train, val, test
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_split.py -v`
Expected: PASS, including `test_split_module_imports_only_stdlib` (no new imports were added).

- [ ] **Step 5: Run the full suite and lint**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: the 60 existing tests plus the 4 new ones pass.

Run: `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add training/split.py tests/test_split.py
git commit -m "feat: add fold triple selection for M2 training runs"
```

---

### Task 2: `train.py`, one run of one configuration

**Files:**
- Modify: `training/requirements-train.txt`
- Create: `training/train.py`

**Interfaces:**
- Consumes: `training.data.Row`, `training.split.train_val_test_folds`.
- Produces:
  - `RunResult` frozen dataclass with fields `held_out: int`, `seed: int`, `lr: float`, `f1_macro: float`, `val_loss: float`, `stopped_epoch: float`, `n_train: int`, `n_val: int`, `n_test: int`.
  - `run_one(rows: list[Row], folds: list[int], held_out: int, seed: int, lr: float, out_dir: str, n_folds: int = 5, max_epochs: int = MAX_EPOCHS) -> RunResult`
  - `token_length_percentiles(texts: list[str]) -> dict[str, int]` with keys `"p50"`, `"p95"`, `"p99"`, `"max"`.
  - Module constants `MODEL`, `MAX_LENGTH`, `BATCH_SIZE`, `GRAD_ACCUM`, `MAX_EPOCHS`.

- [ ] **Step 1: Add the tokenizer dependencies**

DeBERTa-v3 ships a SentencePiece model, not a `tokenizer.json`, so `AutoTokenizer` needs `sentencepiece` and `protobuf` to build the fast tokenizer. Neither is in the current pin.

```bash
uv pip install --python .venv-train/Scripts/python.exe sentencepiece protobuf
```

- [ ] **Step 2: Verify the tokenizer loads and re-pin**

Run:

```bash
.venv-train/Scripts/python.exe -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); print(type(t).__name__, len(t('a test sentence')['input_ids']))"
```

Expected: a tokenizer class name and a small integer. If it raises `ImportError` naming a missing converter dependency, install what it names and note it in the commit message.

Then re-pin. `uv venv` installs no pip, so `python -m pip freeze` fails here:

```bash
uv pip freeze --python .venv-train/Scripts/python.exe > training/requirements-train.txt
```

Re-add the two header comment lines and the `--extra-index-url https://download.pytorch.org/whl/cu126` line that `freeze` drops, then confirm with `git diff training/requirements-train.txt` that the only other change is the added packages.

- [ ] **Step 3: Write `training/train.py`**

```python
"""Fine-tune DeBERTa untuk satu kombinasi (fold, seed, learning rate).

Modul ini mengimpor torch dan transformers, jadi ia hanya jalan di .venv-train/.
`tests/` tidak boleh mengimpornya: job CI serving tidak memasang torch. Lihat
catatan di kepala `training/split.py`.

Satu pemanggilan `run_one()` menghasilkan satu `RunResult`. Orkestrasi 27 run ada
di `run_m2.py`, bukan di sini.
"""

import shutil
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from training.data import Row
from training.split import train_val_test_folds

MODEL = "microsoft/deberta-v3-base"

# Spec 3.4. Nilai di sini tidak di-sweep; satu-satunya yang di-sweep adalah
# learning rate, sesuai PRD 6.
MAX_LENGTH = 128
BATCH_SIZE = 16
GRAD_ACCUM = 1
EVAL_BATCH_SIZE = 64
MAX_EPOCHS = 5
PATIENCE = 2
LLRD_DECAY = 0.95
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01


@dataclass(frozen=True)
class RunResult:
    """Hasil satu run, cukup untuk disalin ke JSONL tanpa tambahan."""

    held_out: int
    seed: int
    lr: float
    f1_macro: float
    val_loss: float
    stopped_epoch: float
    n_train: int
    n_val: int
    n_test: int


class _Tokenized(torch.utils.data.Dataset):
    """Kalimat yang sudah ditokenisasi tanpa padding.

    Padding diserahkan ke `DataCollatorWithPadding` supaya kalimat pendek tidak
    dibayar sebagai MAX_LENGTH token. BABE berisi kalimat tunggal, jadi sebagian
    besar batch jauh lebih pendek dari 128.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer) -> None:
        self.encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def llrd_param_groups(model, lr: float) -> list[dict]:
    """Learning rate menurun dari kepala klasifikasi ke embedding.

    PRD 6 meminta layer-wise learning rate decay sebagai mitigasi overfitting,
    dan `TrainingArguments` tidak menyediakannya. Layer teratas paling jauh dari
    fitur umum bahasa, jadi ia yang paling boleh bergerak.
    """
    encoder = model.deberta.encoder.layer
    head = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("deberta.")
    ]
    groups = [{"params": head, "lr": lr}]
    for depth, layer in enumerate(reversed(encoder), start=1):
        groups.append({"params": list(layer.parameters()), "lr": lr * LLRD_DECAY**depth})
    groups.append(
        {
            "params": list(model.deberta.embeddings.parameters()),
            "lr": lr * LLRD_DECAY ** (len(encoder) + 1),
        }
    )
    return groups


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predicted = np.asarray(logits).argmax(axis=-1)
    return {"f1_macro": float(f1_score(labels, predicted, average="macro"))}


def token_length_percentiles(texts: list[str]) -> dict[str, int]:
    """Panjang token, agar MAX_LENGTH punya dasar ukur dan bukan asumsi."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    return {
        "p50": int(np.percentile(lengths, 50)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(max(lengths)),
    }


def run_one(
    rows: list[Row],
    folds: list[int],
    held_out: int,
    seed: int,
    lr: float,
    out_dir: str,
    n_folds: int = 5,
    max_epochs: int = MAX_EPOCHS,
) -> RunResult:
    """Latih satu konfigurasi, kembalikan F1 pada fold test dan validation loss.

    Checkpoint selalu ditulis ke `out_dir`, juga untuk run yang bobotnya tidak
    disimpan. Alasannya bukan kelalaian: `load_best_model_at_end` memerlukan
    penyimpanan, dan mematikannya untuk sebagian run akan membuat run yang
    dilaporkan dan run yang bobotnya dipakai menempuh jalur training berbeda.
    Penghapusan direktori adalah tanggung jawab pemanggil.
    """
    train_folds, val_fold, test_fold = train_val_test_folds(held_out, n_folds)
    parts = {"train": train_folds, "val": [val_fold], "test": [test_fold]}
    subsets = {
        name: [row for row, fold in zip(rows, folds, strict=True) if fold in wanted]
        for name, wanted in parts.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    datasets = {
        name: _Tokenized(
            [row.text for row in subset], [row.label for row in subset], tokenizer
        )
        for name, subset in subsets.items()
    }

    args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,
        data_seed=seed,
        learning_rate=lr,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        report_to=[],
        disable_tqdm=True,
    )
    optimizer = torch.optim.AdamW(
        llrd_param_groups(model, lr), lr=lr, weight_decay=WEIGHT_DECAY
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
        optimizers=(optimizer, None),
    )
    trainer.train()

    # `load_best_model_at_end` mengembalikan bobot terbaik ke memori, tetapi yang
    # ada di disk hanya `out_dir/checkpoint-N/`. Simpan sekali lagi ke
    # `out_dir/final` supaya yang dipindahkan ke models/ adalah direktori model
    # yang dapat dimuat langsung, bukan direktori checkpoint Trainer.
    trainer.save_model(f"{out_dir}/final")
    tokenizer.save_pretrained(f"{out_dir}/final")

    history = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    best = min(history, key=lambda entry: entry["eval_loss"])
    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")

    return RunResult(
        held_out=held_out,
        seed=seed,
        lr=lr,
        f1_macro=float(test_metrics["test_f1_macro"]),
        val_loss=float(best["eval_loss"]),
        stopped_epoch=float(history[-1]["epoch"]),
        n_train=len(subsets["train"]),
        n_val=len(subsets["val"]),
        n_test=len(subsets["test"]),
    )


def _self_check() -> None:
    """Buktikan harness benar-benar melatih, pada data yang jawabannya diketahui.

    Ini bukan test performa. Yang diuji: label tidak tertukar, learning rate tidak
    nol, dan bobot benar-benar berubah. Kesalahan seperti itu menghasilkan angka
    yang tampak masuk akal tanpa memunculkan error apa pun.
    """
    biased = "so-called experts peddle this outrageous scheme"
    neutral = "the committee met on tuesday to review the budget"
    rows = [
        Row(
            text=biased if i % 2 == 0 else neutral,
            label=1 if i % 2 == 0 else 0,
            outlet="synthetic",
            lean="None",
            group_key=f"synthetic/{i}",
            official_split="train",
        )
        for i in range(90)
    ]
    folds = [i % 3 for i in range(90)]
    out_dir = "runs/selfcheck"
    try:
        result = run_one(
            rows, folds, held_out=0, seed=0, lr=2e-5, out_dir=out_dir, n_folds=3,
            max_epochs=2,
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    assert result.f1_macro > 0.9, f"dua kelas terpisah sempurna gagal dipelajari: {result}"
    assert result.val_loss == result.val_loss, f"val_loss NaN: {result}"
    assert result.n_train == 30 and result.n_val == 30 and result.n_test == 30, result
    print("self-check ok:", result)


if __name__ == "__main__":
    _self_check()
```

- [ ] **Step 4: Run the self-check**

Run: `.venv-train/Scripts/python.exe -m training.train`
Expected: `self-check ok: RunResult(...)` with `f1_macro` above 0.9. The first run downloads about 740 MB of weights.

Troubleshooting, in the order to try:

1. `torch.cuda.OutOfMemoryError` — set `BATCH_SIZE = 8` and `GRAD_ACCUM = 2`. Effective batch stays 16, so nothing else in the plan changes. Record the change in the commit message.
2. `TypeError` on a `TrainingArguments` or `Trainer` keyword — transformers 5 removed several older names. Run `.venv-train/Scripts/python.exe -c "from transformers import TrainingArguments; import inspect; print(sorted(inspect.signature(TrainingArguments).parameters))"` and use the name that exists. Do not drop the argument.
3. Loss becomes `nan` — replace `fp16=True` with `bf16=True`. The GPU is Ampere, so bf16 is supported, and this deviates from PRD 6 only in numeric format. Record it as a deviation in Task 7.
4. `Trainer` rejects the `optimizers=(optimizer, None)` keyword — transformers 5 may only accept `optimizer_cls_and_kwargs`, which cannot express per-layer groups. In that case keep the same `llrd_param_groups` function and attach it by subclassing instead, which is the smallest change that preserves LLRD:

```python
class _LlrdTrainer(Trainer):
    """Trainer yang memakai optimizer group per layer dari llrd_param_groups."""

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                llrd_param_groups(self.model, self.args.learning_rate),
                lr=self.args.learning_rate,
                weight_decay=WEIGHT_DECAY,
            )
        return self.optimizer
```

Then build `_LlrdTrainer(...)` without the `optimizers` argument. Do not silently drop LLRD: PRD 6 lists it as one of three overfitting mitigations.

- [ ] **Step 5: Lint**

Run: `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: clean. `training/train.py` is linted but not imported by the serving venv, so ruff never has to resolve torch.

- [ ] **Step 6: Commit**

```bash
git add training/train.py training/requirements-train.txt
git commit -m "feat: add single-run DeBERTa fine-tune harness"
```

---

### Task 3: `run_m2.py` dry run, leakage assert, and JSONL resume

**Files:**
- Create: `training/run_m2.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `training.data.load_babe`, `training.split.group_split`, `training.split.train_val_test_folds`, `training.train.RunResult`, `training.train.token_length_percentiles`.
- Produces: `assert_no_group_overlap(rows, folds, n_folds)`, `load_runs(path) -> dict[tuple[int, int, float], RunResult]`, `append_run(path, result)`, module constants `RUNS_PATH`, `N_FOLDS`, `N_SEEDS`, `LEARNING_RATES`, `KEEP_FOLD`, `MODEL_DIR`.

- [ ] **Step 1: Add `runs/` to `.gitignore`**

`models/` is already there. Append `runs/` under it, so the JSONL and the temporary checkpoints written in Task 5 are never committed.

- [ ] **Step 2: Write the dry-run part of `training/run_m2.py`**

```python
"""Entry point M2: sweep learning rate, 25 run final, laporan terhadap baseline.

Jalankan: .venv-train/Scripts/python.exe -m training.run_m2
Dry run  : .venv-train/Scripts/python.exe -m training.run_m2 --dry-run

Dry run memeriksa data, fold, dan panjang token tanpa menyentuh GPU. Gunakan itu
lebih dulu: assert kebocoran di bawah lebih murah daripada dua jam training di
atas fold yang salah.

Keluaran dicetak ke stdout tanpa pembangkit laporan, mengikuti pola run_m1.py.
"""

import dataclasses
import json
import pathlib
import shutil
import statistics
import sys
from collections import Counter

from training.data import Row, load_babe
from training.split import group_split, train_val_test_folds
from training.train import RunResult, run_one, token_length_percentiles

N_FOLDS = 5
N_SEEDS = 5
LEARNING_RATES = (1e-5, 2e-5, 3e-5)

# Fold yang bobotnya disimpan, ditetapkan sebelum satu run pun berjalan (spec 3.3).
KEEP_FOLD = 0

RUNS_PATH = pathlib.Path("runs/m2_runs.jsonl")
CANDIDATE_DIR = pathlib.Path("runs/candidates")
SCRATCH_DIR = pathlib.Path("runs/scratch")
MODEL_DIR = pathlib.Path("models/deberta-babe")

# Baseline TF-IDF M1, dipakai hanya sebagai pemeriksaan bahwa fold tidak berubah.
# Angka yang dibandingkan di laporan dihitung ulang pada fold yang sama.
M1_BASELINE_MEAN = 0.7420


def assert_no_group_overlap(rows: list[Row], folds: list[int], n_folds: int) -> None:
    """Tidak ada artikel yang muncul di dua bagian mana pun, untuk setiap held_out.

    Fold berasal dari fungsi yang sudah di-test, jadi yang dijaga di sini adalah
    kesalahan indeks di modul ini sendiri. PRD 5.2: tidak ada angka setelah titik
    ini yang layak dilaporkan bila split bocor, jadi assert, bukan peringatan.
    """
    for held_out in range(n_folds):
        train_folds, val_fold, test_fold = train_val_test_folds(held_out, n_folds)
        parts = {"train": train_folds, "val": [val_fold], "test": [test_fold]}
        keys = {
            name: {
                row.group_key
                for row, fold in zip(rows, folds, strict=True)
                if fold in wanted
            }
            for name, wanted in parts.items()
        }
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = keys[a] & keys[b]
            assert not shared, (
                f"held_out={held_out}: {len(shared)} grup ada di {a} dan {b} sekaligus"
            )


def load_runs(path: pathlib.Path) -> dict[tuple[int, int, float], RunResult]:
    """Baca run yang sudah selesai, agar rerun melanjutkan dan tidak mengulang."""
    if not path.exists():
        return {}
    runs: dict[tuple[int, int, float], RunResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = RunResult(**json.loads(line))
        runs[(result.held_out, result.seed, result.lr)] = result
    return runs


def append_run(path: pathlib.Path, result: RunResult) -> None:
    """Tulis satu baris setelah setiap run, bukan sekali di akhir.

    27 run berjalan sekitar dua jam. Crash di run ke-20 tidak boleh menghapus
    hasil 19 run sebelumnya.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dataclasses.asdict(result)) + "\n")


def describe_data(rows: list[Row], folds: list[int]) -> None:
    print("== Data ==")
    print(f"baris dipakai               : {len(rows)}")
    labels = Counter(row.label for row in rows)
    print(f"biased / neutral            : {labels[1]} / {labels[0]}")
    sizes = Counter(folds)
    print(f"ukuran fold                 : {[sizes[f] for f in range(N_FOLDS)]}")
    print()
    print("== Pembagian per held_out ==")
    for held_out in range(N_FOLDS):
        train_folds, val_fold, test_fold = train_val_test_folds(held_out, N_FOLDS)
        n_train = sum(sizes[f] for f in train_folds)
        print(
            f"held_out {held_out}: test fold {test_fold} ({sizes[test_fold]}), "
            f"val fold {val_fold} ({sizes[val_fold]}), train {n_train}"
        )


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    rows, _ = load_babe()
    folds = group_split([row.group_key for row in rows], n_folds=N_FOLDS, seed=0)
    assert_no_group_overlap(rows, folds, N_FOLDS)
    print(f"assert kebocoran lulus untuk {N_FOLDS} held_out")
    print()
    describe_data(rows, folds)

    print()
    print("== Panjang token ==")
    percentiles = token_length_percentiles([row.text for row in rows])
    print(f"p50 / p95 / p99 / max       : {percentiles}")
    print("MAX_LENGTH 128 (spec 3.4); truncation di atas itu")

    if dry_run:
        print()
        print("dry run selesai, GPU tidak disentuh")
        return


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the dry run**

Run: `.venv-train/Scripts/python.exe -m training.run_m2 --dry-run`

Expected: the leakage assert passes, fold sizes read `[751, 883, 743, 770, 727]`, and the per-`held_out` train sizes match the table in spec 3.2 (2240, 2248, 2361, 2377, 2396). If the fold sizes differ, stop: the fold assignment changed and the comparison with the M1 baseline is void.

Expected token percentiles: p99 well under 128. If p99 exceeds 128, record the number and raise `MAX_LENGTH` to the next power of two in `train.py` before Task 4, since truncating a large share of the corpus changes what the model is measured on.

- [ ] **Step 4: Check the resume helpers on a synthetic file**

Run:

```bash
.venv-train/Scripts/python.exe -c "
import pathlib, tempfile
from training.run_m2 import append_run, load_runs
from training.train import RunResult
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / 'runs.jsonl'
    r = RunResult(0, 1, 2e-05, 0.8, 0.4, 3.0, 2240, 883, 751)
    append_run(p, r); append_run(p, r)
    loaded = load_runs(p)
    assert list(loaded) == [(0, 1, 2e-05)], loaded
    assert loaded[(0, 1, 2e-05)] == r, loaded
    assert load_runs(pathlib.Path(d) / 'missing.jsonl') == {}
    print('resume helpers ok')
"
```

Expected: `resume helpers ok`.

- [ ] **Step 5: Lint**

Run: `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add training/run_m2.py .gitignore
git commit -m "feat: add M2 entry point with three-way leakage assert and run resume"
```

---

### Task 4: Learning rate sweep

**Files:**
- Modify: `training/run_m2.py`

**Interfaces:**
- Consumes: everything from Task 3, plus `run_one`.
- Produces: `record_run(rows, folds, held_out, seed, lr, out_dir, done) -> RunResult` and `sweep(rows, folds, done) -> float` returning the chosen learning rate.

- [ ] **Step 1: Add the sweep functions**

Insert above `main()` in `training/run_m2.py`:

```python
def record_run(
    rows: list[Row],
    folds: list[int],
    held_out: int,
    seed: int,
    lr: float,
    out_dir: pathlib.Path,
    done: dict[tuple[int, int, float], RunResult],
) -> RunResult:
    """Jalankan satu run bila belum tercatat, lalu simpan barisnya."""
    key = (held_out, seed, lr)
    if key in done:
        print(f"lewati {key}, sudah tercatat")
        return done[key]
    result = run_one(rows, folds, held_out, seed, lr, out_dir=str(out_dir))
    append_run(RUNS_PATH, result)
    done[key] = result
    print(
        f"held_out {held_out} seed {seed} lr {lr:.0e}: "
        f"F1 {result.f1_macro:.4f}, val_loss {result.val_loss:.4f}, "
        f"stop di epoch {result.stopped_epoch:.0f}"
    )
    return result


def sweep(
    rows: list[Row], folds: list[int], done: dict[tuple[int, int, float], RunResult]
) -> float:
    """Pilih learning rate di satu fold, memakai validation loss.

    Spec 3.1: pemilihan tidak boleh memakai F1 pada fold test. Fold test tidak
    menyentuh keputusan apa pun sampai angka akhir dilaporkan.
    """
    print()
    print("== Sweep learning rate ==")
    results = [
        record_run(rows, folds, KEEP_FOLD, 0, lr, CANDIDATE_DIR / f"sweep{i}", done)
        for i, lr in enumerate(LEARNING_RATES)
    ]
    best = min(results, key=lambda result: result.val_loss)

    # Run sweep yang menang adalah juga anggota grid final (KEEP_FOLD, seed 0, lr
    # terpilih), dan grid akan melewatinya karena sudah tercatat. Bobotnya
    # dipindahkan ke slot seed 0 sekarang, supaya retensi checkpoint di Task 5
    # tidak kehilangan kandidat yang run-nya tidak diulang.
    winner = CANDIDATE_DIR / f"sweep{LEARNING_RATES.index(best.lr)}"
    seed_slot = CANDIDATE_DIR / "seed0"
    if winner.exists():
        shutil.rmtree(seed_slot, ignore_errors=True)
        shutil.move(str(winner), str(seed_slot))
    for index in range(len(LEARNING_RATES)):
        shutil.rmtree(CANDIDATE_DIR / f"sweep{index}", ignore_errors=True)

    for result in results:
        mark = " <-" if result is best else ""
        print(f"lr {result.lr:.0e}: val_loss {result.val_loss:.4f}{mark}")
    print(f"learning rate terpilih      : {best.lr:.0e}")
    return best.lr
```

Peak disk during the sweep is three checkpoints, about 2.2 GB. `SCRATCH_DIR` is not used here; Task 5 uses it for the folds whose weights are discarded.

- [ ] **Step 2: Call it from `main()`**

Replace the dry-run early return block in `main()` with:

```python
    if dry_run:
        print()
        print("dry run selesai, GPU tidak disentuh")
        return

    done = load_runs(RUNS_PATH)
    if done:
        print(f"{len(done)} run sudah tercatat di {RUNS_PATH}, dilanjutkan")
    lr = sweep(rows, folds, done)
```

- [ ] **Step 3: Run the sweep**

Run: `.venv-train/Scripts/python.exe -m training.run_m2`
Expected: three runs, three validation losses, one chosen learning rate, and three lines in `runs/m2_runs.jsonl`.

This is also the first real memory test. If the first run raises `torch.cuda.OutOfMemoryError`, apply the Task 2 troubleshooting step 1 (batch 8, accumulation 2), delete `runs/m2_runs.jsonl` so no run recorded under the old batch size survives, and start the sweep again.

Watch the reported stop epochs. If all three stop at epoch 1, the learning rates are too high for this data or LLRD decay is suppressing learning; check `LLRD_DECAY` first, as spec 3.4 flags it as an unmeasured starting point.

- [ ] **Step 4: Verify resume works**

Run the same command again.
Expected: three `lewati (0, 0, ...)` lines, no training, and the same chosen learning rate. This proves the 25 final runs can survive a crash.

- [ ] **Step 5: Commit**

```bash
git add training/run_m2.py
git commit -m "feat: add M2 learning rate sweep on a single fold"
```

Record the chosen learning rate and the three validation losses in the commit body. They belong in the history even though `runs/` is not committed.

---

### Task 5: The 25 final runs and checkpoint retention

**Files:**
- Modify: `training/run_m2.py`

**Interfaces:**
- Consumes: Task 4's `record_run`.
- Produces: `final_runs(rows, folds, lr, done) -> list[RunResult]` and `keep_best_checkpoint(results) -> RunResult`.

- [ ] **Step 1: Add the final-run loop and the retention step**

Insert above `main()`:

```python
def final_runs(
    rows: list[Row],
    folds: list[int],
    lr: float,
    done: dict[tuple[int, int, float], RunResult],
) -> list[RunResult]:
    """5 fold x 5 seed pada learning rate terpilih.

    Bobot ditulis ke direktori per seed hanya untuk KEEP_FOLD; sisanya memakai
    satu direktori scratch yang ditimpa terus. Puncak pemakaian disk sekitar
    3,7 GB, bukan 18 GB (spec 3.3).
    """
    print()
    print("== 25 run final ==")
    results = []
    for held_out in range(N_FOLDS):
        for seed in range(N_SEEDS):
            out_dir = (
                CANDIDATE_DIR / f"seed{seed}"
                if held_out == KEEP_FOLD
                else SCRATCH_DIR
            )
            results.append(record_run(rows, folds, held_out, seed, lr, out_dir, done))
            if held_out != KEEP_FOLD:
                shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    return results


def keep_best_checkpoint(results: list[RunResult]) -> RunResult:
    """Sisakan satu bobot: KEEP_FOLD, seed dengan validation loss terbaik.

    Seleksi memakai validation loss dari fold val, bukan F1 dari fold test, jadi
    fold yang dilaporkan tidak ikut memilih model (spec 3.3).
    """
    candidates = [result for result in results if result.held_out == KEEP_FOLD]
    best = min(candidates, key=lambda result: result.val_loss)
    source = CANDIDATE_DIR / f"seed{best.seed}" / "final"
    if not source.exists():
        print(f"CATATAN: {source} tidak ada, bobot tidak dipindahkan")
        return best
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    shutil.move(str(source), str(MODEL_DIR))
    shutil.rmtree(CANDIDATE_DIR, ignore_errors=True)
    (MODEL_DIR / "m2_selection.json").write_text(
        json.dumps(dataclasses.asdict(best), indent=2), encoding="utf-8"
    )
    print()
    print("== Checkpoint ==")
    print(f"disimpan                    : {MODEL_DIR}")
    print(
        f"asal                        : held_out {best.held_out}, seed {best.seed}, "
        f"val_loss {best.val_loss:.4f}"
    )
    return best
```

- [ ] **Step 2: Call both from `main()`**

Append after the `lr = sweep(...)` line:

```python
    results = final_runs(rows, folds, lr, done)
    keep_best_checkpoint(results)
```

- [ ] **Step 3: Run it**

Run: `.venv-train/Scripts/python.exe -m training.run_m2`

This is the long one, roughly two hours. Expected at the end: 27 lines in `runs/m2_runs.jsonl`, `models/deberta-babe/` holding one checkpoint plus `m2_selection.json`, and `runs/candidates/` and `runs/scratch/` gone.

27, not 28: the sweep already trained `(fold 0, seed 0, chosen lr)`, and the final grid reuses that recorded line instead of training it again. Task 4 moved that run's weights into `runs/candidates/seed0`, so retention still sees all five fold-0 candidates even though only four of them trained in this task.

If it dies partway, rerun the same command. Completed runs are skipped. One caveat after a crash inside the `KEEP_FOLD` block: `runs/candidates/seed{n}` may hold weights from a run whose JSONL line was never written. Before rerunning, delete only those directories, so retention can never pick weights that no recorded run produced:

```bash
.venv-train/Scripts/python.exe -c "
import shutil
from training.run_m2 import CANDIDATE_DIR, KEEP_FOLD, RUNS_PATH, load_runs
recorded = {seed for held_out, seed, _ in load_runs(RUNS_PATH) if held_out == KEEP_FOLD}
for path in sorted(CANDIDATE_DIR.glob('seed*')):
    if int(path.name.removeprefix('seed')) not in recorded:
        shutil.rmtree(path, ignore_errors=True)
        print('dihapus, tidak tercatat:', path)
print('sisa:', [p.name for p in sorted(CANDIDATE_DIR.glob('seed*'))])
"
```

- [ ] **Step 4: Verify what landed**

Run:

```bash
.venv-train/Scripts/python.exe -c "
from training.run_m2 import RUNS_PATH, load_runs
runs = load_runs(RUNS_PATH)
print('run tercatat:', len(runs))
print('lr dipakai  :', sorted({k[2] for k in runs}))
"
ls models/deberta-babe/
```

Expected: 27 recorded runs, all three learning rates present because the sweep lines stay in the file, and `models/deberta-babe/` containing `m2_selection.json`, `config.json`, a `model.safetensors`, and the tokenizer files.

Confirm the saved model actually loads, since this is what M4b and M5 consume:

```bash
.venv-train/Scripts/python.exe -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m = AutoModelForSequenceClassification.from_pretrained('models/deberta-babe')
t = AutoTokenizer.from_pretrained('models/deberta-babe')
print(m.config.num_labels, type(t).__name__)
"
```

Expected: `2` and a tokenizer class name.

- [ ] **Step 5: Commit**

```bash
git add training/run_m2.py
git commit -m "feat: add M2 final runs and single-checkpoint retention"
```

---

### Task 6: Report and the stop criterion

**Files:**
- Modify: `training/run_m2.py`

**Interfaces:**
- Consumes: `training.baseline_tfidf.run_baseline`, Task 5's results.
- Produces: `report(rows, folds, results) -> bool`, returning whether the spec 6 criterion passed.

- [ ] **Step 1: Add the report**

Add `from training.baseline_tfidf import run_baseline` to the imports, then insert above `main()`:

```python
# Spec 6: kriteria lulus ditetapkan sebelum satu run dijalankan, agar tidak
# ditafsirkan setelah melihat angkanya.
MIN_FOLD_WINS = 4
MIN_MEAN_DELTA = 0.02


def report(rows: list[Row], folds: list[int], results: list[RunResult]) -> bool:
    """Cetak angka M2 dan uji kriteria berhenti terhadap baseline TF-IDF.

    Baseline dihitung ulang di sini, bukan disalin dari catatan M1, supaya
    perbandingannya dijamin berada di fold yang sama persis.
    """
    baseline = run_baseline(
        [row.text for row in rows], [row.label for row in rows], folds, N_FOLDS
    )
    per_fold = {
        held_out: [r.f1_macro for r in results if r.held_out == held_out]
        for held_out in range(N_FOLDS)
    }
    all_scores = [r.f1_macro for r in results]

    print()
    print("== Hasil M2 ==")
    print(f"F1 macro DeBERTa            : {statistics.mean(all_scores):.4f} "
          f"+/- {statistics.stdev(all_scores):.4f}  (n={len(all_scores)})")
    print(f"F1 macro baseline TF-IDF    : {statistics.mean(baseline):.4f} "
          f"+/- {statistics.stdev(baseline):.4f}")
    if abs(statistics.mean(baseline) - M1_BASELINE_MEAN) > 0.001:
        print(
            f"CATATAN: baseline bergeser dari {M1_BASELINE_MEAN} yang dicatat M1 — "
            "fold atau filter data berubah, dan angka M2 tidak sebanding dengan M1"
        )

    print()
    print("fold | DeBERTa mean +/- std | seed terburuk..terbaik | baseline | delta")
    wins = 0
    for held_out in range(N_FOLDS):
        scores = per_fold[held_out]
        delta = statistics.mean(scores) - baseline[held_out]
        wins += delta > 0
        print(
            f"  {held_out}  | {statistics.mean(scores):.4f} +/- "
            f"{statistics.stdev(scores):.4f}      | {min(scores):.4f}..{max(scores):.4f}"
            f"        | {baseline[held_out]:.4f}   | {delta:+.4f}"
        )

    print()
    print("== Sebaran, dipisah ==")
    fold_means = [statistics.mean(per_fold[f]) for f in range(N_FOLDS)]
    seed_spreads = [statistics.stdev(per_fold[f]) for f in range(N_FOLDS)]
    print(f"std antar-fold (atas mean)  : {statistics.stdev(fold_means):.4f}")
    print(f"std antar-seed (rerata)     : {statistics.mean(seed_spreads):.4f}")
    print(f"epoch berhenti              : "
          f"{sorted({r.stopped_epoch for r in results})}")

    mean_delta = statistics.mean(all_scores) - statistics.mean(baseline)
    passed = wins >= MIN_FOLD_WINS and mean_delta >= MIN_MEAN_DELTA
    print()
    print("== Kriteria berhenti M2 (spec 6) ==")
    print(f"menang di fold              : {wins} dari {N_FOLDS}, minimal {MIN_FOLD_WINS}")
    print(f"delta mean                  : {mean_delta:+.4f}, minimal {MIN_MEAN_DELTA}")
    if passed:
        print("LULUS. Lanjut ke M3 (baseline NELA).")
    else:
        print(
            "GAGAL. PRD 9: berhenti dan periksa kualitas data, jangan menambah "
            "sweep. Angka di atas tetap dilaporkan apa adanya."
        )
    print()
    print(
        "Target 0,78 di PRD 8.1 dihitung pada split resmi BABE yang bocor "
        "(spec M1 6.1), jadi ia rujukan literatur, bukan ambang yang setara."
    )
    return passed
```

- [ ] **Step 2: Call it from `main()`**

Append after `keep_best_checkpoint(results)`:

```python
    report(rows, folds, results)
```

- [ ] **Step 3: Run it**

Run: `.venv-train/Scripts/python.exe -m training.run_m2`

All 27 runs are already recorded, so this skips training and prints the report in about a minute, most of which is the TF-IDF baseline.

Expected: a per-fold table, separated fold and seed spreads, and an explicit pass or fail line. Copy the whole stdout block into the scratchpad; Task 7 transcribes it.

- [ ] **Step 4: Lint and run the test suite**

Run: `.venv/Scripts/python.exe -m ruff check src tests training`
Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: clean, and 64 tests pass. No test in `tests/` imports torch, so the serving venv still runs them.

- [ ] **Step 5: Commit**

```bash
git add training/run_m2.py
git commit -m "feat: add M2 report and stop-criterion check"
```

Put the headline numbers in the commit body: DeBERTa mean and std, fold wins, and pass or fail.

---

### Task 7: Record the results

**Files:**
- Modify: `docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the stdout from Task 6.
- Produces: nothing in code.

- [ ] **Step 1: Add a measured-results section to the spec**

Append this section, filling every angle-bracket slot from the Task 6 stdout. Keep the shape of section 6.4 in the M1 spec: numbers first, then the decision they force.

```markdown
---

## 12. Hasil, diukur pada <tanggal>

Dijalankan lewat `.venv-train/Scripts/python.exe -m training.run_m2`.

```
learning rate terpilih      : <lr>   (val_loss <a> / <b> / <c> untuk 1e-5 / 2e-5 / 3e-5)
F1 macro DeBERTa            : <mean> +/- <std>   (25 run)
F1 macro baseline TF-IDF    : <mean> +/- <std>   (dihitung ulang pada fold yang sama)
per fold, DeBERTa vs baseline: <lima baris delta>
std antar-fold              : <angka>
std antar-seed              : <angka>
epoch berhenti              : <daftar>
menang di fold              : <n> dari 5
delta mean                  : <angka>
```

**Kriteria bagian 6: <LULUS atau GAGAL>.** <Satu paragraf: apa yang terukur, dan
apa yang tidak. Bila gagal, sebutkan bahwa pekerjaan berikutnya adalah pemeriksaan
kualitas data sesuai PRD 9, bukan M3.>

**Deviasi dari bagian 3.4 yang dipakai saat menjalankan:** <batch 8 dengan accum 2,
atau bf16 alih-alih fp16, atau "tidak ada". Sebutkan alasannya.>
```

- [ ] **Step 2: Add the M2 numbers to the README**

Add a section with this content, again filling the slots from the Task 6 stdout:

```markdown
### Hasil M2: DeBERTa fine-tuned

| | F1 macro |
|---|---|
| TF-IDF + logistic regression | <mean> +/- <std> |
| DeBERTa-v3-base, 5 fold x 5 seed | <mean> +/- <std> |

Learning rate <lr>, dipilih lewat validation loss pada satu fold. Setiap angka
adalah mean atas 5 seed, dan deviasi antar-fold serta antar-seed dilaporkan
terpisah karena artinya berbeda: <angka> antar-fold, <angka> antar-seed.

Fold dibangun per artikel (`news_link`), bukan per baris, jadi angka di atas tidak
dapat dibandingkan langsung dengan angka mana pun yang dihitung pada split resmi
BABE. Split resmi itu sendiri bocor: 384 dari 2.151 kunci grup muncul di train dan
test sekaligus. Target 0,78 di literatur dihitung pada split itu.
```

Do not touch the bold limitations claim that says the reported numbers have no predictive value. That claim covers the numbers the running app reports, and the app still serves `StubClassifier`: `NEUTRAL_MODEL_BACKEND` defaults to stub, the Dockerfile pins it, and no ONNX file exists until M5. Rewriting it now would be false.

State next to the M2 numbers that they were computed on this project's group-clean folds and are not comparable to figures computed on the official BABE split.

- [ ] **Step 3: Update `CLAUDE.md`**

In the "Current state" section, replace the four closed M2 decisions with the measured outcome: F1 macro mean and std, whether the stop criterion passed, the chosen learning rate, and where the checkpoint lives. Keep the decisions that still constrain later work, in particular that seed averaging is a reporting device and that serving loads a single ONNX file.

If the criterion failed, say so plainly and record that the next work is a data quality check, not M3.

- [ ] **Step 4: Verify no data leaked into the repo**

Run: `git status --short`
Expected: only the three documentation files. If `runs/`, `models/`, or `data/` appears, stop and fix `.gitignore` before committing.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md README.md CLAUDE.md
git commit -m "docs: record M2 fine-tuning results"
```

- [ ] **Step 6: Check the authorship of the whole branch**

Run:

```bash
git log --format='%an|%cn' main..HEAD | sort -u
git log --grep='Co-Authored-By' -i main..HEAD
```

Expected: exactly `Ihsan-p1|Ihsan-p1`, and no output from the second command.

---

## Notes for whoever executes this

**`gh` is not authenticated in the session where this plan was written.** Opening the M2 pull request needs `gh auth login` first, or the PR gets created in the browser. PR #1 for M1 is still open against `main`, and this branch is stacked on `training-m1`, so the M2 PR should target `training-m1` until PR #1 merges.

**Tasks 4 and 5 are the only expensive steps.** Task 4 costs three runs and answers the memory question. Do not start Task 5 until Task 4 has finished cleanly, and do not start either before the Task 3 dry run reports fold sizes `[751, 883, 743, 770, 727]`.

**If the stop criterion fails, the plan still completes.** Task 7 records the failure. What changes is the work after this plan: data quality checks per PRD 9, not M3.
