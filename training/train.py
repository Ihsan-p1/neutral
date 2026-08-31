"""Fine-tune DeBERTa untuk satu kombinasi (fold, seed, learning rate).

Modul ini mengimpor torch dan transformers, jadi ia hanya jalan di .venv-train/.
`tests/` tidak boleh mengimpornya: job CI serving tidak memasang torch. Lihat
catatan di kepala `training/split.py`.

Satu pemanggilan `run_one()` menghasilkan satu `RunResult`. Orkestrasi 27 run ada
di `run_m2.py`, bukan di sini.
"""

import math
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
        groups.append(
            {"params": list(layer.parameters()), "lr": lr * LLRD_DECAY**depth}
        )
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
    # dtype fp32 dinyatakan eksplisit. transformers 5 memuat bobot dengan dtype
    # checkpoint, dan checkpoint deberta-v3-base berisi fp16; melatih parameter
    # fp16 langsung membuat GradScaler gagal dengan "Attempting to unscale FP16
    # gradients". Mixed precision tetap aktif lewat `fp16=True`: yang dibutuhkan
    # adalah master weight fp32, bukan matematika fp32.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, dtype=torch.float32
    )
    datasets = {
        name: _Tokenized(
            [row.text for row in subset], [row.label for row in subset], tokenizer
        )
        for name, subset in subsets.items()
    }

    # transformers 5 membuang `warmup_ratio` dan hanya menyediakan
    # `warmup_steps`, jadi rasio spec 3.4 dihitung menjadi step di sini. Dasarnya
    # epoch maksimum: early stopping bisa memotong lebih awal, dan warmup yang
    # ikut memendek akan membuat learning rate berbeda antar run.
    steps_per_epoch = math.ceil(len(subsets["train"]) / (BATCH_SIZE * GRAD_ACCUM))
    warmup_steps = max(1, round(WARMUP_RATIO * steps_per_epoch * max_epochs))

    args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,
        data_seed=seed,
        learning_rate=lr,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_steps=warmup_steps,
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
    # 600 baris, bukan 90. Dengan 90 baris dan n_folds=3, train hanya 30 baris
    # dan seluruh training berjumlah 4 step optimizer — terlalu sedikit untuk
    # membedakan harness yang bekerja dari harness yang rusak.
    n_rows = 600
    rows = [
        Row(
            text=biased if i % 2 == 0 else neutral,
            label=1 if i % 2 == 0 else 0,
            outlet="synthetic",
            lean="None",
            group_key=f"synthetic/{i}",
            official_split="train",
        )
        for i in range(n_rows)
    ]
    folds = [i % 3 for i in range(n_rows)]
    out_dir = "runs/selfcheck"
    try:
        result = run_one(
            rows,
            folds,
            held_out=0,
            seed=0,
            lr=5e-5,
            out_dir=out_dir,
            n_folds=3,
            max_epochs=3,
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    assert result.f1_macro > 0.9, (
        f"dua kelas terpisah sempurna gagal dipelajari: {result}"
    )
    assert result.val_loss == result.val_loss, f"val_loss NaN: {result}"
    third = n_rows // 3
    assert result.n_train == third, result
    assert result.n_val == third and result.n_test == third, result
    print("self-check ok:", result)


if __name__ == "__main__":
    _self_check()
