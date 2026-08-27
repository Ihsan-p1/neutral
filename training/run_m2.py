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
import sys
from collections import Counter

from training.data import Row, load_babe
from training.split import group_split, train_val_test_folds
from training.train import RunResult, token_length_percentiles

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
