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

from training.baseline_tfidf import run_baseline
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
    # dipindahkan ke slot seed 0 sekarang, supaya retensi checkpoint di tahap
    # berikutnya tidak kehilangan kandidat yang run-nya tidak diulang.
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
                CANDIDATE_DIR / f"seed{seed}" if held_out == KEEP_FOLD else SCRATCH_DIR
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
    print(
        f"F1 macro DeBERTa            : {statistics.mean(all_scores):.4f} "
        f"+/- {statistics.stdev(all_scores):.4f}  (n={len(all_scores)})"
    )
    print(
        f"F1 macro baseline TF-IDF    : {statistics.mean(baseline):.4f} "
        f"+/- {statistics.stdev(baseline):.4f}"
    )
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
    print(
        f"epoch berhenti              : {sorted({r.stopped_epoch for r in results})}"
    )

    mean_delta = statistics.mean(all_scores) - statistics.mean(baseline)
    passed = wins >= MIN_FOLD_WINS and mean_delta >= MIN_MEAN_DELTA
    print()
    print("== Kriteria berhenti M2 (spec 6) ==")
    print(
        f"menang di fold              : {wins} dari {N_FOLDS}, minimal {MIN_FOLD_WINS}"
    )
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

    done = load_runs(RUNS_PATH)
    if done:
        print(f"{len(done)} run sudah tercatat di {RUNS_PATH}, dilanjutkan")
    lr = sweep(rows, folds, done)
    results = final_runs(rows, folds, lr, done)
    keep_best_checkpoint(results)
    report(rows, folds, results)


if __name__ == "__main__":
    main()
