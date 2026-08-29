"""Entry point M3: baseline NELA terhadap TF-IDF, gabungan keduanya, dan DeBERTa.

Jalankan: .venv-train/Scripts/python.exe -m training.run_m3

Jalan pertama memakan sekitar 46 menit untuk ekstraksi fitur. Jalan berikutnya
membaca cache dan selesai dalam hitungan detik.

Keluaran dicetak ke stdout tanpa pembangkit laporan dan tanpa file log run,
mengikuti pola run_m1.py. M2 memerlukan `m2_runs.jsonl` karena 27 run memakan
dua jam dan harus dapat dilanjutkan setelah crash; di M3 bagian mahalnya hanya
ekstraksi, dan itu sudah dijaga cache.
"""

import json
import pathlib
import statistics
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from training.baseline_tfidf import run_baseline
from training.data import load_babe
from training.nela import extract_features, group_map
from training.split import group_split

N_FOLDS = 5

# Baseline M1, dipakai sebagai pemeriksaan bahwa pembagian fold belum bergeser.
# Bukan sebagai angka yang dilaporkan: yang dilaporkan adalah hasil hitung ulang.
M1_BASELINE_MEAN = 0.7420
BASELINE_TOLERANCE = 0.001

M2_RUNS_PATH = pathlib.Path("runs/m2_runs.jsonl")
# Learning rate run final M2. Dua titik sweep, 1e-5 dan 2e-5, tidak ikut
# dirata-ratakan; filter ini yang menyisakan tepat 25 run.
FINAL_LR = 3e-5


def assert_no_group_overlap(group_keys: list[str], folds: list[int]) -> int:
    """Tidak ada kunci grup yang muncul di dua fold. Kembalikan jumlah grup.

    Fold berasal dari fungsi yang sudah di-test, jadi yang dijaga di sini adalah
    kesalahan indeks di modul ini sendiri. PRD 5.2: tidak ada angka setelah titik
    ini yang layak dilaporkan bila split bocor, jadi assert, bukan peringatan.
    """
    seen: dict[str, int] = {}
    for key, fold in zip(group_keys, folds, strict=True):
        first = seen.setdefault(key, fold)
        assert first == fold, f"grup {key} ada di fold {first} dan {fold}"
    return len(seen)


def _classifier() -> LogisticRegression:
    """Setelan identik dengan baseline_tfidf.py, tanpa sweep (spec M3 3.1).

    Model yang perlu di-tuning agar terlihat baik tidak lagi sebanding dengan
    model yang tidak di-tuning, jadi ketiga konfigurasi memakai setelan ini.
    """
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def _fold_indices(folds: list[int], held_out: int) -> tuple[list[int], list[int]]:
    """Indeks train dan test untuk satu fold yang di-hold-out."""
    train_idx = [i for i, fold in enumerate(folds) if fold != held_out]
    test_idx = [i for i, fold in enumerate(folds) if fold == held_out]
    if not train_idx or not test_idx:
        raise ValueError(
            f"fold {held_out} kosong: train={len(train_idx)} test={len(test_idx)}"
        )
    return train_idx, test_idx


def _prepare_nela(
    features: np.ndarray, train_idx: list[int], test_idx: list[int]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Buang kolom varians-nol lalu standardisasi, keduanya di-fit pada train saja.

    Mem-fit keduanya pada seluruh matriks akan memakai informasi test untuk
    memutuskan kolom mana yang hidup dan berapa mean tiap kolom. Kebocoran itu
    tidak memunculkan error apa pun (spec M3 3.5).

    Kembalikan matriks train, matriks test, dan jumlah kolom yang bertahan.
    """
    selector = VarianceThreshold().fit(features[train_idx])
    if not selector.get_support().any():
        raise ValueError("semua kolom varians-nol pada bagian train")
    scaler = StandardScaler().fit(selector.transform(features[train_idx]))
    train = scaler.transform(selector.transform(features[train_idx]))
    test = scaler.transform(selector.transform(features[test_idx]))
    return train, test, int(selector.get_support().sum())


def score_nela(
    features: np.ndarray,
    labels: list[int],
    folds: list[int],
    n_folds: int = N_FOLDS,
) -> tuple[list[float], list[int]]:
    """Latih satu logistic regression per fold di atas fitur NELA.

    Kembalikan F1 macro per fold dan jumlah kolom yang bertahan per fold. Jumlah
    itu bisa berbeda antar fold, karena kolom dibuang berdasarkan bagian train.
    """
    scores: list[float] = []
    kept: list[int] = []
    for held_out in range(n_folds):
        train_idx, test_idx = _fold_indices(folds, held_out)
        train, test, n_kept = _prepare_nela(features, train_idx, test_idx)
        model = _classifier()
        model.fit(train, [labels[i] for i in train_idx])
        predicted = model.predict(test)
        scores.append(
            float(f1_score([labels[i] for i in test_idx], predicted, average="macro"))
        )
        kept.append(n_kept)
    return scores, kept


def score_combined(
    texts: list[str],
    features: np.ndarray,
    labels: list[int],
    folds: list[int],
    n_folds: int = N_FOLDS,
) -> list[float]:
    """TF-IDF disambung fitur NELA dalam satu matriks desain.

    `scipy.sparse.hstack` dipakai, bukan `ColumnTransformer`: yang terakhir
    menuntut DataFrame, dan itu memasukkan pandas hanya untuk keperluan ini
    (spec M3 3.6). Vectorizer di-fit ulang di dalam tiap iterasi, hanya pada
    bagian train, sama seperti di `baseline_tfidf.py`.
    """
    scores: list[float] = []
    for held_out in range(n_folds):
        train_idx, test_idx = _fold_indices(folds, held_out)
        dense_train, dense_test, _ = _prepare_nela(features, train_idx, test_idx)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        sparse_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        sparse_test = vectorizer.transform([texts[i] for i in test_idx])
        train = hstack([sparse_train, csr_matrix(dense_train)]).tocsr()
        test = hstack([sparse_test, csr_matrix(dense_test)]).tocsr()
        model = _classifier()
        model.fit(train, [labels[i] for i in train_idx])
        predicted = model.predict(test)
        scores.append(
            float(f1_score([labels[i] for i in test_idx], predicted, average="macro"))
        )
    return scores


def deberta_per_fold(
    path: pathlib.Path = M2_RUNS_PATH, n_folds: int = N_FOLDS
) -> tuple[list[float], int]:
    """Rata-rata F1 macro DeBERTa per fold, dibaca dari log run M2.

    Angka ini dibaca ulang dari log, bukan disalin dari README, supaya tabel M3
    tidak pernah memuat angka yang tidak dapat ditelusuri (spec M3 2).

    Kembalikan mean per fold dan jumlah run yang ikut dihitung.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} tidak ada. Kolom DeBERTa dibaca dari log run M2; jalankan "
            "training.run_m2 lebih dulu, atau kembalikan log itu ke tempatnya."
        )
    per_fold: dict[int, list[float]] = {fold: [] for fold in range(n_folds)}
    n_runs = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if abs(float(record["lr"]) - FINAL_LR) > 1e-12:
            continue
        per_fold[int(record["held_out"])].append(float(record["f1_macro"]))
        n_runs += 1
    empty = [fold for fold, scores in per_fold.items() if not scores]
    if empty:
        raise SystemExit(f"tidak ada run lr={FINAL_LR} untuk fold {empty}")
    return [statistics.mean(per_fold[fold]) for fold in range(n_folds)], n_runs


def _print_model_row(name: str, scores: list[float], reference: float) -> None:
    """Satu baris tabel perbandingan model."""
    mean = statistics.mean(scores)
    print(
        f"{name:<18} | {mean:.4f} +/- {statistics.stdev(scores):.4f} | "
        f"{mean - reference:+.4f}      | {[round(s, 4) for s in scores]}"
    )


def main() -> None:
    rows, _ = load_babe()
    texts = [row.text for row in rows]
    labels = [row.label for row in rows]
    group_keys = [row.group_key for row in rows]
    folds = group_split(group_keys, n_folds=N_FOLDS, seed=0)

    print("== Fold ==")
    n_groups = assert_no_group_overlap(group_keys, folds)
    print(f"assert kebocoran lulus      : {n_groups} grup, tidak ada di dua fold")
    sizes = Counter(folds)
    print(f"baris                       : {len(rows)}")
    print(f"ukuran fold                 : {[sizes[f] for f in range(N_FOLDS)]}")

    print()
    print("== Fitur NELA ==")
    features, names = extract_features(texts)
    print(f"matriks                     : {features.shape[0]} x {features.shape[1]}")
    mapping = group_map()
    per_group = Counter(mapping[name] for name in names)
    print(f"fitur per grup              : {dict(per_group)}")

    print()
    print("== Baseline TF-IDF, dihitung ulang ==")
    tfidf = run_baseline(texts, labels, folds, N_FOLDS)
    tfidf_mean = statistics.mean(tfidf)
    print(f"F1 macro per fold           : {[round(s, 4) for s in tfidf]}")
    print(f"F1 macro                    : {tfidf_mean:.4f} +/- {statistics.stdev(tfidf):.4f}")
    if abs(tfidf_mean - M1_BASELINE_MEAN) > BASELINE_TOLERANCE:
        raise SystemExit(
            f"baseline bergeser dari {M1_BASELINE_MEAN} yang dicatat M1 ke "
            f"{tfidf_mean:.4f}. Pembagian fold atau filter data sudah berubah, dan "
            "seluruh perbandingan M3 tidak sah. Berhenti di sini."
        )
    print("cocok dengan angka M1, pembagian fold tidak bergeser")

    nela, kept = score_nela(features, labels, folds)
    combined = score_combined(texts, features, labels, folds)
    deberta, n_runs = deberta_per_fold()

    print()
    print("== Empat model pada fold yang sama ==")
    print("model              | F1 macro            | delta TF-IDF | per fold")
    _print_model_row("TF-IDF + LR", tfidf, tfidf_mean)
    _print_model_row("NELA + LR", nela, tfidf_mean)
    _print_model_row("TF-IDF + NELA + LR", combined, tfidf_mean)
    _print_model_row("DeBERTa", deberta, tfidf_mean)
    span = f"{min(kept)}" if min(kept) == max(kept) else f"{min(kept)}..{max(kept)}"
    print()
    print(f"kolom NELA bertahan         : {span} dari {features.shape[1]}")
    print(f"run DeBERTa yang dirata2    : {n_runs} pada lr={FINAL_LR}")


if __name__ == "__main__":
    main()
