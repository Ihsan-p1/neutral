"""Entry point M3: baseline NELA terhadap TF-IDF, gabungan keduanya, dan DeBERTa.

Jalankan: .venv-train/Scripts/python.exe -m training.run_m3

Jalan pertama memakan sekitar 46 menit untuk ekstraksi fitur. Jalan berikutnya
membaca cache dan selesai dalam hitungan detik.

Keluaran dicetak ke stdout tanpa pembangkit laporan dan tanpa file log run,
mengikuti pola run_m1.py. M2 memerlukan `m2_runs.jsonl` karena 27 run memakan
dua jam dan harus dapat dilanjutkan setelah crash; di M3 bagian mahalnya hanya
ekstraksi, dan itu sudah dijaga cache.
"""

import statistics
from collections import Counter

from training.baseline_tfidf import run_baseline
from training.data import load_babe
from training.nela import extract_features, group_map
from training.split import group_split

N_FOLDS = 5

# Baseline M1, dipakai sebagai pemeriksaan bahwa pembagian fold belum bergeser.
# Bukan sebagai angka yang dilaporkan: yang dilaporkan adalah hasil hitung ulang.
M1_BASELINE_MEAN = 0.7420
BASELINE_TOLERANCE = 0.001


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


if __name__ == "__main__":
    main()
