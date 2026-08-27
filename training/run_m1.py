"""Entry point M1: muat BABE, laporkan bentuk data sebenarnya, buktikan split
tidak bocor, lalu hasilkan angka baseline yang menjadi pembanding M2.

Jalankan: .venv-train/Scripts/python.exe -m training.run_m1

Keluaran sengaja dicetak ke stdout tanpa pembangkit laporan. Angkanya disalin ke
README sesuai kewajiban transparansi PRD 5.1; sebuah pembangkit laporan tidak
menambah apa pun pada kewajiban itu.
"""

import statistics
from collections import Counter

from training.baseline_tfidf import run_baseline
from training.data import Row, load_babe
from training.split import group_split

N_FOLDS = 5

# PRD 8.3 mensyaratkan minimal 12 outlet agar korelasi outlet bermakna.
MIN_OUTLETS_FOR_CORRELATION = 12


def check_official_split(rows: list[Row]) -> tuple[bool, int]:
    """Apakah split train/test resmi BABE bersih terhadap kunci grup.

    Spec 6.1 menetapkan konsekuensinya lebih dulu: bila bersih, split resmi
    dipertahankan agar angka sebanding dengan literatur BABE; bila bocor, split
    resmi dibuang dan split sendiri dipakai.
    """
    splits_per_group: dict[str, set[str]] = {}
    for row in rows:
        splits_per_group.setdefault(row.group_key, set()).add(row.official_split)
    shared = [key for key, splits in splits_per_group.items() if len(splits) > 1]
    return not shared, len(shared)


def main() -> None:
    rows, report = load_babe()

    print("== Data ==")
    print(f"baris mentah                : {report.n_raw}")
    print(f"dibuang, news_link kosong   : {report.n_missing_link}")
    print(f"dibuang, agreement rendah   : {report.n_low_agreement}")
    print(f"dibuang, label di luar 0/1  : {report.n_unexpected_label}")
    print(f"dipakai                     : {report.n_kept}")
    print(f"nilai label ditemukan       : {report.label_values}")
    print(f"nilai type ditemukan        : {report.lean_values}")
    print(f"nilai label_opinion         : {len(report.opinion_values)} nilai")

    print()
    print("== Kunci grup ==")
    print(f"news_link distinct          : {report.n_groups_raw}")
    print(f"setelah normalisasi URL     : {report.n_groups_normalised}")
    collapsed = report.n_groups_raw - report.n_groups_normalised
    print(f"grup yang menyatu           : {collapsed}")

    print()
    print("== Distribusi ==")
    labels = Counter(row.label for row in rows)
    total = sum(labels.values())
    for label in sorted(labels):
        name = "biased" if label == 1 else "neutral"
        print(
            f"label {label} ({name:7})          : {labels[label]} "
            f"({labels[label] / total:.1%})"
        )

    outlets = Counter(row.outlet for row in rows)
    print(f"outlet distinct             : {len(outlets)}")
    print(f"5 outlet terbesar           : {outlets.most_common(5)}")
    if len(outlets) < MIN_OUTLETS_FOR_CORRELATION:
        print(
            f"CATATAN: {len(outlets)} outlet, di bawah {MIN_OUTLETS_FOR_CORRELATION} "
            "yang disyaratkan PRD 8.3 — korelasi outlet tidak dapat dijalankan "
            "pada BABE saja"
        )
    print(f"lean distinct               : {dict(Counter(row.lean for row in rows))}")

    print()
    print("== Split resmi BABE ==")
    clean, n_shared = check_official_split(rows)
    if clean:
        print("bersih terhadap kunci grup — split resmi dapat dipertahankan")
    else:
        print(
            f"BOCOR: {n_shared} kunci grup muncul di train dan test resmi sekaligus. "
            "Spec 6.1: split resmi dibuang, split sendiri dipakai, dan "
            "ketidakcocokan dengan angka literatur dicatat di README"
        )

    print()
    print("== Group split ==")
    folds = group_split([row.group_key for row in rows], n_folds=N_FOLDS)

    # PRD 5.2: tidak ada satu pun angka setelah titik ini yang layak dilaporkan
    # bila split bocor. Karena itu assert, bukan peringatan.
    seen: dict[str, int] = {}
    for row, fold in zip(rows, folds, strict=True):
        first = seen.setdefault(row.group_key, fold)
        assert first == fold, f"grup {row.group_key} ada di fold {first} dan {fold}"
    print(f"{len(seen)} grup, tidak ada yang muncul di dua fold")

    sizes = Counter(folds)
    print(f"ukuran fold                 : {[sizes[f] for f in range(N_FOLDS)]}")
    for fold in range(N_FOLDS):
        in_fold = [row.label for row, f in zip(rows, folds, strict=True) if f == fold]
        share = sum(in_fold) / len(in_fold)
        print(f"  fold {fold}: {len(in_fold)} baris, {share:.1%} biased")

    print()
    print("== Baseline TF-IDF ==")
    scores = run_baseline(
        [row.text for row in rows], [row.label for row in rows], folds, N_FOLDS
    )
    print(f"F1 macro per fold           : {[round(s, 4) for s in scores]}")
    # "+/-" bukan "±": stdout Windows memakai cp1252 dan merusak karakter itu.
    print(
        f"F1 macro                    : {statistics.mean(scores):.4f} "
        f"+/- {statistics.stdev(scores):.4f}"
    )
    print()
    print("Angka di atas adalah pembanding untuk kriteria berhenti M2 (PRD 9).")


if __name__ == "__main__":
    main()
