"""Entry point M4a: eval cross-dataset BABE -> BASIL.

Jalankan: .venv-train/Scripts/python.exe -m training.run_m4a

Tidak ada training di sini. Yang dipakai checkpoint M2 yang akan dilayani, plus
baseline TF-IDF yang di-fit pada seluruh BABE. Keluaran dicetak ke stdout dan
disalin ke runs/m4a_report.txt.

Guard reproduksi di bagian pertama berjalan sebelum apa pun yang menyentuh
BASIL. Tanpa itu, penurunan angka di BASIL tidak bisa dibedakan dari urutan
label yang tertukar atau preprocessing inferensi yang berbeda dari saat
training, dan dua-duanya tidak menimbulkan error apa pun.
"""

import pathlib

import torch
from sklearn.metrics import average_precision_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.baseline_tfidf import fit_predict_proba
from training.basil import Row as BasilRow
from training.basil import compare_with_hf, load_basil, normalise
from training.data import load_babe
from training.split import fold_of, group_split, train_val_test_folds
from training.train import MAX_LENGTH

N_FOLDS = 5
MODEL_DIR = "models/deberta-babe"

# F1 macro fold 0 seed 3 dari models/deberta-babe/m2_selection.json. Bukan angka
# yang dilaporkan M4a: perannya membuktikan inferensi di modul ini setara dengan
# inferensi saat training.
M2_FOLD0_F1 = 0.8765
REPRODUCTION_TOLERANCE = 0.01

# Di atas ambang ini yang bermasalah bukan beberapa baris, melainkan klaim
# cross-dataset-nya sendiri.
OVERLAP_LIMIT = 0.01

# Pembagian dev/test BASIL, spec 3.4. Kuncinya event, bukan artikel, bukan baris.
DEV_FOLDS = 10
DEV_CUTOFF = 3

REPORT_PATH = pathlib.Path("runs/m4a_report.txt")

# Kandidat threshold. Rentang tetap, bukan probabilitas unik, supaya nilai yang
# dilaporkan sama persis di setiap jalan dan mudah dibaca di tabel.
THRESHOLDS = [value / 100 for value in range(1, 100)]


def predict_deberta(
    texts: list[str], model_dir: str = MODEL_DIR, batch_size: int = 32
) -> list[float]:
    """Kembalikan P(label == 1) untuk tiap teks, dalam urutan masukan.

    `MAX_LENGTH` diimpor dari train.py, bukan ditulis ulang, supaya truncation
    saat inferensi identik dengan saat training. Nilai berbeda tidak menimbulkan
    error dan hanya menggeser angkanya.

    Temperature tidak diterapkan di sini. Kalibrasi adalah M4b, dan M4a hanya
    memerlukan urutan probabilitas plus threshold yang dicari sendiri.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
    return probabilities


def assert_checkpoint_reproduces() -> float:
    """Uji checkpoint pada test split fold 0 BABE. Kembalikan F1 macro terukur."""
    rows, _ = load_babe()
    folds = group_split([row.group_key for row in rows], N_FOLDS, seed=0)
    _, _, test_fold = train_val_test_folds(0, N_FOLDS)
    subset = [row for row, fold in zip(rows, folds, strict=True) if fold == test_fold]

    probabilities = predict_deberta([row.text for row in subset])
    predicted = [1 if value >= 0.5 else 0 for value in probabilities]
    measured = float(f1_score([row.label for row in subset], predicted, average="macro"))

    delta = abs(measured - M2_FOLD0_F1)
    if delta > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"checkpoint memberi F1 macro {measured:.4f} pada test split fold 0, "
            f"selisih {delta:.4f} dari {M2_FOLD0_F1} yang tercatat M2; "
            "inferensi atau urutan label berbeda dari saat training"
        )
    print(f"guard reproduksi   : F1 macro {measured:.4f} pada {len(subset)} baris")
    return measured


def drop_babe_overlap(
    rows: list[BasilRow], babe_texts: list[str]
) -> tuple[list[BasilRow], int]:
    """Buang kalimat BASIL yang identik dengan kalimat BABE. Kembalikan (sisa, jumlah).

    BABE dan BASIL dua korpus berbeda, tapi keduanya menyerap artikel berita AS
    dari periode yang berdekatan. Irisan diperiksa, bukan diasumsikan tidak ada.
    """
    seen = {normalise(text) for text in babe_texts}
    kept = [row for row in rows if normalise(row.text) not in seen]
    return kept, len(rows) - len(kept)


def split_dev_test(rows: list[BasilRow]) -> tuple[list[int], list[int]]:
    """Bagi indeks baris jadi dev dan test, dikelompokkan per event.

    `fold_of()` men-hash kuncinya dengan SHA-256, jadi satu event selalu jatuh
    utuh di satu sisi dan hasilnya sama di setiap mesin tanpa menyimpan daftar
    fold ke disk. Pembagian 30/70-nya perkiraan atas 100 event, bukan tepat.
    """
    dev: list[int] = []
    test: list[int] = []
    for index, row in enumerate(rows):
        bucket = dev if fold_of(row.event_key, DEV_FOLDS, seed=0) < DEV_CUTOFF else test
        bucket.append(index)
    if not dev or not test:
        raise RuntimeError(f"sisi kosong: dev={len(dev)} test={len(test)}")
    return dev, test


def best_threshold(
    probabilities: list[float], gold: list[int], indices: list[int]
) -> float:
    """Cari threshold yang memaksimalkan F1 macro pada indeks yang diberikan.

    Dipanggil hanya dengan indeks dev. Memanggilnya dengan indeks test akan
    memilih threshold pada data yang sama tempat angkanya dilaporkan, dan itu
    tidak menimbulkan error apa pun.
    """
    subset_gold = [gold[index] for index in indices]
    scored = []
    for threshold in THRESHOLDS:
        predicted = [1 if probabilities[index] >= threshold else 0 for index in indices]
        scored.append(
            (
                f1_score(subset_gold, predicted, average="macro", zero_division=0),
                threshold,
            )
        )
    return max(scored)[1]


def score(
    probabilities: list[float],
    gold: list[int],
    dev: list[int],
    test: list[int],
) -> dict:
    """Hitung kedua angka PRD 8.2 plus AP, semuanya pada indeks test.

    Angka apa adanya dan angka rekalibrasi dihitung pada baris yang sama supaya
    selisihnya murni efek threshold. Dev hanya dipakai memilih threshold.
    """
    threshold = best_threshold(probabilities, gold, dev)
    test_gold = [gold[index] for index in test]
    test_prob = [probabilities[index] for index in test]
    asis = [1 if value >= 0.5 else 0 for value in test_prob]
    recal = [1 if value >= threshold else 0 for value in test_prob]

    return {
        "f1_asis": float(f1_score(test_gold, asis, average="macro", zero_division=0)),
        "f1_recal": float(f1_score(test_gold, recal, average="macro", zero_division=0)),
        "threshold": threshold,
        "ap": float(average_precision_score(test_gold, test_prob)),
        "gold_rate": sum(test_gold) / len(test_gold),
        "pred_rate": sum(asis) / len(asis),
    }


def constant_floor(gold: list[int], indices: list[int]) -> float:
    """F1 macro bila seluruh baris diprediksi neutral.

    Dicetak bersama tabel: pada prior positif serendah BASIL, angka di sekitar
    0,49 setara dengan tidak memprediksi apa pun, dan tanpa lantai ini ia
    terbaca seperti ada sinyal.
    """
    subset = [gold[index] for index in indices]
    return float(f1_score(subset, [0] * len(subset), average="macro", zero_division=0))


def main() -> None:
    assert_checkpoint_reproduces()

    lines: list[str] = []
    babe_rows, _ = load_babe()
    rows, report = load_basil()
    lines.append(f"artikel BASIL      : {report.n_articles}, event {report.n_events}")
    lines.append(
        f"kalimat            : {report.n_sentences_raw}, "
        f"kosong dibuang {report.n_blank_dropped}"
    )
    lines.append(
        f"span               : {report.n_spans}, tidak sesuai {report.n_span_mismatch}"
    )
    lines.append(f"positif lex        : {report.n_lex} ({report.n_lex / len(rows):.1%})")
    lines.append(
        f"positif lex or inf : {report.n_lex_or_info} "
        f"({report.n_lex_or_info / len(rows):.1%})"
    )
    lines.append(f"sumber             : {report.sources}")

    comparison = compare_with_hf(rows)
    lines.append(
        f"banding HuggingFace: {comparison.n_hf} baris, cocok "
        f"{comparison.matched_ratio:.1%}, label beda {comparison.differs_ratio:.1%}"
    )

    rows, dropped = drop_babe_overlap(rows, [row.text for row in babe_rows])
    ratio = dropped / (len(rows) + dropped)
    lines.append(f"irisan dengan BABE : {dropped} kalimat dibuang ({ratio:.2%})")
    if ratio > OVERLAP_LIMIT:
        raise RuntimeError(
            f"{ratio:.1%} kalimat BASIL juga ada di BABE; "
            "klaim cross-dataset tidak berlaku pada proporsi ini"
        )

    dev, test = split_dev_test(rows)
    dev_events = len({rows[index].event_key for index in dev})
    test_events = len({rows[index].event_key for index in test})
    lines.append(f"dev                : {len(dev)} kalimat, {dev_events} event")
    lines.append(f"test               : {len(test)} kalimat, {test_events} event")

    texts = [row.text for row in rows]
    predictions = {
        "DeBERTa": predict_deberta(texts),
        "TF-IDF": fit_predict_proba(
            [row.text for row in babe_rows],
            [row.label for row in babe_rows],
            texts,
        ),
    }
    golds = {
        "lex": [row.lex for row in rows],
        "lex or inf": [1 if row.lex or row.info else 0 for row in rows],
    }

    lines.append("")
    lines.append(
        f"{'model':8} {'gold':11} {'F1 apa adanya':>13} {'F1 rekal':>9} "
        f"{'thr':>5} {'AP':>6} {'gold+':>6} {'pred+':>6}"
    )
    for model_name, probabilities in predictions.items():
        for gold_name, gold in golds.items():
            result = score(probabilities, gold, dev, test)
            lines.append(
                f"{model_name:8} {gold_name:11} {result['f1_asis']:13.4f} "
                f"{result['f1_recal']:9.4f} {result['threshold']:5.2f} "
                f"{result['ap']:6.4f} {result['gold_rate']:6.1%} "
                f"{result['pred_rate']:6.1%}"
            )

    lines.append("")
    for gold_name, gold in golds.items():
        lines.append(
            f"lantai constant predictor, gold {gold_name}: "
            f"F1 macro {constant_floor(gold, test):.4f}"
        )
    lines.append("")
    lines.append(
        "pembanding BABE: TF-IDF 0,7420 +/- 0,0194 (M1), "
        "DeBERTa 0,8477 +/- 0,0277 (M2)"
    )
    # Empat catatan spec 7.3, dicetak bersama tabel dan bukan di catatan kaki.
    lines.append(
        "catatan: label dihitung dari rilis resmi launchnlp/BASIL dengan "
        "segmentasi kalimat milik dataset itu sendiri; re-upload HuggingFace "
        "hanya dibandingkan, tidak dipakai sebagai label"
    )
    lines.append(
        "catatan: aturan pemetaan - kalimat ke-k biased bila ada span "
        "bias == 'lex' dengan id == 'p{k}'"
    )
    lines.append(
        "catatan: BASIL beranotasi dalam konteks artikel, BABE berisi kalimat "
        "berdiri sendiri; sebagian selisih adalah artefak label"
    )
    lines.append(
        "catatan: angka DeBERTa berasal dari satu checkpoint, bukan mean "
        "beberapa seed; M2 mengukur spread antar-seed 0,0187 pada BABE"
    )

    print("\n".join(lines))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlaporan disimpan   : {REPORT_PATH}")


if __name__ == "__main__":
    main()
