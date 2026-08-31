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

import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def main() -> None:
    assert_checkpoint_reproduces()

    babe_rows, _ = load_babe()
    rows, report = load_basil()
    print(f"artikel BASIL      : {report.n_articles}, event {report.n_events}")
    print(
        f"kalimat            : {report.n_sentences_raw}, "
        f"kosong dibuang {report.n_blank_dropped}"
    )
    print(f"span               : {report.n_spans}, tidak sesuai {report.n_span_mismatch}")
    print(f"positif lex        : {report.n_lex} ({report.n_lex / len(rows):.1%})")
    print(
        f"positif lex or inf : {report.n_lex_or_info} "
        f"({report.n_lex_or_info / len(rows):.1%})"
    )
    print(f"sumber             : {report.sources}")

    comparison = compare_with_hf(rows)
    print(
        f"banding HuggingFace: {comparison.n_hf} baris, cocok "
        f"{comparison.matched_ratio:.1%}, label beda {comparison.differs_ratio:.1%}"
    )

    rows, dropped = drop_babe_overlap(rows, [row.text for row in babe_rows])
    ratio = dropped / (len(rows) + dropped)
    print(f"irisan dengan BABE : {dropped} kalimat dibuang ({ratio:.2%})")
    if ratio > OVERLAP_LIMIT:
        raise RuntimeError(
            f"{ratio:.1%} kalimat BASIL juga ada di BABE; "
            "klaim cross-dataset tidak berlaku pada proporsi ini"
        )

    dev, test = split_dev_test(rows)
    dev_events = len({rows[index].event_key for index in dev})
    test_events = len({rows[index].event_key for index in test})
    print(f"dev                : {len(dev)} kalimat, {dev_events} event")
    print(f"test               : {len(test)} kalimat, {test_events} event")


if __name__ == "__main__":
    main()
