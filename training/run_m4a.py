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

from training.data import load_babe
from training.split import group_split, train_val_test_folds
from training.train import MAX_LENGTH

N_FOLDS = 5
MODEL_DIR = "models/deberta-babe"

# F1 macro fold 0 seed 3 dari models/deberta-babe/m2_selection.json. Bukan angka
# yang dilaporkan M4a: perannya membuktikan inferensi di modul ini setara dengan
# inferensi saat training.
M2_FOLD0_F1 = 0.8765
REPRODUCTION_TOLERANCE = 0.01


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


def main() -> None:
    assert_checkpoint_reproduces()


if __name__ == "__main__":
    main()
