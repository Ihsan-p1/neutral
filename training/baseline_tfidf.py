"""TF-IDF ditambah logistic regression, dievaluasi pada fold yang sama dengan
yang akan dipakai DeBERTa.

Peran modul ini ditentukan oleh kriteria berhenti M2 di PRD 9: bila model yang
di-fine-tune tidak mengungguli angka di sini secara meyakinkan, yang diperiksa
adalah kualitas data, bukan hyperparameter. Tanpa baseline ini, instruksi itu
tidak dapat dijalankan.

Metrik yang dipakai F1 macro, sama dengan metrik utama PRD 8.1, agar kedua angka
dapat dibandingkan tanpa konversi.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


def _model():
    # class_weight="balanced" menangani ketimpangan kelas yang diminta PRD 5.1
    # tanpa menyentuh data. Bigram disertakan karena bias leksikal sering
    # berupa pasangan kata, misalnya "so-called expert".
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def run_baseline(
    texts: list[str], labels: list[int], folds: list[int], n_folds: int = 5
) -> list[float]:
    """Latih satu model per fold dengan fold itu sebagai test, kembalikan F1 macro.

    Vectorizer di-fit ulang di dalam tiap iterasi, hanya pada bagian train.
    Mem-fit vectorizer pada seluruh data lebih dulu akan membocorkan vocabulary
    test ke train, dan kebocoran itu tidak memunculkan error apa pun.
    """
    scores: list[float] = []
    for held_out in range(n_folds):
        train_idx = [i for i, fold in enumerate(folds) if fold != held_out]
        test_idx = [i for i, fold in enumerate(folds) if fold == held_out]
        if not train_idx or not test_idx:
            raise ValueError(
                f"fold {held_out} kosong: train={len(train_idx)} test={len(test_idx)}"
            )
        model = _model()
        model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        predicted = model.predict([texts[i] for i in test_idx])
        scores.append(
            float(f1_score([labels[i] for i in test_idx], predicted, average="macro"))
        )
    return scores


def _self_check() -> None:
    """Periksa loop fold pada data yang jawabannya sudah diketahui."""
    biased = ["so-called experts peddle this outrageous scheme"] * 50
    neutral = ["the committee met on tuesday to review the budget"] * 50
    texts = biased + neutral
    labels = [1] * 50 + [0] * 50
    folds = [i % 5 for i in range(100)]

    scores = run_baseline(texts, labels, folds)
    assert len(scores) == 5, scores
    # Dua kelas yang terpisah sempurna harus terpisah sempurna.
    assert min(scores) > 0.95, scores

    try:
        run_baseline(texts, labels, [0] * 100)
    except ValueError:
        pass
    else:
        raise AssertionError("fold kosong seharusnya menimbulkan ValueError")

    print("self-check ok:", [round(s, 4) for s in scores])


if __name__ == "__main__":
    _self_check()
