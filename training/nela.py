"""Ekstraksi fitur NELA untuk teks sepanjang kalimat, dengan cache di disk.

Modul ini mengimpor `nela_features` dan `nltk`, jadi ia memerlukan venv training
dan tidak boleh diimpor dari `tests/`. Batas yang sama sudah berlaku untuk
`baseline_tfidf.py` dan `train.py`.

Ekstraksi memakan sekitar 714 ms per kalimat, atau 46 menit untuk 3.874 baris
BABE (spec M3 3.3). Karena itu hasilnya disimpan ke `data/nela_cache.npz` dengan
kunci hash dari daftar teks: filter data yang berubah menghasilkan hash berbeda,
sehingga cache lama tidak pernah terpakai diam-diam. `data/` ada di `.gitignore`,
jadi cache tidak bisa ikut terkomit.
"""

import hashlib
import pathlib
import tempfile
from collections import Counter

import nltk
import numpy as np
from nela_features.nela_features import NELAFeatureExtractor

CACHE_PATH = pathlib.Path("data/nela_cache.npz")

# Urutan ini sama dengan urutan kolom yang dikembalikan `extract_all`, dan
# kesamaan itu diperiksa `_self_check()`.
GROUPS = ("style", "complexity", "bias", "affect", "moral", "event")

# Jumlah kolom per grup pada nela-features 3.0.1. PRD 6.1 menyebut 141; itu
# angka paper Horne et al. 2018, bukan angka paket ini (spec M3 7.3).
GROUP_SIZES = {
    "style": 50,
    "complexity": 7,
    "bias": 8,
    "affect": 9,
    "moral": 11,
    "event": 2,
}
N_FEATURES = sum(GROUP_SIZES.values())

# Resource nltk yang dipakai NELA: tokenizer kalimat, POS tagger, NER chunker,
# dan dua korpus. Tanpa unduhan ini M3 gagal di mesin bersih (spec M3 9). Lima
# ini sudah cukup; itu diuji pada direktori nltk_data kosong, bukan diperkirakan.
NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("corpora/words", "words"),
    ("corpora/stopwords", "stopwords"),
)

# Kalimat contoh untuk membangun peta kolom ke grup. Isinya dipilih agar tiap
# grup punya sesuatu untuk dihitung, termasuk lokasi dan tanggal untuk `event`.
_PROBE = "The so-called experts met in Paris on Monday to peddle this scheme."


def ensure_nltk_resources() -> None:
    """Unduh resource nltk yang belum ada. Idempoten dan diam bila sudah lengkap.

    `nltk.data.find` diperiksa lebih dulu supaya jalan yang normal tidak
    menyentuh jaringan sama sekali.
    """
    for path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def texts_hash(texts: list[str]) -> str:
    """Hash daftar teks, dengan panjang tiap baris ikut di-hash.

    Panjangnya disertakan supaya dua daftar berbeda yang hasil gabungannya sama
    tidak menghasilkan kunci cache yang sama.
    """
    digest = hashlib.sha256()
    digest.update(len(texts).to_bytes(8, "big"))
    for text in texts:
        blob = text.encode("utf-8")
        digest.update(len(blob).to_bytes(8, "big"))
        digest.update(blob)
    return digest.hexdigest()


def group_map() -> dict[str, str]:
    """Peta nama fitur ke nama grupnya, dibangun dari paket, bukan dihardcode."""
    ensure_nltk_resources()
    extractor = NELAFeatureExtractor()
    mapping: dict[str, str] = {}
    for group in GROUPS:
        _, names = getattr(extractor, f"extract_{group}")(_PROBE)
        for name in names:
            mapping[name] = group
    return mapping


def _read_cache(path: pathlib.Path, key: str) -> tuple[np.ndarray, list[str]] | None:
    """Baca cache bila ada dan kuncinya cocok. Kunci tidak cocok berarti tolak."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["key"]) != key:
            return None
        return data["matrix"], [str(name) for name in data["names"]]


def extract_features(
    texts: list[str], cache_path: pathlib.Path = CACHE_PATH
) -> tuple[np.ndarray, list[str]]:
    """Kembalikan matriks (len(texts), 87) dan nama kolomnya.

    Tiap kalimat diproses sendiri tanpa melihat kalimat lain, jadi ekstraksi
    boleh dilakukan di luar loop fold. Yang tidak boleh keluar dari loop fold
    adalah pembuangan kolom dan standardisasi (spec M3 3.5).
    """
    if not texts:
        raise ValueError("daftar teks kosong")
    ensure_nltk_resources()
    key = texts_hash(texts)
    cached = _read_cache(cache_path, key)
    if cached is not None:
        return cached

    extractor = NELAFeatureExtractor()
    rows: list[list[float]] = []
    names: list[str] = []
    for text in texts:
        vector, names = extractor.extract_all(text)
        rows.append(vector)
        if len(rows) % 500 == 0:
            print(f"  ekstraksi {len(rows)}/{len(texts)}", flush=True)
    matrix = np.asarray(rows, dtype=np.float64)

    # ponytail: cache ditulis sekali di akhir. Crash di menit ke-40 berarti
    # mengulang dari nol; upgradenya menulis per potongan bila itu mengganggu.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, key=key, matrix=matrix, names=np.asarray(names))
    return matrix, names


def _self_check() -> None:
    """Periksa bentuk keluaran, peta grup, dan perilaku cache.

    Cache ditulis ke direktori sementara, bukan ke `data/`, supaya self-check
    tidak pernah menimpa cache 46 menit milik run yang sebenarnya.
    """
    texts = [
        "The so-called experts met in Paris on Monday to peddle this scheme.",
        "The committee met on Tuesday to review the budget.",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "nela_cache.npz"

        matrix, names = extract_features(texts, cache_path=path)
        assert matrix.shape == (2, N_FEATURES), matrix.shape
        assert len(names) == N_FEATURES, len(names)

        mapping = group_map()
        assert set(mapping) == set(names), sorted(set(names) ^ set(mapping))
        assert Counter(mapping.values()) == GROUP_SIZES, Counter(mapping.values())

        again, names_again = extract_features(texts, cache_path=path)
        assert np.array_equal(matrix, again), "cache tidak mengembalikan matriks sama"
        assert names_again == names, "nama kolom berubah setelah dibaca dari cache"

        # Daftar teks berbeda menghasilkan hash berbeda, jadi cache lama ditolak
        # dan ekstraksi diulang.
        shorter, _ = extract_features(texts[:1], cache_path=path)
        assert shorter.shape == (1, N_FEATURES), shorter.shape

    print("self-check ok:", matrix.shape, len(names), "kolom")


if __name__ == "__main__":
    _self_check()
