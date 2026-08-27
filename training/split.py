"""Derivasi kunci grup dan penetapan fold.

Modul ini sengaja hanya memakai stdlib. Alasannya bukan kerapian: `tests/
test_split.py` mengimpornya dan berjalan di job CI serving, yang hanya memasang
`requirements-dev.txt`. Test anti-kebocoran adalah pemeriksaan terpenting di
bagian data (PRD 5.2), jadi ia harus berjalan pada setiap push, bukan hanya di
mesin yang punya venv training.

Menambahkan impor `datasets`, `pandas`, `sklearn`, `torch`, atau `numpy` ke sini
akan membuat test di CI gagal. Itu disengaja.
"""

import hashlib
from urllib.parse import urlsplit, urlunsplit


def group_key_from_url(url: str) -> str:
    """Reduksi URL artikel menjadi kunci grup yang stabil.

    Query string, fragment, dan trailing slash dibuang; skema dan host
    di-lowercase. Kolom `news_link` di BABE mencampur URL artikel kanonik dengan
    redirect feedproxy, dan satu artikel bisa muncul dengan parameter tracking
    yang berbeda. Tanpa normalisasi, satu artikel terpecah menjadi beberapa grup
    dan group split kehilangan gunanya.

    Id artikel pada redirect feedproxy berada di path, sehingga dua artikel
    berbeda tetap menghasilkan kunci berbeda.
    """
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
    )


def fold_of(group_key: str, n_folds: int, seed: int = 0) -> int:
    """Petakan satu kunci grup ke nomor fold secara deterministik.

    Hash dipakai, bukan pengacakan, agar hasilnya sama di setiap mesin dan
    setiap kali dijalankan tanpa perlu menyimpan daftar fold ke disk.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds minimal 2, diterima {n_folds}")
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def group_split(group_keys: list[str], n_folds: int = 5, seed: int = 0) -> list[int]:
    """Beri nomor fold untuk tiap baris, berdasarkan kunci grupnya.

    Baris dengan kunci grup sama selalu berakhir di fold yang sama. Itulah yang
    menjaga kalimat-kalimat dari satu artikel tidak tersebar ke dua fold.
    """
    return [fold_of(key, n_folds, seed) for key in group_keys]
