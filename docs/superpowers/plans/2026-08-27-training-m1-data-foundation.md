# Training M1 — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun fondasi data fase training — loader BABE, group split yang terbukti tidak bocor, dan baseline TF-IDF — sampai angka M1 tersedia, tanpa melatih DeBERTa.

**Architecture:** Direktori `training/` baru dengan venv sendiri (`.venv-train/`), terpisah dari venv serving agar PyTorch tidak pernah terpasang di sisi produksi. Fungsi group split ditulis memakai stdlib saja sehingga test anti-kebocoran ikut berjalan di job CI yang sudah ada, tanpa dependency tambahan. Loader dan baseline hidup di modul terpisah yang memerlukan venv training.

**Tech Stack:** Python 3.13, `datasets` (HuggingFace), `scikit-learn`, `torch` (dipasang dan diverifikasi, belum dipakai), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-training-phase-design.md`

## Global Constraints

- **Dataset tidak pernah masuk repo.** `data/` sudah ada di `.gitignore`. Diunduh saat bekerja, tidak pernah di-commit, tidak pernah disalin ke `src/` atau ke image.
- **`training/split.py` hanya boleh mengimpor stdlib.** Dilarang: `datasets`, `pandas`, `sklearn`, `torch`, `numpy`. Batasan ini diuji, bukan sekadar disepakati.
- **Tidak ada perubahan pada `src/neutral/`.** Fase ini tidak menyentuh lapisan serving.
- **Tidak ada perubahan pada `.github/workflows/ci.yml`** dan tidak ada tambahan pada `requirements-dev.txt`.
- **Kunci group split adalah `news_link`**, bukan `article_id` — kolom itu tidak ada di BABE.
- **Metrik baseline adalah F1 macro**, sama dengan metrik utama PRD 8.1.
- **Jumlah fold: 5.**
- **Assert anti-kebocoran menghentikan proses**, bukan mencetak peringatan.
- **Commit hanya atas nama `ihsan-p1`.** Tanpa trailer `Co-Authored-By`, tanpa footer generator.
- **Berhenti sebelum M2.** Tidak ada fine-tune DeBERTa di plan ini.
- Venv serving tetap `.venv/`, venv training `.venv-train/`. Keduanya memakai interpreter di `Scripts/`, bukan `bin/` — ini mesin Windows.

---

## File Structure

| File | Tanggung jawab |
|---|---|
| `training/__init__.py` | Menjadikan `training` paket yang dapat diimpor oleh pytest |
| `training/requirements-train.txt` | Pin dependency training, terpisah dari `requirements.txt` |
| `training/split.py` | Derivasi kunci grup dari URL dan penetapan fold. Stdlib saja |
| `training/data.py` | Load BABE, buang baris tidak layak, laporkan jumlah yang dibuang |
| `training/baseline_tfidf.py` | TF-IDF + logistic regression per fold, F1 macro |
| `training/run_m1.py` | Entry point M1: EDA, split, assert kebocoran, baseline |
| `tests/test_split.py` | Test anti-kebocoran dan derivasi kunci. Berjalan di CI |
| `pyproject.toml` | Menambah `.` ke `pythonpath` agar `training` dapat diimpor pytest |
| `.gitignore` | Menambah `.venv-train/` |

Pemisahan `split.py` dari `data.py` adalah deviasi sadar terhadap spec serving
bagian 3.2, yang menempatkan keduanya di satu file. Alasannya ada di spec
training bagian 3.2: fungsi split harus dapat diimpor tanpa menarik `datasets`,
dan itu tidak mungkin bila ia tinggal di modul yang sama dengan loader.

Derivasi kunci grup dari URL juga ditempatkan di `split.py`, bukan di `data.py`
seperti bunyi spec training bagian 4. Alasannya sama dan sejalan dengan
keputusan 3.2: normalisasi URL murni stdlib, dan menempatkannya di `split.py`
membuat ia ikut teruji di CI.

---

## Task 1: Venv training dan verifikasi GPU

**Files:**
- Create: `training/requirements-train.txt`, `training/__init__.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: tidak ada
- Produces: `.venv-train/Scripts/python.exe` dengan `torch`, `datasets`, `scikit-learn`, `transformers` terpasang; paket `training` dapat diimpor

- [ ] **Step 1: Tambah `.venv-train/` ke `.gitignore`**

Sisipkan setelah baris `.venv/`:

```gitignore
.venv-train/
```

- [ ] **Step 2: Buat paket `training`**

```bash
mkdir -p training
```

Tulis `training/__init__.py` berisi satu docstring:

```python
"""Sisi training. Tidak pernah diimpor oleh `src/neutral/` dan tidak pernah masuk image produksi."""
```

- [ ] **Step 3: Buat venv training**

```bash
uv venv --python 3.13 .venv-train
```

Expected: `Using CPython 3.13.x` dan `Creating virtual environment at: .venv-train`.

- [ ] **Step 4: Pasang torch dari index CUDA**

```bash
uv pip install --python .venv-train/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
```

Bila resolusi gagal karena wheel cu126 tidak tersedia untuk Python 3.13, **jangan
berimprovisasi**. Coba `cu124`, lalu laporkan mana yang berhasil. Bila keduanya
gagal, hentikan dan lapor — konsekuensinya rencana M2 berubah, dan itu justru
alasan langkah ini ada di M1 (spec 3.4).

- [ ] **Step 5: Pasang sisa dependency training**

```bash
uv pip install --python .venv-train/Scripts/python.exe "datasets>=3.0" "scikit-learn>=1.5" transformers
```

- [ ] **Step 6: Smoke check GPU**

```bash
.venv-train/Scripts/python.exe -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); x = torch.zeros(256, 256, device='cuda'); print('alloc ok', x.sum().item())"
```

Expected: `cuda available True`, `device NVIDIA GeForce RTX 3050 6GB Laptop GPU`, dan `alloc ok 0.0`.

Bila `cuda available` bernilai `False`, hentikan dan lapor. Jangan melanjutkan ke
Task 2 — seluruh premis fase training bergantung pada ini, dan menemukannya
sekarang adalah tujuan langkah ini.

- [ ] **Step 7: Bekukan versi hasil resolusi ke `training/requirements-train.txt`**

```bash
.venv-train/Scripts/python.exe -m pip freeze > training/requirements-train.txt
```

Lalu sisipkan tiga baris komentar di paling atas file itu:

```
# Dependency sisi training. Tidak pernah masuk image produksi:
# Dockerfile hanya menyalin src/, dan .dockerignore mengecualikan training/.
# Dipasang ke .venv-train/, bukan ke .venv/ serving.
```

Versi di sini adalah hasil resolusi nyata, bukan angka yang ditebak. Itu sebabnya
file ini dibekukan setelah instalasi, bukan ditulis sebelumnya.

- [ ] **Step 8: Verifikasi venv serving tetap bersih dari torch**

```bash
.venv/Scripts/python.exe -c "import torch" ; echo "exit=$?"
```

Expected: `ModuleNotFoundError: No module named 'torch'` dan `exit=1`.

Inilah bentuk konkret keputusan spec 3.3: invariant dijaga oleh keadaan mesin, bukan hanya oleh Dockerfile.

- [ ] **Step 9: Commit**

```bash
git add .gitignore training/__init__.py training/requirements-train.txt
git commit -m "chore: add training venv dependencies and GPU verification"
```

---

## Task 2: Group split dan test anti-kebocoran

**Files:**
- Create: `training/split.py`
- Test: `tests/test_split.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: tidak ada
- Produces:
  - `group_key_from_url(url: str) -> str`
  - `fold_of(group_key: str, n_folds: int, seed: int = 0) -> int`
  - `group_split(group_keys: list[str], n_folds: int = 5, seed: int = 0) -> list[int]`

`group_split` mengembalikan satu nomor fold per baris masukan, dengan urutan yang sama.

- [ ] **Step 1: Ubah `pyproject.toml` agar pytest dapat mengimpor `training`**

Ganti baris `pythonpath = ["src"]` menjadi:

```toml
pythonpath = ["src", "."]
```

`testpaths = ["tests"]` tidak diubah, sehingga pytest tetap hanya mengumpulkan test dari `tests/`.

- [ ] **Step 2: Tulis test yang gagal**

```python
# tests/test_split.py
import ast
import pathlib

import pytest

from training.split import fold_of, group_key_from_url, group_split


def test_same_group_always_lands_in_one_fold():
    keys = ["a/1", "a/1", "a/1", "b/2", "b/2"]
    folds = group_split(keys, n_folds=5)
    assert folds[0] == folds[1] == folds[2]
    assert folds[3] == folds[4]


def test_no_group_appears_in_two_folds():
    # PRD 5.2 menyebut ini pemeriksaan terpenting di seluruh bagian data:
    # kalimat dari satu artikel tidak boleh berada di train dan test sekaligus.
    keys = [f"article/{i // 7}" for i in range(700)]
    folds = group_split(keys, n_folds=5)
    seen: dict[str, int] = {}
    for key, fold in zip(keys, folds, strict=True):
        if key in seen:
            assert seen[key] == fold, f"grup {key} ada di fold {seen[key]} dan {fold}"
        else:
            seen[key] = fold


def test_split_is_deterministic():
    keys = [f"article/{i}" for i in range(50)]
    assert group_split(keys, n_folds=5) == group_split(keys, n_folds=5)


def test_seed_changes_assignment():
    keys = [f"article/{i}" for i in range(200)]
    assert group_split(keys, n_folds=5, seed=0) != group_split(keys, n_folds=5, seed=1)


def test_every_fold_is_used_when_groups_are_plentiful():
    keys = [f"article/{i}" for i in range(500)]
    assert set(group_split(keys, n_folds=5)) == {0, 1, 2, 3, 4}


def test_folds_are_roughly_balanced():
    # Hash tidak menjamin fold sama besar. Yang perlu dijamin hanya bahwa tidak
    # ada fold yang nyaris kosong, karena F1 per fold akan tidak stabil.
    keys = [f"article/{i}" for i in range(1000)]
    folds = group_split(keys, n_folds=5)
    sizes = [folds.count(f) for f in range(5)]
    assert min(sizes) > 1000 / 5 * 0.7, f"fold terlalu timpang: {sizes}"


def test_rejects_fewer_than_two_folds():
    with pytest.raises(ValueError):
        fold_of("a", n_folds=1)


def test_group_key_drops_query_and_fragment():
    a = group_key_from_url("https://example.com/news/story?utm_source=x#top")
    b = group_key_from_url("https://example.com/news/story")
    assert a == b


def test_group_key_drops_trailing_slash_and_lowercases_host():
    a = group_key_from_url("https://Example.COM/news/story/")
    b = group_key_from_url("https://example.com/news/story")
    assert a == b


def test_group_key_keeps_distinct_articles_distinct():
    # Redirect feedproxy membawa id artikel di path, jadi keduanya tidak boleh
    # menyatu menjadi satu grup. Dua URL ini diambil dari BABE.
    a = group_key_from_url("http://feedproxy.google.com/~r/breitbart/~3/F5vueG8YCQQ/")
    b = group_key_from_url("http://feedproxy.google.com/~r/breitbart/~3/ksXbL6hU-Ko/")
    assert a != b


def test_split_module_imports_only_stdlib():
    # split.py harus tetap dapat diimpor di job CI serving, yang hanya memasang
    # requirements-dev.txt. Bila ia menumbuhkan impor datasets/pandas/sklearn/
    # torch/numpy, test ini gagal di sini, bukan gagal diam-diam di CI.
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "training" / "split.py"
    ).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"hashlib", "urllib"}, f"impor tidak diizinkan: {imported}"
```

- [ ] **Step 3: Jalankan test, pastikan gagal**

Run: `.venv/Scripts/python.exe -m pytest tests/test_split.py -v`
Expected: FAIL dengan `ModuleNotFoundError: No module named 'training.split'`

Perhatikan: test ini dijalankan dengan venv **serving**, bukan venv training. Itu memang inti keputusan spec 3.2.

- [ ] **Step 4: Tulis `training/split.py`**

```python
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
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def group_split(group_keys: list[str], n_folds: int = 5, seed: int = 0) -> list[int]:
    """Beri nomor fold untuk tiap baris, berdasarkan kunci grupnya.

    Baris dengan kunci grup sama selalu berakhir di fold yang sama. Itulah yang
    menjaga kalimat-kalimat dari satu artikel tidak tersebar ke dua fold.
    """
    return [fold_of(key, n_folds, seed) for key in group_keys]
```

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `.venv/Scripts/python.exe -m pytest tests/test_split.py -v`
Expected: 11 passed

- [ ] **Step 6: Jalankan seluruh test dan lint, pastikan tidak ada yang rusak**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests training
```

Expected: 60 passed (49 serving + 11 baru), ruff bersih.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml training/split.py tests/test_split.py
git commit -m "feat: add group split by article URL with leakage test in CI"
```

---

## Task 3: Loader BABE

**Files:**
- Create: `training/data.py`

**Interfaces:**
- Consumes: `training.split.group_key_from_url`
- Produces:
  - `Row` — dataclass frozen, field: `text: str`, `label: int`, `outlet: str`, `lean: str`, `group_key: str`, `official_split: str`
  - `LoadReport` — dataclass frozen, field: `n_raw: int`, `n_missing_link: int`, `n_low_agreement: int`, `n_unexpected_label: int`, `n_kept: int`, `label_values: list[str]`, `lean_values: list[str]`, `opinion_values: list[str]`, `n_groups_raw: int`, `n_groups_normalised: int`
  - `load_babe(cache_dir: str = "data/babe") -> tuple[list[Row], LoadReport]`

`LoadReport` memuat semua angka yang harus dilaporkan M1, sehingga `run_m1.py` tidak perlu menghitung ulang apa pun.

- [ ] **Step 1: Tulis `training/data.py`**

```python
"""Muat BABE dan susutkan menjadi baris serta kolom yang dibutuhkan M1.

Modul ini mengimpor `datasets`, jadi ia memerlukan venv training dan tidak boleh
diimpor dari `tests/`. Lihat catatan di kepala `training/split.py`.

Dataset diunduh ke `data/`, yang ada di `.gitignore`. BABE tidak menyatakan
lisensi apa pun (PRD 5.0), sehingga pengamanannya adalah tidak pernah
meredistribusikannya: tidak masuk repo, tidak masuk image.
"""

from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

from training.split import group_key_from_url

DATASET = "mediabiasgroup/BABE"

# Nilai `label_opinion` yang menandai annotator tidak sepakat. PRD 5.1: label
# noisy lebih merusak daripada data yang lebih sedikit.
LOW_AGREEMENT = "No agreement"


@dataclass(frozen=True)
class Row:
    text: str
    label: int
    outlet: str
    lean: str
    group_key: str
    official_split: str


@dataclass(frozen=True)
class LoadReport:
    """Angka yang harus dilaporkan M1, termasuk setiap baris yang dibuang."""

    n_raw: int
    n_missing_link: int
    n_low_agreement: int
    n_unexpected_label: int
    n_kept: int
    label_values: list[str]
    lean_values: list[str]
    opinion_values: list[str]
    n_groups_raw: int
    n_groups_normalised: int


def load_babe(cache_dir: str = "data/babe") -> tuple[list[Row], LoadReport]:
    """Muat BABE, buang baris yang tidak layak, dan laporkan jumlahnya.

    Split resmi BABE digabung menjadi satu pool lebih dulu, bukan dipakai apa
    adanya: split itu belum diketahui bersih terhadap kunci grup, dan itu
    diperiksa terpisah di `run_m1.py`. Asal tiap baris tetap disimpan di
    `official_split` supaya pemeriksaan itu mungkin dilakukan.
    """
    dataset = load_dataset(DATASET, cache_dir=cache_dir)
    names = sorted(dataset.keys())
    pool = concatenate_datasets([dataset[name] for name in names])
    origin = [name for name in names for _ in range(len(dataset[name]))]

    # Kolom diambil sekali menjadi list. Mengakses `pool["label"]` di dalam loop
    # akan menyalin seluruh kolom pada setiap iterasi.
    texts = pool["text"]
    labels = pool["label"]
    outlets = pool["outlet"]
    leans = pool["type"]
    opinions = pool["label_opinion"]
    links = [str(link or "").strip() for link in pool["news_link"]]

    present = [link for link in links if link]
    n_groups_raw = len(set(present))
    n_groups_normalised = len({group_key_from_url(link) for link in present})

    rows: list[Row] = []
    n_missing_link = 0
    n_low_agreement = 0
    n_unexpected_label = 0

    for i in range(len(pool)):
        if not links[i]:
            n_missing_link += 1
            continue
        if str(opinions[i]) == LOW_AGREEMENT:
            n_low_agreement += 1
            continue
        if labels[i] not in (0, 1):
            # Spec 6.2: rilis BABE lain pernah memuat kategori selain 0/1 pada
            # label utama. Baris seperti itu dibuang sebagai agreement rendah.
            n_unexpected_label += 1
            continue
        rows.append(
            Row(
                text=str(texts[i]),
                label=int(labels[i]),
                outlet=str(outlets[i]),
                lean=str(leans[i]),
                group_key=group_key_from_url(links[i]),
                official_split=origin[i],
            )
        )

    report = LoadReport(
        n_raw=len(pool),
        n_missing_link=n_missing_link,
        n_low_agreement=n_low_agreement,
        n_unexpected_label=n_unexpected_label,
        n_kept=len(rows),
        label_values=sorted({str(value) for value in labels}),
        lean_values=sorted({str(value) for value in leans}),
        opinion_values=sorted({str(value) for value in opinions}),
        n_groups_raw=n_groups_raw,
        n_groups_normalised=n_groups_normalised,
    )
    return rows, report
```

- [ ] **Step 2: Jalankan loader dan periksa hasilnya**

```bash
.venv-train/Scripts/python.exe -c "from training.data import load_babe; rows, rep = load_babe(); print(rep); print(rows[0])"
```

Expected: unduhan BABE ke `data/babe`, lalu `LoadReport` tercetak dengan `n_raw=4121` dan `n_kept` lebih kecil atau sama.

Catat empat angka dari keluaran ini — keempatnya adalah jawaban atas pertanyaan
di spec bagian 6, dan dipakai di Task 5: `label_values`, `n_missing_link`,
`n_groups_raw` dibanding `n_groups_normalised`, dan `n_kept`.

Bila `n_kept` turun di bawah sekitar 3.000, hentikan dan lapor sebelum
melanjutkan: spec bagian 11 menandai penyusutan dataset sebagai risiko, dan
ambang tempat WNC menjadi cadangan (PRD 5.0) perlu dibicarakan, bukan diputuskan
sendiri.

- [ ] **Step 3: Pastikan dataset tidak masuk git**

```bash
git status --short
```

Expected: `data/` tidak muncul sama sekali dalam keluaran.

- [ ] **Step 4: Lint dan commit**

```bash
.venv/Scripts/python.exe -m ruff check training
git add training/data.py
git commit -m "feat: add BABE loader with drop counts and group key derivation"
```

---

## Task 4: Baseline TF-IDF

**Files:**
- Create: `training/baseline_tfidf.py`

**Interfaces:**
- Consumes: tidak ada
- Produces: `run_baseline(texts: list[str], labels: list[int], folds: list[int], n_folds: int = 5) -> list[float]` — satu F1 macro per fold, urut menurut nomor fold yang di-hold-out

- [ ] **Step 1: Tulis `training/baseline_tfidf.py` beserta self-check-nya**

```python
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
            raise ValueError(f"fold {held_out} kosong: train={len(train_idx)} test={len(test_idx)}")
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
```

- [ ] **Step 2: Jalankan self-check**

Run: `.venv-train/Scripts/python.exe -m training.baseline_tfidf`
Expected: `self-check ok: [1.0, 1.0, 1.0, 1.0, 1.0]`

Self-check ini tidak diletakkan di `tests/` karena ia memerlukan `sklearn`, dan
menambahkan `sklearn` ke `requirements-dev.txt` akan membawa dependency training
ke sisi serving.

- [ ] **Step 3: Lint dan commit**

```bash
.venv/Scripts/python.exe -m ruff check training
git add training/baseline_tfidf.py
git commit -m "feat: add TF-IDF baseline evaluated per fold"
```

---

## Task 5: Entry point M1

**Files:**
- Create: `training/run_m1.py`

**Interfaces:**
- Consumes: `training.data.load_babe`, `training.split.group_split`, `training.baseline_tfidf.run_baseline`
- Produces: `check_official_split(rows: list[Row]) -> tuple[bool, int]`, `main() -> None`

- [ ] **Step 1: Tulis `training/run_m1.py`**

```python
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
    print(f"nilai label_opinion         : {report.opinion_values}")

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
        print(f"label {label} ({name:7}) : {labels[label]} ({labels[label] / total:.1%})")

    outlets = Counter(row.outlet for row in rows)
    print(f"outlet distinct             : {len(outlets)}")
    print(f"5 outlet terbesar           : {outlets.most_common(5)}")
    if len(outlets) < MIN_OUTLETS_FOR_CORRELATION:
        print(
            f"CATATAN: {len(outlets)} outlet, di bawah {MIN_OUTLETS_FOR_CORRELATION} "
            "yang disyaratkan PRD 8.3 — korelasi outlet tidak dapat dijalankan pada BABE saja"
        )
    print(f"lean distinct               : {Counter(row.lean for row in rows)}")

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
    print(
        f"F1 macro                    : {statistics.mean(scores):.4f} "
        f"± {statistics.stdev(scores):.4f}"
    )
    print()
    print("Angka di atas adalah pembanding untuk kriteria berhenti M2 (PRD 9).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Jalankan M1 secara utuh**

Run: `.venv-train/Scripts/python.exe -m training.run_m1`

Expected: seluruh bagian tercetak, assert anti-kebocoran lulus, dan F1 macro
baseline muncul beserta deviasinya.

Bila assert anti-kebocoran gagal, jangan menambal `run_m1.py`. Kegagalan itu
berarti `group_split` atau `group_key_from_url` salah, dan perbaikannya ada di
Task 2 beserta test-nya.

- [ ] **Step 3: Catat jawaban atas ketiga pertanyaan spec bagian 6**

Dari keluaran Step 2, tuliskan di pesan commit atau catatan sesi:

1. Split resmi BABE bersih atau bocor
2. `label_values` — hanya `0`/`1` atau ada nilai lain
3. Jumlah outlet distinct dan jumlah baris tanpa `news_link`

Ketiganya menentukan isi rencana M2, jadi jangan hanya dibaca lalu dilewati.

- [ ] **Step 4: Pastikan seluruh test serving masih lulus**

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests training
```

Expected: 60 passed, ruff bersih.

- [ ] **Step 5: Commit**

```bash
git add training/run_m1.py
git commit -m "feat: add M1 entry point with leakage assertion and EDA report"
```

---

## Task 6: Koreksi dokumen

**Files:**
- Modify: `PRD_Media_Neutrality_Classifier.md` (bagian 5.2 dan 12.1)
- Modify: `docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md` (bagian 3.2)
- Modify: `docs/superpowers/plans/2026-08-27-serving-foundation.md` (bagian tertunda)
- Modify: `CLAUDE.md` (bagian Current state dan Architecture)

**Interfaces:**
- Consumes: hasil Task 5
- Produces: dokumen yang cocok dengan kode yang benar-benar ada

- [ ] **Step 1: Koreksi PRD 5.2**

Ganti seluruh daftar ketentuan di bagian 5.2 menjadi:

```markdown
Ketentuan:

- Split dilakukan **per artikel sumber**. BABE tidak memiliki kolom
  `article_id`; penanda artikel yang tersedia adalah `news_link`, dan kunci grup
  diturunkan darinya
- `news_link` dinormalisasi lebih dulu: query string, fragment, dan trailing
  slash dibuang, skema dan host di-lowercase. Tanpa itu satu artikel yang muncul
  dengan parameter tracking berbeda terpecah menjadi beberapa grup
- Baris tanpa `news_link` dibuang, bukan digabung menjadi satu grup — menggabungkan
  justru menciptakan grup palsu. Jumlahnya dilaporkan
- Gunakan group-aware CV, bukan `train_test_split` biasa
- Sertakan satu test yang secara eksplisit memverifikasi tidak ada kunci grup
  yang muncul di dua fold berbeda
- Split resmi train/test BABE tidak dipakai apa adanya sebelum diverifikasi
  bersih terhadap kunci grup
```

- [ ] **Step 2: Koreksi PRD 12.1**

Ganti dua baris verifikasi tersisa di bagian 12.1 menjadi:

```markdown
*Verifikasi lisensi sudah dilakukan pada 2026-08-27:* dataset card BABE di
HuggingFace tetap tidak menyatakan lisensi apa pun. Pengamanannya karena itu
tidak berubah — tidak ada redistribusi, sesuai bagian 5.0.
```

- [ ] **Step 3: Koreksi spec serving bagian 3.2**

Di blok struktur repositori, ganti baris `data.py` pada bagian `training/` menjadi dua baris:

```
│   ├── split.py       # derivasi kunci grup dan fold; stdlib saja
│   ├── data.py        # loading, filter kualitas
```

Lalu tambahkan satu paragraf setelah blok itu:

```markdown
`split.py` dipisahkan dari `data.py` setelah implementasi: fungsi split harus
dapat diimpor tanpa menarik `datasets`, agar test anti-kebocoran ikut berjalan di
CI. Alasan lengkapnya ada di
`docs/superpowers/specs/2026-08-27-training-phase-design.md` bagian 3.2.
```

- [ ] **Step 4: Koreksi bagian tertunda di plan serving**

Ganti baris `training/data.py` menjadi:

```markdown
- `training/split.py` dan `training/data.py` — loading BABE, group split per
  artikel sumber lewat `news_link`, test anti-kebocoran. Sudah dikerjakan; lihat
  `docs/superpowers/plans/2026-08-27-training-m1-data-foundation.md`
```

- [ ] **Step 5: Perbarui CLAUDE.md**

Pada bagian "Current state", ganti paragraf yang menyatakan `training/` belum ada
dengan keadaan sebenarnya setelah Task 5: `training/` berisi `split.py`,
`data.py`, `baseline_tfidf.py`, `run_m1.py`; M1 selesai; M2 belum punya rencana.

Pada bagian "Architecture", tambahkan satu baris di bawah paragraf pemisahan
training dan serving: venv training terpisah di `.venv-train/`, dan `import torch`
di venv serving gagal secara lokal.

Tambahkan satu baris ke daftar "Invariants":

```markdown
- **`training/split.py` hanya mengimpor stdlib.** Ia diimpor oleh `tests/`, yang
  berjalan di CI dengan `requirements-dev.txt` saja. Menambah impor `datasets`,
  `pandas`, `sklearn`, `torch`, atau `numpy` membuat test di CI gagal.
```

- [ ] **Step 6: Commit**

```bash
git add PRD_Media_Neutrality_Classifier.md docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md docs/superpowers/plans/2026-08-27-serving-foundation.md CLAUDE.md
git commit -m "docs: correct group split key to news_link and record M1 state"
```

- [ ] **Step 7: Verifikasi atribusi commit sebelum berhenti**

```bash
git log --format='%an|%cn' origin/main..HEAD | sort -u
git log --grep='Co-Authored-By' -i origin/main..HEAD
```

Expected: baris pertama hanya `Ihsan-p1|Ihsan-p1`, perintah kedua tidak mengeluarkan apa pun.

---

## Berhenti di sini

M2 dan seterusnya tidak ada di plan ini. Setelah Task 6, yang tersedia adalah
angka baseline dan jawaban atas ketiga pertanyaan spec bagian 6 — dan itulah
masukan yang dibutuhkan untuk menulis rencana M2.

Jangan melanjutkan ke fine-tune DeBERTa tanpa rencana tersendiri. Kriteria
berhenti M2 di PRD 9 mensyaratkan perbandingan terhadap baseline, dan
perbandingan itu baru mungkin setelah angka di Task 5 diketahui.

---

## Self-Review

**Spec coverage.** Spec bagian 2 temuan skema → Task 3, dilaporkan lewat
`LoadReport` dan dicetak di Task 5. Bagian 3.1 kunci `news_link` →
`group_key_from_url` di Task 2, koreksi PRD di Task 6. Bagian 3.2 split
stdlib-only → Task 2, dijaga `test_split_module_imports_only_stdlib`. Bagian 3.3
venv kedua → Task 1, diverifikasi Step 8. Bagian 3.4 torch dipasang dan diuji →
Task 1 Step 4 dan 6. Bagian 4 struktur `training/` → Task 1 sampai 5. Bagian 5
alur data → Task 3 dan Task 5. Bagian 6.1 split resmi bersih atau bocor →
`check_official_split` di Task 5. Bagian 6.2 nilai `label` → `label_values` di
Task 3, dicetak Task 5. Bagian 6.3 jumlah outlet dan baris tanpa link → Task 5,
dengan ambang 12 outlet diperiksa eksplisit. Bagian 7 baseline → Task 4. Bagian 8
pengujian → Task 2 untuk CI, Task 4 untuk self-check, Task 5 untuk assert. Bagian
9 batas berhenti → bagian "Berhenti di sini". Bagian 10 koreksi dokumen → Task 6.

**Placeholder scan.** Tidak ada TBD, TODO, atau "serupa dengan Task N". Setiap
langkah kode memuat kode utuh. Satu-satunya angka yang tidak ditulis di muka
adalah versi dependency di `training/requirements-train.txt`, dan itu disengaja:
Task 1 Step 7 membekukannya dari hasil resolusi nyata, karena menuliskan versi
yang ditebak akan menghasilkan pin yang salah.

**Type consistency.** `group_key_from_url(str) -> str`, `fold_of(str, int, int)
-> int`, dan `group_split(list[str], int, int) -> list[int]` dipakai identik di
Task 2, 3, dan 5. `Row` memiliki field `text`, `label`, `outlet`, `lean`,
`group_key`, `official_split` di Task 3 dan dibaca dengan nama yang sama di Task
5. `LoadReport` memiliki sepuluh field di Task 3, dan sepuluh field itu juga yang
dicetak di Task 5. `run_baseline(list[str], list[int], list[int], int) ->
list[float]` sama di Task 4 dan Task 5. `N_FOLDS = 5` konsisten dengan default
`n_folds=5` di kedua modul.
