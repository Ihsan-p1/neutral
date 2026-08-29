# M3 NELA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how far explicit linguistic features get on the M1 folds, so the +0.1058 gap between TF-IDF and DeBERTa can be attributed to something rather than left unexplained.

**Architecture:** `training/nela.py` extracts the 87 NELA features once and caches the matrix to `data/nela_cache.npz`, keyed on a hash of the text list. `training/run_m3.py` loops the folds and fits nine logistic regressions per fold group: three model configurations (TF-IDF recomputed, NELA, TF-IDF+NELA) and six per-group ablations. The DeBERTa column is read back out of `runs/m2_runs.jsonl`, not copied from the README. Everything prints to stdout; there is no run log file.

**Tech Stack:** `nela-features==3.0.1`, `nltk==3.10.3`, `numpy==2.5.2`, `scipy==1.18.1`, `scikit-learn==1.9.0`. The last three are already pinned; the first two and their transitive dependencies get pinned in Task 1.

**Spec:** `docs/superpowers/specs/2026-08-29-training-m3-nela-design.md`

## Global Constraints

- **Two interpreters.** Anything importing `nela_features`, `nltk`, `datasets`, or `sklearn` runs under `.venv-train/Scripts/python.exe`. Tests and lint run under `.venv/Scripts/python.exe`. This is a Windows machine: `Scripts/`, not `bin/`.
- **`tests/` must never import `nela_features`.** The serving CI job installs `requirements-dev.txt` only. Checks that need training dependencies live as `__main__` self-checks inside `training/`, the way `baseline_tfidf.py` and `train.py` already do.
- **`training/split.py` imports only stdlib** and is not touched by this plan. `tests/test_split.py` asserts its import set.
- **Never commit data or caches.** `data/`, `models/`, and `runs/` are already in `.gitignore`. `data/nela_cache.npz` lands inside `data/`, so it is covered.
- **Fold assignment is `group_split(keys, n_folds=5, seed=0)`, identical to M1 and M2.** Changing any argument invalidates every comparison in this milestone.
- **Fixed logistic regression settings, spec 3.1:** `class_weight="balanced"`, `max_iter=1000`, no sweep, for all three model configurations and all six group ablations. TF-IDF settings stay `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`.
- **`VarianceThreshold`, `StandardScaler`, and `TfidfVectorizer` are fit inside the fold, on train indices only.** Fitting any of them on the full matrix leaks test information and raises nothing.
- **Metric is F1 macro**, matching PRD 8.1, so the four numbers compare without conversion.
- **No stop criterion to pass, spec 6.** The delta is reported as measured. What can fail is the reporting, not the number.
- **Comments and docstrings in Indonesian**, matching `split.py`, `data.py`, `baseline_tfidf.py`, and `run_m2.py`. Commit messages in English, conventional prefixes.
- **Only `Ihsan-p1` as author and committer.** No `Co-Authored-By`, no generator footer.
- **No sentence text in printed output.** Counts, shapes, and distributions only.
- **One commit per task.** Do not batch tasks.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `training/requirements-train.txt` | pin `nela-features` and its five new transitive dependencies | 1 |
| `training/nela.py` | create: nltk resource download, feature extraction, group map, disk cache, self-check | 1 |
| `training/run_m3.py` | create: data load, leakage assert, cache warm, TF-IDF recompute guard | 2 |
| `training/run_m3.py` | add: NELA and combined scoring, DeBERTa reader, four-model table | 3 |
| `training/run_m3.py` | add: six-group breakdown table, reading note | 4 |
| `docs/superpowers/specs/2026-08-29-training-m3-nela-design.md` | add measured-results section | 5 |
| `PRD_Media_Neutrality_Classifier.md` | correct 141 features to 87 in section 6.1 | 5 |
| `README.md` | M3 numbers section | 5 |
| `CLAUDE.md` | M3 results carried forward | 5 |

`split.py`, `data.py`, `baseline_tfidf.py`, `train.py`, `run_m1.py`, and `run_m2.py` are not modified. `run_m3.py` consumes `load_babe()`, `group_split()`, and `run_baseline()` exactly as they are. Nothing under `src/neutral/`, `requirements.txt`, `Dockerfile`, or `.dockerignore` changes.

**One deviation from spec section 4.** The spec has `extract_features()` return the matrix, the feature names, and the column-to-group map together. Here the map is a separate `group_map()`. Building it costs one probe extraction through the `event` group, about 0.8 seconds, and folding that into `extract_features()` would charge it on every cache hit — including the six group ablations, which call nothing else. The two callers that need the map ask for it once.

---

### Task 1: Feature extraction and cache in `training/nela.py`

**Files:**
- Modify: `training/requirements-train.txt`
- Create: `training/nela.py`
- Test: `training/nela.py` `_self_check()` under `__main__` — there is no file in `tests/`, because `tests/` runs in the serving venv, which has no `nela-features` and must not get one.

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces, all used by Tasks 2, 3, and 4:
  - `GROUPS: tuple[str, ...]` — `("style", "complexity", "bias", "affect", "moral", "event")`, in the same order `extract_all` returns its columns.
  - `ensure_nltk_resources() -> None`
  - `texts_hash(texts: list[str]) -> str`
  - `group_map() -> dict[str, str]` — feature name to group name, 87 entries.
  - `extract_features(texts: list[str], cache_path: pathlib.Path = CACHE_PATH) -> tuple[np.ndarray, list[str]]` — matrix of shape `(len(texts), 87)` and the 87 column names.

- [ ] **Step 1: Pin the new dependencies**

`nela-features` and its transitive dependencies were installed into `.venv-train` during the design measurements but never written down. Six lines are missing. Insert each one into `training/requirements-train.txt` in the existing alphabetical order, below the `--extra-index-url` line and the comment block:

```
datefinder==1.0.0
defusedxml==0.7.1
nela-features==3.0.1
nltk==3.10.3
pytz==2026.3.post1
vadersentiment==3.3.2
```

`datefinder`, `pytz`, and `defusedxml` arrive through `nela-features`; `vadersentiment` backs the affect group; `nltk` backs the style, complexity, and event groups.

- [ ] **Step 2: Verify the pin file matches the installed environment**

Run:

```bash
cd "d:/Project/College Project/neutral"
uv pip freeze --python .venv-train/Scripts/python.exe | sort > /tmp/freeze.txt
grep -v '^--extra-index-url' training/requirements-train.txt | grep -v '^#' | grep -v '^$' | sort > /tmp/pinned.txt
comm -3 /tmp/freeze.txt /tmp/pinned.txt
```

Expected: no output. Any line printed is a package installed but unpinned, or pinned but not installed.

- [ ] **Step 3: Write `training/nela.py`**

Create the file with exactly this content:

```python
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
```

- [ ] **Step 4: Run the self-check**

Run: `.venv-train/Scripts/python.exe -m training.nela`
Expected: `self-check ok: (2, 87) 87 kolom`, and nothing else. It takes about five seconds, most of it the `event` group's NER chunker.

If it fails on `LookupError: Resource ... not found`, `ensure_nltk_resources` has the wrong resource path for that name. The path is the argument to `nltk.data.find`, the name is the argument to `nltk.download`; they differ.

- [ ] **Step 5: Run the existing tests and lint**

Run:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests training
```

Expected: `64 passed`, and `All checks passed!`. M3 touches nothing the 64 tests cover, so they are here as a check that scope has not widened.

- [ ] **Step 6: Verify nothing from `data/` is staged**

Run: `git status --short`
Expected: only `training/nela.py` and `training/requirements-train.txt`. If `data/nela_cache.npz` appears, stop and fix `.gitignore` before committing.

- [ ] **Step 7: Commit**

```bash
git add training/nela.py training/requirements-train.txt
git commit -m "feat: add M3 NELA feature extraction with disk cache"
```

---

### Task 2: `run_m3.py` entry point, leakage assert, and baseline guard

**Files:**
- Create: `training/run_m3.py`
- Test: run the module; there is no file in `tests/`.

**Interfaces:**
- Consumes: `extract_features()`, `group_map()`, `GROUPS`, `GROUP_SIZES` from Task 1. Also `load_babe()` from `training/data.py`, `group_split()` from `training/split.py`, `run_baseline()` from `training/baseline_tfidf.py`, all unchanged.
- Produces: the module skeleton and `main()`, which Tasks 3 and 4 extend. Constants `N_FOLDS = 5`, `M1_BASELINE_MEAN = 0.7420`, `BASELINE_TOLERANCE = 0.001`.

- [ ] **Step 1: Write `training/run_m3.py`**

Create the file with exactly this content. Tasks 3 and 4 append to it; nothing here gets rewritten later.

```python
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
    print("cocok dengan angka M1 — pembagian fold tidak bergeser")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it, first pass, and warm the cache**

Run: `.venv-train/Scripts/python.exe -m training.run_m3`

This is the expensive step in the whole plan. Expect about 46 minutes, with an `ekstraksi N/3874` line every 500 rows.

Expected at the end:

```
ukuran fold                 : [751, 883, 743, 770, 727]
matriks                     : 3874 x 87
fitur per grup              : {'style': 50, 'complexity': 7, 'bias': 8, 'affect': 9, 'moral': 11, 'event': 2}
F1 macro                    : 0.7420 +/- 0.0194
cocok dengan angka M1 — pembagian fold tidak bergeser
```

If the run exits with `baseline bergeser`, do not continue to Task 3. Something upstream of this plan changed the folds or the row filter, and every comparison M3 makes would be against a different dataset than M1 and M2 measured.

- [ ] **Step 3: Run it again and confirm the cache is used**

Run: `.venv-train/Scripts/python.exe -m training.run_m3`
Expected: same output, no `ekstraksi N/3874` lines, and the whole thing finishes in under a minute. If the extraction lines reappear, the cache key is unstable — check that `load_babe()` returns rows in the same order twice.

- [ ] **Step 4: Confirm the cache file exists and is ignored**

Run:

```bash
ls -la data/nela_cache.npz
git status --short
```

Expected: the file exists, and `git status` lists only `training/run_m3.py`.

- [ ] **Step 5: Run the existing tests and lint**

Run:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests training
```

Expected: `64 passed`, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add training/run_m3.py
git commit -m "feat: add M3 entry point with fold leakage and baseline guards"
```

---

### Task 3: NELA, combined, and DeBERTa — the four-model table

**Files:**
- Modify: `training/run_m3.py` (add five functions above `main`, extend `main`)
- Test: run the module.

**Interfaces:**
- Consumes: everything Task 2 produced, plus `runs/m2_runs.jsonl` written by M2. Each line of that file is a JSON object with the keys `held_out`, `seed`, `lr`, `f1_macro`, `val_loss`, `stopped_epoch`, `n_train`, `n_val`, `n_test`. There is no `kind` field, so sweep runs are separated by their `lr` alone.
- Produces, used by Task 4:
  - `_classifier() -> LogisticRegression`
  - `_fold_indices(folds: list[int], held_out: int) -> tuple[list[int], list[int]]`
  - `_prepare_nela(features: np.ndarray, train_idx: list[int], test_idx: list[int]) -> tuple[np.ndarray, np.ndarray, int]`
  - `score_nela(features: np.ndarray, labels: list[int], folds: list[int], n_folds: int = N_FOLDS) -> tuple[list[float], list[int]]`
  - `_print_model_row(name: str, scores: list[float], reference: float) -> None`

- [ ] **Step 1: Extend the import block**

Replace the import block at the top of `training/run_m3.py` with this. The new lines are `json`, `pathlib`, `numpy`, and the six `scipy`/`sklearn` lines. Do not import `GROUPS` or `GROUP_SIZES` yet — nothing in this task uses them, and ruff's `F401` fails the lint step in Step 6 on an unused import. Task 4 adds them when it needs them.

```python
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
```

- [ ] **Step 2: Add the two new constants**

Below `BASELINE_TOLERANCE = 0.001`, add:

```python
M2_RUNS_PATH = pathlib.Path("runs/m2_runs.jsonl")
# Learning rate run final M2. Dua titik sweep, 1e-5 dan 2e-5, tidak ikut
# dirata-ratakan; filter ini yang menyisakan tepat 25 run.
FINAL_LR = 3e-5
```

- [ ] **Step 3: Add the scoring functions**

Insert these five functions between `assert_no_group_overlap` and `main`:

```python
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
```

- [ ] **Step 4: Extend `main` with the four-model table**

Append to the end of `main`, after the `print("cocok dengan angka M1 ...")` line:

```python
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
```

- [ ] **Step 5: Run it**

Run: `.venv-train/Scripts/python.exe -m training.run_m3`

Expected: the Task 2 output, then a four-row table. The cache is warm, so the whole thing runs in a minute or two. Two things must hold, and both are checks, not predictions:

- `run DeBERTa yang dirata2    : 25 pada lr=3e-05`. Any other number means the filter caught sweep runs or missed final ones.
- The DeBERTa row reads `0.8477 +/- ...`. The mean is fixed by M2; only the std differs, because this row averages seeds within each fold first and M2's headline std does not.

Do not treat the NELA and combined numbers as expected values. They are the measurement this milestone exists to make.

- [ ] **Step 6: Run the existing tests and lint**

Run:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests training
```

Expected: `64 passed`, `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add training/run_m3.py
git commit -m "feat: add M3 four-model comparison table"
```

---

### Task 4: Per-group breakdown table

**Files:**
- Modify: `training/run_m3.py` (extend `main`)
- Test: run the module.

**Interfaces:**
- Consumes: `score_nela()` from Task 3; `GROUPS` and `GROUP_SIZES` from Task 1.
- Produces: nothing further tasks depend on. This is the last code change in the plan.

- [ ] **Step 1: Extend the `training.nela` import line**

Change it to pull in the two names this task uses:

```python
from training.nela import GROUP_SIZES, GROUPS, extract_features, group_map
```

- [ ] **Step 2: Append the breakdown table to `main`**

Append to the end of `main`:

```python
    print()
    print("== Breakdown per grup fitur ==")
    print("grup       | fitur | bertahan | F1 macro")
    columns = {
        group: [i for i, name in enumerate(names) if mapping[name] == group]
        for group in GROUPS
    }
    for group in GROUPS:
        assert len(columns[group]) == GROUP_SIZES[group], (
            f"grup {group}: {len(columns[group])} kolom, bukan {GROUP_SIZES[group]}"
        )
        group_scores, group_kept = score_nela(features[:, columns[group]], labels, folds)
        group_span = (
            f"{min(group_kept)}"
            if min(group_kept) == max(group_kept)
            else f"{min(group_kept)}..{max(group_kept)}"
        )
        print(
            f"{group:<10} | {len(columns[group]):>5} | {group_span:>8} | "
            f"{statistics.mean(group_scores):.4f} +/- "
            f"{statistics.stdev(group_scores):.4f}"
        )

    print()
    print("== Catatan pembacaan (spec M3 7.2) ==")
    print(
        f"Std DeBERTa mencampur sebaran antar-fold dan antar-seed dari {n_runs} run."
    )
    print(
        "Std ketiga model logistic regression hanya sebaran antar-fold dari 5 run, "
        "karena lbfgs pada data tetap bersifat deterministik dan seed tidak "
        "menghasilkan variasi apa pun."
    )
    print(
        "Kedua angka itu tidak sebanding sebagai ukuran ketidakpastian. Yang "
        "sebanding adalah mean per fold, karena keempatnya dihitung pada pembagian "
        "data yang sama, jadi delta antar-model dibaca dari mean itu."
    )
```

- [ ] **Step 3: Run it**

Run: `.venv-train/Scripts/python.exe -m training.run_m3`

Expected: both tables and the reading note. The `fitur` column must read `50, 7, 8, 9, 11, 2` top to bottom; the assert catches it if not. Total fits this run: 45, five folds times nine configurations.

If a group raises `semua kolom varians-nol pada bagian train`, that group is entirely dead at sentence length. Report it as such in Task 5 rather than working around it — that outcome is itself a result.

- [ ] **Step 4: Save the output for Task 5**

Run: `.venv-train/Scripts/python.exe -m training.run_m3 > runs/m3_report.txt`

`runs/` is in `.gitignore`, so this file stays local. Task 5 copies numbers out of it; retyping them from a scrollback is how transcription errors get into a README.

- [ ] **Step 5: Run the existing tests and lint**

Run:

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests training
```

Expected: `64 passed`, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add training/run_m3.py
git commit -m "feat: add M3 per-feature-group breakdown"
```

---

### Task 5: Record the results

**Files:**
- Modify: `docs/superpowers/specs/2026-08-29-training-m3-nela-design.md` (append section 11)
- Modify: `PRD_Media_Neutrality_Classifier.md` (section 6.1)
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `runs/m3_report.txt` from Task 4.
- Produces: nothing in code.

- [ ] **Step 1: Add section 11 to the M3 spec**

Append a section `## 11. Hasil yang terukur` after section 10. Include both tables verbatim from `runs/m3_report.txt`, the number of surviving NELA columns, and two or three sentences reading the result: whether NELA closes the gap to DeBERTa, whether the combined configuration beats TF-IDF alone, and which feature group carries the most signal.

Spec 6 rules out a failing outcome here. Whatever the numbers say, they go in as measured. If NELA lands close to DeBERTa, PRD 6.1 already calls that an interesting finding for the README rather than a problem.

Write in Indonesian, matching sections 1 through 10.

- [ ] **Step 2: Correct PRD section 6.1**

`nela-features==3.0.1` returns 87 features from `extract_all`, not the 141 the PRD claims. Change the count to 87 and keep 141 in the same sentence, attributed to Horne et al. 2018, so the origin of the older number is still traceable. Add the per-group split: style 50, complexity 7, bias 8, affect 9, moral 11, event 2.

This follows the pattern M1 used when it recorded that BABE has no `article_id`: correct the document, keep the trail.

- [ ] **Step 3: Update `README.md`**

Add an M3 section next to the existing M2 numbers. Give the four-model table, the per-group table, and the note from spec 7.2 that the four standard deviations do not measure the same thing.

State next to the numbers, as the M2 section already does, that they were computed on this project's group-clean folds and are not comparable to figures computed on the official BABE split.

- [ ] **Step 4: Update `CLAUDE.md`**

In the "Current state" section, add M3 below M2. Carry forward only what constrains later work: the headline NELA number against DeBERTa, what that says about where DeBERTa's advantage comes from, the 87-versus-141 correction, and that `data/nela_cache.npz` is a local cache keyed on a hash of the text list, so a changed data filter silently invalidates it rather than reusing it.

Add the M3 command to the training block:

```bash
.venv-train/Scripts/python.exe -m training.run_m3          # M3 report, 46 min first run
```

Change the closing line about plans: the serving-foundation, M1, M2, and M3 plans are all fully executed, and M4a has no plan document yet.

- [ ] **Step 5: Verify no data leaked into the repo**

Run: `git status --short`
Expected: only the four documentation files. If `runs/`, `models/`, or `data/` appears, stop and fix `.gitignore` before committing.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-08-29-training-m3-nela-design.md PRD_Media_Neutrality_Classifier.md README.md CLAUDE.md
git commit -m "docs: record M3 NELA baseline results"
```

- [ ] **Step 7: Check the authorship of the whole branch**

Run:

```bash
git log --format='%an|%cn' training-m2..HEAD | sort -u
git log --grep='Co-Authored-By' -i training-m2..HEAD
```

Expected: exactly `Ihsan-p1|Ihsan-p1`, and no output from the second command. The range starts at `training-m2` rather than `main` because this branch is stacked on M2.

---

## Notes for whoever executes this

**This branch is stacked on `training-m2`, which is stacked on `main`.** PR #2 for M2 was still open when this plan was written. The M3 pull request should target `training-m2` until PR #2 merges, then be retargeted at `main`.

**Task 2 Step 2 is the only slow step, at about 46 minutes.** Everything after it reads the cache and finishes in a minute or two. Do not start it until Task 1's self-check passes: a wrong feature matrix costs the full 46 minutes to discover.

**The cache is keyed on the text list, not on a version number.** Any change to `load_babe()`'s filters produces a different key and triggers a full re-extraction. That is the intended behavior, not a bug to route around.

**There is no criterion to pass.** Spec 6 fixes the obligation as a place, not a value: the numbers land in the spec's results section, the README, and `CLAUDE.md`. A small delta is a finding; a missing table is the only failure available here.
