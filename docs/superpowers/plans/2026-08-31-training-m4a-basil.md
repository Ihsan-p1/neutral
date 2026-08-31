# M4a BASIL Cross-Dataset Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the classifier trained on BABE holds up on BASIL, and report the two numbers PRD 8.2 requires — as-is, and after the threshold is recalibrated on a BASIL dev split — so a drop can be told apart from a label-taxonomy artifact.

**Architecture:** `training/basil.py` downloads the official `launchnlp/BASIL` release as one tarball, flattens `body-paragraphs` into sentences, and labels sentence `k` biased when a span with `bias == 'lex'` carries `id == 'p{k}'`. `training/run_m4a.py` loads the shipped DeBERTa checkpoint, proves it reproduces its own BABE fold-0 number, then scores all of BASIL with both DeBERTa and a TF-IDF baseline refit on all of BABE. No training happens, and no model is written.

**Tech Stack:** stdlib `urllib.request` and `tarfile` for the download, `datasets==5.0.1` for the one-shot HuggingFace comparison, `torch==2.13.0+cu126` and `transformers==5.16.1` for inference, `scikit-learn==1.9.0` for metrics. Every one of these is already pinned in `training/requirements-train.txt`; this plan adds no dependency.

**Spec:** `docs/superpowers/specs/2026-08-31-training-m4a-basil-design.md`

## Global Constraints

- **Two interpreters.** Anything importing `datasets`, `torch`, `transformers`, or `sklearn` runs under `.venv-train/Scripts/python.exe`. Tests and lint run under `.venv/Scripts/python.exe`. This is a Windows machine: `Scripts/`, not `bin/`.
- **`tests/` must never import `datasets`, `torch`, or `transformers`.** The serving CI job installs `requirements-dev.txt` only. Checks needing training dependencies live as `__main__` self-checks inside `training/`, the way `baseline_tfidf.py`, `train.py`, and `nela.py` already do.
- **`training/split.py` imports only stdlib** and is not modified by this plan. `tests/test_split.py` asserts its import set.
- **Never commit data.** `data/`, `models/`, and `runs/` are already in `.gitignore`. The BASIL release lands in `data/basil/`, so it is covered. BASIL states no license in either release repo (PRD 5.0) — it is downloaded at eval time, never redistributed, never copied into the image.
- **No training, no fine-tuning, no model written.** M4a reads `models/deberta-babe` and never writes to it.
- **BABE fold assignment stays `group_split(keys, n_folds=5, seed=0)` plus `train_val_test_folds(0)`,** identical to M1 and M2. Any other argument compares against a split that did not produce 0.8765.
- **BASIL dev/test split is `fold_of(triplet_uuid, n_folds=10, seed=0)`,** folds 0-2 dev and folds 3-9 test. The key is the event (`triplet-uuid`), never the article, never the row.
- **Label rule, spec 3.1b:** sentence `k` is biased when a span with `bias == 'lex'` has `id == 'p{k}'`. The second gold definition accepts `'lex'` or `'inf'`. Nothing else sets a label.
- **Metric is F1 macro**, matching PRD 8.1, so BASIL numbers sit beside 0.8477 and 0.7420 without conversion. Average precision is reported alongside it.
- **No stop criterion to pass, spec 6.** PRD 8.2 already decided the number is reported as measured. What can fail is a guard, not the F1.
- **Comments and docstrings in Indonesian**, matching `split.py`, `data.py`, `baseline_tfidf.py`, `nela.py`, and `run_m3.py`. Commit messages in English, conventional prefixes.
- **Only `Ihsan-p1` as author and committer.** No `Co-Authored-By`, no generator footer.
- **No sentence text in printed output.** Counts, proportions, and shapes only.
- **One commit per task.** Do not batch tasks.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `training/basil.py` | create: tarball download, sentence flattening, label rule, span verification, `LoadReport`, self-check | 1 |
| `training/basil.py` | add: one-shot comparison against `horychtom/BASIL` | 2 |
| `training/baseline_tfidf.py` | add: `fit_predict_proba()` for an external test set | 3 |
| `training/run_m4a.py` | create: BABE load, DeBERTa inference, reproduction guard | 4 |
| `training/run_m4a.py` | add: BASIL load, HuggingFace comparison, overlap guard, dev/test split | 5 |
| `training/run_m4a.py` | add: thresholds, four-row table, constant-predictor floor, report file | 6 |
| `docs/superpowers/specs/2026-08-31-training-m4a-basil-design.md` | fill section 11 with measured results | 7 |
| `README.md` | M4a cross-dataset section | 7 |
| `CLAUDE.md` | M4a results carried forward, next milestone | 7 |

`split.py`, `data.py`, `train.py`, `nela.py`, `run_m1.py`, `run_m2.py`, and `run_m3.py` are not modified. `run_m4a.py` consumes `load_babe()`, `group_split()`, `train_val_test_folds()`, and `fold_of()` exactly as they are, and imports `MAX_LENGTH` from `train.py` so inference truncation matches training. Nothing under `src/neutral/`, `requirements.txt`, `requirements-dev.txt`, `Dockerfile`, or `.dockerignore` changes.

---

### Task 1: BASIL loader in `training/basil.py`

**Files:**
- Create: `training/basil.py`
- Test: `training/basil.py` `_self_check()` under `__main__` (no file in `tests/` — the serving venv has no `datasets`)

**Interfaces:**
- Consumes: nothing from earlier tasks. `pathlib`, `tarfile`, `urllib.request`, `json`, `re`, `dataclasses` from stdlib.
- Produces:
  - `Row` frozen dataclass: `text: str`, `lex: int`, `info: int`, `event_key: str`, `article_key: str`, `source: str`, `year: str`
  - `LoadReport` frozen dataclass: `n_articles: int`, `n_events: int`, `n_sentences_raw: int`, `n_blank_dropped: int`, `n_spans: int`, `n_span_mismatch: int`, `n_lex: int`, `n_info: int`, `n_lex_or_info: int`, `bias_values: list[str]`, `sources: dict[str, int]`
  - `load_basil(cache_dir: str = "data/basil") -> tuple[list[Row], LoadReport]`
  - `normalise(text: str) -> str`
  - `download_release(cache_dir: str = "data/basil") -> pathlib.Path` returning the extracted `BASIL-main` directory
  - `SPAN_MISMATCH_LIMIT = 0.01`

- [ ] **Step 1: Write the failing self-check**

Append to `training/basil.py` (the file does not exist yet, so this is the first thing in it — the implementation lands in Step 3 above it):

```python
def _fake_release(root: pathlib.Path) -> None:
    """Tulis satu triplet tiruan dengan struktur identik rilis resmi.

    Dipakai self-check supaya aturan pemetaan diuji tanpa mengunduh 300 artikel.
    Kalimat sengaja dibuat pendek dan hambar: keluaran M4a tidak boleh memuat
    teks berita.
    """
    for index in (1, 2, 3):
        article = {
            "title": f"judul {index}",
            "date": "2010-07-02",
            "uuid": f"artikel-{index}",
            "source": ["nyt", "fox", "hpo"][index - 1],
            "triplet-uuid": "event-a",
            "body-paragraphs": [
                ["kalimat nol di sini", "kalimat satu di sini"],
                ["kalimat dua di sini"],
            ],
        }
        annotation = {
            "triplet-uuid": "event-a",
            "uuid": f"artikel-{index}",
            "phrase-level-annotations": [
                {"id": "p0", "bias": "lex", "start": 8, "end": 11, "txt": "nol"},
                {"id": "p2", "bias": "inf", "start": 8, "end": 11, "txt": "dua"},
            ],
        }
        (root / "articles" / "2010").mkdir(parents=True, exist_ok=True)
        (root / "annotations" / "2010").mkdir(parents=True, exist_ok=True)
        name = f"event-a_{index}"
        (root / "articles" / "2010" / f"{name}.json").write_text(
            json.dumps(article), encoding="utf-8"
        )
        (root / "annotations" / "2010" / f"{name}_ann.json").write_text(
            json.dumps(annotation), encoding="utf-8"
        )


def _self_check() -> None:
    """Periksa aturan pemetaan dan normalisasi pada data yang jawabannya diketahui."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "BASIL-main"
        _fake_release(root)
        rows, report = _rows_from_release(root)

        # Tiga artikel x tiga kalimat, urutan paragraf dipertahankan.
        assert report.n_articles == 3, report
        assert report.n_events == 1, report
        assert len(rows) == 9, len(rows)
        assert [row.text for row in rows[:3]] == [
            "kalimat nol di sini",
            "kalimat satu di sini",
            "kalimat dua di sini",
        ], rows[:3]

        # p0 menyalakan lex dan hanya kalimat 0. p2 bertipe inf, jadi tidak
        # pernah menyalakan label utama.
        assert [row.lex for row in rows[:3]] == [1, 0, 0], rows[:3]
        assert [row.info for row in rows[:3]] == [0, 0, 1], rows[:3]
        assert report.n_lex == 3 and report.n_info == 3, report
        assert report.n_lex_or_info == 6, report
        assert report.n_span_mismatch == 0, report
        assert sorted(report.bias_values) == ["inf", "lex"], report

        # Kunci grup adalah event, sama untuk ketiga artikel.
        assert {row.event_key for row in rows} == {"event-a"}, rows[0]
        assert len({row.article_key for row in rows}) == 3, rows[0]

    # Indeks di luar rentang dihitung sebagai ketidaksesuaian, bukan IndexError.
    sents = ["kalimat nol di sini"]
    spans = [{"id": "p9", "bias": "lex", "start": 0, "end": 3, "txt": "kal"}]
    lex, info, mismatch = _labels_from_spans(sents, spans)
    assert lex == [0] and info == [0], (lex, info)
    assert mismatch == 1, mismatch

    # Span yang teksnya tidak cocok juga dihitung, dan tidak melabeli apa pun.
    spans = [{"id": "p0", "bias": "lex", "start": 0, "end": 3, "txt": "xyz"}]
    lex, info, mismatch = _labels_from_spans(sents, spans)
    assert lex == [0], lex
    assert mismatch == 1, mismatch

    # Id yang tidak berbentuk p<angka> ditolak, bukan diterima diam-diam.
    spans = [{"id": "s0", "bias": "lex", "start": 0, "end": 3, "txt": "kal"}]
    lex, info, mismatch = _labels_from_spans(sents, spans)
    assert lex == [0], lex
    assert mismatch == 1, mismatch

    # Normalisasi: whitespace diciutkan, kutip typografis dan dash ke ASCII.
    assert normalise("  a\n b\tc ") == "a b c"
    assert normalise("“dia” — ‘itu’") == '"dia" - \'itu\''
    assert normalise("sama") == normalise("sama")

    print("self-check ok")


if __name__ == "__main__":
    _self_check()
```

- [ ] **Step 2: Run the self-check to verify it fails**

Run: `.venv-train/Scripts/python.exe -m training.basil`
Expected: FAIL with `NameError: name '_rows_from_release' is not defined` (or `ImportError` for the missing stdlib imports if you have not added them yet).

- [ ] **Step 3: Write the implementation**

Put this above the self-check in `training/basil.py`:

```python
"""Muat BASIL dari rilis resmi dan susutkan menjadi kalimat berlabel.

Modul ini mengimpor `datasets` hanya di fungsi perbandingan HuggingFace, jadi
jalur muat utamanya berjalan dengan stdlib saja. Tetap tidak boleh diimpor dari
`tests/`: batasnya sama dengan `data.py`, dan perbandingan itu bagian dari modul.

Rilis diunduh ke `data/basil/`, yang ada di `.gitignore`. BASIL tidak menyatakan
lisensi apa pun di kedua repo rilisnya (PRD 5.0), sehingga pengamanannya adalah
tidak pernah meredistribusikannya: tidak masuk repo, tidak masuk image.

Struktur yang diandalkan, diperiksa langsung pada 2026-08-31:

    articles/<tahun>/<triplet-uuid>_<1|2|3>.json
      body-paragraphs : list paragraf, tiap paragraf list KALIMAT
      triplet-uuid    : id event      uuid : id artikel      source : 'nyt' | ...

    annotations/<tahun>/<triplet-uuid>_<1|2|3>_ann.json
      phrase-level-annotations : [{id: 'p0', bias: 'lex' | 'inf',
                                   start: 81, end: 87, txt: 'fierce', ...}]

`id` bukan sekadar nama: `p{k}` menunjuk kalimat ke-k pada daftar kalimat yang
sudah diratakan, dan `start`/`end` relatif terhadap kalimat itu. Itu hipotesis
yang terbukti pada satu artikel saat spec ditulis, jadi `load_basil()`
memverifikasinya pada seluruh 300 artikel dan `raise` bila tidak berlaku.
"""

import json
import pathlib
import re
import tarfile
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass

TARBALL = "https://codeload.github.com/launchnlp/BASIL/tar.gz/refs/heads/main"
RELEASE_DIR = "BASIL-main"

# Proporsi span yang boleh tidak memenuhi `sents[k][start:end] == txt` sebelum
# aturan pemetaannya sendiri yang dianggap salah, bukan beberapa barisnya.
SPAN_MISMATCH_LIMIT = 0.01

_SPAN_ID = re.compile(r"^p(\d+)$")

# Kutip typografis, apostrof, dan dash diseragamkan ke ASCII. BASIL memuat
# karakter ini apa adanya dari artikel aslinya, dan encoding-nya bisa berbeda
# antar rilis maupun antar re-upload.
_ASCII = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        " ": " ",
    }
)


@dataclass(frozen=True)
class Row:
    text: str
    lex: int
    info: int
    event_key: str
    article_key: str
    source: str
    year: str


@dataclass(frozen=True)
class LoadReport:
    """Angka yang harus dilaporkan M4a, termasuk setiap baris yang dibuang."""

    n_articles: int
    n_events: int
    n_sentences_raw: int
    n_blank_dropped: int
    n_spans: int
    n_span_mismatch: int
    n_lex: int
    n_info: int
    n_lex_or_info: int
    bias_values: list[str]
    sources: dict[str, int]


def normalise(text: str) -> str:
    """Ciutkan whitespace dan seragamkan karakter typografis ke ASCII.

    Casing dipertahankan: pilihan huruf besar adalah bagian dari bias leksikal.
    """
    return " ".join(text.translate(_ASCII).split())


def download_release(cache_dir: str = "data/basil") -> pathlib.Path:
    """Unduh dan ekstrak tarball rilis resmi. Idempoten.

    Satu tarball, bukan 600 permintaan file per file. Bila direktori hasil
    ekstraksi sudah ada, tidak ada yang diunduh.
    """
    root = pathlib.Path(cache_dir) / RELEASE_DIR
    if (root / "articles").is_dir():
        return root

    target = pathlib.Path(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = pathlib.Path(tmp) / "basil.tar.gz"
        urllib.request.urlretrieve(TARBALL, archive)
        with tarfile.open(archive) as handle:
            # filter="data" menolak entri absolut, path di luar target, symlink,
            # dan device file. Tarball pihak ketiga tidak diekstrak tanpa ini.
            handle.extractall(target, filter="data")

    if not (root / "articles").is_dir():
        raise RuntimeError(f"ekstraksi tidak menghasilkan {root / 'articles'}")
    return root


def _labels_from_spans(
    sents: list[str], spans: list[dict]
) -> tuple[list[int], list[int], int]:
    """Terapkan aturan pemetaan spec 3.1b. Kembalikan (lex, info, jumlah gagal).

    Span dihitung gagal bila id-nya bukan `p<angka>`, indeksnya di luar rentang,
    atau `sents[k][start:end]` tidak sama dengan `txt`. Span gagal tidak
    melabeli apa pun: label yang salah lebih merusak daripada label yang hilang,
    dan jumlahnya dilaporkan.
    """
    lex = [0] * len(sents)
    info = [0] * len(sents)
    mismatch = 0
    for span in spans:
        found = _SPAN_ID.match(str(span.get("id", "")))
        if found is None:
            mismatch += 1
            continue
        index = int(found.group(1))
        if index >= len(sents):
            mismatch += 1
            continue
        if sents[index][span["start"] : span["end"]] != span["txt"]:
            mismatch += 1
            continue
        if span["bias"] == "lex":
            lex[index] = 1
        elif span["bias"] == "inf":
            info[index] = 1
    return lex, info, mismatch


def _rows_from_release(root: pathlib.Path) -> tuple[list[Row], LoadReport]:
    """Baca seluruh artikel dan anotasi di bawah `root`. Tanpa unduhan."""
    rows: list[Row] = []
    events: set[str] = set()
    articles = 0
    sentences_raw = 0
    blank = 0
    spans_total = 0
    mismatch_total = 0
    bias_values: set[str] = set()
    sources: Counter[str] = Counter()

    for article_path in sorted((root / "articles").glob("*/*.json")):
        year = article_path.parent.name
        ann_path = root / "annotations" / year / f"{article_path.stem}_ann.json"
        if not ann_path.is_file():
            raise RuntimeError(f"anotasi hilang untuk {article_path}")

        article = json.loads(article_path.read_text(encoding="utf-8"))
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        spans = annotation.get("phrase-level-annotations", [])

        sents = [
            sentence for para in article["body-paragraphs"] for sentence in para
        ]
        lex, info, mismatch = _labels_from_spans(sents, spans)

        articles += 1
        sentences_raw += len(sents)
        spans_total += len(spans)
        mismatch_total += mismatch
        bias_values.update(str(span.get("bias")) for span in spans)
        events.add(article["triplet-uuid"])
        sources[article["source"]] += 1

        for text, lex_label, info_label in zip(sents, lex, info, strict=True):
            if not text.strip():
                blank += 1
                continue
            rows.append(
                Row(
                    text=text,
                    lex=lex_label,
                    info=info_label,
                    event_key=article["triplet-uuid"],
                    article_key=article["uuid"],
                    source=article["source"],
                    year=year,
                )
            )

    report = LoadReport(
        n_articles=articles,
        n_events=len(events),
        n_sentences_raw=sentences_raw,
        n_blank_dropped=blank,
        n_spans=spans_total,
        n_span_mismatch=mismatch_total,
        n_lex=sum(row.lex for row in rows),
        n_info=sum(row.info for row in rows),
        n_lex_or_info=sum(1 for row in rows if row.lex or row.info),
        bias_values=sorted(bias_values),
        sources=dict(sources),
    )
    return rows, report


def load_basil(cache_dir: str = "data/basil") -> tuple[list[Row], LoadReport]:
    """Unduh bila perlu, muat, dan verifikasi aturan pemetaan pada data nyata.

    Verifikasi di sini, bukan di self-check: yang diuji adalah apakah hipotesis
    indeks `p{k}` berlaku pada 300 artikel sungguhan, dan itu tidak bisa
    dijawab data tiruan.
    """
    rows, report = _rows_from_release(download_release(cache_dir))
    if report.n_spans == 0:
        raise RuntimeError("tidak ada span sama sekali; rilis kemungkinan rusak")
    ratio = report.n_span_mismatch / report.n_spans
    if ratio > SPAN_MISMATCH_LIMIT:
        raise RuntimeError(
            f"{report.n_span_mismatch} dari {report.n_spans} span "
            f"({ratio:.1%}) tidak memenuhi sents[k][start:end] == txt; "
            "aturan pemetaan spec 3.1b tidak berlaku"
        )
    return rows, report
```

- [ ] **Step 4: Run the self-check to verify it passes**

Run: `.venv-train/Scripts/python.exe -m training.basil`
Expected: PASS, printing `self-check ok`.

- [ ] **Step 5: Run the loader once against the real release**

Run:
```bash
.venv-train/Scripts/python.exe -c "from training.basil import load_basil; rows, report = load_basil(); print(report); print('baris:', len(rows))"
```
Expected: 300 articles, 100 events, roughly 8,100 sentences, `n_span_mismatch` at or near 0, `bias_values` `['inf', 'lex']`, and three sources at 100 articles each. If it raises on the mismatch limit, stop and report the number — the label rule is wrong and no later task is worth running.

- [ ] **Step 6: Verify nothing else regressed**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add training/basil.py
git commit -m "feat: add M4a BASIL loader with span-to-sentence label rule"
```

---

### Task 2: One-shot comparison against the HuggingFace re-upload

**Files:**
- Modify: `training/basil.py` (add one function plus its self-check block)

**Interfaces:**
- Consumes: `Row`, `normalise()`, `_rows_from_release()` from Task 1.
- Produces:
  - `HfComparison` frozen dataclass: `n_official: int`, `n_hf: int`, `n_matched: int`, `n_ambiguous: int`, `n_label_differs: int`, `matched_ratio: float`, `differs_ratio: float`
  - `compare_with_hf(rows: list[Row], cache_dir: str = "data/basil") -> HfComparison`
  - `HF_MATCH_FLOOR = 0.50`

- [ ] **Step 1: Write the failing self-check**

Add to `_self_check()` in `training/basil.py`, immediately before the final `print("self-check ok")`:

```python
    # Perbandingan HuggingFace dihitung dari peta teks, jadi bagian yang bisa
    # salah diuji tanpa jaringan: baris ganda dan baris yang labelnya beda.
    rows = [
        Row("kalimat nol di sini", 1, 0, "event-a", "artikel-1", "nyt", "2010"),
        Row("kalimat satu di sini", 0, 0, "event-a", "artikel-1", "nyt", "2010"),
        Row("kalimat dua di sini", 0, 1, "event-a", "artikel-1", "nyt", "2010"),
    ]
    hf = [
        # cocok, label sama
        {"text": "kalimat nol di sini", "lex_label": 1},
        # cocok setelah normalisasi, label beda
        {"text": "  kalimat   satu di sini ", "lex_label": 1},
        # tidak cocok ke kalimat mana pun
        {"text": "kalimat lain sekali", "lex_label": 0},
    ]
    comparison = _compare_rows(rows, hf)
    assert comparison.n_official == 3, comparison
    assert comparison.n_hf == 3, comparison
    assert comparison.n_matched == 2, comparison
    assert comparison.n_label_differs == 1, comparison
    assert comparison.matched_ratio == 2 / 3, comparison
    assert comparison.differs_ratio == 1 / 2, comparison

    # Teks resmi yang muncul dua kali tidak dipakai membandingkan apa pun:
    # tidak ada cara menentukan baris HuggingFace itu pasangan yang mana.
    doubled = rows + [
        Row("kalimat nol di sini", 0, 0, "event-b", "artikel-9", "fox", "2011")
    ]
    comparison = _compare_rows(doubled, hf[:1])
    assert comparison.n_ambiguous == 1, comparison
    assert comparison.n_matched == 0, comparison
```

- [ ] **Step 2: Run the self-check to verify it fails**

Run: `.venv-train/Scripts/python.exe -m training.basil`
Expected: FAIL with `NameError: name '_compare_rows' is not defined`.

- [ ] **Step 3: Write the implementation**

Add to `training/basil.py`, after `load_basil()`:

```python
HF_DATASET = "horychtom/BASIL"

# Di bawah ini, yang rusak adalah kode normalisasinya sendiri, bukan datanya,
# dan angka perbandingan apa pun jadi tidak berarti.
HF_MATCH_FLOOR = 0.50


@dataclass(frozen=True)
class HfComparison:
    """Hasil perbandingan sekali jalan spec 3.3. Tidak menghasilkan label."""

    n_official: int
    n_hf: int
    n_matched: int
    n_ambiguous: int
    n_label_differs: int
    matched_ratio: float
    differs_ratio: float


def _compare_rows(rows: list[Row], hf_rows: list[dict]) -> HfComparison:
    """Bandingkan label resmi dengan `lex_label` HuggingFace lewat teks ternormalisasi.

    Teks resmi yang muncul lebih dari sekali dikeluarkan dari perbandingan: tidak
    ada cara menentukan baris HuggingFace mana pasangannya, dan menebak akan
    mengarang angka ketidaksesuaian.
    """
    counts: Counter[str] = Counter(normalise(row.text) for row in rows)
    official = {
        normalise(row.text): row.lex
        for row in rows
        if counts[normalise(row.text)] == 1
    }
    matched = 0
    differs = 0
    for hf_row in hf_rows:
        key = normalise(str(hf_row["text"]))
        if key not in official:
            continue
        matched += 1
        if int(hf_row["lex_label"]) != official[key]:
            differs += 1

    return HfComparison(
        n_official=len(rows),
        n_hf=len(hf_rows),
        n_matched=matched,
        n_ambiguous=sum(1 for key, count in counts.items() if count > 1),
        n_label_differs=differs,
        matched_ratio=matched / len(hf_rows) if hf_rows else 0.0,
        differs_ratio=differs / matched if matched else 0.0,
    )


def compare_with_hf(
    rows: list[Row], cache_dir: str = "data/basil"
) -> HfComparison:
    """Muat re-upload HuggingFace sekali dan bandingkan. Tidak mengubah label.

    Import `datasets` sengaja lokal: jalur muat utama modul ini stdlib saja, dan
    kegagalan unduhan di sini tidak boleh menjatuhkan `load_basil()`.
    """
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET, cache_dir=cache_dir, split="train")
    hf_rows = [
        {"text": row["text"], "lex_label": row["lex_label"]} for row in dataset
    ]
    comparison = _compare_rows(rows, hf_rows)
    if comparison.matched_ratio < HF_MATCH_FLOOR:
        raise RuntimeError(
            f"hanya {comparison.matched_ratio:.1%} baris HuggingFace cocok; "
            "normalisasi teks kemungkinan rusak, angkanya tidak berarti"
        )
    return comparison
```

- [ ] **Step 4: Run the self-check to verify it passes**

Run: `.venv-train/Scripts/python.exe -m training.basil`
Expected: PASS, printing `self-check ok`.

- [ ] **Step 5: Run the comparison once against the real dataset**

Run:
```bash
.venv-train/Scripts/python.exe -c "from training.basil import compare_with_hf, load_basil; rows, _ = load_basil(); print(compare_with_hf(rows))"
```
Expected: `n_hf` 8018, `matched_ratio` well above 0.50, and whatever `differs_ratio` turns out to be. Record the numbers — Task 5 prints them and Task 7 reports them. A high `differs_ratio` is a finding, not a failure: nothing downstream depends on it.

- [ ] **Step 6: Verify nothing else regressed**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add training/basil.py
git commit -m "feat: compare M4a BASIL labels against the HuggingFace re-upload"
```

---

### Task 3: TF-IDF prediction for an external test set

**Files:**
- Modify: `training/baseline_tfidf.py` (add one function, extend `_self_check()`)

**Interfaces:**
- Consumes: `_model()` from `baseline_tfidf.py`, unchanged.
- Produces: `fit_predict_proba(train_texts: list[str], train_labels: list[int], test_texts: list[str]) -> list[float]` returning P(label == 1) per test text, in input order.

- [ ] **Step 1: Write the failing test**

Add to `_self_check()` in `training/baseline_tfidf.py`, before its final `print(...)`:

```python
    # Fit pada seluruh data, prediksi teks yang tidak ada di dalamnya. Dipakai
    # M4a untuk test set eksternal, di mana tidak ada fold sama sekali.
    probabilities = fit_predict_proba(
        texts,
        labels,
        ["outrageous so-called experts", "the budget was reviewed on tuesday"],
    )
    assert len(probabilities) == 2, probabilities
    assert all(0.0 <= value <= 1.0 for value in probabilities), probabilities
    assert probabilities[0] > probabilities[1], probabilities
```

- [ ] **Step 2: Run the self-check to verify it fails**

Run: `.venv-train/Scripts/python.exe -m training.baseline_tfidf`
Expected: FAIL with `NameError: name 'fit_predict_proba' is not defined`.

- [ ] **Step 3: Write the implementation**

Add to `training/baseline_tfidf.py`, after `run_baseline()`:

```python
def fit_predict_proba(
    train_texts: list[str], train_labels: list[int], test_texts: list[str]
) -> list[float]:
    """Latih pada seluruh data latih, kembalikan P(label == 1) untuk tiap teks test.

    `run_baseline()` terikat loop fold dan tidak bisa dipakai untuk test set
    eksternal seperti BASIL, di mana tidak ada fold. Yang dipakai ulang di sini
    setelan `_model()`, jadi baseline M4a identik dengan baseline M1 selain
    bagian train-nya yang memakai seluruh BABE.

    Kebocoran yang dijaga `run_baseline()` tidak berlaku di sini: teks test
    berasal dari korpus lain, jadi mem-fit vectorizer pada seluruh data latih
    tidak membocorkan apa pun tentang test.
    """
    model = _model()
    model.fit(train_texts, train_labels)
    return [float(value) for value in model.predict_proba(test_texts)[:, 1]]
```

- [ ] **Step 4: Run the self-check to verify it passes**

Run: `.venv-train/Scripts/python.exe -m training.baseline_tfidf`
Expected: PASS, printing `self-check ok: [...]`.

- [ ] **Step 5: Verify nothing else regressed**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean. M3 imports `run_baseline` and is untouched.

- [ ] **Step 6: Commit**

```bash
git add training/baseline_tfidf.py
git commit -m "feat: add TF-IDF prediction for an external test set"
```

---

### Task 4: `run_m4a.py` entry point and the reproduction guard

**Files:**
- Create: `training/run_m4a.py`

**Interfaces:**
- Consumes: `load_babe()` from `data.py`, `group_split()` and `train_val_test_folds()` from `split.py`, `MAX_LENGTH` from `train.py`.
- Produces:
  - `predict_deberta(texts: list[str], model_dir: str = MODEL_DIR, batch_size: int = 32) -> list[float]` returning P(label == 1)
  - `assert_checkpoint_reproduces() -> float` returning the measured F1 macro
  - `MODEL_DIR = "models/deberta-babe"`, `M2_FOLD0_F1 = 0.8765`, `REPRODUCTION_TOLERANCE = 0.01`

- [ ] **Step 1: Write the entry point with the guard as its only content**

Create `training/run_m4a.py`:

```python
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
            probabilities.extend(
                torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            )
    return probabilities


def assert_checkpoint_reproduces() -> float:
    """Uji checkpoint pada test split fold 0 BABE. Kembalikan F1 macro terukur."""
    rows, _ = load_babe()
    folds = group_split([row.group_key for row in rows], N_FOLDS, seed=0)
    _, _, test_fold = train_val_test_folds(0, N_FOLDS)
    subset = [row for row, fold in zip(rows, folds, strict=True) if fold == test_fold]

    probabilities = predict_deberta([row.text for row in subset])
    predicted = [1 if value >= 0.5 else 0 for value in probabilities]
    measured = float(
        f1_score([row.label for row in subset], predicted, average="macro")
    )

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
```

- [ ] **Step 2: Run it**

Run: `.venv-train/Scripts/python.exe -m training.run_m4a`
Expected: prints `guard reproduksi   : F1 macro 0.87xx pada 751 baris`. If it raises, stop: something about inference differs from training and every later number would be meaningless. Check the label order in `models/deberta-babe/config.json` first.

- [ ] **Step 3: Verify nothing else regressed**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add training/run_m4a.py
git commit -m "feat: add M4a entry point with checkpoint reproduction guard"
```

---

### Task 5: BASIL load, comparison, overlap guard, dev/test split

**Files:**
- Modify: `training/run_m4a.py`

**Interfaces:**
- Consumes: `load_basil()`, `compare_with_hf()`, `normalise()` from `basil.py`; `fold_of()` from `split.py`; `assert_checkpoint_reproduces()` from Task 4.
- Produces:
  - `drop_babe_overlap(rows: list[BasilRow], babe_texts: list[str]) -> tuple[list[BasilRow], int]`
  - `split_dev_test(rows: list[BasilRow]) -> tuple[list[int], list[int]]` returning index lists
  - `OVERLAP_LIMIT = 0.01`, `DEV_FOLDS = 10`, `DEV_CUTOFF = 3`

- [ ] **Step 1: Add the imports and the two functions**

Add to the import block in `training/run_m4a.py`:

```python
from training.basil import Row as BasilRow
from training.basil import compare_with_hf, load_basil, normalise
from training.split import fold_of
```

Add the constants beside the existing ones:

```python
# Di atas ambang ini yang bermasalah bukan beberapa baris, melainkan klaim
# cross-dataset-nya sendiri.
OVERLAP_LIMIT = 0.01

# Pembagian dev/test BASIL, spec 3.4. Kuncinya event, bukan artikel, bukan baris.
DEV_FOLDS = 10
DEV_CUTOFF = 3
```

Add the two functions after `assert_checkpoint_reproduces()`:

```python
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
```

- [ ] **Step 2: Extend `main()` to print everything through the split**

Replace `main()` in `training/run_m4a.py`:

```python
def main() -> None:
    assert_checkpoint_reproduces()

    babe_rows, _ = load_babe()
    rows, report = load_basil()
    print(f"artikel BASIL      : {report.n_articles}, event {report.n_events}")
    print(f"kalimat            : {report.n_sentences_raw}, kosong dibuang {report.n_blank_dropped}")
    print(f"span               : {report.n_spans}, tidak sesuai {report.n_span_mismatch}")
    print(f"positif lex        : {report.n_lex} ({report.n_lex / len(rows):.1%})")
    print(f"positif lex or inf : {report.n_lex_or_info} ({report.n_lex_or_info / len(rows):.1%})")
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
```

- [ ] **Step 3: Run it**

Run: `.venv-train/Scripts/python.exe -m training.run_m4a`
Expected: the guard line, then the BASIL report, the comparison line, the overlap line, and dev/test sizes. Roughly 30 events in dev and 70 in test, with no event in both. Record every printed number — Task 7 reports them.

- [ ] **Step 4: Verify nothing else regressed**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add training/run_m4a.py
git commit -m "feat: add M4a BASIL load, overlap guard, and event-grouped split"
```

---

### Task 6: Thresholds, the four-row table, and the report file

**Files:**
- Modify: `training/run_m4a.py`

**Interfaces:**
- Consumes: everything from Tasks 4 and 5, plus `fit_predict_proba()` from Task 3.
- Produces:
  - `best_threshold(probabilities: list[float], gold: list[int], indices: list[int]) -> float`
  - `score(probabilities, gold, dev, test) -> dict` with keys `f1_asis`, `f1_recal`, `threshold`, `ap`, `gold_rate`, `pred_rate`
  - `constant_floor(gold: list[int], indices: list[int]) -> float`
  - `REPORT_PATH = pathlib.Path("runs/m4a_report.txt")`

- [ ] **Step 1: Add the metric functions**

Add these imports to `training/run_m4a.py`:

```python
import pathlib

from sklearn.metrics import average_precision_score
```

Add the constant beside the others:

```python
REPORT_PATH = pathlib.Path("runs/m4a_report.txt")

# Kandidat threshold. Rentang tetap, bukan probabilitas unik, supaya nilai yang
# dilaporkan sama persis di setiap jalan dan mudah dibaca di tabel.
THRESHOLDS = [value / 100 for value in range(1, 100)]
```

Add the functions after `split_dev_test()`:

```python
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
        predicted = [
            1 if probabilities[index] >= threshold else 0 for index in indices
        ]
        scored.append(
            (f1_score(subset_gold, predicted, average="macro", zero_division=0), threshold)
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
    return float(
        f1_score(subset, [0] * len(subset), average="macro", zero_division=0)
    )
```

- [ ] **Step 2: Extend `main()` to score, tabulate, and save**

Replace the end of `main()` in `training/run_m4a.py` — everything from the `dev, test = split_dev_test(rows)` line onward:

```python
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
    lines.append("pembanding BABE: TF-IDF 0,7420 +/- 0,0194 (M1), "
                 "DeBERTa 0,8477 +/- 0,0277 (M2)")
    # Empat catatan spec 7.3, dicetak bersama tabel dan bukan di catatan kaki.
    lines.append("catatan: label dihitung dari rilis resmi launchnlp/BASIL dengan "
                 "segmentasi kalimat milik dataset itu sendiri; re-upload "
                 "HuggingFace hanya dibandingkan, tidak dipakai sebagai label")
    lines.append("catatan: aturan pemetaan — kalimat ke-k biased bila ada span "
                 "bias == 'lex' dengan id == 'p{k}'")
    lines.append("catatan: BASIL beranotasi dalam konteks artikel, BABE berisi "
                 "kalimat berdiri sendiri; sebagian selisih adalah artefak label")
    lines.append("catatan: angka DeBERTa berasal dari satu checkpoint, bukan mean "
                 "beberapa seed; M2 mengukur spread antar-seed 0,0187 pada BABE")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlaporan disimpan   : {REPORT_PATH}")
```

For that to work, `main()` has to collect its output instead of printing it directly. Change the top of `main()` so every existing `print(...)` becomes `lines.append(...)`, and print the joined block once:

```python
def main() -> None:
    assert_checkpoint_reproduces()

    lines: list[str] = []
    babe_rows, _ = load_babe()
    rows, report = load_basil()
    lines.append(f"artikel BASIL      : {report.n_articles}, event {report.n_events}")
    lines.append(f"kalimat            : {report.n_sentences_raw}, kosong dibuang {report.n_blank_dropped}")
    lines.append(f"span               : {report.n_spans}, tidak sesuai {report.n_span_mismatch}")
    lines.append(f"positif lex        : {report.n_lex} ({report.n_lex / len(rows):.1%})")
    lines.append(f"positif lex or inf : {report.n_lex_or_info} ({report.n_lex_or_info / len(rows):.1%})")
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
```

and add `print("\n".join(lines))` immediately before the `REPORT_PATH.write_text(...)` call.

Add the import for the baseline:

```python
from training.baseline_tfidf import fit_predict_proba
```

- [ ] **Step 3: Run the whole thing**

Run: `.venv-train/Scripts/python.exe -m training.run_m4a`
Expected: the guard line, the report block, a four-row table, two constant-predictor floors, and `runs/m4a_report.txt` written. Read the table before moving on: `gold+` should sit near 6% for gold `lex`, and `thr` should come out well below 0.5 if the prior shift is what moves the numbers.

- [ ] **Step 4: Verify nothing else regressed**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add training/run_m4a.py
git commit -m "feat: add M4a threshold recalibration and cross-dataset table"
```

---

### Task 7: Record the results

**Files:**
- Modify: `docs/superpowers/specs/2026-08-31-training-m4a-basil-design.md` (section 11)
- Modify: `README.md` (cross-dataset section)
- Modify: `CLAUDE.md` (current state)

**Interfaces:**
- Consumes: `runs/m4a_report.txt` from Task 6.
- Produces: nothing code depends on.

- [ ] **Step 1: Fill section 11 of the spec**

Replace `Diisi setelah run_m4a.py dijalankan.` with the measured output: the printed block verbatim in a fenced code block, then the four-row table as a Markdown table, then a reading of it. The reading has to answer three questions and nothing else:

1. Does the drop survive threshold recalibration? Compare `f1_asis` against `f1_recal` per row.
2. Is the ranking intact? Read AP against the positive rate — AP near the gold rate means no better than chance ordering.
3. Does the taxonomy explain part of it? Compare gold `lex` against gold `lex or inf` per model.

State what the numbers do not license. The number came from one checkpoint on one dataset with a mapping rule this project chose, and 24 sibling checkpoints no longer exist to check the spread against.

- [ ] **Step 2: Add the README section**

Add a cross-dataset subsection beside the existing M3 numbers. It carries the two PRD 8.2 numbers for gold `lex`, the constant-predictor floor, the mapping rule in one sentence, and the fact that BASIL labels were computed from the official release rather than taken from the HuggingFace re-upload.

Do not touch the limitations paragraph that says the reported numbers have no predictive value. That claim is about the served model, which still has no ONNX export, and M4a does not change it.

- [ ] **Step 3: Update `CLAUDE.md`**

Add an M4a block to "Current state" in the same shape as the M2 and M3 blocks: the design and plan paths, the files `training/` gained, the branch, and the results that carry forward. Add `run_m4a` to the command list. Mark M4a done and name what is next — the outlet-correlation analysis (PRD 8.3), which this milestone deliberately dropped, and M4b.

Carry forward exactly the facts a later milestone cannot rederive from the code:

- the two PRD 8.2 numbers and the constant-predictor floor
- the mapping rule and that `p{k}` indexes the flattened sentence list
- the BASIL positive rate against BABE's 54.6%
- the HuggingFace comparison numbers, and that the re-upload is not the label source
- that 8.3 was split out of M4a and still has no design

- [ ] **Step 4: Verify**

Run: `.venv/Scripts/python.exe -m pytest -q` then `.venv/Scripts/python.exe -m ruff check src tests training`
Expected: 64 passed, ruff clean. Then re-read the three edited files for numbers that disagree with `runs/m4a_report.txt`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-31-training-m4a-basil-design.md README.md CLAUDE.md
git commit -m "docs: record M4a BASIL cross-dataset results"
```

- [ ] **Step 6: Confirm attribution before any push**

```bash
git log --format='%an|%cn' main..HEAD | sort -u
git log --grep='Co-Authored-By' -i main..HEAD
```
Expected: exactly `Ihsan-p1|Ihsan-p1`, and no output from the second command.
