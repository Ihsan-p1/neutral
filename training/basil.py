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

        # 12 dari 300 artikel menulis outlet dalam huruf besar ('NYT' bukan
        # 'nyt'). Tanpa penyeragaman, laporan menampilkan enam outlet padahal
        # ada tiga.
        source = str(article["source"]).lower()

        sents = [sentence for para in article["body-paragraphs"] for sentence in para]
        lex, info, mismatch = _labels_from_spans(sents, spans)

        articles += 1
        sentences_raw += len(sents)
        spans_total += len(spans)
        mismatch_total += mismatch
        bias_values.update(str(span.get("bias")) for span in spans)
        events.add(article["triplet-uuid"])
        sources[source] += 1

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
                    source=source,
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
    indeks `p{k}` berlaku pada 300 artikel sungguhan, dan itu tidak bisa dijawab
    data tiruan.
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
