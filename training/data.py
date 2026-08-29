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
