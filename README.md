# Media Neutrality Classifier

Deteksi bias leksikal pada level kalimat berita berbahasa Inggris — mengukur
seberapa netral cara sebuah kalimat menyampaikan informasi berdasarkan pilihan
kata, bukan berdasarkan konten faktualnya.

**Status:** dalam pengembangan. Lapisan serving dibangun lebih dulu dengan
classifier stub; model terlatih menyusul.

## Dokumen

| Dokumen | Isi |
|---|---|
| [PRD](PRD_Media_Neutrality_Classifier.md) | Problem statement, scope, dataset, metrik, milestone |
| [Desain deployment](docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md) | Arsitektur serving, kontrak API, container, portabilitas |
| [Rencana implementasi](docs/superpowers/plans/2026-08-27-serving-foundation.md) | Task-by-task untuk lapisan serving |

## Menjalankan secara lokal

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn neutral.app:app --reload
```

UI tersedia di `http://localhost:8000`, API di `http://localhost:8000/analyze`.

Menjalankan test:

```bash
.venv/Scripts/python.exe -m pytest -v
.venv/Scripts/python.exe -m ruff check src tests
```

## Konfigurasi

Seluruh konfigurasi dibaca dari environment variable berawalan `NEUTRAL_`.

| Variable | Default | Keterangan |
|---|---|---|
| `NEUTRAL_MODEL_BACKEND` | `stub` | `stub` atau `onnx` |
| `NEUTRAL_MODEL_DIR` | `models/current` | Lokasi `model.onnx`, `tokenizer.json`, `metadata.json` |
| `NEUTRAL_DECISION_THRESHOLD` | `0.5` | Ambang probabilitas untuk label `biased` |
| `NEUTRAL_TEMPERATURE` | `1.0` | Temperature hasil kalibrasi |
| `NEUTRAL_MAX_CHARS` | `50000` | Batas ukuran teks per request |
| `NEUTRAL_MAX_SENTENCES` | `300` | Batas jumlah kalimat per request |
| `NEUTRAL_LOG_LEVEL` | `INFO` | Level logging |
| `PORT` | `8000` | Port HTTP (dibaca oleh container) |

## Keterbatasan

**Selama `NEUTRAL_MODEL_BACKEND=stub`, seluruh angka yang ditampilkan tidak
memiliki nilai prediktif apa pun.** Stub bersifat deterministik terhadap isi
kalimat sehingga pipeline dan API dapat diuji sepenuhnya, tetapi keluarannya
bukan hasil analisis bias. Angka baru bermakna setelah backend `onnx` aktif
dengan bobot terlatih.

Keterbatasan lain yang berlaku juga setelah model tersedia:

- Hanya mendeteksi **bias leksikal**, bukan bias seleksi/omisi maupun stance politik.
- Hanya untuk teks berbahasa **Inggris**. Input non-Inggris ditandai, bukan ditolak.
- `article_score` adalah **proporsi kalimat yang ditandai**. Proporsi dari
  sedikit kalimat kurang dapat diandalkan, karena itu jumlah kalimat selalu
  ditampilkan bersama skornya.

## Lisensi

Belum ditentukan.
