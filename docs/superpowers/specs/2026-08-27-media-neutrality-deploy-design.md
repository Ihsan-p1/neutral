# Design Spec: Media Neutrality Classifier — Serving & Deployment

**Tanggal:** 2026-08-27
**Status:** Draft untuk review
**Terkait:** `PRD_Media_Neutrality_Classifier.md`

---

## 1. Konteks dan Tujuan

Dokumen ini melengkapi PRD dengan desain sisi serving dan deployment. PRD
menjawab "model apa yang dibangun"; dokumen ini menjawab "bagaimana model itu
sampai ke tangan orang lain dalam bentuk yang berjalan".

Tujuan yang disepakati: **portofolio lebih dulu, user adalah bonus.** Sukses
diukur dari apakah reviewer teknis membuka link demo dan mendapati sesuatu yang
langsung berfungsi — bukan dari jumlah pengguna. Target angka pengguna dihapus
dan diganti menjadi **deployment-ready**: artifact siap di-deploy kapan saja,
ke platform mana saja, tanpa perubahan kode.

Karena penulis memiliki beberapa project lain yang juga akan di-deploy, pola
dalam dokumen ini sengaja dirancang agar dapat dipakai ulang. Keputusan
platform dan biaya sengaja **ditunda** dan diambil sekali untuk semua project.

### 1.1 Estimasi Beban

Perhitungan kasar untuk memberi konteks skala:

```
1.000 pengguna / 30 hari                     = ~33 pengguna/hari
33 pengguna x ~100 kalimat                   = ~3.300 kalimat/hari
DeBERTa-v3-base, ONNX INT8, CPU (batched)    = ~40-60 ms/kalimat
                                             = ~3 menit CPU per hari
```

Kesimpulan: kapasitas bukan masalah. Satu container CPU terkecil sudah jauh
melebihi kebutuhan. Yang perlu ditangani adalah **lonjakan sesaat** (satu
tautan viral mengirim ratusan orang dalam satu jam), **cold start**, dan
**penyalahgunaan endpoint publik** — bukan volume rata-rata.

Karena itu tidak ada GPU, tidak ada autoscaling, dan tidak ada orkestrasi
dalam desain ini.

---

## 2. Keputusan yang Ditutup

Keputusan berikut diambil dalam sesi desain dan menjadi dasar seluruh dokumen.

### 2.1 Taxonomy v1 adalah biner

PRD versi awal menargetkan tiga kelas: `lexical_bias`, `informational_bias`,
`neutral`. Ini tidak dapat dilaksanakan. BABE dan MBIC memberi label biner
(`biased` / `non-biased`) pada level kalimat, ditambah label sekunder mengenai
opini. Pemisahan lexical versus informational adalah taxonomy milik BASIL, dan
BASIL berperan sebagai test set lintas-dataset, bukan data latih.

Dengan kata lain, kelas ketiga tidak memiliki data latih sama sekali.

**Keputusan:** v1 adalah klasifikasi biner `biased` / `neutral`. Pemisahan
lexical versus informational tetap dianalisis, tetapi hanya sebagai evaluasi
pada BASIL, dan dilaporkan di bagian limitation.

Konsekuensi yang menguntungkan: target F1 naik dari 0,65 menjadi sekitar
0,78-0,82 sesuai literatur BABE, dan output biner jauh lebih mudah dijelaskan
di antarmuka demo.

### 2.2 Input hanya teks yang di-paste

PRD awal mengandung kontradiksi: bagian Non-Goals menolak crawling, tetapi
bagian Scope menerima input berupa URL. Input URL tetap memerlukan pengambilan
konten di sisi server, dengan seluruh konsekuensinya: paywall, blokir
Cloudflare, halaman yang dirender JavaScript, latensi 2-10 detik yang tidak
dapat diprediksi, risiko IP server diblokir, serta area abu-abu legal terkait
pengambilan konten berita.

**Keputusan:** v1 hanya menerima teks yang di-paste pengguna. Pengambilan URL
menjadi kandidat v2 bila memang terbukti dibutuhkan.

### 2.3 Pola deployment: satu container

Satu image berisi FastAPI yang melayani API sekaligus antarmuka Gradio yang
di-mount ke aplikasi yang sama. Model ONNX ikut dibakar ke dalam image saat
build.

Alternatif yang ditolak:

- **API dan UI sebagai dua service.** Menggandakan biaya operasional untuk
  pemisahan yang belum dibutuhkan, dan menambah urusan CORS.
- **Container tipis yang mengunduh model saat start.** Image lebih kecil,
  tetapi startup menjadi 30-60 detik dan HF Hub berubah menjadi titik kegagalan
  saat boot.

---

## 3. Arsitektur

### 3.1 Pemisahan training dari serving

Keputusan paling berdampak dalam dokumen ini: **image produksi tidak memuat
PyTorch.**

PyTorch beserta wheel CUDA berukuran sekitar 2,5 GB. ONNX Runtime versi CPU
sekitar 40 MB. Selisih ini menentukan apakah hasilnya image 3 GB yang lambat
di-pull di platform mana pun, atau image sekitar 700 MB yang siap melayani
dalam kurang dari 10 detik.

```
training/     -> torch, transformers, datasets   (lokal, GPU, di luar image)
src/neutral/  -> onnxruntime, tokenizers         (di dalam image)
```

### 3.2 Struktur repositori

```
neutral/
├── src/neutral/
│   ├── config.py         # pydantic-settings; seluruh konfigurasi via env
│   ├── schemas.py        # model request/response pydantic
│   ├── segment.py        # pemecah artikel menjadi kalimat (pysbd)
│   ├── detect.py         # deteksi bahasa dan batas ukuran input
│   ├── classifier.py     # sesi ONNX, backend stub, temperature scaling
│   ├── pipeline.py       # segmentasi + klasifikasi + agregasi
│   ├── logging_setup.py  # formatter JSON, membuang teks pengguna
│   ├── api.py            # rute FastAPI dan rate limiting
│   ├── ui.py             # Gradio Blocks
│   └── app.py            # menggabungkan Gradio ke FastAPI, satu proses
├── training/
│   ├── split.py       # derivasi kunci grup dan fold; stdlib saja
│   ├── data.py        # loading, filter kualitas
│   ├── train.py
│   ├── calibrate.py   # temperature scaling
│   ├── evaluate.py
│   └── export_onnx.py # ekspor, kuantisasi, uji paritas
├── tests/
├── docs/superpowers/specs/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

Pemisahan `src/` dan `training/` bukan sekadar kerapian. Dockerfile hanya
menyalin `src/`, sehingga dependensi training tidak mungkin bocor ke image
produksi tanpa disengaja.

`split.py` dipisahkan dari `data.py` setelah implementasi: fungsi split harus
dapat diimpor tanpa menarik `datasets`, agar test anti-kebocoran ikut berjalan di
CI. Alasan lengkapnya ada di
`docs/superpowers/specs/2026-08-27-training-phase-design.md` bagian 3.2.

### 3.3 Alur permintaan

```
teks masuk
  -> validasi ukuran (tolak di atas 50.000 karakter)
  -> deteksi bahasa (tandai bila bukan Inggris, jangan tolak)
  -> segmentasi kalimat (pysbd)
  -> batasi 300 kalimat, sisanya dipotong dengan peringatan
  -> tokenisasi, padding dinamis, max_len 128
  -> satu panggilan ONNX untuk seluruh kalimat artikel
  -> logits dibagi temperature, lalu softmax
  -> agregasi menjadi skor artikel
  -> respons
```

Satu artikel menghasilkan **satu** panggilan ONNX berisi seluruh kalimatnya,
bukan satu panggilan per kalimat. Pada artikel 30 kalimat, perbedaannya besar.

---

## 4. Kontrak API

### 4.1 `POST /analyze`

Request:

```jsonc
{
  "text": "...",
  "return_sentences": true
}
```

Response:

```jsonc
{
  "article_score": 0.34,
  "n_sentences": 29,
  "n_biased": 10,
  "confidence_mean": 0.81,
  "sentences": [
    { "text": "...", "label": "biased", "prob": 0.87, "start": 0, "end": 112 }
  ],
  "model_version": "babe-deberta-v3-int8-2026.03",
  "warnings": []
}
```

`warnings` dapat berisi `non_english`, `too_short`, atau `truncated`.

### 4.2 Definisi `article_score`

`article_score` adalah **proporsi kalimat berlabel biased**, bukan rata-rata
probabilitas. Alasannya: proporsi ternormalisasi terhadap panjang, sehingga
artikel 5 kalimat dan 50 kalimat dapat dibandingkan secara langsung.

Keterbatasan yang harus ditampilkan, bukan disembunyikan: 1 dari 3 kalimat dan
13 dari 40 kalimat sama-sama bernilai 0,33 dengan reliabilitas yang sangat
berbeda. Karena itu `n_sentences` selalu disertakan dalam respons dan
ditampilkan di antarmuka, dan artikel dengan kurang dari 5 kalimat memicu
peringatan `too_short`.

`confidence_mean` disediakan sebagai informasi sekunder dan tidak ditampilkan
sebagai angka utama.

### 4.3 `GET /health` dan `GET /version`

`/health` harus benar-benar memverifikasi bahwa model telah dimuat dan mampu
melakukan inferensi, bukan sekadar mengembalikan status 200. Platform hosting
menggunakan endpoint ini untuk memutuskan restart; health check yang selalu
hijau membuat fitur tersebut tidak berguna.

### 4.4 Tidak ada `/compare`

Perbandingan beberapa artikel dilakukan dengan memanggil `/analyze` beberapa
kali dari sisi antarmuka. Tidak ada fungsionalitas yang hilang, dan ada satu
endpoint lebih sedikit yang harus ditulis, diuji, dan dipelihara.

---

## 5. Pipeline Model

```
train (GPU lokal)
  -> temperature scaling pada validation set
  -> ekspor ONNX
  -> kuantisasi INT8
  -> uji paritas
  -> push ke HF Hub dengan versi
  -> dibakar ke image saat docker build
```

### 5.1 Gate 1 — temperature ikut terekspor

Hasil kalibrasi hidup sebagai satu konstanta yang membagi logits sebelum
softmax. Konstanta ini mudah tertinggal ketika model dipindahkan ke ONNX.

Bila tertinggal, seluruh pekerjaan kalibrasi hilang tanpa memunculkan error apa
pun: model tetap mengembalikan probabilitas yang tampak wajar, hanya saja tidak
terkalibrasi. Nilai temperature disimpan bersama metadata model dan
diverifikasi oleh test.

### 5.2 Gate 2 — uji paritas ONNX INT8 terhadap PyTorch

Kuantisasi menggeser prediksi. Sebelum model dianggap layak rilis, wajib
diverifikasi pada test set:

- tingkat kesepakatan label minimal 99%
- selisih probabilitas maksimum di bawah 0,02

Bila gagal, turunkan ke kuantisasi dynamic per-channel, atau gunakan fp16.

Alasan gate ini penting melampaui urusan teknis: angka F1 yang dilaporkan di
README berasal dari model PyTorch, sedangkan yang dilayani ke pengguna adalah
model INT8. Tanpa verifikasi paritas, README menjadi tidak jujur — persis jenis
masalah yang PRD ini nyatakan ingin dihindari.

### 5.3 Versioning

Model disimpan di HF Hub dan disematkan ke image berdasarkan **commit SHA**,
bukan `main`. Referensi ke `main` dapat berubah sewaktu-waktu dan membuat build
berhenti reproducible tanpa pemberitahuan.

---

## 6. Container dan Portabilitas

### 6.1 Dockerfile

Multi-stage, dua tahap:

```dockerfile
FROM python:3.13-slim AS builder
# pip install -r requirements.txt -> venv dengan versi ter-pin
# snapshot_download(revision="<commit-sha>")

FROM python:3.13-slim
# salin venv dan model dari builder; pengguna non-root
CMD uvicorn neutral.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

Tiga detail yang menentukan portabilitas:

- **`python:3.13-slim`, bukan alpine.** musl memaksa numpy dan onnxruntime
  dibangun dari sumber; hasilnya image lebih besar dan build jauh lebih lama —
  kebalikan dari tujuan memakai alpine.
- **`${PORT:-8000}`.** HF Spaces memberi 7860, Cloud Run dan Fly memberi 8080.
  Port yang di-hardcode mengunci project ke satu platform.
- **Pengguna non-root**, dan tidak ada build tooling di image akhir.

### 6.2 Checklist portabilitas

Checklist ini berlaku untuk seluruh project yang akan di-deploy, bukan hanya
project ini.

| Aturan | Alasan |
|---|---|
| Port dibaca dari `$PORT` | Setiap platform berbeda |
| Tidak menulis ke filesystem selain `/tmp` | Banyak runtime memakai filesystem read-only |
| Tidak ada network call saat startup | Startup tidak boleh bergantung pada layanan lain |
| `/health` memverifikasi model, bukan sekadar 200 | Restart otomatis bergantung padanya |
| SIGTERM ditangani | uvicorn sudah menanganinya; jangan dirusak |
| Seluruh konfigurasi lewat env, tanpa default rahasia | Ganti platform cukup ganti env |

Enam baris ini membuat pemilihan platform menjadi keputusan dua puluh menit di
kemudian hari, bukan pekerjaan refactor.

---

## 7. Operasional

### 7.1 Concurrency

Handler inferensi ditulis sebagai `def`, **bukan `async def`**.

FastAPI menjalankan handler sinkron di threadpool, sehingga inferensi ONNX yang
CPU-bound tidak memblokir event loop. Sebaliknya, `async def` yang memanggil
ONNX secara sinkron akan membekukan seluruh server selama inferensi
berlangsung: satu pengguna membuat semua pengguna lain menunggu.

Bug ini tidak muncul saat pengujian sendirian dan baru terlihat di bawah beban
nyata. Menuliskannya sebagai `def` menyelesaikan masalah tanpa satu baris kode
tambahan.

### 7.2 Rate limiting dan batas input

`slowapi` dengan penyimpanan in-memory. Redis tidak digunakan: hanya ada satu
container, tidak ada state yang perlu dibagi.

| Batas | Nilai | Alasan |
|---|---|---|
| Per IP | 20/menit, 300/jam | Longgar untuk pemakaian wajar |
| Ukuran teks | 50.000 karakter | Menolak payload besar sebelum tokenisasi |
| Kalimat per request | 300, sisanya dipotong | Mencegah satu request menyandera CPU |

Permintaan yang melewati batas mendapat status 429 dengan pesan yang jelas.

### 7.3 Logging

Modul `logging` bawaan dengan formatter JSON, keluaran ke stdout. Seluruh
platform membaca stdout, sehingga tidak diperlukan konfigurasi tambahan.

Field: `request_id`, `path`, `latency_ms`, `n_sentences`, `status`.

**Teks artikel tidak pernah dicatat.** Itu data pengguna. Yang dicatat hanya
panjangnya. Sekali teks masuk ke log, ia ikut ke mana pun log tersebut dikirim
atau disimpan.

---

## 8. CI dan Pengujian

### 8.1 Pipeline

```
ruff -> pytest -> docker build -> smoke test
```

Tahap paling bernilai adalah smoke test: menjalankan image hasil build,
memanggil `/health`, memanggil `/analyze` dengan artikel contoh, lalu
memverifikasi bentuk respons. Tahap ini membuktikan bahwa artifact benar-benar
berjalan, bukan sekadar bahwa kode lolos pengujian.

Push ke registry belum dilakukan sampai platform dipilih.

### 8.2 Test yang ditulis

Setiap test berikut menangkap kerusakan nyata yang tidak memunculkan error:

- `segment`: `"U.S."`, `"Mr."`, `"Inc."` tidak memecah kalimat
- `model`: temperature benar-benar diterapkan pada logits
- `api`: bentuk respons, penolakan input berlebih, pemicuan 429
- `parity`: ONNX INT8 terhadap PyTorch — berada di `training/`, di luar CI
  karena membutuhkan torch

Yang sengaja tidak ditulis: suite per-fungsi, fixture berlapis, dan mock untuk
ONNX. Tidak ada di antaranya yang menangkap bug nyata.

---

## 9. Risiko

| Risiko | Mitigasi |
|---|---|
| Kuantisasi INT8 menurunkan akurasi tanpa terdeteksi | Gate paritas wajib sebelum rilis; turun ke fp16 bila gagal |
| Temperature hilang saat ekspor ONNX | Disimpan sebagai metadata model, diverifikasi test |
| Cold start merusak kesan pertama reviewer | Model dibakar ke image; tanpa network call saat boot |
| Endpoint publik disalahgunakan | Rate limit per IP dan batas ukuran input |
| Teks pengguna bocor lewat log | Kebijakan tegas: catat panjang, jangan catat isi |
| Image membengkak karena torch ikut terbawa | Dockerfile hanya menyalin `src/`; `training/` tidak pernah masuk |
| Terkunci pada satu platform | Checklist portabilitas bagian 6.2 |

---

## 10. Di Luar Scope

Dinyatakan eksplisit agar tidak dianggap sebagai celah yang tidak disadari:

- Pengambilan konten dari URL (kandidat v2)
- Autentikasi dan akun pengguna
- Penyimpanan riwayat analisis
- Autoscaling, orkestrasi, atau GPU serving
- Analytics, SEO, dan mekanisme pertumbuhan pengguna
- Endpoint `/compare` sebagai endpoint tersendiri

---

## 11. Langkah Berikutnya

Setelah spec ini disetujui, tahap berikutnya adalah penyusunan rencana
implementasi melalui skill `writing-plans`. Belum ada kode yang ditulis sampai
tahap tersebut disetujui.

Catatan: direktori ini belum merupakan repositori git. Inisialisasi git
disarankan sebelum implementasi dimulai, agar spec dan seluruh perubahan
berikutnya terekam.
