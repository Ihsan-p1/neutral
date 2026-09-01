# Media Neutrality Classifier

Deteksi bias leksikal pada level kalimat berita berbahasa Inggris — mengukur
seberapa netral cara sebuah kalimat menyampaikan informasi berdasarkan pilihan
kata, bukan berdasarkan konten faktualnya.

**Status:** dalam pengembangan. Lapisan serving dibangun lebih dulu dengan
classifier stub. Model DeBERTa sudah dilatih dan dievaluasi (lihat Hasil
evaluasi), tetapi belum diekspor ke ONNX, jadi aplikasi yang berjalan masih
memakai stub.

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

## Menjalankan lewat Docker

```bash
docker build -t neutral .
docker run --rm -p 8000:8000 neutral
```

Port dibaca dari `$PORT`, sehingga image yang sama berjalan di HF Spaces (7860),
Cloud Run dan Fly (8080), atau VPS mana pun tanpa perubahan:

```bash
docker run --rm -e PORT=7860 -p 7860:7860 neutral
```

## Arsitektur

Tiga batas menopang sebagian besar desain:

**Training terpisah dari serving.** `training/` memakai PyTorch; `src/neutral/`
hanya memakai onnxruntime. `.dockerignore` mengecualikan `training/`, dan CI
memastikan `import torch` gagal di dalam image. Selisihnya sekitar 2,5 GB
berbanding 40 MB — itulah pembeda antara image yang siap melayani di bawah 10
detik dan yang tidak.

**Classifier berada di balik satu interface dengan dua implementasi.**
`StubClassifier` bersifat deterministik dan membuat seluruh pipeline, API, UI,
dan CI dapat berjalan sebelum ada bobot model. `OnnxClassifier` adalah jalur
produksi. Ketika bobot siap, satu-satunya perubahan di sisi serving adalah
menyetel `NEUTRAL_MODEL_BACKEND=onnx`.

**`pipeline.py` memuat logika analisis; `api.py` hanya menangani HTTP.**
Antarmuka Gradio memanggil fungsi `analyze()` yang sama dengan yang dipanggil
endpoint, sehingga keduanya tidak dapat berbeda perilaku.

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

## Hasil evaluasi

### Hasil M2: DeBERTa fine-tuned

| | F1 macro |
|---|---|
| TF-IDF + logistic regression | 0,7420 +/- 0,0194 |
| DeBERTa-v3-base, 5 fold x 5 seed | 0,8477 +/- 0,0277 |

Learning rate 3e-5, dipilih lewat validation loss pada satu fold. Setiap angka
per fold adalah mean atas 5 seed, dan deviasi antar-fold serta antar-seed
dilaporkan terpisah karena artinya berbeda: 0,0195 antar-fold dan 0,0187
antar-seed. Keduanya hampir sama besar, jadi satu angka dari satu fold dengan
satu seed akan menyembunyikan sebagian besar sebarannya.

Fold dibangun per artikel (`news_link`), bukan per baris, jadi angka di atas
tidak dapat dibandingkan langsung dengan angka mana pun yang dihitung pada split
resmi BABE. Split resmi itu sendiri bocor: 384 dari 2.151 kunci grup muncul di
train dan test sekaligus. Target 0,78 di literatur dihitung pada split itu.

Yang diukur hanya F1 macro pada fold BABE. Generalisasi lintas dataset,
kalibrasi, dan held-out berita terbaru belum diukur. Rinciannya di
[bagian 12 desain M2](docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md).

### Hasil M3: baseline NELA

Fitur linguistik eksplisit, diukur pada fold yang sama.

| | F1 macro | Delta TF-IDF |
|---|---|---|
| TF-IDF + logistic regression | 0,7420 +/- 0,0194 | +0,0000 |
| NELA + logistic regression | 0,6837 +/- 0,0116 | -0,0583 |
| TF-IDF + NELA + logistic regression | 0,7294 +/- 0,0130 | -0,0126 |
| DeBERTa-v3-base | 0,8477 +/- 0,0195 | +0,1058 |

Per grup fitur, semuanya dengan logistic regression yang sama:

| Grup | Fitur | Bertahan | F1 macro |
|---|---|---|---|
| style | 50 | 40 | 0,6599 +/- 0,0089 |
| affect | 9 | 9 | 0,6194 +/- 0,0211 |
| bias | 8 | 8 | 0,5884 +/- 0,0240 |
| complexity | 7 | 7 | 0,5382 +/- 0,0375 |
| moral | 11 | 11 | 0,5028 +/- 0,0310 |
| event | 2 | 1 | 0,4401 +/- 0,0175 |

NELA tertinggal 0,0583 di bawah TF-IDF dan 0,1640 di bawah DeBERTa.
Menyambungkannya ke TF-IDF menurunkan angka 0,0126, bukan menaikkannya, jadi
fitur ini bukan sinyal kuat yang kebetulan tumpang tindih dengan bag-of-words:
ia memang lemah pada teks sependek kalimat. Keunggulan DeBERTa karena itu datang
dari konteks, bukan dari hitungan kata kerja factive dan hedge.

Sinyal yang ada terkumpul di grup style: 40 kolomnya sendiri mencapai 0,6599 dari
0,6837 milik 87 fitur penuh. Grup `event` memberi 0,4401, sekitar 0,09 di atas
prediktor konstan (0,3534 bila selalu menebak `biased`), sementara memakan 92%
waktu ekstraksi.

Keempat deviasi standar itu tidak mengukur hal yang sama. Milik DeBERTa mencampur
sebaran antar-fold dan antar-seed dari 25 run; milik ketiga logistic regression
hanya sebaran antar-fold dari 5 run, karena lbfgs pada data tetap bersifat
deterministik. Yang sebanding adalah mean per fold, dan delta di tabel dibaca
dari situ. Rinciannya di
[bagian 11 desain M3](docs/superpowers/specs/2026-08-29-training-m3-nela-design.md).

Paket `nela-features==3.0.1` memberi 87 fitur, bukan 141 seperti yang tertulis di
PRD versi awal. Angka 141 adalah angka paper Horne et al. 2018.

### Hasil M4a: eval cross-dataset ke BASIL

Model dilatih pada BABE, diuji pada BASIL tanpa penyesuaian apa pun. BASIL berisi
300 artikel dari NYT, Fox News, dan HuffPost, 2010-2019. Label dihitung dari
rilis resmi `launchnlp/BASIL` dengan aturan **kalimat dianggap biased bila memuat
minimal satu span bertipe lexical**; segmentasi kalimatnya milik dataset itu
sendiri, bukan hasil segmenter tambahan. Re-upload di HuggingFace hanya
dibandingkan (99,3% cocok, 0,6% label berbeda), tidak dipakai sebagai sumber
label.

Diukur pada 5.780 kalimat dari 73 event; 27 event sisanya dipakai memilih
threshold. Prior positifnya 5,1%, terhadap 54,6% biased di BABE.

| Model | F1 apa adanya | F1 setelah rekalibrasi | AP |
|---|---|---|---|
| DeBERTa-v3-base | 0,5656 | 0,5854 | 0,1523 |
| TF-IDF + logistic regression | 0,3912 | 0,4963 | 0,0524 |

Prediktor konstan "semua neutral" mencapai F1 macro **0,4868** pada test set yang
sama. Angka itu perlu dibaca lebih dulu: DeBERTa hanya 0,079 di atasnya, dan
TF-IDF apa adanya berada di bawahnya.

Tiga hal yang dilaporkan bersama angka tersebut:

- **Rekalibrasi threshold tidak menyelamatkan penurunannya.** DeBERTa naik 0,0198
  saja, sehingga terhadap 0,8477 di BABE tetap turun 0,26. PRD 8.2 memperkirakan
  penurunan sebagian besar akan berupa artefak threshold dan ketidakcocokan
  label; perkiraan itu hanya berlaku sebagian kecil.
- **Sinyalnya masih ada, tipis.** AP DeBERTa 0,1523 pada prior 0,051 adalah 2,99
  kali prediktor acak, jadi urutannya belum rusak. TF-IDF mencapai 1,03 kali
  acak, yang berarti tidak memindahkan apa pun.
- **Angkanya berasal dari satu checkpoint,** bukan mean beberapa seed. M2
  mengukur sebaran antar-seed 0,0187 pada BABE; sebaran di BASIL tidak diukur.

Bacaan lengkap, termasuk apa yang tidak boleh disimpulkan dari angka ini, ada di
[bagian 11 desain M4a](docs/superpowers/specs/2026-08-31-training-m4a-basil-design.md).

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

## Data

Model dilatih pada **BABE** (Bias Annotations By Experts), sekitar 3.700 kalimat
berita beranotasi ahli dari Media Bias Group.

Dataset **tidak disertakan di repositori ini maupun di image Docker** — diunduh
saat training dan tidak pernah di-commit. Card HuggingFace BABE tidak menyatakan
lisensi, sementara repositori GitHub papernya berlisensi AGPL-3.0 pada level
repo. Dengan tidak melakukan redistribusi, klausul redistribusi kedua sumber
tidak pernah aktif. Rinciannya di bagian 5.0 PRD.

Evaluasi memakai tiga set: test split BABE (group split per artikel), BASIL
sebagai cross-dataset, dan satu held-out kecil berisi berita terbaru yang
dilabeli manual mengikuti panduan anotasi BABE. Set terakhir hanya dipakai untuk
menguji, tidak pernah untuk melatih atau memilih model.

Bila memakai dataset ini, sitasi papernya:

```bibtex
@inproceedings{spinde2021neural,
    title     = {Neural Media Bias Detection Using Distant Supervision With BABE -- Bias Annotations By Experts},
    author    = {Spinde, Timo and Plank, Manuel and Krieger, Jan-David and Ruas, Terry and Gipp, Bela and Aizawa, Akiko},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2021},
    year      = {2021},
    publisher = {Association for Computational Linguistics}
}
```

## Lisensi

Belum ditentukan.
