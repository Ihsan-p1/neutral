# Design Spec: Media Neutrality Classifier — Fase Training (M1)

**Tanggal:** 2026-08-27
**Status:** Draft untuk review
**Terkait:** `PRD_Media_Neutrality_Classifier.md`, `docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md`

---

## 1. Konteks dan Cakupan

Spec serving menjawab bagaimana model sampai ke pengguna. PRD menjawab model apa
yang dibangun. Dokumen ini menjawab satu hal yang lebih sempit: bagaimana data
BABE masuk ke project ini dengan cara yang membuat angka hasilnya layak
dilaporkan.

Lapisan serving sudah selesai dan berjalan di atas `StubClassifier`. Yang belum
ada sama sekali adalah direktori `training/`.

**Cakupan implementasi dokumen ini adalah M1 saja:** data loading, EDA,
pemeriksaan kualitas, group split, test anti-kebocoran, dan baseline TF-IDF.
Pekerjaan berhenti tepat sebelum M2 (fine-tune DeBERTa). Keputusan arsitektur di
sini tetap dibuat dengan M2–M5 dalam pandangan, agar tidak perlu dibongkar
nanti, tetapi M2 ke atas mendapat rencana tersendiri.

Alasan berhenti di batas itu bukan kehati-hatian berlebihan. Kriteria berhenti M1
di PRD bagian 9 berbunyi: test anti-kebocoran harus lulus sebelum training apa
pun dimulai. Dan kriteria lanjut M2 mensyaratkan perbandingan terhadap baseline
TF-IDF, yang berarti baseline itu prasyarat M2, bukan bagian darinya.

---

## 2. Temuan terhadap data nyata

Skema BABE diperiksa langsung lewat HuggingFace datasets-server sebelum dokumen
ini ditulis. Hasilnya mengubah satu ketentuan PRD.

```
dataset : mediabiasgroup/BABE
splits  : train 3.121, test 1.000  (total 4.121)
kolom   : text, outlet, label, topic, news_link, biased_words, uuid,
          type, label_opinion
lisensi : tidak ada tag lisensi
```

| Kolom | Isi | Peran di project ini |
|---|---|---|
| `text` | kalimat | input |
| `label` | `0` / `1` | target; `1` = biased |
| `news_link` | URL artikel sumber | **kunci group split** |
| `outlet` | Breitbart, Federalist, Alternet, … | stratifikasi, dan korelasi outlet PRD 8.3 |
| `type` | `left` / `center` / `right` / `None` | lean outlet, bahan absolute lean di PRD 8.3 |
| `label_opinion` | 5 nilai, termasuk `No agreement` | filter kualitas PRD 5.1 |
| `biased_words` | daftar kata pemicu, mis. `indoctrinating`, `Marxist` | bahan contrast set M4b |
| `uuid`, `topic` | identitas dan topik | pelacakan, analisis error |

Daftar nilai untuk `type` dan `label_opinion` di atas berasal dari sampel 100
baris pertama, bukan dari keseluruhan dataset. Keduanya diverifikasi ulang
terhadap data penuh sebagai bagian dari M1.

Pemeriksaan lisensi menutup satu-satunya verifikasi yang masih terbuka di PRD
12.1: dataset card BABE di HuggingFace tetap tidak menyatakan lisensi apa pun.
Karena itu pengamanannya tidak berubah — dataset diunduh saat bekerja, tidak
pernah masuk repo, tidak pernah masuk image.

---

## 3. Keputusan yang Ditutup

### 3.1 Kunci group split adalah `news_link`, bukan `article_id`

PRD 5.2 mewajibkan split per `article_id` dan menyebutnya poin terpenting di
seluruh bagian data. Kolom itu tidak ada di BABE.

Yang tersedia sebagai penanda artikel sumber hanya `news_link`. Karena itu
`news_link` menjadi kunci grouping, dan PRD 5.2 dikoreksi.

Ini bukan penggantian yang setara tanpa syarat. Sebagian `news_link` berupa URL
redirect feedproxy, dan dua URL berbeda pada prinsipnya masih bisa menunjuk satu
artikel yang sama. Konsekuensinya dua hal harus diukur di M1, bukan diasumsikan:
berapa baris yang `news_link`-nya kosong, dan apakah normalisasi URL mengubah
jumlah grup secara berarti. Bila jumlah grup turun banyak setelah normalisasi,
normalisasi itulah yang dipakai sebagai kunci.

Baris tanpa `news_link` tidak boleh digabung menjadi satu grup besar — itu justru
menciptakan grup palsu. Baris seperti itu dibuang, dan jumlahnya dilaporkan.

### 3.2 Fungsi split memakai stdlib saja

Group split diimplementasikan sebagai hash deterministik dari kunci grup ke nomor
fold, tanpa `scikit-learn`.

Alasannya bukan menghindari dependency. Alasannya agar test anti-kebocoran dapat
hidup di `tests/` dan ikut berjalan di CI yang sudah ada, tanpa menambah satu
dependency pun ke `requirements-dev.txt`. Test terpenting di project ini menjadi
test yang berjalan otomatis pada setiap push.

Sebagai konsekuensi, `split.py` tidak boleh mengimpor `datasets`, `pandas`,
`sklearn`, atau `torch` — langsung maupun tidak langsung. Ini batasan yang dijaga
oleh test itu sendiri: bila `split.py` menarik dependency berat, test di CI gagal
saat impor.

Stratifikasi per outlet yang disebut PRD 5.2 sebagai "idealnya" tidak
diimplementasikan lebih dulu. Ia ditambahkan hanya bila EDA menunjukkan
distribusi outlet antar fold benar-benar timpang. Hash pada kunci grup sudah
menyebar outlet secara acak; menambah stratifikasi tanpa bukti hanya menambah
kode.

### 3.3 Isolasi dependency lewat venv kedua

`.venv-train/` terpisah dari `.venv/` serving, dengan pin sendiri di
`requirements-train.txt`.

Invariant "PyTorch tidak pernah masuk serving" sebelumnya dijaga oleh Dockerfile
dan satu langkah CI. Dengan venv terpisah, ia juga dijaga oleh keadaan fisik
mesin: torch memang tidak terpasang di venv serving, sehingga `import torch` yang
salah letak di `src/neutral/` gagal secara lokal, bukan baru gagal di CI.

Alternatif yang ditolak: satu venv untuk keduanya, yang membuat invariant hanya
dijaga Dockerfile; dan optional-dependency group di `pyproject.toml`, yang
menambah mekanisme pin kedua sementara tabel Deviations di CLAUDE.md sudah
memutuskan `requirements.txt` sebagai satu-satunya sumber pin.

### 3.4 Torch dipasang di M1 meski belum dipakai

M1 tidak memerlukan torch sama sekali — hanya `datasets` dan `scikit-learn`.
Torch tetap dipasang dan diverifikasi di fase ini.

Yang ditutup lebih awal adalah risiko yang mengubah isi rencana M2: PRD 6
menyatakan DeBERTa-v3-base muat untuk full fine-tune di 6 GB VRAM dengan fp16 dan
batch 16. Klaim itu belum pernah diuji di mesin ini. Bila wheel CUDA tidak cocok
dengan driver 581.86, atau bila 6 GB ternyata tidak cukup pada batch 16, rencana
M2 berubah — dan lebih murah mengetahuinya sebelum rencana itu ditulis daripada
sesudahnya.

Verifikasinya satu smoke check, bukan training: `torch.cuda.is_available()`, nama
device, dan satu alokasi tensor di GPU.

---

## 4. Struktur `training/`

```
training/
├── requirements-train.txt   # pin: torch (cu12x), transformers, datasets,
│                            #      scikit-learn
├── split.py                 # group split; stdlib saja, tanpa dependency berat
├── data.py                  # load BABE, filter kualitas, normalisasi kunci
├── baseline_tfidf.py        # TF-IDF + logistic regression
└── run_m1.py                # entry point: EDA, split, assert kebocoran, baseline
```

Spec serving bagian 3.2 menempatkan loading, group split, dan pemeriksaan
kebocoran seluruhnya di `data.py`. Dokumen ini memisahkan `split.py` sebagai file
sendiri. Deviasi ini diambil karena keputusan 3.2: fungsi split harus dapat
diimpor tanpa menarik `datasets`, dan itu tidak mungkin bila ia tinggal di modul
yang sama dengan loader.

Nama `train.py`, `calibrate.py`, `evaluate.py`, dan `export_onnx.py` di spec
serving tetap berlaku dan belum dibuat di fase ini. `nela-features` juga belum
dipin: ia baru dipakai di M3, dan `transformers` dipin lebih dulu hanya karena
ikut terpasang bersama torch di keputusan 3.4.

`.gitignore` menambahkan `.venv-train/`. `data/` sudah ada di sana. `training/`
sudah ada di `.dockerignore`.

---

## 5. Alur data M1

```
load_dataset("mediabiasgroup/BABE")
  -> gabungkan split train dan test menjadi satu pool
  -> buang baris tanpa news_link
  -> buang baris dengan agreement rendah (label_opinion == "No agreement")
  -> normalisasi news_link menjadi kunci grup
  -> group split by kunci grup  ->  fold
  -> assert: tidak ada kunci grup yang muncul di dua fold
  -> baseline TF-IDF + logistic regression
```

Split resmi BABE (3.121/1.000) digabung lebih dulu, bukan dipakai apa adanya.
Alasannya ada di bagian 6.1: split itu belum diketahui bersih terhadap
`news_link`, dan memakai split yang bocor menghasilkan angka yang tidak layak
dilaporkan. Bila ternyata bersih, split resmi dipertahankan agar angka project
ini dapat dibandingkan dengan literatur BABE.

---

## 6. Yang diputuskan oleh data, bukan oleh dokumen ini

Tiga hal berikut tidak bisa dijawab dari luar data. M1 menjawabnya, dan tiap
cabang punya konsekuensi yang ditentukan lebih dulu agar hasilnya tidak
ditafsirkan setelah melihat angkanya.

### 6.1 Apakah split resmi BABE bersih terhadap `news_link`

- **Bersih** — split resmi dipertahankan. Angka yang dilaporkan dapat
  dibandingkan dengan literatur BABE.
- **Bocor** — split resmi dibuang, split sendiri dibuat dari pool gabungan.
  Ketidakcocokan dengan angka literatur dicatat di README sebagai konsekuensi,
  bukan disembunyikan.

### 6.2 Apakah `label` hanya bernilai 0 dan 1

Sampel 100 baris hanya menampilkan 0 dan 1. Rilis BABE lain pernah memuat
kategori "No agreement" pada label utama.

- **Hanya 0/1** — tidak ada tindakan.
- **Ada nilai lain** — baris tersebut dibuang sebagai agreement rendah, jumlahnya
  dilaporkan, dan keputusan ini masuk README sesuai PRD 5.1.

### 6.3 Berapa outlet distinct, dan berapa baris tanpa `news_link`

PRD 8.3 mensyaratkan minimal 12–15 outlet agar korelasi outlet bermakna.

- **12 outlet atau lebih** — analisis 8.3 tetap di rencana M4a.
- **Kurang dari 12** — 8.3 dinyatakan tidak dapat dijalankan pada BABE saja, dan
  keterbatasan itu dicatat sekarang, bukan ditemukan di M4a.

Jumlah baris tanpa `news_link` dilaporkan apa adanya. Bila proporsinya besar,
ukuran dataset efektif turun dan itu mempengaruhi ekspektasi F1.

---

### 6.4 Hasil, diukur pada 2026-08-27

Dijalankan lewat `.venv-train/Scripts/python.exe -m training.run_m1`.

```
baris mentah                : 4.121
dibuang, news_link kosong   : 32
dibuang, agreement rendah   : 215
dibuang, label di luar 0/1  : 0
dipakai                     : 3.874
distribusi                  : 2.117 biased (54,6%), 1.757 neutral (45,4%)
news_link distinct          : 2.240  ->  2.219 setelah normalisasi
grup setelah filter         : 2.151
outlet distinct             : 18
lean                        : 1.431 None, 902 right, 884 left, 657 center
ukuran fold                 : 751, 883, 743, 770, 727
F1 macro baseline           : 0,7420 +/- 0,0194
```

Jawaban atas ketiga pertanyaan:

**6.1 — split resmi BABE bocor.** 384 dari 2.151 kunci grup muncul di sisi train
dan test resmi sekaligus. Cabang yang sudah ditetapkan berlaku: split resmi
dibuang, fold dibuat sendiri dari pool gabungan, dan konsekuensinya dicatat.

Yang boleh dan tidak boleh disimpulkan dari angka ini perlu dipisahkan. Yang
terukur adalah bahwa split yang didistribusikan di HuggingFace tidak bersih
terhadap `news_link`. Yang **tidak** terukur adalah bagaimana angka di paper BABE
dihasilkan — penulisnya bisa saja memakai skema split lain. Karena itu
pernyataan yang layak masuk README adalah bahwa angka project ini dihitung pada
fold yang group-clean dan tidak dapat dibandingkan langsung dengan angka mana pun
yang dihitung pada split resmi, bukan tuduhan bahwa angka literatur keliru.

**6.2 — `label` hanya bernilai 0 dan 1.** Tidak ada tindakan; 0 baris dibuang
karena label tak terduga.

**6.3 — 18 outlet, di atas ambang 12.** Analisis korelasi outlet PRD 8.3 tetap
dapat dijalankan, dengan dua catatan yang baru terlihat di sini: distribusinya
sangat berat di tiga outlet teratas (Breitbart 904, Reuters 809, Alternet 748 —
64% dari seluruh baris), dan kolom `type` yang menjadi sumber lean kosong pada
1.431 baris, yaitu 37%. Keduanya menyusutkan n efektif untuk 8.3 dan harus
dilaporkan bersama nilai korelasinya.

Satu angka lain yang mengubah rencana M2: baseline TF-IDF sudah mencapai 0,7420,
sementara target PRD 8.1 untuk model yang di-fine-tune adalah 0,78. Margin yang
harus dilewati M2 karena itu hanya sekitar 0,04 — jauh lebih sempit daripada yang
tersirat saat kriteria berhenti M2 ditulis, dan itu perlu diperhitungkan saat
memutuskan apa yang dianggap "mengungguli baseline secara meyakinkan".

---

## 7. Baseline TF-IDF

TF-IDF pada kata, diikuti logistic regression, dievaluasi pada fold yang sama
dengan yang akan dipakai DeBERTa.

Perannya bukan pembanding formalitas. Kriteria berhenti M2 di PRD 9 berbunyi:
bila DeBERTa tidak mengungguli baseline ini secara meyakinkan, yang diperiksa
adalah kualitas data, bukan hyperparameter. Baseline ini yang membuat instruksi
itu dapat dijalankan.

Metrik yang dilaporkan adalah F1 macro, sama dengan metrik utama PRD 8.1, agar
angka baseline dan angka DeBERTa dapat dibandingkan langsung tanpa konversi.

Baseline dijalankan pada seluruh fold dan dilaporkan sebagai mean dan deviasi,
bukan satu angka. Pada dataset sekecil ini satu angka tunggal menyesatkan dengan
cara yang sama seperti yang dijelaskan PRD 6 untuk seed.

---

## 8. Pengujian

| Test | Lokasi | Berjalan di CI |
|---|---|---|
| Tidak ada kunci grup di dua fold, pada fixture sintetis | `tests/test_split.py` | ya |
| Fold berjumlah sesuai permintaan dan seluruh baris terpakai | `tests/test_split.py` | ya |
| Split deterministik: input sama menghasilkan fold sama | `tests/test_split.py` | ya |
| Kebocoran pada data BABE nyata | assert di `run_m1.py` | tidak, butuh data |
| Paritas ONNX terhadap PyTorch | `training/`, di M5 | tidak, butuh torch |

Test di `tests/` berjalan di job CI yang sudah ada tanpa perubahan pada `ci.yml`
dan tanpa tambahan pada `requirements-dev.txt`. Itu konsekuensi langsung dari
keputusan 3.2.

Assert di `run_m1.py` sengaja berupa assert yang menghentikan proses, bukan
peringatan yang tercetak. Split yang bocor membuat seluruh angka setelahnya tidak
layak dilaporkan, jadi tidak ada alasan melanjutkan.

Keluaran M1 dicetak ke stdout. Tidak ada generator laporan: angkanya disalin ke
README sesuai kewajiban transparansi PRD 5.1, dan sebuah pembangkit laporan tidak
menambah apa pun pada kewajiban itu.

---

## 9. Batas berhenti fase ini

Fase ini selesai ketika:

- `.venv-train/` ada, torch mengenali GPU, dan smoke check GPU lulus
- `run_m1.py` berjalan sampai selesai di atas data BABE nyata
- assert anti-kebocoran lulus
- test di `tests/test_split.py` lulus bersama 49 test serving yang sudah ada
- angka baseline TF-IDF ada, beserta jawaban atas ketiga pertanyaan di bagian 6

Yang tidak dikerjakan di fase ini: fine-tune DeBERTa, sweep learning rate, dan
seluruh M2 ke atas. Semuanya mendapat rencana tersendiri, ditulis setelah angka
M1 diketahui.

---

## 10. Dokumen yang dikoreksi

| Dokumen | Koreksi |
|---|---|
| PRD 5.2 | `article_id` menjadi `news_link`, beserta syarat normalisasi dan penanganan baris kosong |
| PRD 12.1 | verifikasi lisensi HF selesai: tetap tanpa lisensi |
| Spec serving 3.2 | `split.py` dicatat sebagai file tambahan di `training/` |
| Plan serving, bagian tertunda | "group split per artikel" mengikuti koreksi PRD 5.2 |

Koreksi dokumen dilakukan sebagai commit `docs:` tersendiri, terpisah dari commit
kode.

---

## 11. Risiko

| Risiko | Mitigasi |
|---|---|
| `news_link` tidak setara `article_id`, satu artikel terpecah menjadi beberapa grup | Normalisasi URL diukur di M1; jumlah grup sebelum dan sesudah dilaporkan |
| Split resmi BABE ternyata bocor dan angka tidak lagi sebanding dengan literatur | Konsekuensinya ditetapkan lebih dulu di 6.1, bukan setelah melihat angka |
| Wheel CUDA tidak cocok dengan driver, atau 6 GB tidak cukup | Diuji di M1 lewat smoke check, sebelum rencana M2 ditulis |
| Torch bocor ke serving | Venv terpisah, ditambah pemeriksaan CI dan `.dockerignore` yang sudah ada |
| Dataset ikut ter-commit | `data/` di `.gitignore`; tidak ada langkah yang menyalinnya ke `src/` atau image |
| Filter agreement membuang terlalu banyak baris | Jumlah sebelum dan sesudah filter dilaporkan; bila dataset efektif terlalu kecil, WNC sebagai cadangan sudah dicatat di PRD 5.0 |

---

## 12. Di luar cakupan fase ini

- Fine-tune DeBERTa dan seluruh M2 ke atas
- BASIL, held-out set buatan sendiri, dan korelasi outlet — data eval, bukan M1
- NELA features sebagai baseline terpisah — M3
- Stratifikasi per outlet, kecuali bila EDA membuktikan perlu
- Perubahan apa pun pada `src/neutral/`
