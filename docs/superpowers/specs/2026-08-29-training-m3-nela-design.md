# Design Spec: Media Neutrality Classifier — Fase Training (M3)

**Tanggal:** 2026-08-29
**Status:** Draft untuk review
**Terkait:** `PRD_Media_Neutrality_Classifier.md`, `docs/superpowers/specs/2026-08-27-training-phase-design.md`, `docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md`

---

## 1. Konteks dan cakupan

M2 menjawab seberapa jauh DeBERTa mengungguli bag-of-words. Dokumen ini
menjawab pertanyaan lanjutannya: keunggulan itu datang dari mana. Bila fitur
linguistik eksplisit sudah cukup untuk mendekati DeBERTa, arsitektur transformer
tidak menjelaskan apa pun yang tidak dijelaskan hitungan kata kerja factive dan
hedge. Bila fitur itu tertinggal jauh, keunggulan DeBERTa datang dari konteks.

PRD 6.1 sudah menutup pilihan fiturnya: NELA features dari Horne et al. 2018,
dipakai sebagai model baseline terpisah dengan logistic regression di atasnya,
bukan dikonkatenasi ke dalam DeBERTa. Yang belum ditutup adalah daftar
konfigurasi yang dibandingkan, cara fitur diekstrak dan disimpan, penanganan
fitur yang mati di teks sependek kalimat, dan bentuk laporannya. Empat hal itu
isi dokumen ini.

**Cakupan implementasi: M3 saja.** Tidak ada training ulang DeBERTa. Pekerjaan
berhenti sebelum cross-dataset eval (M4a), temperature scaling dan ablation
shortcut learning (M4b), dan ekspor ONNX (M5).

---

## 2. Titik awal: angka yang mengikat

Tiga angka dari milestone sebelumnya menjadi pembanding tetap M3. Ketiganya
diukur pada pembagian fold yang sama, dibangun `group_split()` dengan seed 0.

```
TF-IDF + logistic regression : F1 macro 0,7420 +/- 0,0194  (M1, 5 fold)
DeBERTa fine-tuned           : F1 macro 0,8477 +/- 0,0277  (M2, 5 fold x 5 seed)
selisih                      : +0,1058
```

Angka DeBERTa dibaca ulang dari `runs/m2_runs.jsonl`, bukan disalin dari README.
Filter `lr == 3e-5` menyisakan tepat 25 run final; dua run sweep pada 1e-5 dan
2e-5 tidak ikut. Mean per fold dihitung dari 5 seed di fold itu.

Angka TF-IDF dihitung ulang di M3 lewat `run_baseline()` yang sudah ada, bukan
disalin. Alasannya ada di bagian 9: hasil hitung ulang itu berfungsi sebagai
pemeriksaan bahwa pembagian fold belum bergeser.

---

## 3. Keputusan yang ditutup

### 3.1 Empat konfigurasi, satu tabel

| Konfigurasi | Matriks desain | Sumber angka |
|---|---|---|
| TF-IDF + LR | unigram dan bigram, `min_df=2`, `sublinear_tf` | dihitung ulang di M3 |
| NELA + LR | 87 fitur, kolom varians-nol dibuang, distandardisasi | dihitung di M3 |
| TF-IDF + NELA + LR | keduanya disambung `scipy.sparse.hstack` | dihitung di M3 |
| DeBERTa | — | dibaca dari `runs/m2_runs.jsonl` |

Konfigurasi gabungan awalnya tidak direncanakan. Ia masuk karena tiga titik saja
tidak cukup untuk menunjuk asal selisih. Bila NELA tertinggal dari DeBERTa, tanpa
konfigurasi gabungan tidak bisa dibedakan dua kemungkinan: fitur linguistik
memang lemah, atau fitur itu kuat tetapi tumpang tindih dengan sinyal yang sudah
ditangkap bag-of-words. Konfigurasi gabungan memisahkan keduanya dengan satu
matriks tambahan.

Batasnya: gabungan adalah satu konfigurasi, bukan proyek tersendiri. Skala,
regularisasi, dan `class_weight` sama dengan tiga model lain, tanpa sweep
sendiri. Model yang perlu di-tuning agar terlihat baik tidak lagi sebanding
dengan model yang tidak di-tuning.

Ketiga logistic regression memakai setelan identik dengan M1:
`class_weight="balanced"`, `max_iter=1000`, tanpa sweep. Metriknya F1 macro, sama
dengan PRD 8.1, sehingga keempat angka sebanding tanpa konversi.

### 3.2 Breakdown per grup fitur

Enam run tambahan, satu per grup NELA, masing-masing 5 fold. Yang dilaporkan:
nama grup, jumlah fitur, jumlah yang bertahan setelah pembuangan varians-nol, dan
F1 macro mean +/- std.

PRD 10 mencantumkan baris risiko "ablation dilewati karena terasa tidak penting",
dengan mitigasi "dijadwalkan eksplisit di M3". Tabel ini yang menjawab baris itu.
Yang ingin diketahui: sinyalnya terkumpul di grup mana, dan apakah satu grup saja
sudah mendekati NELA penuh. Bila iya, 87 fitur adalah kemasan untuk sinyal yang
jauh lebih kecil, dan itu perlu diketahui sebelum fitur ini pernah
dipertimbangkan masuk jalur serving.

Total 45 fit logistic regression: 5 fold dikali 9 konfigurasi, yaitu tiga model
yang dihitung di bagian 3.1 ditambah enam grup. Angka NELA penuh dipakai di
kedua tabel dan tidak dihitung dua kali. Hitungan detik setelah cache terisi.

### 3.3 Fitur diekstrak sekali dan disimpan ke disk

Ekstraksi memakan 714 ms per kalimat, atau sekitar 46 menit untuk 3.874 baris.
Angka itu diukur pada 300 kalimat BABE, bukan diperkirakan. Tanpa penyimpanan, 50
fit berarti 50 kali ekstraksi.

Hasilnya disimpan ke `data/nela_cache.npz`, dengan kunci hash SHA-256 dari daftar
teks. Filter data yang berubah menghasilkan hash berbeda, sehingga cache lama
tidak pernah terpakai diam-diam. `data/` sudah ada di `.gitignore`, jadi cache
tidak bisa ikut terkomit.

Ekstraksi ditulis sekali di akhir. Crash di menit ke-40 berarti mengulang dari
nol. Ini diterima dengan sadar, ditandai komentar `ponytail:` di kodenya, dan
upgradenya menulis per potongan bila ternyata mengganggu.

### 3.4 Grup `event` dipertahankan meski memakan 92% waktu ekstraksi

Biaya per grup, diukur pada 40 kalimat:

```
style          19,7 ms/kalimat
complexity      2,2 ms/kalimat
bias            9,6 ms/kalimat
affect         27,9 ms/kalimat
moral           1,7 ms/kalimat
event         762,9 ms/kalimat
```

`event` memakai NER chunker nltk dan menyumbang 2 fitur, satu di antaranya
konstan pada teks sepanjang kalimat. Membuangnya menurunkan ekstraksi dari 46
menit ke sekitar 4 menit.

Grup itu tetap dipakai. Biayanya dibayar sekali per milestone, bukan per run, dan
cache membuat sisa pekerjaan tetap dalam hitungan detik. Membuangnya menghemat 42
menit sekali dan menambah catatan kaki yang harus ikut di setiap tabel
sesudahnya, termasuk di README M7.

### 3.5 Fitur mati dibuang di dalam fold, bukan sebelumnya

NELA dirancang untuk artikel utuh. Pada 300 kalimat BABE, 15 dari 87 fitur
konstan nol dan 11 lagi nol di lebih dari 95% baris. Yang mati hampir semuanya di
grup style: tag POS yang jarang muncul dalam satu kalimat (`FW`, `LS`, `SYM`,
`UH`) dan penghitung tanda baca.

Kolom varians-nol dibuang `VarianceThreshold`, di-fit hanya pada bagian train di
dalam tiap iterasi fold. Mem-fit-nya pada seluruh data akan memakai informasi
test untuk memutuskan kolom mana yang hidup, dan itu kebocoran yang tidak
memunculkan error apa pun. Alasan yang sama berlaku untuk `StandardScaler` dan
`TfidfVectorizer`.

Ekstraksi fiturnya sendiri boleh dilakukan di luar loop, karena tiap kalimat
diproses sendiri tanpa melihat kalimat lain.

### 3.6 Loop fold manual, bukan `ColumnTransformer`

Konfigurasi gabungan butuh dua jenis input dalam satu model: teks mentah untuk
`TfidfVectorizer` dan matriks numerik untuk NELA. Menjodohkan keduanya lewat
`ColumnTransformer` menuntut DataFrame, yang berarti pandas masuk hanya untuk
keperluan ini. Loop fold manual yang membangun matriks per konfigurasi memakan
sekitar sepuluh baris dan tidak menambah dependency.

---

## 4. Struktur `training/` setelah M3

```
training/
  split.py                tidak berubah
  data.py                 tidak berubah
  baseline_tfidf.py       tidak berubah, run_baseline() dipakai ulang
  train.py                tidak berubah
  run_m1.py               tidak berubah
  run_m2.py               tidak berubah
  nela.py                 BARU  ekstraksi fitur, unduhan resource nltk, cache
  run_m3.py               BARU  loop fold, 4 konfigurasi, 6 grup, laporan
  requirements-train.txt  nela-features dan turunannya dipin
```

`nela.py` mengekspor satu fungsi publik, `extract_features(texts)`, yang
mengembalikan matriks 87 kolom, nama fitur, dan peta kolom ke grup. Unduhan lima
resource nltk dilakukan di dalamnya, idempoten dan diam.

Serving tidak tersentuh. `src/neutral/`, `requirements.txt`, `Dockerfile`, dan
`.dockerignore` tidak berubah satu baris pun.

---

## 5. Alur `run_m3.py`

1. `load_babe()`, lalu `group_split()` dengan seed 0. Fold M3 identik dengan fold
   M1 dan M2 karena seed dan fungsi hash-nya sama.
2. Assert anti-kebocoran fold: tidak ada kunci grup yang muncul di dua fold.
3. `extract_features()`. Baca cache bila hash cocok, ekstrak dan tulis bila tidak.
4. Hitung ulang baseline TF-IDF lewat `run_baseline()`. Assert hasilnya cocok
   dengan angka M1. Bila tidak cocok, pembagian fold sudah berubah dan seluruh
   perbandingan M3 tidak sah, jadi program berhenti di sini.
5. Loop fold untuk NELA dan untuk gabungan.
6. Loop fold untuk enam grup.
7. Baca `runs/m2_runs.jsonl`, filter `lr == 3e-5`, rata-ratakan per `held_out`.
8. Cetak tabel perbandingan empat model dan tabel breakdown enam grup ke stdout.

Tidak ada file log run. M2 memerlukan `m2_runs.jsonl` karena 27 run memakan dua
jam dan harus dapat dilanjutkan setelah crash. Di M3 bagian mahalnya hanya
ekstraksi, dan itu sudah dijaga cache; log run di sini berarti memelihara file
yang tidak pernah dibaca.

---

## 6. Kriteria berhenti M3

PRD 9 menetapkan kriterianya sebagai "delta dilaporkan apa adanya, termasuk bila
kecil". Tidak ada ambang yang harus dilewati, dan tidak ada hasil yang membuat M3
gagal.

Yang bisa gagal di milestone ini bukan angkanya, melainkan pelaporannya. Karena
itu kewajiban M3 dinyatakan sebagai tempat, bukan sebagai nilai: angka masuk ke
bagian hasil spec ini, bagian hasil README, dan ringkasan CLAUDE.md, sama seperti
yang dilakukan M2.

Dua kemungkinan hasil sudah punya bacaannya masing-masing. Bila NELA mendekati
DeBERTa, PRD 6.1 sudah menyebutnya temuan menarik untuk README, bukan kegagalan.
Bila NELA tertinggal dan gabungan TF-IDF+NELA juga tidak mendekat, kesimpulannya
keunggulan DeBERTa datang dari konteks, bukan dari fitur permukaan.

---

## 7. Yang dilaporkan

### 7.1 Dua tabel

Tabel pertama, empat model pada 5 fold: F1 macro mean +/- std, delta terhadap
TF-IDF, dan angka per fold.

Tabel kedua, enam grup fitur: jumlah fitur, jumlah yang bertahan, F1 macro
mean +/- std.

### 7.2 Std keempat model tidak mengukur hal yang sama

Ini harus tertulis di laporan, bukan disimpulkan pembaca.

Std DeBERTa (+/- 0,0277) mencampur sebaran antar-fold dan antar-seed dari 25 run.
Std ketiga model logistic regression hanya sebaran antar-fold dari 5 run, karena
logistic regression dengan lbfgs pada data tetap bersifat deterministik dan seed
tidak menghasilkan variasi apa pun. Kedua angka itu tidak sebanding sebagai
ukuran ketidakpastian.

Yang tetap sebanding adalah mean per fold, karena keempatnya dihitung pada
pembagian data yang sama. Delta antar-model dilaporkan dari mean itu.

### 7.3 Koreksi PRD

PRD 6.1 menyebut 141 fitur linguistik. `nela-features==3.0.1` mengembalikan 87
dari `extract_all`: style 50, complexity 7, bias 8, affect 9, moral 11, event 2.
PRD dikoreksi sebagai bagian dari M3, dengan angka 141 tetap disebut sebagai
angka paper Horne et al. 2018 agar jejaknya tidak hilang. Polanya sama dengan
cara M1 mengoreksi asumsi `article_id` yang ternyata tidak ada di BABE.

---

## 8. Pengujian

`nela.py` mendapat `_self_check()` di bawah `__main__`, mengikuti pola
`baseline_tfidf.py`. Tiga hal diperiksa pada data yang jawabannya sudah
diketahui:

1. Ekstraksi mengembalikan 87 kolom dengan nama yang cocok dengan peta grup.
2. Cache yang ditulis lalu dibaca menghasilkan matriks identik.
3. Hash yang berbeda menolak cache lama.

Tidak ada test di `tests/`. Direktori itu dijalankan di venv serving, yang tidak
punya `nela-features` dan tidak boleh punya. Batas yang sama sudah berlaku untuk
`baseline_tfidf.py` dan `train.py`.

Sebelum commit: 64 test yang sudah ada tetap lulus dan ruff bersih. M3 tidak
menyentuh apa pun yang mereka uji, jadi keduanya berfungsi sebagai pemeriksaan
bahwa cakupan tidak melebar.

---

## 9. Risiko

| Risiko | Mitigasi |
|---|---|
| Kebocoran lewat preprocessing yang di-fit pada seluruh data | `VarianceThreshold`, `StandardScaler`, dan `TfidfVectorizer` di-fit di dalam fold, hanya pada indeks train |
| Pembagian fold bergeser tanpa disadari | Baseline TF-IDF dihitung ulang dan di-assert cocok dengan 0,7420 +/- 0,0194 |
| Cache basi terpakai setelah filter data berubah | Kunci cache adalah hash SHA-256 dari daftar teks |
| Ekstraksi 46 menit hilang karena crash | Diterima. Ditandai komentar `ponytail:`; upgradenya menulis per potongan |
| M3 gagal di mesin bersih karena resource nltk | Lima resource diunduh di awal `nela.py`, idempoten |
| Fitur mati membuat angka NELA terlihat lebih buruk dari seharusnya | 26 dari 87 fitur konstan atau hampir konstan; yang varians-nol dibuang, sisanya dilaporkan apa adanya |
| Cakupan melebar ke tuning | Bagian 10 menyatakan tuning di luar cakupan; setelan LR dikunci sama dengan M1 |

---

## 10. Di luar cakupan fase ini

- Tuning hyperparameter untuk model mana pun, termasuk konfigurasi gabungan
- Seleksi fitur selain pembuangan varians-nol
- Koefisien dan daftar fitur teratas. Ditunda, dan dapat diambil belakangan dari
  model yang sudah dilatih di sini tanpa training ulang
- Konkatenasi fitur NELA ke dalam DeBERTa. PRD 6.1 sudah membatalkannya
- Cross-dataset eval ke BASIL (M4a)
- Temperature scaling, ECE, ablation shortcut learning, contrast set (M4b)
- Ekspor ONNX dan uji paritas (M5)

---

## 11. Hasil yang terukur

Diukur 2026-08-29 pada 3.874 baris, fold `[751, 883, 743, 770, 727]`, dari
`training/run_m3.py`. Baseline TF-IDF hasil hitung ulang persis 0,7420 +/- 0,0194,
sama dengan angka M1, jadi pembagian fold tidak bergeser dan keempat kolom di
bawah dihitung pada data yang sama.

### 11.1 Empat model

| Model | F1 macro | Delta TF-IDF | Per fold |
|---|---|---|---|
| TF-IDF + LR | 0,7420 +/- 0,0194 | +0,0000 | 0,7445 0,7095 0,7449 0,7614 0,7496 |
| NELA + LR | 0,6837 +/- 0,0116 | -0,0583 | 0,6960 0,6712 0,6802 0,6753 0,6959 |
| TF-IDF + NELA + LR | 0,7294 +/- 0,0130 | -0,0126 | 0,7252 0,7163 0,7405 0,7193 0,7455 |
| DeBERTa | 0,8477 +/- 0,0195 | +0,1058 | 0,8693 0,8397 0,8184 0,8550 0,8563 |

Kolom NELA yang bertahan setelah pembuangan varians-nol: 76 dari 87, sama di
kelima fold. Kolom DeBERTa dirata-ratakan dari 25 run pada `lr=3e-05`, dibaca
dari `runs/m2_runs.jsonl`.

### 11.2 Enam grup fitur

| Grup | Fitur | Bertahan | F1 macro |
|---|---|---|---|
| style | 50 | 40 | 0,6599 +/- 0,0089 |
| affect | 9 | 9 | 0,6194 +/- 0,0211 |
| bias | 8 | 8 | 0,5884 +/- 0,0240 |
| complexity | 7 | 7 | 0,5382 +/- 0,0375 |
| moral | 11 | 11 | 0,5028 +/- 0,0310 |
| event | 2 | 1 | 0,4401 +/- 0,0175 |

Lantai perbandingannya: prediktor konstan pada data ini memberi F1 macro 0,3534
bila selalu menebak `biased` dan 0,3120 bila selalu `neutral`.

### 11.3 Bacaan

**Keunggulan DeBERTa tidak datang dari fitur permukaan.** NELA tertinggal 0,0583
di bawah TF-IDF dan 0,1640 di bawah DeBERTa. Hitungan kata kerja factive, hedge,
dan kata bias tidak menjelaskan selisih yang diukur M2.

**Konfigurasi gabungan menutup kemungkinan kedua.** Bagian 3.1 menyebut dua
bacaan yang tidak bisa dipisahkan tanpa matriks gabungan: fitur linguistik memang
lemah, atau fitur itu kuat tetapi tumpang tindih dengan sinyal yang sudah
ditangkap bag-of-words. Jawabannya yang pertama. Menyambung 76 kolom NELA ke
TF-IDF bukan hanya tidak menambah apa pun, tetapi menurunkan angkanya 0,0126.
Kolom padat yang distandardisasi ikut menarik bobot dari ribuan kolom TF-IDF yang
jarang, dan yang dibawanya tidak sepadan.

**Sinyal NELA terkumpul di satu grup.** Style sendiri mencapai 0,6599 dari 0,6837
milik 87 fitur penuh, dengan 40 kolom. Lima grup sisanya menyumbang 0,0238
di atas grup terbesar. Bila fitur ini pernah dipertimbangkan masuk jalur serving,
yang dibawa cukup grup style; itu juga membuang grup `event`, yang memakan 92%
waktu ekstraksi.

**Grup `event` tidak mengukur apa pun pada teks sependek kalimat.** Satu dari dua
kolomnya konstan dan terbuang, dan yang tersisa memberi 0,4401, sekitar 0,09 di
atas prediktor konstan. Itu membenarkan keputusan 3.4 dari arah yang berlawanan
dengan alasan aslinya: grup itu dipertahankan supaya tidak ada catatan kaki di
tabel-tabel berikutnya, dan hasilnya adalah baris yang bisa dilaporkan apa adanya
alih-alih lubang yang harus dijelaskan.

**Tidak ada yang gagal di sini.** Bagian 6 sudah menetapkan tidak ada ambang yang
harus dilewati. PRD 6.1 menyiapkan bacaan untuk kemungkinan NELA mendekati
DeBERTa; yang terjadi kebalikannya, dan kesimpulannya sudah tertulis di bagian 6:
keunggulan DeBERTa datang dari konteks, bukan dari fitur permukaan.
