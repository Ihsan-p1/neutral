# Design Spec: Media Neutrality Classifier — Fase Training (M4a)

**Tanggal:** 2026-08-31
**Status:** Draft untuk review
**Terkait:** `PRD_Media_Neutrality_Classifier.md`, `docs/superpowers/specs/2026-08-27-training-phase-design.md`, `docs/superpowers/specs/2026-08-28-training-m2-finetune-design.md`, `docs/superpowers/specs/2026-08-29-training-m3-nela-design.md`

---

## 1. Konteks dan cakupan

M2 mengukur seberapa jauh DeBERTa mengungguli bag-of-words pada BABE. M3
menunjukkan keunggulan itu tidak datang dari fitur permukaan. Keduanya diukur
pada satu dataset, dengan fold yang dibangun dari pool BABE. Dokumen ini menjawab
pertanyaan yang tidak bisa dijawab dataset itu sendiri: apakah yang dipelajari
model berlaku di luar BABE, atau yang terukur selama ini adalah distribusi BABE.

PRD 8.2 sudah menutup beberapa pilihan. BASIL adalah cross-dataset test set,
bukan data latih. Angkanya dilaporkan apa adanya walaupun turun, sebagai bagian
limitation section. Dan karena BASIL beranotasi span-level dalam konteks artikel
sementara BABE berisi kalimat berdiri sendiri, penurunan F1 sebagian merupakan
artefak ketidakcocokan label — karena itu dilaporkan dua angka: apa adanya, dan
setelah threshold direkalibrasi pada dev split BASIL.

Yang belum ditutup dan menjadi isi dokumen ini: dari mana data BASIL diambil,
bagaimana span dipetakan ke label kalimat dan bagaimana pemetaan itu
diverifikasi, bagaimana dev/test BASIL dibagi, model mana yang dievaluasi, dan
bentuk laporannya.

**Cakupan implementasi: M4a saja, dan hanya bagian 8.2.** Tabel milestone PRD
menempatkan validasi korelasi outlet (8.3) di M4a juga. Dua analisis itu tidak
berbagi apa pun — beda data, beda metrik, beda risiko — sehingga 8.3 dikeluarkan
dan dijadwalkan terpisah. Tidak ada training di M4a.

---

## 2. Titik awal: angka yang mengikat

Empat angka dari milestone sebelumnya menjadi pembanding tetap. Semuanya diukur
pada pembagian fold yang sama, dibangun `group_split()` dengan seed 0.

```
BABE, baris terpakai         : 3.874  (54,6% biased)
TF-IDF + logistic regression : F1 macro 0,7420 +/- 0,0194  (M1, 5 fold)
DeBERTa fine-tuned           : F1 macro 0,8477 +/- 0,0277  (M2, 5 fold x 5 seed)
NELA + logistic regression   : F1 macro 0,6837 +/- 0,0116  (M3, 5 fold)
```

Checkpoint yang dievaluasi adalah `models/deberta-babe`, satu-satunya yang
bertahan di disk: fold 0, seed 3, F1 macro 0,8765 pada test split fold 0-nya
sendiri, val loss 0,3331 (`models/deberta-babe/m2_selection.json`).
`run_m2.py:211` menghapus `CANDIDATE_DIR` setelah seleksi, jadi 24 checkpoint
lainnya tidak ada lagi dan mereproduksinya berarti training ulang.

---

## 3. Keputusan yang ditutup

### 3.1 Sumber data: rilis resmi, karena BASIL membawa segmentasi kalimatnya sendiri

Sumber tunggal label adalah rilis resmi `launchnlp/BASIL`, diunduh sebagai satu
tarball ke `data/basil/`. Struktur berikut diperiksa langsung pada 2026-08-31,
bukan disimpulkan dari paper:

```
articles/<tahun>/<triplet-uuid>_<1|2|3>.json
  body-paragraphs : list paragraf, tiap paragraf berupa list KALIMAT
  triplet-uuid    : id event      uuid : id artikel      source : 'nyt' | ...

annotations/<tahun>/<triplet-uuid>_<1|2|3>_ann.json
  phrase-level-annotations : [{id: 'p0', bias: 'lex' | 'inf',
                               start: 81, end: 87, txt: 'fierce', quote: 'no', ...}]
```

Terhitung dari daftar isi repo: 300 artikel, 300 file anotasi, 100 triplet, tahun
2010 sampai 2019.

Dua fakta di struktur itu menentukan seluruh keputusan berikutnya. Pertama,
`body-paragraphs` adalah list dari list kalimat, jadi segmentasi kalimat datang
dari dataset dan tidak perlu dipilih. Kedua, `id` pada span bukan sekadar nama:
`p{k}` menunjuk kalimat ke-`k` pada daftar kalimat yang sudah diratakan, dan
`start`/`end` relatif terhadap kalimat itu. Diuji pada artikel contoh, 5 dari 5
span memenuhi `sents[k][start:end] == txt`.

Konsekuensinya, memetakan span ke label kalimat bukan pekerjaan perkiraan
melainkan pembacaan indeks. Alternatifnya — memakai re-upload
`horychtom/BASIL` di HuggingFace (8.018 kalimat, kolom `text`, `lex_label`,
`info_label`, `original_label`) sebagai sumber label — ditolak karena menukar
indeks yang pasti dengan pemetaan pihak ketiga yang tidak terdokumentasi, dan
menambah kode pencocokan teks untuk label yang bisa dihitung langsung.

Versi HuggingFace tetap dipakai satu kali, sebagai perbandingan sekali jalan di
3.3. Perannya melaporkan seberapa dekat label project ini dengan label yang
beredar, bukan menghasilkan label.

### 3.1b Aturan pemetaan

PRD 8.2 mensyaratkan aturan pemetaan span ke kalimat dinyatakan eksplisit.
Aturannya: **kalimat ke-`k` dilabeli biased bila ada span dengan `bias == 'lex'`
yang `id`-nya `p{k}`.** Untuk baris kedua tabel (3.2), `bias` yang diterima
`'lex'` atau `'inf'`.

Kalimat diratakan dari `body-paragraphs` dengan urutan paragraf dipertahankan,
karena urutan itulah yang dipakai indeks `p{k}`.

### 3.2 Label positif: `lex_label` utama, `lex OR info` sebagai baris kedua

Angka utama memakai `lex_label`. Itu definisi tugas project ini: bias leksikal
tingkat kalimat, pilihan kata yang bermuatan. Bias informational — penyertaan
atau penghilangan fakta secara selektif — adalah tugas lain yang model ini tidak
pernah dilatihkan.

`lex_label OR info_label` dilaporkan sebagai baris kedua pada tabel yang sama.
Biayanya nol: prediksinya identik, yang berbeda hanya vektor gold. Nilainya ada
pada selisih dua baris itu. Kalau F1 pada gold gabungan jauh lebih tinggi,
sebagian penurunan berasal dari kalimat yang BASIL sebut informational sementara
model menandainya biased — artefak taxonomy, bukan kegagalan generalisasi. Kalau
kedua baris sama-sama rendah, penjelasan itu tidak tersedia.

Baris kedua tidak pernah menjadi angka utama. Menilai model pada gold yang memuat
tugas yang tidak pernah dilatihkan akan menghasilkan angka yang lebih enak dibaca
dan lebih tidak jujur.

### 3.3 Perbandingan sekali jalan terhadap versi HuggingFace

Label sudah selesai dihitung dari rilis resmi sebelum bagian ini berjalan, jadi
perbandingan ini tidak pernah menghasilkan atau mengubah label. Yang dihasilkan
satu baris catatan di laporan.

Baris HuggingFace dicocokkan ke kalimat resmi lewat teks yang dinormalisasi:
whitespace diciutkan, tanda kutip typografis diseragamkan ke ASCII, casing
dipertahankan. Titik ketidakcocokan yang sudah diketahui ada dua — README rilis
kedua menyebut tanda kutip ikut disertakan ketika kalimat biased berupa kutipan,
dan teks BASIL memuat karakter non-ASCII (em dash, curly quote) yang bisa
berbeda encoding antar rilis.

Tiga angka dilaporkan: jumlah kalimat di kedua sisi (sekitar 8.100 dari rilis
resmi lawan 8.018 di HuggingFace), proporsi baris HuggingFace yang cocok ke tepat
satu kalimat resmi, dan proporsi baris cocok yang `lex_label`-nya berbeda dari
hasil hitungan sendiri.

Tidak ada cabang keputusan yang bergantung pada angka ini, dan tidak ada ambang
lulus. Satu-satunya `raise`: proporsi cocok di bawah 50%, yang berarti kode
normalisasinya sendiri rusak dan angka yang dilaporkan tidak berarti apa pun.

Bila selisih labelnya besar tapi pencocokannya sehat, itu dilaporkan apa adanya
sebagai perbedaan antara label project ini dan label yang beredar — bukan
diselesaikan di M4a. Yang menjadi dasar angka M4a tetap rilis resmi, dan aturan
pemetaannya sudah dinyatakan di 3.1b.

### 3.4 Dev/test BASIL dikelompokkan per event, bukan per artikel

BASIL berisi 300 artikel dalam 100 triplet: satu event dibahas Fox News, New York
Times, dan Huffington Post. Membagi per artikel akan menempatkan tiga artikel
tentang event yang sama di dev dan test sekaligus, dan kalimat yang nyaris
identik muncul di dua sisi.

Kunci grupnya karena itu event id, bukan artikel id. Biaya keputusan ini nol:
`triplet-uuid` ada di setiap file artikel dan anotasi, dan juga menjadi nama
filenya. Pembagiannya 30% dev / 70% test atas 100 event, seed tetap.

Yang dituning di dev hanya satu skalar, jadi dampak kebocoran per artikel
sebenarnya kecil. Keputusan ini diambil bukan karena dampaknya besar, tapi karena
invarian yang sudah ditegakkan `run_m1.py` berbunyi fold dikelompokkan per
sumber, bukan per baris, dan pengecualian yang tidak dibutuhkan tidak dibuat.

Pembagian dipakai ulang lewat `fold_of(triplet_uuid, n_folds=10, seed=0)` dari
`training/split.py`: fold 0 sampai 2 menjadi dev, fold 3 sampai 9 menjadi test.
Fungsi itu men-hash kuncinya dengan SHA-256, jadi hasilnya sama di setiap mesin
tanpa menyimpan daftar apa pun ke disk, dan satu event selalu jatuh utuh di satu
sisi. Tidak ada kode split baru, dan pembagian 30/70-nya perkiraan atas 100 event,
bukan tepat — jumlah event per sisi dilaporkan apa adanya.

### 3.5 Average precision dilaporkan bersama F1

F1 pada threshold apa pun mencampur dua hal: kemampuan model mengurutkan kalimat
menurut kadar bias, dan apakah threshold-nya sesuai untuk prior kelas yang baru.
Pada BASIL kedua hal itu bergerak sekaligus, karena prior positif `lex_label`
sekitar 5,6% lawan 54,6% biased di BABE.

Average precision bebas threshold, sehingga memisahkan keduanya. AP tinggi dengan
F1 apa adanya rendah berarti model masih membedakan, kalibrasinya yang bergeser.
AP rendah berarti urutannya sendiri rusak. Dua kesimpulan itu berbeda jauh, dan
tanpa AP keduanya terbaca sebagai satu angka F1 yang jelek.

### 3.6 Tanpa cache inferensi

M3 menyimpan matriks fitur ke `data/nela_cache.npz` karena ekstraksinya 46 menit.
Inferensi DeBERTa atas 8.018 kalimat pada RTX 3050 selesai dalam hitungan menit,
jadi mekanisme cache tidak membayar biayanya di sini. Prediksi dihitung ulang
setiap kali `run_m4a.py` dijalankan.

### 3.7 Tanpa training ulang

Yang dievaluasi checkpoint yang benar-benar akan dilayani, ditambah baseline
TF-IDF yang di-fit pada seluruh BABE lalu memprediksi BASIL.

TF-IDF ikut bukan sebagai formalitas. Kalau kedua model turun sama dalam,
penyebabnya pergeseran distribusi dan ketidakcocokan label, bukan sesuatu yang
khas DeBERTa. Kalau hanya DeBERTa yang turun, yang dicurigai adalah model yang
lebih besar mengunci diri pada distribusi BABE lebih rapat. Tanpa pembanding,
penurunan berapa pun tidak bisa ditafsirkan.

Melatih lima checkpoint baru pada seluruh BABE dengan lima seed akan memberi pita
mean +/- std di BASIL dan lebih konsisten dengan kebiasaan project yang menolak
angka tunggal. Itu ditolak untuk M4a: biayanya sekitar 30 menit GPU dan 3,7 GB
checkpoint sementara, sementara angka yang harus dipertanggungjawabkan README
adalah angka checkpoint yang dilayani. Variansi seed dicatat sebagai batasan yang
diketahui, bukan diukur ulang.

---

## 4. Struktur `training/` setelah M4a

```
training/
  __init__.py
  split.py            (tak berubah — hanya stdlib)
  data.py             (tak berubah)
  baseline_tfidf.py   (+1 fungsi publik: fit pada semua BABE, prediksi teks arbitrer)
  train.py            (tak berubah — MAX_LENGTH dan MODEL diimpor dari sini)
  nela.py             (tak berubah)
  run_m1.py           (tak berubah)
  run_m2.py           (tak berubah)
  run_m3.py           (tak berubah)
  basil.py            (baru)
  run_m4a.py          (baru)
```

`basil.py` mengunduh tarball rilis resmi ke `data/basil/`, meratakan
`body-paragraphs` menjadi kalimat, menerapkan aturan pemetaan 3.1b, dan menyusun
kunci grup `triplet-uuid`. Mengikuti `data.py`, fungsi muatnya mengembalikan
baris plus `LoadReport` yang mencantumkan setiap baris yang dibuang dan alasannya.
Perbandingan terhadap versi HuggingFace (3.3) tinggal di modul yang sama, sebagai
fungsi terpisah yang tidak dipanggil jalur muat.

`run_baseline()` di `baseline_tfidf.py` terikat pada loop fold dan tidak bisa
dipakai apa adanya untuk test set eksternal. Yang ditambahkan satu fungsi yang
memakai ulang `_model()`, fit pada seluruh BABE, dan mengembalikan probabilitas
untuk daftar teks apa pun. Bukan modul baru.

Inferensi DeBERTa ada di `run_m4a.py`. Tokenizer sudah tersimpan di samping bobot
(`train.py:216`), dan `MAX_LENGTH` diimpor dari `train.py` supaya truncation saat
inferensi identik dengan saat training.

---

## 5. Alur `run_m4a.py`

1. Muat BABE lewat `load_babe()`, bangun fold dengan `group_split()` seed 0, dan
   ambil fold test lewat `train_val_test_folds(0)` — fungsi yang sama yang dipakai
   M2, supaya split yang diuji identik dengan split yang melahirkan angka 0,8765.
2. Muat checkpoint `models/deberta-babe`, prediksi test split fold 0 (751 baris
   menurut `m2_selection.json`), dan **assert** selisih absolut F1 macro-nya
   terhadap 0,8765 tidak lebih dari 0,01. Guard ini berjalan sebelum apa pun yang
   menyentuh BASIL.
3. Muat BASIL lewat `load_basil()` dan cetak `LoadReport`.
4. Jalankan perbandingan HuggingFace (3.3), cetak tiga angkanya. **Assert**
   proporsi cocok tidak di bawah 50%.
5. Buang kalimat BASIL yang identik dengan kalimat BABE mana pun setelah
   normalisasi, laporkan jumlahnya, dan **assert** proporsinya tidak lebih dari 1%.
6. Bagi event jadi dev/test lewat `fold_of(triplet_uuid, 10, 0)`, fold 0-2 dev.
   Cetak jumlah event dan kalimat di tiap sisi.
7. Prediksi seluruh BASIL dengan DeBERTa dan dengan TF-IDF yang di-fit pada
   seluruh BABE.
8. Untuk tiap kombinasi model dan definisi gold: hitung F1 macro apa adanya
   (argmax), cari threshold yang memaksimalkan F1 macro di dev, terapkan
   threshold itu di test, hitung AP.
9. Cetak tabel, F1 macro constant predictor, dan simpan seluruh keluaran ke
   `runs/m4a_report.txt`.

---

## 6. Kriteria selesai

Tidak ada gate lulus/gagal pada nilai F1. PRD 8.2 sudah memutuskan angkanya
dilaporkan apa adanya walaupun turun. Menetapkan ambang lulus sekarang hanya akan
menciptakan tekanan untuk memilih definisi label yang membuat ambang itu
terlewati.

M4a selesai bila: tiga guard di bagian 9 lewat, kedua angka PRD 8.2 tercetak
untuk kedua definisi gold, dan interpretasinya ditulis di bagian 11.

---

## 7. Yang dilaporkan

### 7.1 Satu tabel, empat baris

Baris: {DeBERTa, TF-IDF} x {gold `lex`, gold `lex OR info`}. Kolom: F1 macro apa
adanya, F1 macro setelah threshold dari dev, nilai threshold itu, average
precision, proporsi positif gold, proporsi positif prediksi.

Dua kolom terakhir ada supaya besarnya pergeseran prior terlihat langsung di
sebelah F1-nya, bukan harus dihitung pembaca.

### 7.2 Lantai constant predictor dicetak bersama tabel

Pada prior positif sekitar 5,6%, prediksi semua-neutral sudah memberi F1 macro
sekitar 0,49. Tanpa angka itu tercetak, hasil di sekitar 0,5 terbaca seperti ada
sinyal padahal setara dengan tidak memprediksi apa pun. M3 memakai pola yang sama
untuk grup `event`, dan alasannya sama.

### 7.3 Catatan metodologis yang wajib menyertai angka

Empat hal, ditulis bersama tabelnya dan bukan di catatan kaki:

- Label BASIL dihitung dari rilis resmi `launchnlp/BASIL`, dengan segmentasi
  kalimat milik dataset itu sendiri. Perbandingan terhadap re-upload HuggingFace
  dilaporkan sebagai catatan, bukan sebagai sumber label.
- Aturan pemetaan: kalimat ke-`k` biased bila ada span `bias == 'lex'` dengan
  `id == 'p{k}'`.
- BASIL beranotasi dalam konteks artikel; BABE berisi kalimat berdiri sendiri.
- Angka DeBERTa berasal dari satu checkpoint, bukan mean beberapa seed, sementara
  M2 mengukur spread antar-seed 0,0187 pada BABE.

---

## 8. Pengujian

`basil.py` mendapat `_self_check()` di bawah `__main__`, mengikuti pola `nela.py`
dan `baseline_tfidf.py`. Yang diperiksa pada data yang jawabannya sudah
diketahui — anotasi tiruan, bukan unduhan: perataan `body-paragraphs` menjaga
urutan paragraf, aturan 3.1b melabeli kalimat yang ditunjuk `p{k}` dan hanya
kalimat itu, span `bias == 'inf'` tidak pernah menyalakan label utama, indeks
`p{k}` di luar rentang ditolak, dan normalisasi teks menyamakan varian whitespace
dan tanda kutip typografis.

Satu pemeriksaan lagi berjalan pada data sungguhan di dalam `load_basil()`, bukan
di self-check: kesesuaian `sents[k][start:end] == txt` untuk seluruh span di 300
artikel. Itu yang mengubah hipotesis indeks menjadi terverifikasi.

Tidak ada test di `tests/`. Direktori itu dijalankan di venv serving, yang tidak
punya `datasets` dan tidak boleh punya. Batas yang sama sudah berlaku untuk
`data.py`, `baseline_tfidf.py`, `train.py`, dan `nela.py`.

Sebelum commit: 64 test yang sudah ada tetap lulus dan ruff bersih. M4a tidak
menyentuh apa pun yang mereka uji, jadi keduanya berfungsi sebagai pemeriksaan
bahwa cakupan tidak melebar.

---

## 9. Risiko

Tiga risiko pertama ditangani `raise`, bukan warning, mengikuti `run_m1.py`.

| Risiko | Mitigasi |
|---|---|
| Urutan label checkpoint tertukar, atau preprocessing inferensi beda dari training | Guard reproduksi: F1 macro pada test split fold 0 BABE harus berada dalam 0,01 dari 0,8765 sebelum BASIL disentuh. Tanpa ini, penurunan di BASIL tidak bisa dibedakan dari bug inferensi |
| Aturan pemetaan `p{k}` tidak berlaku untuk semua artikel, sehingga label bergeser diam-diam | Setiap span di-assert memenuhi `sents[k][start:end] == txt`; span yang tidak memenuhi dihitung dan dilaporkan, dan `raise` bila melebihi 1% dari seluruh span. Hipotesis indeks ini terbukti pada satu artikel saja, jadi ia diperlakukan sebagai hipotesis sampai seluruh 300 artikel memeriksanya |
| Normalisasi teks rusak sehingga perbandingan HuggingFace tidak berarti | `raise` bila proporsi cocok di bawah 50%. Di atas itu angkanya dilaporkan apa adanya, tanpa cabang keputusan — label tidak bergantung padanya |
| BABE dan BASIL beririsan, sehingga cross-dataset sebagian bukan cross-dataset | Kalimat identik dibuang dan jumlahnya dilaporkan; assert bila proporsinya melebihi 1%. Di atas ambang itu yang bermasalah bukan beberapa baris, melainkan klaim cross-dataset-nya sendiri, dan itu harus dibereskan sebelum angka dilaporkan. Keduanya menyerap artikel berita AS periode berdekatan, jadi ini diperiksa, bukan diasumsikan |
| `horychtom/BASIL` hilang, karena re-upload tidak resmi | Angka M4a tidak bergantung padanya. Bila unduhannya gagal, perbandingan 3.3 dilewati dan dicatat sebagai tidak dijalankan |
| F1 apa adanya jeblok dan terbaca sebagai kegagalan generalisasi | Dua angka PRD 8.2, dua definisi gold, AP, dan lantai constant predictor dicetak bersama |
| Angka DeBERTa berasal dari satu checkpoint | Dinyatakan sebagai batasan di 7.3; spread seed 0,0187 dari M2 dikutip sebagai ukuran ketidakpastiannya |
| Cakupan melebar ke training ulang atau ke 8.3 | Bagian 1 dan 10 menyatakan keduanya di luar cakupan |

---

## 10. Di luar cakupan fase ini

- Validasi korelasi outlet PRD 8.3. Dijadwalkan terpisah; tidak berbagi data
  maupun metrik dengan 8.2
- Training ulang atau fine-tuning apa pun, termasuk melatih ulang untuk pita
  multi-seed di BASIL
- Adaptasi domain, continued pretraining, atau training pada BASIL. BASIL adalah
  test set dan tetap test set
- Bias informational sebagai target latih. PRD 4.5 sudah membatalkannya; di sini
  `info_label` hanya dipakai sebagai gold pembanding
- Held-out set berita terbaru buatan sendiri (PRD 5.3)
- Temperature scaling, ECE, ablation shortcut learning, contrast set (M4b)
- Ekspor ONNX dan uji paritas (M5)

---

## 11. Hasil yang terukur

Diisi setelah `run_m4a.py` dijalankan.
