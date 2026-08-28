# Design Spec: Media Neutrality Classifier — Fase Training (M2)

**Tanggal:** 2026-08-28
**Status:** Draft untuk review
**Terkait:** `PRD_Media_Neutrality_Classifier.md`, `docs/superpowers/specs/2026-08-27-training-phase-design.md`

---

## 1. Konteks dan cakupan

Spec M1 menjawab bagaimana data BABE masuk ke project ini. Dokumen ini menjawab
satu hal berikutnya: bagaimana DeBERTa dilatih di atas fold M1 sehingga angka
hasilnya dapat dibandingkan dengan baseline TF-IDF tanpa syarat tambahan.

PRD 6 sudah menutup pilihan model, strategi fine-tune, jumlah epoch, jumlah seed,
dan daftar learning rate. Yang belum ditutup adalah protokol evaluasi, sumber
validation set, artefak yang dibawa ke M4b dan M5, harness training, dan definisi
angka dari kriteria berhenti M2. Lima hal itu isi dokumen ini.

**Cakupan implementasi: M2 saja.** Pekerjaan berhenti sebelum baseline NELA (M3),
cross-dataset eval (M4a), dan temperature scaling (M4b).

---

## 2. Titik awal: angka M1 yang mengikat

Tiga angka dari bagian 6.4 spec M1 membatasi apa yang boleh dilakukan di sini.

```
baris dipakai            : 3.874  (2.117 biased, 1.757 neutral)
grup setelah filter      : 2.151
ukuran fold              : 751, 883, 743, 770, 727
F1 macro baseline TF-IDF : 0,7420 +/- 0,0194  (mean +/- std atas 5 fold)
target PRD 8.1           : 0,78
```

Dua konsekuensi langsung:

**Margin yang harus dilewati sekitar 0,04, bukan lebih.** Baseline TF-IDF sudah
0,7420. Deviasi antar-fold baseline sendiri 0,0194, dan PRD 6 menyebut deviasi
antar-seed pada dataset sekecil ini bisa mencapai 0,03. Perbedaan sebesar margin
target karena itu berada dalam jangkauan derau, sehingga angka tunggal tidak dapat
menjawab apakah DeBERTa menang.

**Angka 0,78 di PRD 8.1 bukan pembanding langsung.** Bagian 6.1 spec M1
menunjukkan split resmi BABE bocor terhadap `news_link`, dan angka literatur
dihitung pada split resmi itu. Satu-satunya pembanding yang sah untuk project ini
adalah 0,7420, yang dihitung pada fold yang sama dengan yang akan dipakai DeBERTa.
Target 0,78 tetap dilaporkan, tetapi sebagai rujukan literatur, bukan sebagai
ambang yang setara.

---

## 3. Keputusan yang ditutup

### 3.1 Sweep learning rate di satu fold, konfigurasi final di 5 fold x 5 seed

PRD 6 meminta sweep learning rate pada tiga nilai dan 5 seed untuk setiap
konfigurasi final. Dijalankan naif, itu berarti 3 x 5 x 5 = 75 run.

Protokol yang dipakai memisahkan pemilihan dari pelaporan:

```
sweep  : fold 0 sebagai test, seed 0, lr dalam {1e-5, 2e-5, 3e-5}  ->  3 run
final  : fold 0..4 x seed 0..4 pada lr terpilih                    ->  25 run
total  : 27 run yang benar-benar dilatih
```

Jumlahnya 27, bukan 28: kombinasi (fold 0, seed 0, lr terpilih) adalah anggota
sweep sekaligus anggota grid final, dan run kedua memakai hasil yang sudah
tercatat alih-alih melatih ulang. Angka yang dilaporkan tetap 25 run final.

Learning rate dipilih sekali, dan 47 run tambahan dari grid penuh tidak mengubah
pilihan itu. PRD 6 sendiri sudah menyatakan grid search penuh dilewati.

Alternatif yang ditolak: satu split tetap dengan 3 lr x 5 seed (15 run). Lebih
murah, tetapi angkanya mean atas seed pada satu fold, sementara baseline mean atas
lima fold. Pada margin 0,04 perbedaan itu cukup untuk membuat klaim "mengungguli
baseline" tidak dapat dipertahankan.

Pemilihan learning rate memakai validation loss, bukan F1 pada fold test. Fold
test tidak menyentuh keputusan apa pun sampai angka akhir dilaporkan.

### 3.2 Validation set adalah fold tetangga

Early stopping yang diwajibkan PRD 6 memerlukan validation set yang tidak sama
dengan fold test. Fold tetangga dipakai:

| held out | test | val | train |
|---|---|---|---|
| 0 | fold 0 (751) | fold 1 (883) | 2.240 |
| 1 | fold 1 (883) | fold 2 (743) | 2.248 |
| 2 | fold 2 (743) | fold 3 (770) | 2.361 |
| 3 | fold 3 (770) | fold 4 (727) | 2.377 |
| 4 | fold 4 (727) | fold 0 (751) | 2.396 |

Keuntungannya bukan kerapian tabel: fold M1 sudah group-clean terhadap
`news_link`, jadi train, val, dan test otomatis tidak berbagi artikel tanpa satu
baris kode split baru. Ukuran val 727 sampai 883 baris juga membuat kurva
validation loss cukup stabil untuk menghentikan training di titik yang berarti.

Alternatif yang ditolak: mengambil 12,5% dari empat fold train sebagai val, yang
menaikkan train ke sekitar 2.712 baris. Val-nya turun ke sekitar 390 baris, dan
pada ukuran itu validation loss cukup berisik untuk membuat early stopping
berhenti di epoch yang salah. Tambahan 400 baris train tidak sebanding dengan
derau yang masuk ke sinyal penghentian.

Alternatif yang juga ditolak: melatih tanpa val pada epoch tetap. Overfitting
adalah risiko utama yang tercatat di PRD 10 untuk dataset sebesar ini, dan
menghapus early stopping menghapus mitigasi pertamanya.

### 3.3 Satu checkpoint disimpan, dipilih lewat validation loss

25 run final menghasilkan bobot sebesar sekitar 740 MB masing-masing. Menyimpan
semuanya berarti 18 GB untuk satu konsumen yang hanya memerlukan satu model:
serving memuat satu file ONNX, dan M5 mengekspor satu model.

Yang disimpan adalah bobot dari fold 0 pada seed dengan validation loss terbaik,
di `models/deberta-babe/`. Selama lima run fold 0 berjalan, bobot masing-masing
seed ditulis ke direktori sementara, dan empat yang bukan pemenang dihapus setelah
seed terakhir selesai. Puncak pemakaian disk sekitar 3,7 GB, bukan 18 GB.

Fold 0 dipilih sebagai sumber checkpoint sebelum satu run pun dijalankan, dan
seed dipilih lewat validation loss fold 1. Fold 0 tetap menjadi fold test yang
dilaporkan, dan tidak ikut memilih apa pun.

Alternatif yang ditolak: tidak menyimpan bobot sama sekali dan melatih ulang di
M4b. Menghemat disk, tetapi bobot yang dikalibrasi lalu tidak lagi dijamin
identik dengan bobot yang angkanya dilaporkan, karena kernel cuDNN tidak
deterministik secara default.

**Seed averaging tidak menghasilkan model yang dikirim.** PRD 6 menyebut seed
averaging sebagai mitigasi overfitting; di project ini ia berlaku pada pelaporan,
yaitu mean dan deviasi atas lima seed. Ensemble lima bobot akan melipatgandakan
latensi dan ukuran image, dan spec serving bagian 2.3 sudah memutuskan satu file
ONNX yang dibakar ke dalam image.

### 3.4 HuggingFace Trainer, bukan loop manual

`transformers` sudah terpin di `requirements-train.txt` sejak M1. `Trainer`
menyediakan fp16, gradient accumulation, penetapan seed, evaluasi per epoch,
early stopping lewat callback, dan pemuatan checkpoint terbaik di akhir training.
Menulis semuanya sendiri berarti sekitar 150 baris yang setiap barisnya bisa
salah tanpa memunculkan error.

Satu bagian tetap manual: layer-wise learning rate decay yang diminta PRD 6 tidak
ada di `TrainingArguments`, sehingga optimizer group per layer dibangun sendiri.

Parameter yang tetap di seluruh 27 run, dinyatakan di satu tempat dalam kode:

```
model                 : microsoft/deberta-v3-base
max_length            : 128, padding dinamis
batch size            : 16  (fallback 8 x accum 2 bila OOM)
fp16                  : ya
epoch maksimum        : 5
early stopping        : patience 2 pada validation loss
llrd decay per layer  : 0,95 dari layer teratas ke bawah
warmup ratio          : 0,06
weight decay          : 0,01
```

Nilai yang diambil dari PRD 6 adalah model, epoch maksimum, dan daftar learning
rate. Sisanya adalah default yang lazim untuk fine-tune encoder dan tidak
di-sweep: satu-satunya hyperparameter yang di-sweep adalah learning rate, sesuai
PRD 6. Decay 0,95 dipilih sebagai titik awal, bukan hasil pengukuran; bila
validation loss stagnan di seluruh sweep, nilai itu yang diperiksa lebih dulu.

Alternatif yang ditolak: `hyperparameter_search` dengan Optuna. Menambah satu
dependency untuk memilih di antara tiga nilai yang sudah ditentukan PRD 6.

### 3.5 Fungsi pemilihan fold tinggal di `split.py`

Pemetaan `held_out -> (train, val, test)` diimplementasikan sebagai
`train_val_test_folds()` di `training/split.py`, bukan di `train.py`.

Alasannya sama dengan keputusan 3.2 spec M1: `split.py` hanya memakai stdlib,
sehingga `tests/test_split.py` dapat mengujinya di job CI serving yang tidak
memasang torch. Bila fungsi ini tinggal di `train.py`, satu-satunya pemeriksaan
bahwa val dan test tidak pernah fold yang sama akan berjalan hanya di mesin yang
punya GPU.

Batasan impor `split.py` tetap berlaku: tanpa `datasets`, `pandas`, `sklearn`,
`torch`, atau `numpy`.

---

## 4. Struktur `training/` setelah M2

```
training/
├── requirements-train.txt
├── split.py                 # + train_val_test_folds(held_out, n_folds)
├── data.py                  # tidak berubah
├── baseline_tfidf.py        # tidak berubah
├── run_m1.py                # tidak berubah
├── train.py                 # run_one(rows, folds, held_out, seed, lr) -> RunResult
└── run_m2.py                # entry point: sweep, 25 run final, laporan
```

`train.py` memuat tokenisasi, pembangunan optimizer group untuk layer-wise LR
decay, konfigurasi `Trainer`, dan satu fungsi `run_one()` yang mengembalikan F1
macro pada fold test beserta validation loss terbaiknya.

`run_m2.py` memuat orkestrasi: memuat data lewat `data.load_babe()`, membangun
fold lewat `split.group_split()` dengan parameter yang sama seperti M1,
menjalankan sweep, menjalankan 25 run final, dan mencetak laporan.

`.gitignore` menambahkan `models/` dan `runs/`. Bobot dan metrik tidak masuk repo,
konsisten dengan `data/` yang sudah ada di sana.

Nama `calibrate.py`, `evaluate.py`, dan `export_onnx.py` di spec serving tetap
berlaku dan belum dibuat di fase ini.

---

## 5. Alur `run_m2.py`

```
load_babe()
  -> group_split(group_keys, n_folds=5, seed=0)        # identik dengan M1
  -> assert: group_key train, val, test tidak beririsan
  -> cetak persentil panjang token
  -> sweep  : 3 run pada held_out=0, seed=0            -> lr terpilih = argmin val_loss
  -> final  : 25 run, held_out 0..4 x seed 0..4
       tulis satu baris JSON ke runs/m2_runs.jsonl setelah setiap run selesai
       simpan bobot untuk held_out=0, lalu sisakan satu seed terbaik
  -> cetak laporan: per fold, per seed, mean +/- std, dan hasil uji kriteria berhenti
```

Assert kebocoran memakai pola yang sama dengan `run_m1.py`: assert yang
menghentikan proses, bukan peringatan yang tercetak. Fold berasal dari fungsi yang
sudah di-test, tetapi assert ini menjaga terhadap kesalahan indeks di `run_m2.py`
sendiri, dan biayanya satu operasi himpunan.

Persentil panjang token dicetak agar `max_length` yang dipakai punya dasar ukur.
BABE berisi kalimat tunggal, sehingga 128 token diperkirakan cukup, tetapi
perkiraan itu diperiksa sekali, bukan diasumsikan. Padding dinamis lewat
`DataCollatorWithPadding` membuat kalimat pendek tidak dibayar sebagai 128 token.

JSONL ditulis setelah setiap run, bukan sekali di akhir. Dua puluh tujuh run
berjalan sekitar dua jam, dan crash di run ke-20 tidak boleh menghapus hasil 19
run sebelumnya. Karena itu `run_m2.py` membaca JSONL yang sudah ada saat mulai dan
melewati kombinasi `(held_out, seed, lr)` yang barisnya sudah tercatat. Menjalankan
ulang setelah crash melanjutkan, bukan mengulang dari nol. Estimasi dua jam itu
perkiraan dari ukuran dataset dan bukan hasil ukur; angka sebenarnya diketahui dari
run sweep pertama.

---

## 6. Kriteria berhenti M2, dinyatakan sebagai angka

PRD 9 menulis kriteria M2 sebagai: bila DeBERTa tidak mengungguli baseline TF-IDF
secara meyakinkan, berhenti dan periksa kualitas data sebelum melanjutkan tuning.
Frasa "secara meyakinkan" ditetapkan sebagai angka di sini, sebelum satu run
dijalankan, agar tidak ditafsirkan setelah melihat hasilnya.

```
lulus  : DeBERTa menang di >= 4 dari 5 fold
         DAN  mean(DeBERTa) - 0,7420 >= 0,02
gagal  : berhenti. Periksa kualitas data sesuai PRD 9. Tidak menambah sweep.
```

Perbandingan per fold membandingkan mean lima seed pada fold itu dengan skor
baseline TF-IDF pada fold yang sama, bukan dengan mean baseline. Baseline sudah
dijalankan per fold di M1, jadi kelima angkanya tersedia.

Ambang 0,02 kira-kira satu deviasi antar-fold baseline (0,0194). Menang tipis di
bawah itu tidak dapat dipisahkan dari variansi fold, dan kemenangan yang tidak
dapat dipisahkan dari derau bukan kemenangan yang layak dilaporkan.

Syarat "menang di minimal 4 dari 5 fold" ada karena mean saja dapat didorong oleh
satu fold yang mudah. Dua syarat itu menutup dua cara berbeda untuk menang secara
kebetulan.

---

## 7. Yang dilaporkan

| Angka | Bentuk |
|---|---|
| F1 macro DeBERTa | mean +/- std atas 25 run |
| F1 macro per fold | lima baris, masing-masing mean +/- std atas 5 seed |
| Sebaran antar-seed | dilaporkan terpisah dari sebaran antar-fold |
| Learning rate terpilih | beserta tiga validation loss dari sweep |
| Epoch saat early stopping | per run, untuk menunjukkan overfitting bila muncul |
| Perbandingan dengan baseline | delta per fold dan hasil uji kriteria bagian 6 |

Sebaran antar-fold dan antar-seed dilaporkan terpisah karena artinya berbeda.
Variansi antar-fold menunjukkan seberapa banyak angka bergantung pada artikel mana
yang kebetulan jadi test. Variansi antar-seed menunjukkan seberapa banyak angka
bergantung pada inisialisasi. Satu std gabungan mencampur keduanya dan
menyembunyikan keduanya.

Seluruh angka dilaporkan dari `runs/m2_runs.jsonl`, yaitu hasil yang benar-benar
diukur. Tidak ada klaim bahwa angkanya dapat direproduksi bit-exact: kernel cuDNN
tidak deterministik secara default, dan menyalakan mode deterministik akan
memperlambat training tanpa mengubah kesimpulan mana pun.

Keluaran dicetak ke stdout dan disalin ke README, mengikuti pola M1. Tidak ada
generator laporan.

---

## 8. Pengujian

| Test | Lokasi | Berjalan di CI |
|---|---|---|
| `train_val_test_folds`: train, val, test disjoint; gabungannya seluruh fold; val bukan test | `tests/test_split.py` | ya |
| Kebocoran group key antar tiga bagian, pada data BABE nyata | assert di `run_m2.py` | tidak, butuh data |
| Loop training menurunkan loss pada 40 baris sintetis | self-check di `training/train.py` | tidak, butuh torch |

Test pertama berjalan di CI tanpa tambahan dependency, konsekuensi keputusan 3.5.

Self-check di `train.py` bukan test performa. Ia melatih dua epoch pada data
sintetis yang terpisah sempurna dan memastikan loss turun. Yang diuji adalah
bahwa harness benar-benar melatih: label tidak tertukar, learning rate tidak nol,
dan bobot benar-benar berubah. Kesalahan seperti itu menghasilkan angka yang
tampak masuk akal tanpa memunculkan error.

---

## 9. Batas berhenti fase ini

Fase ini selesai ketika:

- `run_m2.py` berjalan sampai selesai, 27 run tercatat di `runs/m2_runs.jsonl`
- assert kebocoran lulus
- `tests/test_split.py` lulus bersama seluruh test serving yang sudah ada
- satu checkpoint ada di `models/deberta-babe/` beserta validation loss-nya
- kriteria bagian 6 diuji dan hasilnya dicatat, termasuk bila hasilnya gagal
- angka M2 masuk README bersama catatan bahwa 0,78 bukan pembanding langsung

Bila kriteria bagian 6 gagal, fase ini tetap selesai. Yang berubah adalah
pekerjaan berikutnya: pemeriksaan kualitas data, bukan M3.

---

## 10. Risiko

| Risiko | Mitigasi |
|---|---|
| Batch 16 tidak muat di 6 GB VRAM | Fallback ke batch 8 dengan `gradient_accumulation_steps=2`; effective batch tetap 16. Diketahui di run sweep pertama, bukan di run ke-20 |
| 27 run gagal di tengah jalan | JSONL ditulis per run; run yang sudah tercatat dilewati saat dijalankan ulang |
| Angka tidak dapat direproduksi persis | Dilaporkan apa adanya dari JSONL; tidak ada klaim bit-exact, dan bobot yang disimpan adalah bobot yang angkanya dilaporkan |
| Checkpoint terpilih ikut memilih lewat fold test | Seleksi seed memakai validation loss fold 1; fold 0 hanya dilaporkan |
| DeBERTa hanya menang tipis | Kriteria bagian 6 sudah menetapkan konsekuensinya sebelum angkanya dilihat |
| Early stopping berhenti di epoch pertama | Epoch penghentian dilaporkan per run; pola yang seragam di semua run adalah tanda konfigurasi salah, bukan tanda dataset kecil |
| Torch bocor ke serving | Venv terpisah, `.dockerignore`, dan pemeriksaan CI yang sudah ada |

---

## 11. Di luar cakupan fase ini

- Baseline NELA (M3), cross-dataset BASIL (M4a), temperature scaling dan ECE (M4b)
- Ekspor ONNX dan uji paritas (M5)
- LoRA dan layer freezing, yang PRD 6 tempatkan sebagai ablation, bukan default
- Contrast set dan analisis shortcut learning (M4b)
- Perubahan apa pun pada `src/neutral/` dan pada `Dockerfile`
- Stratifikasi per outlet, kecuali bila hasil per fold membuktikan perlu

---

## 12. Hasil, diukur pada 28 Agustus 2026

Dijalankan lewat `.venv-train/Scripts/python.exe -m training.run_m2`. 27 run
tanpa interupsi, semuanya tercatat di `runs/m2_runs.jsonl`.

```
learning rate terpilih      : 3e-5  (val_loss 0,3580 / 0,3539 / 0,3530 untuk 1e-5 / 2e-5 / 3e-5)
F1 macro DeBERTa            : 0,8477 +/- 0,0277  (25 run)
F1 macro baseline TF-IDF    : 0,7420 +/- 0,0194  (dihitung ulang pada fold yang sama)
per fold, DeBERTa vs baseline:
  fold 0   0,8693 +/- 0,0106   vs 0,7445   delta +0,1248
  fold 1   0,8397 +/- 0,0183   vs 0,7095   delta +0,1302
  fold 2   0,8184 +/- 0,0453   vs 0,7449   delta +0,0735
  fold 3   0,8550 +/- 0,0143   vs 0,7614   delta +0,0936
  fold 4   0,8563 +/- 0,0050   vs 0,7496   delta +0,1067
std antar-fold              : 0,0195
std antar-seed              : 0,0187
epoch berhenti              : 3 atau 4; tidak ada run yang mencapai epoch 5
menang di fold              : 5 dari 5
delta mean                  : +0,1058
checkpoint disimpan         : models/deberta-babe, dari held_out 0 seed 3, val_loss 0,3331
```

**Kriteria bagian 6: LULUS.** Kedua syarat terlampaui dengan jarak jauh: menang di
5 fold dari 5 dengan batas 4, dan delta mean +0,1058 dengan batas 0,02. Baseline
yang dihitung ulang persis sama dengan angka M1 (0,7420 +/- 0,0194), jadi
pembagian fold benar-benar terreproduksi dan kedua angka diukur pada data yang
sama. Yang terukur di sini hanya satu hal: F1 macro pada fold BABE yang bersih
grup. Generalisasi lintas dataset (BASIL, M4a), kalibrasi dan ECE (M4b), serta
held-out berita terbaru belum diukur sama sekali, dan angka 0,8477 tidak
mengatakan apa pun tentang ketiganya.

Dua hal yang perlu dibaca bersama angkanya. Pertama, std antar-fold (0,0195) dan
std antar-seed (0,0187) hampir sama besar, jadi satu run pada satu fold akan
menyembunyikan setengah dari sebaran yang ada; 25 run bukan kemewahan. Kedua,
fold 2 adalah fold terburuk sekaligus paling tidak stabil (0,8184 +/- 0,0453,
rentang seed 0,7626 sampai 0,8607), dan delta terkecil ada di sana. Deltanya tetap
jauh di atas 0,02, tetapi bila nanti ada fold yang gagal, fold 2 kandidat pertama
yang diperiksa.

Pemilihan learning rate nyaris seri: tiga val_loss berjarak kurang dari 0,006. Yang
perlu dicatat, 1e-5 justru memberi F1 test tertinggi pada run sweep (0,8812 lawan
0,8540 untuk 3e-5). Pemilihan tetap memakai validation loss sesuai bagian 3.1, jadi
fold test tidak ikut memilih apa pun. Bila suatu saat angka ini dipertanyakan,
inilah tempat pertanyaannya: 3e-5 menang karena validation loss, bukan karena test.

**Deviasi dari bagian 3.4 saat dijalankan:** tidak ada pada sisi hyperparameter.
Batch 16 dengan fp16 muat di 6 GB VRAM, jadi fallback batch 8 dengan accum 2 tidak
dipakai. Tiga penyesuaian lain dipaksa oleh transformers 5.16.1 dan tidak mengubah
nilai apa pun di bagian 3.4:

- `warmup_ratio` sudah tidak ada. Rasio 0,06 dihitung menjadi `warmup_steps` dari
  ukuran train dikali epoch maksimum, bukan epoch aktual, supaya early stopping
  tidak mengubah jadwal learning rate antar run.
- `from_pretrained` memuat bobot dengan dtype checkpoint, dan checkpoint
  deberta-v3-base berisi fp16. Parameter fp16 membuat `GradScaler` gagal dengan
  "Attempting to unscale FP16 gradients", jadi `dtype=torch.float32` dinyatakan
  eksplisit. Mixed precision tetap aktif lewat `fp16=True`.
- `accelerate` adalah dependency wajib `Trainer`, bukan opsional. Dipin 1.14.0 di
  `training/requirements-train.txt`.
