# PRD: Media Neutrality Classifier

**Author:** Priyayi
**Status:** Draft v2 — direvisi 2026-08-27 setelah sesi desain
**Compute target:** Local — Intel i5-13th Gen HX, RTX 3050 6GB VRAM, 20GB RAM
**Timeline:** Fleksibel, milestone sebagai checkpoint progres bukan hard deadline
**Desain serving/deployment:** `docs/superpowers/specs/2026-08-27-media-neutrality-deploy-design.md`

> **Perubahan utama dari v1** — taxonomy turun dari 3 kelas ke biner karena kelas
> ketiga tidak punya data latih; split wajib per-artikel untuk mencegah kebocoran;
> input hanya teks yang di-paste; target F1 dinaikkan agar sejajar literatur;
> metrik korelasi outlet diperbaiki; open question di bagian 12 ditutup menjadi
> keputusan. Rangkuman lengkap ada di bagian 13.

---

## 1. Problem Statement

Deteksi bias framing/leksikal pada level kalimat berita berbahasa Inggris —
mengukur seberapa netral cara sebuah kalimat atau artikel menyampaikan
informasi berdasarkan pilihan kata (loaded language) dan pola linguistik,
bukan berdasarkan konten faktual yang disampaikan.

Bias media adalah masalah multi-dimensi. Project ini secara sadar membatasi
diri ke satu dimensi yang tractable (bias leksikal pada level kalimat), bukan
mengklaim solusi untuk "netralitas media" secara umum.

## 2. Goals & Non-Goals

**Goals:**

- Classifier biner yang akurat (F1 macro ≥ 0,78) untuk deteksi bias leksikal per kalimat
- Reliability yang terukur — model tahu kapan dia tidak yakin (kalibrasi), bukan cuma akurat di rata-rata
- Evaluasi yang tahan uji — bebas kebocoran data, dilaporkan dengan variansi antar-seed
- Deliverable yang jalan (API dan demo publik), bukan notebook mati
- Portofolio yang menunjukkan ML engineering maturity: validation rigor, ablation, limitation yang jujur

**Non-Goals:**

- Bias seleksi/omisi (butuh cross-article comparison, out of scope)
- Stance/political-lean classification (beda task, butuh label per-topik, tidak generalize)
- Klasifikasi tiga kelas lexical versus informational (lihat bagian 4.3 — tidak ada data latihnya)
- Bahasa Indonesia (tidak ada dataset publik setara BABE/BASIL — future work)
- Pengambilan konten dari URL, baik manual maupun otomatis (lihat bagian 4.4)
- Klaim "nol halusinasi" — target yang benar adalah kalibrasi confidence, bukan zero-error; task ini subjektif dan inter-annotator agreement manusia sendiri tidak 100%

## 3. Target User (Portfolio Framing)

Reviewer teknis (recruiter atau hiring engineer AI-NLP) yang menilai:

1. Bisa fine-tune transformer dari nol, bukan cuma memanggil API
2. Punya validation rigor — cross-dataset eval, kalibrasi, ablation, dan yang paling jarang: kesadaran soal kebocoran data
3. Bisa men-deploy sesuatu yang benar-benar jalan saat link di README diklik
4. Jujur soal keterbatasan (limitation section eksplisit)

Target jumlah pengguna sengaja tidak ditetapkan. Ukuran sukses adalah demo yang
hidup dan meyakinkan saat dibuka, bukan traffic.

## 4. Scope

### 4.1 In Scope (v1)

| Komponen | Detail |
|---|---|
| Model | Fine-tune `microsoft/deberta-v3-base`, **biner**: `biased` / `neutral` |
| Dataset | **BABE** (~3.700-4.100 kalimat, train/val) + BASIL (300 artikel, cross-dataset test) |
| Split | **Group split per artikel** — wajib, lihat bagian 5.2 |
| Baseline | TF-IDF + logistic regression (floor), dan **NELA features + logistic regression** sebagai baseline linguistik terpisah |
| Validasi | 5-fold group CV, dilaporkan sebagai mean ± std across 5 seed |
| Reliability | Temperature scaling, ECE dengan confidence interval, contrast set 50 kalimat, deteksi bahasa untuk input di luar cakupan |
| Serving | FastAPI `/analyze`, `/health`, `/version` |
| Demo | Gradio di-mount ke FastAPI yang sama, satu container |
| Deployment | Docker image portabel, tidak terikat platform — detail di spec terpisah |

### 4.2 Out of Scope (v2 / Future Work — disebut eksplisit di README)

| Item | Kenapa ditunda |
|---|---|
| Pengambilan artikel dari URL | Paywall, blokir Cloudflare, halaman JS-rendered, latensi tidak terprediksi, area abu-abu legal. Nilai tambahnya kecil dibanding biaya operasionalnya |
| Auto-retrieval cross-outlet by event (NewsAPI/GDELT) | Butuh infrastruktur baru (API key, rate limit, event clustering) |
| Qwen 2.5 3B lokal sebagai explanation generator | Task berbeda. Decoder untuk klasifikasi adalah downgrade dari encoder fine-tuned. Kalau dipakai, perannya adalah menghasilkan penjelasan natural-language dari output classifier, bukan menggantikannya |
| Dashboard historis "outlet paling netral 30 hari terakhir" | Butuh data pipeline berkelanjutan, beda project |
| Bias seleksi/omisi dan stance detection | Beda task, beda label taxonomy |

### 4.3 Kenapa biner, bukan tiga kelas

PRD v1 menargetkan `lexical_bias` / `informational_bias` / `neutral`. Target ini
dibatalkan karena tidak dapat dilaksanakan dengan data yang tersedia.

BABE dan MBIC memberi label biner pada level kalimat (`biased` / `non-biased`),
ditambah label sekunder mengenai opini versus fakta. Pemisahan lexical versus
informational adalah taxonomy milik BASIL, dan BASIL berperan sebagai test set
lintas-dataset, bukan data latih. Artinya kelas ketiga tidak memiliki data latih
sama sekali.

Pemisahan lexical versus informational tetap dianalisis, tetapi sebagai
evaluasi kualitatif pada BASIL dan dilaporkan di bagian limitation — sebagai
temuan, bukan sebagai janji yang gagal ditepati.

### 4.4 Kenapa tidak ada input URL

PRD v1 mengandung kontradiksi internal: Non-Goals menolak crawling, sementara
Scope menerima input berupa URL. Input URL tetap memerlukan pengambilan konten
di sisi server dengan seluruh konsekuensinya. Keputusan: v1 hanya menerima teks
yang di-paste.

## 5. Data

- **BABE** (Spinde et al., 2021): sekitar 3.700-4.100 kalimat, expert-annotated,
  superset dari MBIC. Menjadi dataset latih utama menggantikan MBIC — anotasinya
  lebih berkualitas dan distribusinya lewat HuggingFace lebih jelas.
- **BASIL** (Fan et al., 2019): 300 artikel, anotasi span-level. Dipakai sebagai
  cross-dataset test set, bukan data latih.
- **AllSides / Media Bias Fact Check**: bukan ground truth kalimat. Dipakai untuk
  sanity check agregat, dengan koreksi metodologis di bagian 8.3.

### 5.1 Pemeriksaan kualitas sebelum training

- Cek inter-annotator agreement asli tiap dataset; buang kalimat dengan agreement
  rendah. Label noisy lebih merusak daripada data yang lebih sedikit.
- Cek distribusi kelas; siapkan class weighting bila timpang.
- Dokumentasikan setiap keputusan pemetaan label secara transparan di README.

### 5.2 Group split — wajib, bukan opsional

BABE memuat banyak kalimat yang berasal dari artikel yang sama. Split acak biasa
akan menempatkan kalimat-kalimat bertetangga dari satu artikel di train dan test
sekaligus. Akibatnya F1 naik secara palsu, dan angka yang dilaporkan tidak
mencerminkan kemampuan generalisasi apa pun.

Ketentuan:

- Split dilakukan **per `article_id`**, idealnya juga stratified per outlet
- Gunakan group-aware CV, bukan `train_test_split` biasa
- Sertakan satu test yang secara eksplisit memverifikasi tidak ada `article_id`
  yang muncul di dua fold berbeda

Ini poin terpenting di seluruh bagian data. Angka yang dihasilkan tanpa
pengamanan ini tidak layak dilaporkan.

## 6. Model & Training

- Base: `microsoft/deberta-v3-base` (184M params) — muat nyaman untuk full
  fine-tune di 6GB VRAM dengan fp16 dan batch 16
- **Full fine-tune sebagai default**, dengan early stopping pada validation loss
  dan maksimum 3-5 epoch. Masalah pada dataset sebesar ini adalah overfitting,
  bukan keterbatasan memori
- **LoRA dan layer freezing dijadikan ablation, bukan default.** LoRA menyelesaikan
  masalah memori yang tidak dimiliki project ini, sambil menambah hyperparameter
  yang harus di-sweep dengan compute terbatas
- Mitigasi overfitting: early stopping, layer-wise learning rate decay, dan
  **seed averaging**
- Hyperparameter sweep diprioritaskan pada learning rate (1e-5 / 2e-5 / 3e-5).
  Grid search penuh dilewati
- **Setiap konfigurasi final dijalankan dengan 5 seed berbeda.** Pada dataset
  sekecil ini, deviasi antar-seed bisa mencapai ±0,03 F1. Angka tunggal tanpa
  variansi tidak informatif

### 6.1 Feature linguistik sebagai baseline terpisah

PRD v1 merencanakan konkatenasi fitur eksplisit (bias-lexicon density,
subjectivity, quote-ratio) sebelum classification head. Rencana ini diubah.

Konkatenasi membutuhkan custom model class, custom collator, dan custom trainer
— biaya nyata, sementara DeBERTa kemungkinan besar sudah menangkap sinyal loaded
language sehingga delta-nya mendekati nol.

Sebagai gantinya, **NELA features** (Horne et al., 2018 — 141 fitur linguistik
termasuk factives, assertives, implicatives, hedges) dipakai sebagai **model
baseline terpisah**: logistic regression di atas fitur NELA, dibandingkan
langsung dengan DeBERTa.

Cerita ablation-nya tetap utuh, tanpa satu baris pun kode arsitektur custom.
Bila ternyata NELA + logistic regression mendekati performa DeBERTa, itu justru
temuan yang menarik untuk README.

## 7. Reliability

Model ini classifier, bukan generative LLM — istilah "hallucination" tidak
berlaku literal. Tiga failure mode yang relevan:

1. **Miskalibrasi** — temperature scaling post-hoc yang di-fit di validation set.
   ECE dilaporkan bersama confidence interval hasil bootstrap dan memakai adaptive
   binning. Angka target absolut dihindari: dengan validation set beberapa ratus
   sampel, estimasi ECE sendiri punya variansi besar.
2. **Shortcut learning** — ablation kontrol (mask panjang kalimat dan named
   entity, ukur penurunan performa), contrast set 50 kalimat manual (loaded word
   diganti netral, ukur flip rate), dan visualisasi atribusi token.
3. **Input di luar cakupan** — deteksi bahasa dengan `fasttext lid.176-compressed`.
   Input non-Inggris **ditandai dengan peringatan, bukan ditolak**.

Domain gate berbasis kemiripan embedding ke centroid data latih **dihapus dari
v1**: ambangnya arbitrer dan berisiko menolak artikel berita yang sah hanya
karena topiknya berbeda dari data latih.

## 8. Success Metrics

### 8.1 Metrik utama

- **F1 macro ≥ 0,78** pada BABE test split, dilaporkan sebagai mean ± std across
  5 seed. Literatur BABE melaporkan sekitar 0,77-0,80 untuk transformer
  fine-tuned; angka pastinya diverifikasi ulang dari paper sebelum masuk README
- ECE setelah kalibrasi, dilaporkan dengan confidence interval
- Contrast set flip rate — dilaporkan sebagai indikator shortcut learning, bukan
  sebagai kriteria lulus/gagal

### 8.2 Cross-dataset (BABE → BASIL)

Dilaporkan apa adanya walaupun turun, sebagai bagian dari limitation section.

Namun harus disertai catatan metodologis: BASIL beranotasi span-level dalam
konteks artikel dan sangat timpang, sementara BABE berisi kalimat berdiri
sendiri. Pemetaan span BASIL ke label kalimat memerlukan keputusan eksplisit
(misalnya: kalimat dianggap biased bila memuat minimal satu span).

Konsekuensinya, penurunan F1 sebagian besar merupakan artefak ketidakcocokan
label, bukan murni kegagalan generalisasi. Karena itu dilaporkan **dua angka**:
apa adanya, dan setelah threshold direkalibrasi pada dev split BASIL. Tanpa
keduanya, angka tersebut tidak menginformasikan apa pun.

### 8.3 Korelasi dengan rating outlet — metodologi diperbaiki

Rencana v1 mengkorelasikan skor model dengan rating AllSides secara langsung.
Ini keliru secara desain: rating AllSides mengukur **arah politik**
(left/center/right), sedangkan output model mengukur **intensitas bias**. Outlet
condong kiri dan condong kanan sama-sama bisa berintensitas tinggi, sehingga
korelasi terhadap skala arah akan mendekati nol karena alasan matematis, bukan
karena modelnya buruk.

Perbaikan:

- Korelasikan dengan **|lean|** — jarak absolut dari center — atau gunakan skor
  bias-level MBFC yang memang ordinal terhadap intensitas
- **Minimal 12-15 outlet.** Rencana awal (5 outlet) memberi n = 5 untuk korelasi,
  yang tidak bermakna secara statistik
- Yang dinilai adalah arah dan konsistensinya, bukan nilai r yang tinggi

### 8.4 Analisis kualitatif

Kategorikan 50 error terparah menjadi taksonomi kegagalan, dan laporkan di
README. Bagian ini biasanya lebih meyakinkan reviewer teknis dibanding
visualisasi atribusi, dan hampir selalu dilewatkan orang lain.

## 9. Milestones

Checkpoint, bukan deadline. Setiap milestone memiliki kriteria berhenti agar
waktu tidak terbuang ke arah yang salah.

| # | Kerjaan | Kriteria berhenti / lanjut |
|---|---|---|
| M1 | Data loading, EDA, group split, verifikasi tidak ada kebocoran, baseline TF-IDF | Test anti-kebocoran lulus sebelum training apa pun dimulai |
| M2 | Fine-tune DeBERTa, sweep learning rate, 5 seed | Bila tidak mengungguli baseline TF-IDF secara meyakinkan, **berhenti dan periksa kualitas data** sebelum melanjutkan tuning |
| M3 | Baseline NELA + logistic regression, bandingkan dengan DeBERTa | Delta dilaporkan apa adanya, termasuk bila kecil |
| M4a | Cross-dataset eval BABE → BASIL, validasi korelasi outlet | Dua angka dilaporkan sesuai bagian 8.2 |
| M4b | Temperature scaling, ECE dengan CI, ablation shortcut learning, contrast set | Nilai temperature tersimpan bersama metadata model |
| M5 | Ekspor ONNX, kuantisasi INT8, **uji paritas** | Kesepakatan label ≥ 99% dan selisih probabilitas < 0,02, jika tidak turunkan ke fp16 |
| M6 | FastAPI dan Gradio dalam satu container, Dockerfile, rate limit, logging, CI smoke test | Container jalan dari image bersih, `/health` hijau, startup < 10 detik |
| M7 | README: problem statement, scope, metrik dengan variansi, error analysis, limitation section | — |

## 10. Risks

| Risiko | Mitigasi |
|---|---|
| **Kebocoran data lewat split acak** | Group split per artikel, diverifikasi test otomatis (M1) |
| Dataset kecil menyebabkan overfitting | Early stopping, epoch dibatasi, seed averaging, layer-wise LR decay |
| Angka tunggal menyesatkan karena variansi antar-seed | Seluruh metrik dilaporkan mean ± std across 5 seed |
| Kuantisasi INT8 menggeser prediksi tanpa terdeteksi | Uji paritas wajib di M5 sebelum rilis |
| Hasil kalibrasi hilang saat ekspor ONNX | Temperature disimpan sebagai metadata dan diverifikasi test |
| Metrik korelasi outlet rusak secara desain | Diperbaiki di bagian 8.3 — pakai \|lean\|, minimal 12 outlet |
| Ablation dilewati karena terasa tidak penting | Dijadwalkan eksplisit di M3 — ini pembeda dari project tutorial |
| Taxonomy BABE dan BASIL tidak identik | Pemetaan didokumentasikan, dan dilaporkan sebagai batasan di bagian 8.2 |
| Latensi di CPU pada demo publik | ONNX INT8, batching per artikel, cap 300 kalimat per request |
| Reviewer bertanya kenapa Qwen tidak dipakai | README menjelaskan alasan arsitektural: encoder untuk klasifikasi, decoder untuk generasi |

## 11. Tech Stack Summary

**Training (lokal, GPU, tidak masuk image produksi):**
PyTorch, `transformers`, `datasets`, `scikit-learn`, `nela-features`

**Serving (masuk image produksi):**
FastAPI, `onnxruntime` (CPU), `tokenizers`, `pysbd`, `slowapi`, Gradio,
`fasttext` (lid.176-compressed), `pydantic-settings`

Pemisahan ini disengaja dan dijaga oleh Dockerfile: PyTorch tidak pernah masuk
ke image produksi. Detail lengkap ada di spec deployment.

## 12. Keputusan yang Sebelumnya Terbuka

Bagian ini pada v1 berisi pertanyaan yang belum terjawab. Semuanya kini ditutup
agar tidak memblokir eksekusi.

**12.1 Dataset — ditutup.**
Gunakan **BABE** sebagai dataset latih utama, menggantikan MBIC. Alasannya:
lisensi MBIC tidak eksplisit di paper aslinya, sementara BABE adalah superset
yang expert-annotated dengan distribusi HuggingFace yang lebih jelas. BASIL
tetap sebagai cross-dataset test.
*Satu-satunya verifikasi tersisa:* buka dataset card BABE di HuggingFace untuk
memastikan term lisensinya sebelum repo dipublikasikan. Ini pekerjaan lima
menit, bukan open question.
MBIB (Wessel et al., 2023) dicatat sebagai opsi perluasan evaluasi, tidak
dipakai di v1.

**12.2 Lexicon — ditutup.**
Gunakan **NELA features** menggantikan lexicon Recasens 2013 — tersedia sebagai
package siap pakai, gratis, dan memberi 141 fitur linguistik alih-alih density
dari satu lexicon berusia lebih dari sepuluh tahun. LIWC tidak dipakai karena
berbayar. Perannya sebagai baseline terpisah, bukan konkatenasi (lihat 6.1).

## 13. Ringkasan Perubahan v1 → v2

| Area | v1 | v2 | Alasan |
|---|---|---|---|
| Taxonomy | 3 kelas | Biner | Kelas ketiga tidak punya data latih |
| Dataset latih | MBIC | BABE | Lisensi lebih jelas, anotasi lebih baik, data lebih banyak |
| Split | Tidak dispesifikasi | Group split per artikel | Split acak menyebabkan kebocoran |
| Target F1 | ≥ 0,65 | ≥ 0,78 mean ± std | Disesuaikan dengan literatur BABE untuk task biner |
| Pelaporan metrik | Angka tunggal | Mean ± std across 5 seed | Variansi antar-seed besar pada dataset kecil |
| Fitur linguistik | Dikonkatenasi ke head | Baseline terpisah | Biaya arsitektur tinggi, delta kemungkinan nol |
| Mitigasi overfitting | LoRA / freezing sebagai default | Full fine-tune + early stopping; LoRA jadi ablation | LoRA menyelesaikan masalah memori yang tidak ada |
| Korelasi outlet | Rating AllSides langsung, 5 outlet | \|lean\| atau MBFC, minimal 12 outlet | Arah politik bukan intensitas bias; n = 5 tidak bermakna |
| ECE | Target keras < 0,05 | Dilaporkan dengan CI | Estimasi ECE sendiri bervariansi pada val set kecil |
| OOD gate | Bahasa + domain centroid | Bahasa saja | Ambang domain arbitrer, berisiko menolak input sah |
| Input | URL atau teks | Teks saja | Menghilangkan kontradiksi internal dan seluruh risiko scraping |
| Endpoint | `/analyze` dan `/compare` | `/analyze` saja | Perbandingan dilakukan di sisi UI |
| Deployment | Disebut sekilas | Spec tersendiri, portabel lintas platform | Ada beberapa project lain yang akan menyusul |
