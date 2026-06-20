# 🔊 Sound Classifier Web App

> Web app klasifikasi suara berbasis paper **"A Multimodal Prototypical Approach for Unsupervised Sound Classification"** (INTERSPEECH 2023)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![LAION-CLAP](https://img.shields.io/badge/Model-LAION--CLAP-green)
![ESC-50](https://img.shields.io/badge/Dataset-ESC--50-orange)

---

## 📌 Tentang Project

Project ini mengimplementasikan sound classification ESC-50 sebagai web aplikasi interaktif. User dapat mengupload file audio, lalu sistem mengklasifikasikannya ke salah satu dari **50 kelas suara ESC-50** dan membandingkan beberapa metode.

**Cara kerja:**
1. Audio yang diupload di-encode menjadi vektor 512 dimensi menggunakan model **LAION-CLAP**
2. Vektor tersebut dipakai oleh beberapa metode pembanding
3. Label dengan skor tertinggi = prediksi kelas

## Metode Komparasi

| Metode | Jenis | Cara Prediksi |
|---|---|---|
| Zero-Shot CLAP | Zero-shot / tanpa training | Cosine similarity antara audio embedding dan text embedding label |
| Proto-LC | Prototypical / tanpa training | Cosine similarity antara audio embedding dan prototype kelas |
| Supervised MLP | Supervised | Shallow neural network dilatih pada embedding audio LAION-CLAP |

Untuk membuat hasil evaluasi dataset dan artifact Supervised MLP:

```bash
python evaluate_methods.py
```

Output evaluasi disimpan ke `data/demo/comparison_metrics.csv`, `data/demo/comparison_predictions.csv`, dan `data/demo/mlp_esc50_clap.joblib`.

> Zero-Shot CLAP dan Proto-LC berjalan tanpa training tambahan; Supervised MLP membutuhkan training artifact dari `evaluate_methods.py`.

### Confusion Matrix per Kategori (5×5)

50 kelas ESC-50 tergabung ke 5 kategori besar. Untuk analisis (apakah model tertukar
antar-kategori atau hanya salah kelas di dalam kategori yang sama), buat confusion matrix
5×5 dari hasil prediksi (`comparison_predictions.csv`) — tanpa GPU:

```bash
python confusion_matrix.py
```

Output gambar disimpan ke `imgs/confusion_<metode>.png` dan `imgs/confusion_all.png`.
Versi interaktifnya juga tampil di web (endpoint `GET /api/confusion`).

---

## 🎯 50 Kelas Suara (ESC-50)

| Kategori | Kelas |
|---|---|
| 🐾 Animals | dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow |
| 🌿 Natural soundscapes | rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm |
| 🗣️ Human non-speech | crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking sipping |
| 🏠 Interior/domestic | door knock, mouse click, keyboard typing, door wood creak, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking |
| 🏙️ Exterior/urban | helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw |

---

## 🚀 Setup & Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements_app.txt
```

### 3. Download LAION-CLAP Model (~1.78 GB)

Simpan ke `data/input/630k-audioset-fusion-best.pt`

**Windows (PowerShell):**
```powershell
Invoke-WebRequest `
  -Uri "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt" `
  -OutFile "data/input/630k-audioset-fusion-best.pt"
```

**Linux/Mac:**
```bash
wget https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt \
  -P data/input/
```

Atau download manual dari: https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt

### 4. Verifikasi Struktur File

```
audio_text_proto/
├── app.py
├── classifier.py
├── requirements_app.txt
├── data/
│   ├── input/
│   │   ├── 630k-audioset-fusion-best.pt   ✅ download step 3
│   │   └── ESC-50/meta/esc50.csv          ✅ sudah ada di repo
│   └── demo/
│       └── mean_embd_tensor_esc50_clap_zs.pt  ✅ sudah ada di repo
```

### 5. Jalankan App

```bash
streamlit run app.py
```

Buka browser di: **http://localhost:8501**

> Streamlit app (`app.py`) tetap tersedia sebagai versi legacy/fallback.

---

## 🖥️ Frontend Modern (Next.js + FastAPI) — Direkomendasikan

UI modern, responsif (mobile-friendly), dan dapat di-styling penuh. Logika ML
yang sama (`classifier.py`) dipakai ulang lewat API FastAPI; frontend dibuat
dengan Next.js. Butuh **dua proses** berjalan bersamaan.

**Terminal 1 — Backend (FastAPI):**
```bash
pip install -r requirements_api.txt
uvicorn api:app --port 8000
```
Endpoint utama: `GET /api/health`, `/api/methods`, `/api/metrics`,
`/api/categories`, dan `POST /api/classify`. Model LAION-CLAP dimuat sekali saat
startup (~30 detik).

**Terminal 2 — Frontend (Next.js):**
```bash
cd frontend
npm install
npm run dev
```
Buka browser di: **http://localhost:3000**

> Base URL backend dikonfigurasi via `frontend/.env.local`
> (`NEXT_PUBLIC_API_BASE`, default `http://localhost:8000`).

```
Browser (Next.js, :3000)  →  FastAPI (api.py, :8000)  →  classifier.py  →  LAION-CLAP
```

---

## 📁 Struktur Project

```
├── api.py                  ← Backend FastAPI (membungkus classifier.py)
├── app.py                  ← Streamlit web app legacy (entry point)
├── classifier.py           ← Logic klasifikasi (dipakai api.py & app.py)
├── common_utils.py         ← Helper dari repo paper (get_clap_model, get_label_map)
├── demo.py                 ← Script inferensi CLI dari paper
├── evaluate_methods.py     ← Evaluasi ESC-50 + buat artifact Supervised MLP
├── requirements_app.txt    ← Dependencies Streamlit app
├── requirements_api.txt    ← Dependencies backend FastAPI
├── setup_guide.md          ← Panduan setup lengkap
├── frontend/               ← Web app modern (Next.js + Tailwind)
│   ├── src/app/            ← Halaman & layout (App Router)
│   ├── src/components/     ← Komponen UI (uploader, kartu, chart, tabs)
│   └── src/lib/            ← API client + tipe
├── .streamlit/
│   └── config.toml         ← Konfigurasi Streamlit
└── data/
    └── demo/
        └── mean_embd_tensor_esc50_clap_zs.pt  ← Prototype pre-computed (50×512)
```

---

## 🏗️ Arsitektur

```
OFFLINE (sudah dikerjakan peneliti paper):
  ESC-50 dataset → LAION-CLAP encode → prototype, text embedding, dan Supervised MLP artifact

ONLINE (saat user pakai app):
  Upload audio → LAION-CLAP encode → 3 metode pembanding → Top-10 prediksi per metode
```

---

## 📖 Referensi

- **Paper:** [A Multimodal Prototypical Approach for Unsupervised Sound Classification](https://arxiv.org/pdf/2306.12300.pdf) — INTERSPEECH 2023
- **Repo Paper:** [sakshamsingh1/audio_text_proto](https://github.com/sakshamsingh1/audio_text_proto)
- **LAION-CLAP:** [lukewys/laion_clap](https://huggingface.co/lukewys/laion_clap)
- **ESC-50 Dataset:** [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)

---

## ⚙️ Requirements

| Package | Keterangan |
|---|---|
| `streamlit>=1.28` | Web framework |
| `laion-clap` | Model audio-text multimodal |
| `torch>=1.11` | Deep learning |
| `librosa` | Audio processing & format conversion |
| `soundfile` | Baca/tulis file audio |
| `plotly` | Visualisasi bar chart |
| `pandas` | Baca CSV label map |
