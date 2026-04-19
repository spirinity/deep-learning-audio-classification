# 🔊 Sound Classifier Web App

> Web app klasifikasi suara berbasis paper **"A Multimodal Prototypical Approach for Unsupervised Sound Classification"** (INTERSPEECH 2023)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![LAION-CLAP](https://img.shields.io/badge/Model-LAION--CLAP-green)
![ESC-50](https://img.shields.io/badge/Dataset-ESC--50-orange)

---

## 📌 Tentang Project

Project ini mengimplementasikan pendekatan **Proto-LC** dari paper INTERSPEECH 2023 sebagai web aplikasi interaktif. User dapat mengupload file audio, lalu sistem mengklasifikasikannya ke salah satu dari **50 kelas suara ESC-50**.

**Cara kerja:**
1. Audio yang diupload di-encode menjadi vektor 512 dimensi menggunakan model **LAION-CLAP**
2. Vektor tersebut dibandingkan (Euclidean distance) ke **50 prototype** yang sudah pre-computed
3. Label dengan jarak terkecil = prediksi kelas

> ⚡ Tidak ada training, tidak ada fine-tuning — murni inference menggunakan model pre-trained dan prototype pre-computed dari repo paper.

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

### 3. Download LAION-CLAP Model (~600 MB)

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

### 4. Clone ESC-50 (untuk label CSV)

```bash
cd data/input
git clone https://github.com/karolpiczak/ESC-50.git
cd ../..
```

> File yang dibutuhkan hanya: `data/input/ESC-50/meta/esc50.csv`  
> File audio ESC-50 tidak diperlukan untuk inference.

### 5. Verifikasi Struktur File

```
audio_text_proto/
├── app.py
├── classifier.py
├── requirements_app.txt
├── data/
│   ├── input/
│   │   ├── 630k-audioset-fusion-best.pt   ✅ download step 3
│   │   └── ESC-50/meta/esc50.csv          ✅ clone step 4
│   └── demo/
│       └── mean_embd_tensor_esc50_clap_zs.pt  ✅ sudah ada di repo
```

### 6. Jalankan App

```bash
streamlit run app.py
```

Buka browser di: **http://localhost:8501**

---

## 📁 Struktur Project

```
├── app.py                  ← Streamlit web app (entry point)
├── classifier.py           ← Logic klasifikasi (wrapper dari demo.py)
├── common_utils.py         ← Helper dari repo paper (get_clap_model, get_label_map)
├── demo.py                 ← Script inferensi CLI dari paper
├── requirements_app.txt    ← Dependencies untuk web app
├── setup_guide.md          ← Panduan setup lengkap
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
  ESC-50 dataset → LAION-CLAP encode → rata-rata per kelas → prototype .pt

ONLINE (saat user pakai app):
  Upload audio → LAION-CLAP encode → jarak ke 50 prototype → Top-10 prediksi
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
