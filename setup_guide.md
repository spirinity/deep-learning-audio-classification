# 🔊 Sound Classifier — Setup Guide

## Prerequisites
- Python 3.8 (matches the paper's environment)
- Git
- ~2 GB free disk space (for model + ESC-50 data)

---

## Step 1: Install dependencies

```bash
pip install -r requirements_app.txt
```

---

## Step 2: Download LAION-CLAP model (~600 MB)

The model must be placed at `data/input/630k-audioset-fusion-best.pt`.

**Option A — PowerShell (Windows):**
```powershell
Invoke-WebRequest `
  -Uri "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt" `
  -OutFile "data/input/630k-audioset-fusion-best.pt"
```

**Option B — wget (Linux/Mac/WSL):**
```bash
wget https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt \
  -P data/input/
```

**Option C — Manual download:**  
Visit 👉 https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt  
and save the file to `data/input/`.

---

## Step 3: Verify file structure

After setup, your directory should look like:

```
audio_text_proto/
├── app.py
├── classifier.py
├── requirements_app.txt
├── data/
│   ├── input/
│   │   ├── 630k-audioset-fusion-best.pt   ✅ ~600 MB (from step 2)
│   │   └── ESC-50/
│   │       └── meta/
│   │           └── esc50.csv              ✅ ~95 KB (already present)
│   └── demo/
│       ├── mean_embd_tensor_esc50_clap_zs.pt  ✅ already present
│       └── airplane_demo.wav                   ✅ already present
└── temp/                                      (auto-created)
```

---

## Step 4: Run the app

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: 630k-audioset-fusion-best.pt` | Complete Step 2 |
| `FileNotFoundError: esc50.csv` | Complete Step 3 |
| `ModuleNotFoundError: laion_clap` | Run `pip install laion-clap` |
| Audio upload fails | Ensure file is `.wav` or `.mp3`, under 10 MB |
| Audio too short | Minimum audio length is 1 second |
| Slow first classification | Model loads into memory on first run (~10–30s) |

---

## Notes

- **No training required** — the model is pre-trained, prototypes are pre-computed.
- **First run** will be slower (~30s) as LAION-CLAP model loads into memory.
- **Subsequent classifications** are fast (~1–5s per audio file).
- The prototype embedding file (`mean_embd_tensor_esc50_clap_zs.pt`) is already bundled in `data/demo/`.
