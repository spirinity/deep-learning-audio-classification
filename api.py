"""
api.py — FastAPI backend for the ESC-50 sound classification comparison app.

Reuses the existing, UI-agnostic SoundClassifier (classifier.py) and exposes it
over HTTP so a modern frontend (Next.js) can consume it.

Run:
    uvicorn api:app --port 8000 --reload
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from classifier import ESC50_CATEGORIES, LABEL_TO_CATEGORY, SoundClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# ── Paths (mirror app.py) ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "input", "630k-audioset-fusion-best.pt")
PROTOTYPE_PATH = os.path.join(BASE_DIR, "data", "demo", "mean_embd_tensor_esc50_clap_zs.pt")
LABEL_CSV_PATH = os.path.join(BASE_DIR, "data", "labels", "esc50.csv")
LOGREG_PATH = os.path.join(BASE_DIR, "data", "demo", "logreg_esc50_clap.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "data", "demo", "comparison_metrics.csv")

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3"}
TOP_N = 10

# Single source of truth for category colors (frontend reads these via /api/categories).
CATEGORY_COLORS = {
    "Animals": "#f78166",
    "Natural soundscapes": "#56d364",
    "Human non-speech": "#ffa657",
    "Interior/domestic": "#58a6ff",
    "Exterior/urban": "#a371f7",
    "Unknown": "#8b949e",
}

# Required-file flags mirror app.py:check_setup (Logistic Regression is optional).
REQUIRED_FILES = {
    "LAION-CLAP model": (MODEL_PATH, True),
    "Prototype embeddings": (PROTOTYPE_PATH, True),
    "ESC-50 label CSV": (LABEL_CSV_PATH, True),
    "Logistic Regression artifact": (LOGREG_PATH, False),
}

# Populated at startup.
state: dict = {"classifier": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LAION-CLAP model + artifacts once at startup (~30s, ~1.78GB)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        logger.info("Loading SoundClassifier (model + prototypes + text + logreg)...")
        clf = SoundClassifier(MODEL_PATH, PROTOTYPE_PATH, LABEL_CSV_PATH, LOGREG_PATH)
        clf.load_model()
        clf.load_prototypes()
        clf.load_labels()
        clf.load_text_embeddings()
        clf.load_logistic_model()
        state["classifier"] = clf
        logger.info("Classifier ready. logreg_available=%s", clf.is_logistic_ready())
    except Exception as e:  # noqa: BLE001 — surface load failure via /api/health
        state["load_error"] = str(e)
        logger.exception("Failed to load classifier: %s", e)
    yield
    state["classifier"] = None


app = FastAPI(title="Sound Classifier Comparison API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_classifier() -> SoundClassifier:
    clf = state.get("classifier")
    if clf is None:
        raise HTTPException(
            status_code=503,
            detail=state.get("load_error") or "Model is still loading. Try again shortly.",
        )
    return clf


@app.get("/api/health")
def health():
    """Setup status: which files are present, and whether the model is ready."""
    files = []
    all_required_ok = True
    for name, (path, required) in REQUIRED_FILES.items():
        exists = os.path.exists(path)
        if required and not exists:
            all_required_ok = False
        files.append({
            "item": name,
            "required": required,
            "exists": exists,
            "path": os.path.relpath(path, BASE_DIR).replace(os.sep, "/"),
        })
    clf = state.get("classifier")
    return {
        "ready": clf is not None and clf.is_ready(),
        "loading": clf is None and state.get("load_error") is None,
        "load_error": state.get("load_error"),
        "required_ok": all_required_ok,
        "files": files,
    }


@app.get("/api/methods")
def methods():
    """Per-method availability + descriptions (Zero-Shot, Proto-LC, Logistic Regression)."""
    clf = _get_classifier()
    status = clf.get_method_status()
    return {
        "methods": [
            {
                "name": name,
                "available": info["available"],
                "description": info["description"],
                "setup": info["setup"],
            }
            for name, info in status.items()
        ]
    }


@app.get("/api/categories")
def categories():
    """ESC-50 categories, their labels, and the color map used by the UI."""
    return {
        "categories": [
            {"category": cat, "color": CATEGORY_COLORS.get(cat, "#8b949e"), "labels": labels}
            for cat, labels in ESC50_CATEGORIES.items()
        ],
        "colors": CATEGORY_COLORS,
        "label_to_category": LABEL_TO_CATEGORY,
    }


@app.get("/api/metrics")
def metrics():
    """Mean per-method ESC-50 evaluation metrics from comparison_metrics.csv."""
    if not os.path.exists(METRICS_PATH):
        return {"available": False, "rows": []}
    df = pd.read_csv(METRICS_PATH)
    if df.empty:
        return {"available": False, "rows": []}
    summary = (
        df.groupby("method", as_index=False)
        .agg({
            "accuracy": "mean",
            "macro_f1": "mean",
            "top3_accuracy": "mean",
            "avg_inference_time_sec": "mean",
        })
        .sort_values("accuracy", ascending=False)
    )
    rows = [
        {
            "method": r["method"],
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "top3_accuracy": float(r["top3_accuracy"]),
            "avg_inference_time_sec": float(r["avg_inference_time_sec"]),
        }
        for _, r in summary.iterrows()
    ]
    return {"available": True, "rows": rows}


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    """Encode one uploaded audio file with CLAP and compare all available methods."""
    clf = _get_classifier()

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .wav or .mp3.")

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            results = clf.classify_all(tmp_path, top_n=TOP_N)
        except ValueError as e:
            # Too short / unreadable audio — client error.
            raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "results": results,
    }
