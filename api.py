"""
FastAPI backend for the ESC-50 sound classification comparison app.

The LAION-CLAP checkpoint is preloaded when the backend starts. The preload runs
in a background thread so /api/health can report loading and failure states while
the server stays reachable.

Run:
    uvicorn api:app --port 8000
"""

import gc
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from classifier import ESC50_CATEGORIES, LABEL_TO_CATEGORY, SoundClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "input", "630k-audioset-fusion-best.pt")
PROTOTYPE_PATH = os.path.join(BASE_DIR, "data", "demo", "mean_embd_tensor_esc50_clap_zs.pt")
LABEL_CSV_PATH = os.path.join(BASE_DIR, "data", "labels", "esc50.csv")
LOGREG_PATH = os.path.join(BASE_DIR, "data", "demo", "logreg_esc50_clap.joblib")
MLP_PATH = os.path.join(BASE_DIR, "data", "demo", "mlp_esc50_clap.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "data", "demo", "comparison_metrics.csv")

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3"}
TOP_N = 10

CATEGORY_COLORS = {
    "Animals": "#f78166",
    "Natural soundscapes": "#56d364",
    "Human non-speech": "#ffa657",
    "Interior/domestic": "#58a6ff",
    "Exterior/urban": "#a371f7",
    "Unknown": "#8b949e",
}

REQUIRED_FILES = {
    "LAION-CLAP model": (MODEL_PATH, True),
    "Prototype embeddings": (PROTOTYPE_PATH, True),
    "ESC-50 label CSV": (LABEL_CSV_PATH, True),
    "Supervised MLP artifact": (MLP_PATH, False),
}

state: dict = {"classifier": None, "load_error": None, "loading": False}
load_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _start_preload()
    yield
    state["classifier"] = None
    gc.collect()


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


def _rel(path: str) -> str:
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def _required_files_ok() -> bool:
    return all(
        os.path.exists(path)
        for path, required in REQUIRED_FILES.values()
        if required
    )


def _preload_classifier() -> None:
    if not _required_files_ok():
        state["load_error"] = "Required files are missing."
        return
    try:
        _load_classifier()
    except HTTPException:
        pass


def _start_preload() -> None:
    thread = threading.Thread(target=_preload_classifier, name="classifier-preload", daemon=True)
    thread.start()


def _load_classifier() -> SoundClassifier:
    """Load model and artifacts once."""
    clf = state.get("classifier")
    if clf is not None:
        return clf

    with load_lock:
        clf = state.get("classifier")
        if clf is not None:
            return clf

        state["loading"] = True
        state["load_error"] = None
        try:
            logger.info("Loading SoundClassifier (model + prototypes + text + mlp)...")
            clf = SoundClassifier(MODEL_PATH, PROTOTYPE_PATH, LABEL_CSV_PATH, LOGREG_PATH, MLP_PATH)
            clf.load_model()
            clf.load_prototypes()
            clf.load_labels()
            clf.load_text_embeddings()
            clf.load_mlp_model()
            state["classifier"] = clf
            logger.info("Classifier ready. mlp_available=%s", clf.is_mlp_ready())
            return clf
        except Exception as exc:
            state["load_error"] = str(exc)
            logger.exception("Failed to load classifier: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Failed to load LAION-CLAP model. Close Streamlit or other heavy apps, "
                    "then retry. If it still fails, increase Windows virtual memory/pagefile. "
                    f"Error: {exc}"
                ),
            ) from exc
        finally:
            state["loading"] = False
            gc.collect()


def _method_status_without_loaded_model() -> dict:
    return {
        "Zero-Shot CLAP": {
            "available": False,
            "description": "Audio embedding dibandingkan langsung dengan text embedding label ESC-50.",
            "setup": "Preloaded when the backend starts.",
        },
        "Proto-LC": {
            "available": os.path.exists(PROTOTYPE_PATH),
            "description": "Audio embedding dibandingkan dengan prototype kelas ESC-50 berbasis paper.",
            "setup": _rel(PROTOTYPE_PATH),
        },
        "Supervised MLP": {
            "available": os.path.exists(MLP_PATH),
            "description": "Shallow neural network supervised yang dilatih pada embedding audio LAION-CLAP.",
            "setup": _rel(MLP_PATH),
        },
    }


@app.get("/api/health")
def health():
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
            "path": _rel(path),
        })

    clf = state.get("classifier")
    return {
        "ready": clf is not None and clf.is_ready(),
        "loading": bool(state.get("loading")),
        "load_error": state.get("load_error"),
        "required_ok": all_required_ok,
        "files": files,
    }


@app.get("/api/methods")
def methods():
    clf = state.get("classifier")
    status = clf.get_method_status() if clf is not None else _method_status_without_loaded_model()
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
    if not os.path.exists(METRICS_PATH):
        return {"available": False, "rows": []}

    df = pd.read_csv(METRICS_PATH)
    active_methods = {"Zero-Shot CLAP", "Proto-LC", "Supervised MLP"}
    df = df[df["method"].isin(active_methods)]
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
            "method": row["method"],
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "top3_accuracy": float(row["top3_accuracy"]),
            "avg_inference_time_sec": float(row["avg_inference_time_sec"]),
        }
        for _, row in summary.iterrows()
    ]
    return {"available": True, "rows": rows}


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    clf = _load_classifier()

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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
