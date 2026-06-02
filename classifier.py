"""
classifier.py

Sound classification utilities for the Streamlit app and method comparison.

Methods:
  1. Zero-Shot CLAP: audio embedding vs prompted ESC-50 text embeddings
  2. Proto-LC: audio embedding vs pre-computed ESC-50 class prototypes
  3. Logistic Regression: supervised classifier on LAION-CLAP audio embeddings
"""

import logging
import os
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


ESC50_CATEGORIES = {
    "Animals": [
        "dog", "rooster", "pig", "cow", "frog",
        "cat", "hen", "insects", "sheep", "crow",
    ],
    "Natural soundscapes": [
        "rain", "sea waves", "crackling fire", "crickets",
        "chirping birds", "water drops", "wind",
        "pouring water", "toilet flush", "thunderstorm",
    ],
    "Human non-speech": [
        "crying baby", "sneezing", "clapping", "breathing",
        "coughing", "footsteps", "laughing", "brushing teeth",
        "snoring", "drinking sipping",
    ],
    "Interior/domestic": [
        "door knock", "mouse click", "keyboard typing",
        "door wood creak", "can opening", "washing machine",
        "vacuum cleaner", "clock alarm", "clock tick", "glass breaking",
    ],
    "Exterior/urban": [
        "helicopter", "chainsaw", "siren", "car horn",
        "engine", "train", "church bells", "airplane",
        "fireworks", "hand saw",
    ],
}


def _build_label_to_category() -> dict:
    mapping = {}
    for category, labels in ESC50_CATEGORIES.items():
        for label in labels:
            mapping[label] = category
    return mapping


LABEL_TO_CATEGORY = _build_label_to_category()

METHOD_DESCRIPTIONS = {
    "Zero-Shot CLAP": "Audio embedding dibandingkan langsung dengan text embedding label ESC-50.",
    "Proto-LC": "Audio embedding dibandingkan dengan prototype kelas ESC-50 berbasis paper.",
    "Logistic Regression": "Classifier supervised yang dilatih di atas embedding audio LAION-CLAP.",
}


class SoundClassifier:
    """Compare sound classification methods on LAION-CLAP embeddings."""

    TARGET_SR = 48000
    ZERO_SHOT_PROMPT = "This is a sound of {}"

    def __init__(
        self,
        model_path: str,
        prototype_path: str,
        label_csv_path: str,
        logreg_model_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.prototype_path = prototype_path
        self.label_csv_path = label_csv_path
        self.logreg_model_path = logreg_model_path

        self.model = None
        self.mean_embd = None
        self.label_map = None
        self.text_embd = None
        self.logreg_model = None

        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def load_model(self):
        """Load LAION-CLAP model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LAION-CLAP model not found at: {self.model_path}\n"
                "Download it from:\n"
                "  https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt\n"
                f"and place it at: {self.model_path}"
            )

        try:
            import laion_clap

            model = laion_clap.CLAP_Module(enable_fusion=True)
            model.load_ckpt(ckpt=self.model_path)
            self.model = model
            logger.info("LAION-CLAP model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load LAION-CLAP model: {e}") from e

    def load_prototypes(self):
        """Load pre-computed ESC-50 Proto-LC embeddings."""
        if not os.path.exists(self.prototype_path):
            raise FileNotFoundError(
                f"Prototype embeddings not found at: {self.prototype_path}\n"
                "Expected file: mean_embd_tensor_esc50_clap_zs.pt"
            )
        self.mean_embd = torch.load(self.prototype_path, map_location="cpu")
        self.mean_embd = torch.as_tensor(self.mean_embd, dtype=torch.float32)
        logger.info("Prototype embeddings loaded with shape %s.", tuple(self.mean_embd.shape))

    def load_labels(self):
        """Load ESC-50 label map as {target_index: label_name}."""
        if not os.path.exists(self.label_csv_path):
            raise FileNotFoundError(
                f"ESC-50 CSV not found at: {self.label_csv_path}\n"
                "Expected file: data/labels/esc50.csv"
            )

        df = pd.read_csv(self.label_csv_path)
        df["category"] = df["category"].apply(lambda x: " ".join(str(x).split("_")))
        label_map = dict(zip(df["target"].tolist(), df["category"].tolist()))
        self.label_map = {int(k): v for k, v in label_map.items()}
        logger.info("Label map loaded with %d classes.", len(self.label_map))

    def load_text_embeddings(self):
        """Load zero-shot text embeddings for prompted ESC-50 labels."""
        if self.model is None:
            raise RuntimeError("Load LAION-CLAP before loading text embeddings.")
        if self.label_map is None:
            raise RuntimeError("Load labels before loading text embeddings.")

        prompts = [
            self.ZERO_SHOT_PROMPT.format(self.label_map[idx])
            for idx in sorted(self.label_map)
        ]
        text_embd = self.model.get_text_embedding(prompts)
        self.text_embd = torch.as_tensor(text_embd, dtype=torch.float32)
        logger.info("Zero-shot text embeddings loaded.")

    def load_logistic_model(self):
        """Load optional Logistic Regression artifact if it exists."""
        if not self.logreg_model_path or not os.path.exists(self.logreg_model_path):
            self.logreg_model = None
            return

        try:
            import joblib
        except ImportError as e:
            raise RuntimeError(
                "joblib is required to load Logistic Regression. "
                "Install requirements_app.txt again."
            ) from e

        self.logreg_model = joblib.load(self.logreg_model_path)
        logger.info("Logistic Regression model loaded from %s.", self.logreg_model_path)

    def is_ready(self) -> bool:
        return (
            self.model is not None
            and self.mean_embd is not None
            and self.label_map is not None
        )

    def is_zero_shot_ready(self) -> bool:
        return self.text_embd is not None

    def is_logistic_ready(self) -> bool:
        return self.logreg_model is not None

    def check_paths(self) -> dict:
        return {
            "LAION-CLAP model": (self.model_path, os.path.exists(self.model_path)),
            "Prototype embeddings": (self.prototype_path, os.path.exists(self.prototype_path)),
            "ESC-50 CSV": (self.label_csv_path, os.path.exists(self.label_csv_path)),
        }

    def get_method_status(self) -> dict:
        return {
            "Zero-Shot CLAP": {
                "available": self.is_zero_shot_ready(),
                "description": METHOD_DESCRIPTIONS["Zero-Shot CLAP"],
                "setup": "Loaded from LAION-CLAP text encoder.",
            },
            "Proto-LC": {
                "available": self.mean_embd is not None,
                "description": METHOD_DESCRIPTIONS["Proto-LC"],
                "setup": os.path.relpath(self.prototype_path, os.path.dirname(__file__)),
            },
            "Logistic Regression": {
                "available": self.is_logistic_ready(),
                "description": METHOD_DESCRIPTIONS["Logistic Regression"],
                "setup": (
                    os.path.relpath(self.logreg_model_path, os.path.dirname(__file__))
                    if self.logreg_model_path
                    else "Run evaluate_methods.py first."
                ),
            },
        }

    def preprocess_audio(self, audio_path: str) -> str:
        """Convert uploaded audio to 48 kHz mono WAV."""
        try:
            y, _ = librosa.load(audio_path, sr=self.TARGET_SR, mono=True)
        except Exception as e:
            raise ValueError(f"Could not read audio file: {e}") from e

        duration = len(y) / self.TARGET_SR
        if duration < 1.0:
            raise ValueError(
                f"Audio is too short ({duration:.2f}s). Minimum required: 1 second."
            )

        basename = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(self.temp_dir, f"{basename}_preprocessed.wav")
        sf.write(out_path, y, self.TARGET_SR)
        return out_path

    def encode_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess and encode one audio file into a 1D CLAP embedding."""
        processed_path = self.preprocess_audio(audio_path)
        audio_embd = self.model.get_audio_embedding_from_filelist([processed_path])
        audio_embd = torch.as_tensor(audio_embd, dtype=torch.float32)
        if audio_embd.ndim == 2:
            audio_embd = audio_embd.squeeze(0)
        return audio_embd

    def _score_from_cosine(self, similarities: np.ndarray) -> np.ndarray:
        return np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)

    def _rank_scores(
        self,
        raw_scores: np.ndarray,
        top_n: int,
        method: str,
        score_mode: str,
    ) -> list:
        top_n = min(top_n, len(self.label_map))
        raw_scores = np.asarray(raw_scores, dtype=np.float32)
        sorted_indices = np.argsort(-raw_scores)
        display_scores = (
            self._score_from_cosine(raw_scores)
            if score_mode == "cosine"
            else np.clip(raw_scores, 0.0, 1.0)
        )

        results = []
        for rank, idx in enumerate(sorted_indices[:top_n], start=1):
            idx_int = int(idx)
            label_name = self.label_map.get(idx_int, f"class_{idx_int}")
            category = LABEL_TO_CATEGORY.get(label_name, "Unknown")
            raw_score = float(raw_scores[idx_int])

            results.append({
                "method": method,
                "rank": rank,
                "label": label_name,
                "score": float(display_scores[idx_int]),
                "raw_score": raw_score,
                "distance": float(1.0 - raw_score) if score_mode == "cosine" else None,
                "category": category,
            })

        return results

    def _cosine_scores(self, audio_embd: torch.Tensor, references: torch.Tensor) -> np.ndarray:
        references = torch.as_tensor(references, dtype=audio_embd.dtype)
        similarities = torch.nn.functional.cosine_similarity(
            references,
            audio_embd.unsqueeze(0),
            dim=1,
        )
        return similarities.detach().cpu().numpy()

    def classify_zero_shot(self, audio_embd: torch.Tensor, top_n: int = 10) -> list:
        if self.text_embd is None:
            self.load_text_embeddings()
        raw_scores = self._cosine_scores(audio_embd, self.text_embd)
        return self._rank_scores(raw_scores, top_n, "Zero-Shot CLAP", "cosine")

    def classify_proto_lc(self, audio_embd: torch.Tensor, top_n: int = 10) -> list:
        raw_scores = self._cosine_scores(audio_embd, self.mean_embd)
        return self._rank_scores(raw_scores, top_n, "Proto-LC", "cosine")

    def classify_logistic_regression(self, audio_embd: torch.Tensor, top_n: int = 10) -> list:
        if self.logreg_model is None:
            raise RuntimeError(
                "Logistic Regression model is not available. "
                "Run evaluate_methods.py to create data/demo/logreg_esc50_clap.joblib."
            )

        features = audio_embd.detach().cpu().numpy().reshape(1, -1)
        probabilities = self.logreg_model.predict_proba(features)[0]
        class_scores = np.zeros(len(self.label_map), dtype=np.float32)
        classes = getattr(self.logreg_model, "classes_", None)
        if classes is None and hasattr(self.logreg_model, "steps"):
            classes = self.logreg_model.steps[-1][1].classes_
        if classes is None:
            classes = np.arange(len(probabilities))
        for class_idx, probability in zip(classes, probabilities):
            class_scores[int(class_idx)] = float(probability)

        return self._rank_scores(
            class_scores,
            top_n,
            "Logistic Regression",
            "probability",
        )

    def classify(self, audio_path: str, top_n: int = 10) -> list:
        """Backward-compatible single-method classifier using Proto-LC."""
        if not self.is_ready():
            raise RuntimeError(
                "Classifier not fully initialized. "
                "Call load_model(), load_prototypes(), and load_labels() first."
            )
        audio_embd = self.encode_audio(audio_path)
        return self.classify_proto_lc(audio_embd, top_n=top_n)

    def classify_all(self, audio_path: str, top_n: int = 10) -> dict:
        """Run all available comparison methods for one audio file."""
        if not self.is_ready():
            raise RuntimeError(
                "Classifier not fully initialized. "
                "Call load_model(), load_prototypes(), and load_labels() first."
            )

        audio_embd = self.encode_audio(audio_path)
        results = {
            "Zero-Shot CLAP": self.classify_zero_shot(audio_embd, top_n=top_n),
            "Proto-LC": self.classify_proto_lc(audio_embd, top_n=top_n),
        }

        if self.logreg_model is not None:
            results["Logistic Regression"] = self.classify_logistic_regression(
                audio_embd,
                top_n=top_n,
            )

        return results

    def cleanup_temp(self):
        """Remove all files in the temp directory."""
        for filename in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, filename))
            except Exception:
                pass
