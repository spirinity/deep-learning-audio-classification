"""
classifier.py
Wrapper around the Proto-LC inference logic from demo.py (INTERSPEECH 2023).

Logic:
  1. Load LAION-CLAP model (get_clap_model from common_utils)
  2. Load pre-computed ESC-50 prototype embeddings
  3. Load label map from esc50.csv
  4. Preprocess uploaded audio to 48kHz mono WAV
  5. Encode audio → embedding via CLAP
  6. Euclidean distance to each prototype → ranked similarity scores
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tempfile
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ESC-50 Category Mapping (based on ESC-50 paper)
# ─────────────────────────────────────────────
ESC50_CATEGORIES = {
    "Animals": [
        "dog", "rooster", "pig", "cow", "frog",
        "cat", "hen", "insects", "sheep", "crow"
    ],
    "Natural soundscapes": [
        "rain", "sea waves", "crackling fire", "crickets",
        "chirping birds", "water drops", "wind",
        "pouring water", "toilet flush", "thunderstorm"
    ],
    "Human non-speech": [
        "crying baby", "sneezing", "clapping", "breathing",
        "coughing", "footsteps", "laughing", "brushing teeth",
        "snoring", "drinking sipping"
    ],
    "Interior/domestic": [
        "door knock", "mouse click", "keyboard typing",
        "door wood creak", "can opening", "washing machine",
        "vacuum cleaner", "clock alarm", "clock tick", "glass breaking"
    ],
    "Exterior/urban": [
        "helicopter", "chainsaw", "siren", "car horn",
        "engine", "train", "church bells", "airplane",
        "fireworks", "hand saw"
    ]
}

def _build_label_to_category():
    """Build reverse map: label_name → category_name"""
    mapping = {}
    for category, labels in ESC50_CATEGORIES.items():
        for label in labels:
            mapping[label] = category
    return mapping

LABEL_TO_CATEGORY = _build_label_to_category()


class SoundClassifier:
    """
    Proto-LC sound classifier using LAION-CLAP + pre-computed ESC-50 prototypes.
    """

    TARGET_SR = 48000  # LAION-CLAP expects 48 kHz

    def __init__(self, model_path: str, prototype_path: str, label_csv_path: str):
        """
        Parameters
        ----------
        model_path      : path to 630k-audioset-fusion-best.pt
        prototype_path  : path to mean_embd_tensor_esc50_clap_zs.pt
        label_csv_path  : path to ESC-50/meta/esc50.csv
        """
        self.model_path = model_path
        self.prototype_path = prototype_path
        self.label_csv_path = label_csv_path

        self.model = None
        self.mean_embd = None   # shape: [50, embd_dim]
        self.label_map = None   # dict {int_index: label_name}

        # Temp dir for preprocessed audio
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # Loading helpers
    # ──────────────────────────────────────────

    def load_model(self):
        """Load LAION-CLAP model. Mirrors get_clap_model() in common_utils.py."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LAION-CLAP model not found at: {self.model_path}\n"
                "Please download it from:\n"
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
        """Load pre-computed ESC-50 prototype embeddings."""
        if not os.path.exists(self.prototype_path):
            raise FileNotFoundError(
                f"Prototype embeddings not found at: {self.prototype_path}\n"
                "Expected file: mean_embd_tensor_esc50_clap_zs.pt"
            )
        self.mean_embd = torch.load(self.prototype_path, map_location="cpu")
        logger.info(f"Prototype embeddings loaded — shape: {self.mean_embd.shape}")

    def load_labels(self):
        """
        Load ESC-50 label map from esc50.csv.
        Replicates get_esc50_labels() from common_utils.py.
        Returns dict: {int_index: label_name}
        """
        if not os.path.exists(self.label_csv_path):
            raise FileNotFoundError(
                f"ESC-50 CSV not found at: {self.label_csv_path}\n"
                "This file should be bundled with the repository at: data/labels/esc50.csv"
            )
        df = pd.read_csv(self.label_csv_path)
        df["category"] = df["category"].apply(lambda x: " ".join(x.split("_")))
        label_map = dict(zip(df["target"].tolist(), df["category"].tolist()))
        self.label_map = label_map
        logger.info(f"Label map loaded — {len(label_map)} classes.")

    def is_ready(self) -> bool:
        """Check if all components are loaded."""
        return (self.model is not None and
                self.mean_embd is not None and
                self.label_map is not None)

    def check_paths(self) -> dict:
        """
        Returns a dict of {name: (path, exists)} for setup verification in UI.
        """
        return {
            "LAION-CLAP model": (self.model_path, os.path.exists(self.model_path)),
            "Prototype embeddings": (self.prototype_path, os.path.exists(self.prototype_path)),
            "ESC-50 CSV": (self.label_csv_path, os.path.exists(self.label_csv_path)),
        }

    # ──────────────────────────────────────────
    # Audio preprocessing
    # ──────────────────────────────────────────

    def preprocess_audio(self, audio_path: str) -> str:
        """
        Convert audio to 48 kHz mono WAV (required by LAION-CLAP).
        Saves to temp/ and returns the new path.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.TARGET_SR, mono=True)
        except Exception as e:
            raise ValueError(f"Could not read audio file: {e}") from e

        duration = len(y) / self.TARGET_SR
        if duration < 1.0:
            raise ValueError(
                f"Audio is too short ({duration:.2f}s). Minimum required: 1 second."
            )

        # Write preprocessed audio to temp dir
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(self.temp_dir, f"{basename}_preprocessed.wav")
        sf.write(out_path, y, self.TARGET_SR)
        logger.info(f"Preprocessed audio saved to: {out_path} ({duration:.2f}s)")
        return out_path

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────

    def classify(self, audio_path: str, top_n: int = 10) -> list:
        """
        Classify audio file against 50 ESC-50 prototype embeddings.

        Parameters
        ----------
        audio_path : path to .wav or .mp3 file
        top_n      : number of top predictions to return (max 50)

        Returns
        -------
        List of dicts:
            [
                {
                    "rank": 1,
                    "label": "dog",
                    "score": 0.823,    # normalized [0, 1]
                    "distance": 0.234, # raw euclidean distance
                    "category": "Animals"
                },
                ...
            ]
        """
        if not self.is_ready():
            raise RuntimeError(
                "Classifier not fully initialized. "
                "Call load_model(), load_prototypes(), and load_labels() first."
            )

        top_n = min(top_n, len(self.label_map))

        # Step 1: Preprocess audio
        processed_path = self.preprocess_audio(audio_path)

        # Step 2: Encode audio → embedding (mirrors demo.py)
        audio_embd = self.model.get_audio_embedding_from_filelist([processed_path])

        # Step 3: Euclidean distance to each prototype (from demo.py)
        mean_embd = self.mean_embd
        distances = torch.sum((mean_embd - audio_embd) ** 2, dim=1).sqrt()

        # Step 4: Get all 50 distances and scores
        all_distances = distances.detach().cpu().numpy()
        # Convert to similarity: invert distance, normalize to [0, 1]
        similarities = 1.0 / (1.0 + all_distances)
        max_sim = similarities.max()
        if max_sim > 0:
            normalized_scores = similarities / max_sim
        else:
            normalized_scores = similarities

        # Step 5: Get sorted indices (closest first)
        sorted_indices = np.argsort(all_distances)

        results = []
        for rank, idx in enumerate(sorted_indices[:top_n], start=1):
            idx_int = int(idx)
            label_name = self.label_map.get(idx_int, f"class_{idx_int}")
            category = LABEL_TO_CATEGORY.get(label_name, "Unknown")

            results.append({
                "rank": rank,
                "label": label_name,
                "score": float(normalized_scores[idx_int]),
                "raw_score": float(similarities[idx_int]),
                "distance": float(all_distances[idx_int]),
                "category": category,
            })

        return results

    # ──────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────

    def cleanup_temp(self):
        """Remove all files in the temp/ directory."""
        for f in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass
