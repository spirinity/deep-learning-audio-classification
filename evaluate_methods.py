"""
Evaluate three ESC-50 sound-classification methods:
Zero-Shot CLAP, Proto-LC, and Logistic Regression on CLAP embeddings.
"""

import argparse
import gc
import logging
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "input" / "630k-audioset-fusion-best.pt"
LABEL_CSV_PATH = BASE_DIR / "data" / "labels" / "esc50.csv"
AUDIO_DIR = BASE_DIR / "data" / "input" / "ESC-50" / "audio"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEMO_DIR = BASE_DIR / "data" / "demo"
EMBEDDING_CACHE_PATH = PROCESSED_DIR / "esc50_clap_embeddings.pt"
METRICS_PATH = DEMO_DIR / "comparison_metrics.csv"
PREDICTIONS_PATH = DEMO_DIR / "comparison_predictions.csv"
LOGREG_MODEL_PATH = DEMO_DIR / "logreg_esc50_clap.joblib"
ZERO_SHOT_PROMPT = "This is a sound of {}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Balanced smoke-test row limit.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--skip-final-model", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def load_label_frame(limit):
    df = pd.read_csv(LABEL_CSV_PATH)
    df["label"] = df["category"].apply(lambda x: " ".join(str(x).split("_")))
    df["audio_path"] = df["filename"].apply(lambda name: str(AUDIO_DIR / name))

    missing = [path for path in df["audio_path"] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing ESC-50 audio file: {missing[0]}")

    if limit is not None:
        per_fold = max(1, limit // int(df["fold"].nunique()))
        df = (
            df.sort_values(["fold", "target", "filename"])
            .groupby("fold", group_keys=False)
            .head(per_fold)
            .head(limit)
            .reset_index(drop=True)
        )
    return df


def label_map_from_frame(df):
    label_df = df[["target", "label"]].drop_duplicates().sort_values("target")
    return dict(zip(label_df["target"].astype(int), label_df["label"]))


def load_clap_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"LAION-CLAP model not found: {MODEL_PATH}")
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=True)
    model.load_ckpt(ckpt=str(MODEL_PATH))
    return model


def encode_audio_files(model, paths, batch_size):
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        embd = model.get_audio_embedding_from_filelist(batch_paths)
        chunks.append(torch.as_tensor(embd, dtype=torch.float32))
        print(f"Encoded {min(start + batch_size, len(paths))}/{len(paths)} audio files")
    return torch.cat(chunks, dim=0)


def load_or_create_embeddings(args, df, model=None):
    if args.limit is None and EMBEDDING_CACHE_PATH.exists() and not args.rebuild_cache:
        payload = torch.load(EMBEDDING_CACHE_PATH, map_location="cpu")
        return (
            torch.as_tensor(payload["embeddings"], dtype=torch.float32),
            np.asarray(payload["targets"], dtype=np.int64),
            np.asarray(payload["folds"], dtype=np.int64),
            list(payload["filenames"]),
        )

    if model is None:
        model = load_clap_model()
    embeddings = encode_audio_files(model, df["audio_path"].tolist(), args.batch_size)
    targets = df["target"].to_numpy(dtype=np.int64)
    folds = df["fold"].to_numpy(dtype=np.int64)
    filenames = df["filename"].tolist()

    if args.limit is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings,
                "targets": targets,
                "folds": folds,
                "filenames": filenames,
            },
            EMBEDDING_CACHE_PATH,
        )
    return embeddings, targets, folds, filenames


def load_text_embeddings(label_map, model=None):
    if model is None:
        model = load_clap_model()
    prompts = [ZERO_SHOT_PROMPT.format(label_map[idx]) for idx in sorted(label_map)]
    text_embd = model.get_text_embedding(prompts)
    return torch.as_tensor(text_embd, dtype=torch.float32)


def cosine_matrix(left, right):
    left_norm = F.normalize(torch.as_tensor(left, dtype=torch.float32), dim=1)
    right_norm = F.normalize(torch.as_tensor(right, dtype=torch.float32), dim=1)
    return (left_norm @ right_norm.T).detach().cpu().numpy()


def build_proto_lc(train_embeddings, text_embeddings, top_k):
    train_norm = F.normalize(train_embeddings, dim=1)
    text_norm = F.normalize(text_embeddings, dim=1)
    retrieval_scores = train_norm @ text_norm.T
    k = min(top_k, train_embeddings.shape[0])
    prototypes = []
    for class_idx in range(text_embeddings.shape[0]):
        top_indices = retrieval_scores[:, class_idx].topk(k).indices
        prototypes.append(train_embeddings[top_indices].mean(dim=0))
    return torch.stack(prototypes, dim=0)


def train_logistic(features, targets, max_iter):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, solver="lbfgs", multi_class="auto"),
    )
    model.fit(features, targets)
    return model


def logistic_scores(model, features, num_classes):
    probabilities = model.predict_proba(features)
    scores = np.zeros((features.shape[0], num_classes), dtype=np.float32)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = model.steps[-1][1].classes_
    for output_idx, class_idx in enumerate(classes):
        scores[:, int(class_idx)] = probabilities[:, output_idx]
    return scores


def topk_accuracy(scores, targets, k):
    topk = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([target in row for target, row in zip(targets, topk)]))


def metric_row(method, fold, scores, targets, elapsed):
    preds = scores.argmax(axis=1)
    return {
        "method": method,
        "fold": int(fold),
        "n_test": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "top3_accuracy": topk_accuracy(scores, targets, 3),
        "avg_inference_time_sec": float(elapsed / max(len(targets), 1)),
    }


def prediction_rows(method, fold, scores, targets, filenames, label_map):
    rows = []
    top3 = np.argsort(-scores, axis=1)[:, :3]
    for idx, filename in enumerate(filenames):
        pred = int(top3[idx, 0])
        rows.append({
            "method": method,
            "fold": int(fold),
            "filename": filename,
            "target": int(targets[idx]),
            "target_label": label_map.get(int(targets[idx]), f"class_{targets[idx]}"),
            "prediction": pred,
            "prediction_label": label_map.get(pred, f"class_{pred}"),
            "top3_labels": "; ".join(label_map.get(int(i), f"class_{i}") for i in top3[idx]),
            "top1_score": float(scores[idx, pred]),
        })
    return rows


def main():
    args = parse_args()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    df = load_label_frame(args.limit)
    label_map = label_map_from_frame(df)
    num_classes = max(label_map) + 1
    needs_audio_model = args.limit is not None or args.rebuild_cache or not EMBEDDING_CACHE_PATH.exists()
    model = load_clap_model()
    embeddings, targets, folds, filenames = load_or_create_embeddings(
        args,
        df,
        model=model if needs_audio_model else None,
    )
    text_embeddings = load_text_embeddings(label_map, model=model)
    del model
    gc.collect()

    metrics = []
    predictions = []
    for fold in sorted(np.unique(folds)):
        train_idx = folds != fold
        test_idx = folds == fold
        if not train_idx.any() or not test_idx.any():
            continue

        train_embeddings = embeddings[train_idx]
        test_embeddings = embeddings[test_idx]
        train_targets = targets[train_idx]
        test_targets = targets[test_idx]
        test_filenames = [name for keep, name in zip(test_idx, filenames) if keep]
        print(f"Evaluating fold {fold}: train={train_idx.sum()} test={test_idx.sum()}")

        start = time.perf_counter()
        scores = cosine_matrix(test_embeddings, text_embeddings)
        elapsed = time.perf_counter() - start
        metrics.append(metric_row("Zero-Shot CLAP", fold, scores, test_targets, elapsed))
        predictions.extend(prediction_rows("Zero-Shot CLAP", fold, scores, test_targets, test_filenames, label_map))

        start = time.perf_counter()
        prototypes = build_proto_lc(train_embeddings, text_embeddings, args.top_k)
        scores = cosine_matrix(test_embeddings, prototypes)
        elapsed = time.perf_counter() - start
        metrics.append(metric_row("Proto-LC", fold, scores, test_targets, elapsed))
        predictions.extend(prediction_rows("Proto-LC", fold, scores, test_targets, test_filenames, label_map))

        if len(np.unique(train_targets)) >= 2:
            start = time.perf_counter()
            logreg = train_logistic(
                train_embeddings.detach().cpu().numpy(),
                train_targets,
                args.max_iter,
            )
            scores = logistic_scores(logreg, test_embeddings.detach().cpu().numpy(), num_classes)
            elapsed = time.perf_counter() - start
            metrics.append(metric_row("Logistic Regression", fold, scores, test_targets, elapsed))
            predictions.extend(prediction_rows("Logistic Regression", fold, scores, test_targets, test_filenames, label_map))

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    pd.DataFrame(predictions).to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote metrics: {METRICS_PATH}")
    print(f"Wrote predictions: {PREDICTIONS_PATH}")

    if args.skip_final_model:
        print(
            "Skipped Logistic Regression artifact because --skip-final-model was used. "
            f"Web app will stay missing until this file exists: {LOGREG_MODEL_PATH}"
        )
    else:
        import joblib

        final_model = train_logistic(
            embeddings.detach().cpu().numpy(),
            targets,
            args.max_iter,
        )
        joblib.dump(final_model, LOGREG_MODEL_PATH)
        if not LOGREG_MODEL_PATH.exists():
            raise RuntimeError(f"Failed to write Logistic Regression model: {LOGREG_MODEL_PATH}")
        print(f"Wrote Logistic Regression model: {LOGREG_MODEL_PATH}")


if __name__ == "__main__":
    main()
