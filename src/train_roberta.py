#!/usr/bin/env python3
"""
Task 14: Fine-tune RoBERTa for sarcasm detection.

This script trains RoBERTa on the two controlled preprocessing versions:
- E03: Version A, stopwords kept
- E04: Version B, stopwords selectively removed

The design intentionally mirrors Task 13 BERTweet training so that BERTweet
and RoBERTa can be compared fairly under the same split, sample size,
max length, and evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "task14_roberta_config.json"


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names without changing their meaning."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Find a column robustly using case/spacing-insensitive matching."""
    normalized_to_original = {
        str(col).strip().lower().replace(" ", "_"): col for col in df.columns
    }
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized_to_original:
            return str(normalized_to_original[key])
    raise KeyError(
        f"Could not find any of {list(candidates)} in columns: {list(df.columns)}"
    )


def find_label_col(df: pd.DataFrame) -> str:
    return find_column(
        df,
        [
            "label",
            "labels",
            "class",
            "target",
            "is_sarcastic",
            "sarcasm",
        ],
    )


def build_text_input(df: pd.DataFrame) -> List[str]:
    """Build text-only input from parent_comment + comment when available."""
    comment_col = find_column(df, ["comment", "text", "body", "reply"])

    parent_col: Optional[str]
    try:
        parent_col = find_column(df, ["parent_comment", "parent", "context"])
    except KeyError:
        parent_col = None

    comments = df[comment_col].fillna("").astype(str).str.strip()

    if parent_col is None:
        texts = comments
    else:
        parents = df[parent_col].fillna("").astype(str).str.strip()
        texts = "Parent: " + parents + " Reply: " + comments

    texts = texts.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return texts.tolist()


def load_split(
    path: Path,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)

    label_col = find_label_col(df)

    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].copy()

    texts = build_text_input(df)
    labels = df[label_col].astype(int).tolist()

    metadata = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_col": label_col,
        "label_distribution": {
            str(k): int(v) for k, v in df[label_col].value_counts().sort_index().to_dict().items()
        },
    }
    return texts, labels, metadata


def choose_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        preferred = preferred.lower().strip()
        if preferred == "cpu":
            return torch.device("cpu")
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_texts(
    tokenizer: Any,
    texts: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def make_loader(
    encodings: Dict[str, torch.Tensor],
    labels: List[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TextClassificationDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["non_sarcastic", "sarcastic"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": {
            "non_sarcastic_0": {
                "precision": float(class_precision[0]),
                "recall": float(class_recall[0]),
                "f1": float(class_f1[0]),
                "support": int(class_support[0]),
            },
            "sarcastic_1": {
                "precision": float(class_precision[1]),
                "recall": float(class_recall[1]),
                "f1": float(class_f1[1]),
                "support": int(class_support[1]),
            },
        },
        "confusion_matrix": {
            "labels": ["non_sarcastic_0", "sarcastic_1"],
            "matrix": cm.astype(int).tolist(),
        },
        "classification_report": report,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    losses: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            losses.append(float(loss.detach().cpu().item()))
            all_preds.extend(preds.detach().cpu().numpy().astype(int).tolist())
            all_labels.extend(batch["labels"].detach().cpu().numpy().astype(int).tolist())

    avg_loss = float(np.mean(losses)) if losses else math.nan
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def train_one_experiment(
    experiment: Dict[str, Any],
    config: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    training_args = config["training_args"].copy()

    if args.full_data:
        max_train_samples = None
        max_valid_samples = None
    else:
        max_train_samples = args.max_train_samples
        max_valid_samples = args.max_valid_samples
        if max_train_samples is None:
            max_train_samples = training_args.get("max_train_samples")
        if max_valid_samples is None:
            max_valid_samples = training_args.get("max_valid_samples")

    epochs = int(training_args["epochs"])
    batch_size = int(training_args["batch_size"])
    grad_accum = int(training_args["gradient_accumulation_steps"])
    learning_rate = float(training_args["learning_rate"])
    weight_decay = float(training_args["weight_decay"])
    warmup_ratio = float(training_args["warmup_ratio"])
    max_length = int(training_args["max_length"])
    log_every = int(training_args.get("log_every", 100))

    train_path = resolve_path(experiment["train_path"])
    valid_path = resolve_path(experiment["valid_path"])
    output_dir = resolve_path(experiment["output_dir"])
    metrics_path = resolve_path(experiment["metrics_path"])

    if output_dir.exists() and not args.overwrite:
        existing_metrics = metrics_path if metrics_path.exists() else None
        if existing_metrics:
            print(f"Skipping {experiment['name']} because output exists: {output_dir}")
            return load_json(existing_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Training {experiment['name']}")
    print(f"Dataset version: {experiment['dataset_version']}")
    print(f"Train path: {train_path}")
    print(f"Valid path: {valid_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    print("Loading split data...")
    train_texts, train_labels, train_meta = load_split(train_path, max_train_samples)
    valid_texts, valid_labels, valid_meta = load_split(valid_path, max_valid_samples)

    print(f"Train rows used: {len(train_texts):,} | label distribution: {train_meta['label_distribution']}")
    print(f"Valid rows used: {len(valid_texts):,} | label distribution: {valid_meta['label_distribution']}")

    model_name = config["model_name"]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print("Tokenizing training and validation data...")
    train_encodings = tokenize_texts(tokenizer, train_texts, max_length)
    valid_encodings = tokenize_texts(tokenizer, valid_texts, max_length)

    train_loader = make_loader(train_encodings, train_labels, batch_size=batch_size, shuffle=True)
    valid_loader = make_loader(valid_encodings, valid_labels, batch_size=batch_size, shuffle=False)

    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(config.get("num_labels", 2)),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_training_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(
        "Training settings: "
        f"epochs={epochs}, batch_size={batch_size}, grad_accum={grad_accum}, "
        f"lr={learning_rate}, max_length={max_length}, device={device}"
    )

    start_time = time.time()
    best_macro_f1 = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            outputs = model(**batch)
            original_loss = outputs.loss
            loss = original_loss / grad_accum
            loss.backward()

            running_loss += float(original_loss.detach().cpu().item())
            step_count += 1

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                progress.set_postfix({"loss": f"{running_loss / max(step_count, 1):.4f}"})

        train_loss = running_loss / max(step_count, 1)
        valid_loss, valid_metrics = evaluate(model, valid_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "valid_metrics": valid_metrics,
        }
        history.append(epoch_result)

        macro_f1 = float(valid_metrics["macro_f1"])
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
            f"accuracy={valid_metrics['accuracy']:.4f}, macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best checkpoint to {output_dir}")

    duration_seconds = time.time() - start_time
    best_metrics = history[best_epoch - 1]["valid_metrics"] if best_epoch > 0 else {}

    result = {
        "task": "Task 14 - Train RoBERTa",
        "experiment_id": experiment["id"],
        "experiment_name": experiment["name"],
        "model_name": model_name,
        "dataset_version": experiment["dataset_version"],
        "stopword_setting": experiment["stopword_setting"],
        "text_input_strategy": config["text_input"]["strategy"],
        "train_metadata": train_meta,
        "valid_metadata": valid_meta,
        "training_args": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_length": max_length,
            "random_seed": training_args.get("random_seed"),
            "max_train_samples": max_train_samples,
            "max_valid_samples": max_valid_samples,
            "device": str(device),
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "best_validation_metrics": best_metrics,
        "history": history,
        "duration_seconds": duration_seconds,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
    }

    save_json(metrics_path, result)
    print(f"Saved metrics: {metrics_path}")
    return result


def write_summary(results: List[Dict[str, Any]]) -> None:
    report_dir = PROJECT_ROOT / "reports" / "task14"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_md = report_dir / "task14_roberta_training_summary.md"
    summary_json = report_dir / "task14_roberta_training_summary.json"
    progress_note = report_dir / "task14_progress_note.txt"

    save_json(summary_json, {"task": "Task 14 - Train RoBERTa", "results": results})

    lines: List[str] = []
    lines.append("# Task 14: Train RoBERTa")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "This task fine-tunes RoBERTa on Version A and Version B using the same controlled setup as BERTweet. "
        "The goal is to support a fair model comparison between BERTweet and RoBERTa."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Experiment | Dataset Version | Stopword Setting | Accuracy | Macro-F1 | Precision Macro | Recall Macro |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for result in results:
        metrics = result.get("best_validation_metrics", {})
        lines.append(
            f"| {result.get('experiment_name')} "
            f"| {result.get('dataset_version')} "
            f"| {result.get('stopword_setting')} "
            f"| {metrics.get('accuracy', float('nan')):.4f} "
            f"| {metrics.get('macro_f1', float('nan')):.4f} "
            f"| {metrics.get('precision_macro', float('nan')):.4f} "
            f"| {metrics.get('recall_macro', float('nan')):.4f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Version A keeps stopwords.")
    lines.append("- Version B selectively removes stopwords while preserving negation words.")
    lines.append("- The best final model should be selected later after comparing Task 13 and Task 14 results.")
    lines.append("- Local model checkpoints are saved under `models/roberta/` and are intentionally ignored by Git.")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(results) == 2:
        r1, r2 = results
        m1 = r1.get("best_validation_metrics", {})
        m2 = r2.get("best_validation_metrics", {})
        note = (
            "RoBERTa training completed for Version A and Version B using the controlled experiment design. "
            f"Version A achieved validation accuracy {m1.get('accuracy', float('nan')):.4f} and macro-F1 {m1.get('macro_f1', float('nan')):.4f}; "
            f"Version B achieved validation accuracy {m2.get('accuracy', float('nan')):.4f} and macro-F1 {m2.get('macro_f1', float('nan')):.4f}. "
            "Metrics were saved under reports/task14, and local checkpoints were saved under models/roberta."
        )
    else:
        note = (
            "RoBERTa training completed for the selected experiment(s). Metrics were saved under reports/task14, "
            "and local checkpoints were saved under models/roberta."
        )
    progress_note.write_text(note + "\n", encoding="utf-8")

    print(f"Saved summary: {summary_md}")
    print(f"Saved JSON summary: {summary_json}")
    print(f"Saved progress note: {progress_note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 14: Train RoBERTa")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--experiment",
        default=os.environ.get("TASK14_EXPERIMENT", "all"),
        help="Experiment to run: all, E03, E04, or experiment name.",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        default=os.environ.get("TASK14_FULL_DATA", "0") == "1",
        help="Use full train/validation splits instead of practical sample sizes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=os.environ.get("TASK14_OVERWRITE", "0") == "1",
        help="Overwrite existing local checkpoints and metrics.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("TASK14_DEVICE"),
        help="Optional device override: cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=int(os.environ["TASK14_MAX_TRAIN_SAMPLES"])
        if os.environ.get("TASK14_MAX_TRAIN_SAMPLES")
        else None,
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=int(os.environ["TASK14_MAX_VALID_SAMPLES"])
        if os.environ.get("TASK14_MAX_VALID_SAMPLES")
        else None,
    )
    return parser.parse_args()


def select_experiments(config: Dict[str, Any], request: str) -> List[Dict[str, Any]]:
    experiments = config["experiments"]
    request = request.strip()
    if request.lower() == "all":
        return experiments

    selected = [
        exp
        for exp in experiments
        if exp["id"].lower() == request.lower()
        or exp["name"].lower() == request.lower()
    ]
    if not selected:
        raise ValueError(f"No experiment matched request: {request}")
    return selected


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_json(config_path)

    seed = int(config["training_args"].get("random_seed", 42))
    set_seed(seed)
    device = choose_device(args.device)

    print("Task 14: Train RoBERTa")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Experiment request: {args.experiment}")
    print(f"Full data: {args.full_data}")
    print(f"Max train samples override: {args.max_train_samples}")
    print(f"Max valid samples override: {args.max_valid_samples}")

    selected = select_experiments(config, args.experiment)
    results = [train_one_experiment(exp, config, args, device) for exp in selected]
    write_summary(results)
    print("\nDone. Task 14 RoBERTa training completed.")


if __name__ == "__main__":
    main()
