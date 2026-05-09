#!/usr/bin/env python3
"""
Task 13: Train BERTweet for sarcasm detection.

This script trains BERTweet on the two Task 11 split versions:
- Version A: stopwords kept
- Version B: selective stopword removal

It is intentionally robust against small column-name differences in CSV files.
Expected target label: 0 = non-sarcastic, 1 = sarcastic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "task13_bertweet_config.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "task13"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        requested = requested.lower().strip()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_col(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    normalized = {normalize_col(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_col(candidate)
        if key in normalized:
            return normalized[key]
    return None


def detect_label_col(df: pd.DataFrame) -> str:
    candidates = [
        "label", "labels", "is_sarcastic", "sarcasm", "target", "class", "y", "Label"
    ]
    col = find_col(df, candidates)
    if col is None:
        raise KeyError(
            "Could not find label column. Available columns: " + ", ".join(map(str, df.columns))
        )
    return col


def convert_labels(series: pd.Series) -> pd.Series:
    """Convert common label formats to integer 0/1."""
    s = series.copy()
    if s.dtype == object:
        lowered = s.astype(str).str.strip().str.lower()
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "non_sarcastic": 0,
            "non-sarcastic": 0,
            "not_sarcastic": 0,
            "not sarcastic": 0,
            "sarcastic": 1,
            "yes": 1,
            "no": 0,
        }
        mapped = lowered.map(mapping)
        if mapped.notna().sum() > 0:
            return mapped
    return pd.to_numeric(s, errors="coerce")


def build_text_series(df: pd.DataFrame) -> pd.Series:
    """Build text-only input from SARC split columns."""
    text_col = find_col(df, ["text", "input_text", "combined_text", "sentence"])
    comment_col = find_col(df, ["comment", "body", "reply"])
    parent_col = find_col(df, ["parent_comment", "parent", "context"])

    if comment_col and parent_col:
        parent = df[parent_col].fillna("").astype(str).str.strip()
        comment = df[comment_col].fillna("").astype(str).str.strip()
        # BERTweet is RoBERTa-like; </s> is a natural separator token.
        return (parent + " </s> " + comment).str.replace(r"\s+", " ", regex=True).str.strip()

    if text_col:
        return df[text_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    if comment_col:
        return df[comment_col].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    raise KeyError(
        "Could not find text/comment columns. Available columns: " + ", ".join(map(str, df.columns))
    )


def stratified_sample(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    # Stratified sampling keeps class balance stable for the practical local run.
    _, sampled = train_test_split(
        df,
        test_size=max_samples,
        random_state=seed,
        stratify=df["_label"],
    )
    return sampled.reset_index(drop=True)


def load_split(csv_path: Path, max_samples: int | None, seed: int) -> Tuple[List[str], List[int], Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    original_columns = list(df.columns)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    label_col = detect_label_col(df)
    labels = convert_labels(df[label_col])
    text = build_text_series(df)

    prepared = pd.DataFrame({"_text": text, "_label": labels})
    prepared = prepared.dropna(subset=["_label"])
    prepared["_label"] = prepared["_label"].astype(int)
    prepared = prepared[prepared["_label"].isin([0, 1])]
    prepared["_text"] = prepared["_text"].fillna("").astype(str).str.strip()
    prepared = prepared[prepared["_text"].str.len() > 0]

    if prepared.empty:
        raise ValueError(f"No valid rows after preparing split: {csv_path}")

    sampled = stratified_sample(prepared, max_samples=max_samples, seed=seed)

    info = {
        "csv_path": str(csv_path),
        "original_columns": original_columns,
        "clean_columns": list(df.columns),
        "label_column_used": label_col,
        "rows_loaded": int(len(df)),
        "rows_after_cleaning": int(len(prepared)),
        "rows_used": int(len(sampled)),
        "label_distribution_used": {str(k): int(v) for k, v in sampled["_label"].value_counts().sort_index().items()},
    }
    return sampled["_text"].tolist(), sampled["_label"].astype(int).tolist(), info


class EncodedDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def tokenize_dataset(tokenizer, texts: List[str], labels: List[int], max_length: int) -> EncodedDataset:
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return EncodedDataset(encodings, labels)


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1],
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "class_metrics": {
            "0_non_sarcastic": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "1_sarcastic": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": {
            "labels": ["0_non_sarcastic", "1_sarcastic"],
            "matrix": cm.tolist(),
        },
    }


def evaluate(model, dataloader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, object]]:
    model.eval()
    losses: List[float] = []
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=-1)
            losses.append(float(loss.detach().cpu().item()))
            all_labels.extend(batch["labels"].detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return float(np.mean(losses)) if losses else math.nan, metrics


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_markdown_summary(results: List[Dict[str, object]]) -> None:
    md_path = REPORT_DIR / "task13_bertweet_training_summary.md"
    lines = [
        "# Task 13: BERTweet Training Summary",
        "",
        "This report summarizes BERTweet training for the two preprocessing versions.",
        "",
        "| Experiment | Dataset | Train Rows | Valid Rows | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m = r["validation_metrics"]
        lines.append(
            f"| {r['experiment_id']} | {r['dataset_version']} | {r['train_rows_used']:,} | "
            f"{r['valid_rows_used']:,} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} |"
        )
    lines += [
        "",
        "## Methodological note",
        "",
        "Both BERTweet experiments use the same Task 11 stratified split, max sequence length 128, and identical training settings. The difference between E01 and E02 is only the preprocessing version: Version A keeps stopwords, while Version B removes stopwords selectively while preserving negation words.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    progress = (
        "BERTweet training completed for Version A and Version B using the controlled experiment design. "
        "Both runs used the same split, text-only input, max length 128, and identical training settings. "
        "Validation metrics and local checkpoints were saved under reports/task13 and models/bertweet."
    )
    (REPORT_DIR / "task13_progress_note.txt").write_text(progress + "\n", encoding="utf-8")


def train_one_experiment(exp: Dict[str, object], config: Dict[str, object], args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    training_cfg = config["training"]
    seed = int(training_cfg["random_seed"])
    set_seed(seed)

    model_name = str(config["model_name"])
    max_length = int(args.max_length or training_cfg["max_length"])
    batch_size = int(args.batch_size or training_cfg["batch_size"])
    epochs = int(args.epochs or training_cfg["epochs"])
    grad_accum = int(args.gradient_accumulation_steps or training_cfg["gradient_accumulation_steps"])
    lr = float(args.learning_rate or training_cfg["learning_rate"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else training_cfg["weight_decay"])
    warmup_ratio = float(args.warmup_ratio if args.warmup_ratio is not None else training_cfg["warmup_ratio"])
    log_every = int(args.log_every or training_cfg["log_every"])

    max_train = args.max_train_samples
    if max_train is None:
        max_train = None if args.full_data else int(training_cfg["max_train_samples"])
    max_valid = args.max_valid_samples
    if max_valid is None:
        max_valid = None if args.full_data else int(training_cfg["max_valid_samples"])

    train_path = PROJECT_ROOT / str(exp["train_path"])
    valid_path = PROJECT_ROOT / str(exp["valid_path"])
    output_dir = PROJECT_ROOT / str(exp["output_dir"])
    metrics_path = PROJECT_ROOT / str(exp["metrics_path"])

    print("\n" + "=" * 60)
    print(f"Training {exp['experiment_id']}")
    print(f"Train path: {train_path}")
    print(f"Valid path: {valid_path}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        existing = metrics_path if metrics_path.exists() else None
        if existing:
            print(f"Existing model/metrics found and --overwrite not set. Loading metrics from {metrics_path}")
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Output directory already exists and is not empty: {output_dir}. Use --overwrite.")

    print("Loading split data...")
    train_texts, train_labels, train_info = load_split(train_path, max_train, seed)
    valid_texts, valid_labels, valid_info = load_split(valid_path, max_valid, seed)
    print(f"Train rows used: {len(train_texts):,} | label distribution: {train_info['label_distribution_used']}")
    print(f"Valid rows used: {len(valid_texts):,} | label distribution: {valid_info['label_distribution_used']}")

    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenizing training and validation data...")
    train_ds = tokenize_dataset(tokenizer, train_texts, train_labels, max_length)
    valid_ds = tokenize_dataset(tokenizer, valid_texts, valid_labels, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "non_sarcastic", 1: "sarcastic"},
        label2id={"non_sarcastic": 0, "sarcastic": 1},
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_update_steps = max(1, update_steps_per_epoch * epochs)
    warmup_steps = int(total_update_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    best_macro_f1 = -1.0
    best_metrics: Dict[str, object] | None = None
    training_history: List[Dict[str, object]] = []
    start_time = time.time()

    print(
        f"Training settings: epochs={epochs}, batch_size={batch_size}, grad_accum={grad_accum}, "
        f"lr={lr}, max_length={max_length}, device={device}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu().item()) * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                progress.set_postfix({"loss": f"{running_loss / step:.4f}"})

        train_loss = running_loss / max(1, len(train_loader))
        valid_loss, valid_metrics = evaluate(model, valid_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
        }
        training_history.append(epoch_record)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
            f"accuracy={valid_metrics['accuracy']:.4f}, macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        if float(valid_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(valid_metrics["macro_f1"])
            best_metrics = valid_metrics
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best checkpoint to {output_dir}")

    elapsed_seconds = time.time() - start_time
    assert best_metrics is not None

    result = {
        "experiment_id": exp["experiment_id"],
        "dataset_version": exp["dataset_version"],
        "preprocessing": exp["preprocessing"],
        "model_name": model_name,
        "device": str(device),
        "train_rows_used": len(train_texts),
        "valid_rows_used": len(valid_texts),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "train_info": train_info,
        "valid_info": valid_info,
        "training_history": training_history,
        "validation_metrics": best_metrics,
        "best_checkpoint_local_path": str(output_dir),
        "elapsed_seconds": elapsed_seconds,
    }
    save_json(metrics_path, result)
    print(f"Saved metrics: {metrics_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERTweet for Task 13.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--experiment", default="all", help="all, E01, E02, E01_BERTweet_VersionA, or E02_BERTweet_VersionB")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--full-data", action="store_true", help="Use all train/valid rows instead of configured sample limits.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--device", default=os.environ.get("TASK13_DEVICE"))
    return parser.parse_args()


def select_experiments(config: Dict[str, object], requested: str) -> List[Dict[str, object]]:
    experiments = list(config["experiments"])
    req = requested.strip().lower()
    if req == "all":
        return experiments
    selected = []
    for exp in experiments:
        exp_id = str(exp["experiment_id"]).lower()
        short = exp_id.split("_")[0].lower()
        if req == exp_id or req == short:
            selected.append(exp)
    if not selected:
        raise ValueError(f"Unknown experiment '{requested}'. Valid: all, E01, E02")
    return selected


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    set_seed(int(config["training"]["random_seed"]))
    device = choose_device(args.device)

    print("Task 13: Train BERTweet")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Experiment request: {args.experiment}")
    print(f"Full data: {args.full_data}")
    print(f"Max train samples override: {args.max_train_samples}")
    print(f"Max valid samples override: {args.max_valid_samples}")

    selected = select_experiments(config, args.experiment)
    results = []
    for exp in selected:
        result = train_one_experiment(exp, config, args, device)
        results.append(result)

    # If only one experiment was run, preserve existing other metrics when writing summary if available.
    all_results = []
    for exp in config["experiments"]:
        metrics_path = PROJECT_ROOT / str(exp["metrics_path"])
        if metrics_path.exists():
            all_results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not all_results:
        all_results = results

    write_markdown_summary(all_results)
    save_json(REPORT_DIR / "task13_bertweet_training_summary.json", all_results)
    print("\nDone. Task 13 BERTweet training completed.")


if __name__ == "__main__":
    main()
