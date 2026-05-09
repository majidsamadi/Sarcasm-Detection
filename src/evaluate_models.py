#!/usr/bin/env python3
"""
Task 15: Model Evaluation

Evaluates four trained transformer experiments on held-out test splits:
E01 BERTweet Version A
E02 BERTweet Version B
E03 RoBERTa Version A
E04 RoBERTa Version B

This script intentionally avoids depending on the previous buggy function signature.
It robustly loads labels, builds text-only inputs, evaluates models, and writes
metrics, confusion matrices, and a comparison report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


LABEL_CANDIDATES = [
    "label",
    "Label",
    "LABEL",
    "class",
    "Class",
    "target",
    "Target",
    "is_sarcastic",
    "sarcasm",
]

TEXT_CANDIDATES = ["comment", "text", "sentence", "content", "body"]
CONTEXT_CANDIDATES = ["parent_comment", "context", "parent", "previous_comment"]


@dataclass
class LoadedSplit:
    texts: List[str]
    labels: List[int]
    dataframe: pd.DataFrame
    label_column: str
    text_columns_used: List[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    columns = list(df.columns)
    # exact match first
    for candidate in candidates:
        if candidate in columns:
            return candidate
    # case-insensitive / stripped match
    normalized_map = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_map:
            return normalized_map[key]
    return None


def find_label_column(df: pd.DataFrame) -> str:
    col = find_column(df, LABEL_CANDIDATES)
    if col is not None:
        return col

    # fallback: detect a binary numeric column with values 0/1
    for candidate in df.columns:
        series = df[candidate].dropna()
        if series.empty:
            continue
        try:
            values = set(series.astype(int).unique().tolist())
        except Exception:
            continue
        if values.issubset({0, 1}) and len(values) == 2:
            return candidate

    raise ValueError(
        "Could not find label column. Available columns: " + ", ".join(map(str, df.columns))
    )


def build_text_inputs(df: pd.DataFrame, joiner: str) -> Tuple[List[str], List[str]]:
    text_col = find_column(df, TEXT_CANDIDATES)
    if text_col is None:
        raise ValueError("Could not find text column. Available columns: " + ", ".join(map(str, df.columns)))

    context_col = find_column(df, CONTEXT_CANDIDATES)
    if context_col is not None:
        texts = (
            df[context_col].fillna("").astype(str).str.strip()
            + joiner
            + df[text_col].fillna("").astype(str).str.strip()
        )
        return texts.tolist(), [context_col, text_col]

    texts = df[text_col].fillna("").astype(str).str.strip()
    return texts.tolist(), [text_col]


def load_test_split(test_path: Path, max_samples: Optional[int], joiner: str) -> LoadedSplit:
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")

    df = pd.read_csv(test_path)
    df = normalize_columns(df)

    label_col = find_label_column(df)
    labels = df[label_col].astype(int).tolist()
    texts, text_cols = build_text_inputs(df, joiner=joiner)

    clean = pd.DataFrame({"text": texts, "label": labels})
    clean = clean[clean["text"].astype(str).str.strip().ne("")].reset_index(drop=True)

    if max_samples is not None and max_samples > 0:
        clean = clean.head(max_samples).reset_index(drop=True)

    return LoadedSplit(
        texts=clean["text"].astype(str).tolist(),
        labels=clean["label"].astype(int).tolist(),
        dataframe=clean,
        label_column=label_col,
        text_columns_used=text_cols,
    )


def select_device(force: Optional[str] = None) -> torch.device:
    if force:
        force = force.lower().strip()
        if force == "cpu":
            return torch.device("cpu")
        if force == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if force == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        print(f"WARNING: Requested device '{force}' is unavailable. Falling back automatically.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(checkpoint_dir: Path, fallback_name: str):
    try:
        return AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    except Exception as exc:
        print(f"Tokenizer not fully available in checkpoint. Falling back to {fallback_name}. Reason: {exc}")
        return AutoTokenizer.from_pretrained(fallback_name, use_fast=True)


def predict(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)

    logits = np.vstack(all_logits)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs


def compute_metrics(labels: List[int], preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    y_true = np.array(labels)
    y_pred = np.array(preds)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": {
            "0_non_sarcastic": {
                "precision": float(per_p[0]),
                "recall": float(per_r[0]),
                "f1": float(per_f1[0]),
                "support": int(per_support[0]),
            },
            "1_sarcastic": {
                "precision": float(per_p[1]),
                "recall": float(per_r[1]),
                "f1": float(per_f1[1]),
                "support": int(per_support[1]),
            },
        },
        "probability_summary": {
            "sarcastic_probability_mean": float(probs[:, 1].mean()),
            "sarcastic_probability_std": float(probs[:, 1].std()),
        },
    }


def save_confusion_matrix_figure(cm: List[List[int]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["Non-sarcastic", "Sarcastic"],
    )
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_prediction_sample(
    exp_id: str,
    loaded: LoadedSplit,
    preds: np.ndarray,
    probs: np.ndarray,
    output_dir: Path,
    limit: int = 200,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = loaded.dataframe.copy()
    df["prediction"] = preds.tolist()
    df["prob_non_sarcastic"] = probs[:, 0]
    df["prob_sarcastic"] = probs[:, 1]
    df["is_correct"] = df["label"].astype(int) == df["prediction"].astype(int)

    # Save a small safe sample only. Full predictions are not committed.
    sample = df.head(limit)
    sample.to_csv(output_dir / f"{exp_id}_prediction_sample.csv", index=False)


def evaluate_experiment(exp: Dict[str, Any], config: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    root = project_root()
    exp_id = exp["id"]
    checkpoint_dir = root / exp["checkpoint_dir"]
    test_path = root / exp["test_path"]
    joiner = config.get("text_input", {}).get("joiner", " </s> ")
    max_samples = args.max_test_samples

    print("\n" + "=" * 60)
    print(f"Evaluating {exp_id}")
    print(f"Model family: {exp['model_family']}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Test split: {test_path}")
    print(f"Max test samples: {max_samples if max_samples else 'FULL TEST SET'}")
    print("=" * 60)

    loaded = load_test_split(test_path, max_samples=max_samples, joiner=joiner)
    print(f"Rows evaluated: {len(loaded.labels):,}")
    print(f"Label column: {loaded.label_column}")
    print(f"Text columns used: {loaded.text_columns_used}")
    print(f"Label distribution: {pd.Series(loaded.labels).value_counts().to_dict()}")

    tokenizer = load_tokenizer(checkpoint_dir, exp["tokenizer_name"])
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)

    preds, probs = predict(
        model=model,
        tokenizer=tokenizer,
        texts=loaded.texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    metrics = compute_metrics(loaded.labels, preds, probs)

    report_dir = root / "reports" / "task15"
    figures_dir = report_dir / "figures"
    predictions_dir = report_dir / "predictions"

    result = {
        "experiment_id": exp_id,
        "short_id": exp.get("short_id"),
        "model_family": exp["model_family"],
        "dataset_version": exp["dataset_version"],
        "preprocessing": exp["preprocessing"],
        "checkpoint_dir": exp["checkpoint_dir"],
        "test_path": exp["test_path"],
        "rows_evaluated": len(loaded.labels),
        "max_test_samples": max_samples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "label_column": loaded.label_column,
        "text_columns_used": loaded.text_columns_used,
        "metrics": metrics,
    }

    metrics_path = report_dir / f"{exp_id}_test_metrics.json"
    write_json(metrics_path, result)
    print(f"Saved metrics: {metrics_path}")

    cm_path = figures_dir / f"{exp_id}_confusion_matrix.png"
    save_confusion_matrix_figure(metrics["confusion_matrix"], cm_path, title=exp_id)
    print(f"Saved confusion matrix: {cm_path}")

    # Classification report text.
    cr_path = report_dir / f"{exp_id}_classification_report.txt"
    with cr_path.open("w", encoding="utf-8") as f:
        f.write(classification_report(loaded.labels, preds, target_names=["non-sarcastic", "sarcastic"], zero_division=0))
    print(f"Saved classification report: {cr_path}")

    save_prediction_sample(exp_id, loaded, preds, probs, predictions_dir)
    return result


def create_summary(results: List[Dict[str, Any]], max_test_samples: Optional[int]) -> None:
    root = project_root()
    report_dir = root / "reports" / "task15"
    report_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r["metrics"]["macro_f1"], reverse=True)
    best = sorted_results[0]

    rows = []
    for result in sorted_results:
        m = result["metrics"]
        rows.append({
            "Experiment": result["experiment_id"],
            "Model": result["model_family"],
            "Version": result["dataset_version"],
            "Preprocessing": result["preprocessing"],
            "Rows": result["rows_evaluated"],
            "Accuracy": m["accuracy"],
            "Macro Precision": m["macro_precision"],
            "Macro Recall": m["macro_recall"],
            "Macro F1": m["macro_f1"],
            "Weighted F1": m["weighted_f1"],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_csv = report_dir / "task15_model_evaluation_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    summary_json = {
        "task": "Task 15 - Model Evaluation",
        "evaluation_scope": "full test set" if not max_test_samples else f"first {max_test_samples} test rows",
        "best_experiment": best["experiment_id"],
        "best_model_family": best["model_family"],
        "best_dataset_version": best["dataset_version"],
        "best_macro_f1": best["metrics"]["macro_f1"],
        "best_accuracy": best["metrics"]["accuracy"],
        "results": results,
    }
    write_json(report_dir / "task15_model_evaluation_summary.json", summary_json)

    md = []
    md.append("# Task 15: Model Evaluation Summary\n")
    md.append("## Evaluation Setup\n")
    md.append("- Task: binary sarcasm classification.\n")
    md.append("- Labels: 0 = non-sarcastic, 1 = sarcastic.\n")
    md.append("- Input: text-only input built from `parent_comment + comment` when both are available.\n")
    md.append("- Test set: held-out split from Task 11.\n")
    md.append(f"- Evaluation scope: {'full held-out test set' if not max_test_samples else f'first {max_test_samples} test rows'}.\n")
    md.append("- Metrics: accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class metrics, and confusion matrix.\n\n")

    md.append("## Model Comparison\n\n")
    md.append("| Rank | Experiment | Model | Version | Accuracy | Macro-F1 | Weighted-F1 | Rows |\n")
    md.append("|---:|---|---|---|---:|---:|---:|---:|\n")
    for i, result in enumerate(sorted_results, start=1):
        m = result["metrics"]
        md.append(
            f"| {i} | {result['experiment_id']} | {result['model_family']} | "
            f"{result['dataset_version']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['weighted_f1']:.4f} | {result['rows_evaluated']:,} |\n"
        )

    md.append("\n## Best Model\n\n")
    md.append(
        f"The best-performing experiment based on macro-F1 is **{best['experiment_id']}** "
        f"with macro-F1 **{best['metrics']['macro_f1']:.4f}** and accuracy "
        f"**{best['metrics']['accuracy']:.4f}**.\n"
    )

    # Stopword impact summary by model family.
    md.append("\n## Stopword Impact\n\n")
    for family in ["BERTweet", "RoBERTa"]:
        fam_results = [r for r in results if r["model_family"] == family]
        if len(fam_results) == 2:
            by_version = {r["dataset_version"]: r for r in fam_results}
            if "versionA" in by_version and "versionB" in by_version:
                a_f1 = by_version["versionA"]["metrics"]["macro_f1"]
                b_f1 = by_version["versionB"]["metrics"]["macro_f1"]
                diff = a_f1 - b_f1
                direction = "better" if diff > 0 else "worse"
                md.append(
                    f"- {family}: Version A (stopwords kept) macro-F1 = {a_f1:.4f}; "
                    f"Version B (stopwords removed) macro-F1 = {b_f1:.4f}. "
                    f"Version A is {direction} by {abs(diff):.4f}.\n"
                )

    md.append("\n## Outputs\n\n")
    md.append("- Per-model JSON metrics: `reports/task15/*_test_metrics.json`\n")
    md.append("- Confusion matrix figures: `reports/task15/figures/*_confusion_matrix.png`\n")
    md.append("- Classification reports: `reports/task15/*_classification_report.txt`\n")
    md.append("- Comparison CSV: `reports/task15/task15_model_evaluation_comparison.csv`\n")

    summary_md = report_dir / "task15_model_evaluation_summary.md"
    summary_md.write_text("".join(md), encoding="utf-8")

    progress = (
        "Model evaluation completed for all four experiments on the held-out test split. "
        "Accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class metrics, "
        "classification reports, and confusion matrices were generated. Results were saved under "
        "reports/task15, and the best model was selected based mainly on test macro-F1."
    )
    (report_dir / "task15_progress_note.txt").write_text(progress, encoding="utf-8")

    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved summary JSON: {report_dir / 'task15_model_evaluation_summary.json'}")
    print(f"Saved summary MD: {summary_md}")
    print(f"Saved progress note: {report_dir / 'task15_progress_note.txt'}")


def select_experiments(config: Dict[str, Any], request: str) -> List[Dict[str, Any]]:
    experiments = config["experiments"]
    request = request.strip()
    if request.lower() == "all":
        return experiments

    selected = []
    for exp in experiments:
        if request in {exp["id"], exp.get("short_id", "")}:
            selected.append(exp)

    if not selected:
        valid = [exp["id"] for exp in experiments] + [exp.get("short_id", "") for exp in experiments]
        raise ValueError(f"Unknown experiment request: {request}. Valid values: all, {valid}")
    return selected


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Task 15 model evaluation")
    parser.add_argument("--config", default=str(root / "configs" / "task15_evaluation_config.json"))
    parser.add_argument("--experiment", default=os.environ.get("TASK15_EXPERIMENT", "all"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TASK15_BATCH_SIZE", "16")))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("TASK15_MAX_LENGTH", "128")))
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=int(os.environ["TASK15_MAX_TEST_SAMPLES"]) if os.environ.get("TASK15_MAX_TEST_SAMPLES") else None,
        help="Optional limit for faster evaluation. Omit for full test set.",
    )
    parser.add_argument("--device", default=os.environ.get("TASK15_DEVICE"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    config = read_json(config_path)
    device = select_device(args.device)

    print("Task 15: Model Evaluation")
    print(f"Project root: {root}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Experiment request: {args.experiment}")
    print(f"Max test samples: {args.max_test_samples if args.max_test_samples else 'FULL TEST SET'}")

    experiments = select_experiments(config, args.experiment)
    results = [evaluate_experiment(exp, config, args, device) for exp in experiments]

    # Create global summary only when all four are evaluated.
    if args.experiment.lower() == "all":
        create_summary(results, args.max_test_samples)

    print("\nDone. Task 15 model evaluation completed successfully.")


if __name__ == "__main__":
    main()
