#!/usr/bin/env python3
"""Task 18: Error analysis for the selected final candidate model.

This script evaluates the selected best model from Task 17 on the held-out
Version A test split, extracts false positives and false negatives, and
summarizes error patterns by confidence, text length, and simple sarcasm-related
surface indicators.

Raw text error examples are saved locally for the team, but they are ignored by
Git to avoid pushing Reddit text samples publicly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "task18_error_analysis_config.json"


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    texts: List[str]
    labels: List[int]
    label_col: str
    text_cols: List[str]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def detect_label_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    # fallback: binary-looking numeric column
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        vals = set(series.astype(str).str.strip().unique()[:10])
        if vals.issubset({"0", "1"}):
            return col
    raise ValueError(f"Could not detect label column. Columns available: {list(df.columns)}")


def build_text(df: pd.DataFrame, requested_cols: Sequence[str], separator: str) -> Tuple[List[str], List[str]]:
    available_cols = [c for c in requested_cols if c in df.columns]
    if not available_cols:
        # fallback to any likely text column
        for candidate in ["comment", "text", "body", "content", "parent_comment"]:
            if candidate in df.columns:
                available_cols = [candidate]
                break
    if not available_cols:
        raise ValueError(f"Could not detect text columns. Columns available: {list(df.columns)}")

    if len(available_cols) == 1:
        texts = df[available_cols[0]].fillna("").astype(str).tolist()
    else:
        texts = []
        for _, row in df[available_cols].fillna("").astype(str).iterrows():
            parts = [str(row[col]).strip() for col in available_cols if str(row[col]).strip()]
            texts.append(separator.join(parts))
    return texts, available_cols


def load_dataset(path: Path, config: dict, max_samples: Optional[int]) -> LoadedDataset:
    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}")
    df = pd.read_csv(path)
    label_col = detect_label_col(df, config["input_format"]["label_column_candidates"])
    texts, text_cols = build_text(df, config["input_format"]["text_columns"], config["input_format"]["separator"])

    # Create a clean working copy with stable text/label columns.
    work = df.copy()
    work["_text"] = texts
    work["_label"] = work[label_col].astype(int)
    work = work[work["_text"].astype(str).str.strip().ne("")].copy()

    if max_samples is not None and max_samples > 0:
        # Stratified-ish deterministic sample: take first N after shuffle with fixed seed.
        work = work.sample(n=min(max_samples, len(work)), random_state=42).reset_index(drop=True)

    return LoadedDataset(
        df=work.reset_index(drop=True),
        texts=work["_text"].astype(str).tolist(),
        labels=work["_label"].astype(int).tolist(),
        label_col=label_col,
        text_cols=text_cols,
    )


def choose_device(preference: str) -> torch.device:
    env_device = os.environ.get("TASK18_DEVICE", "").strip().lower()
    if env_device:
        if env_device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if env_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if env_device == "cpu":
            return torch.device("cpu")
        print(f"Requested TASK18_DEVICE={env_device!r} is not available. Falling back to auto.")

    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_iter(items: Sequence[str], batch_size: int) -> Iterable[Tuple[int, Sequence[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def predict(
    checkpoint_dir: Path,
    fallback_model_name: str,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    # Prefer local tokenizer saved with checkpoint. If unavailable, fallback to base model tokenizer.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    except Exception as exc:
        print(f"Could not load tokenizer from checkpoint ({exc}). Falling back to {fallback_model_name}.")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)
    model.eval()

    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for _, batch_texts in tqdm(batch_iter(texts, batch_size), total=math.ceil(len(texts) / batch_size), desc="Predicting"):
            encoded = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)
    preds = probs_arr.argmax(axis=1)
    confidence = probs_arr.max(axis=1)
    return preds, probs_arr, confidence


def surface_features(text: str) -> Dict[str, object]:
    text = str(text)
    words = re.findall(r"\b\w+\b", text)
    caps_words = [w for w in words if len(w) >= 3 and w.isupper()]
    return {
        "char_len": len(text),
        "word_len": len(words),
        "has_exclamation": "!" in text,
        "has_question": "?" in text,
        "has_ellipsis": "..." in text or "…" in text,
        "has_quote": '"' in text or "'" in text or "“" in text or "”" in text,
        "has_all_caps_word": len(caps_words) > 0,
        "has_laughter": bool(re.search(r"\b(lol|lmao|haha|hehe|rofl)\b", text.lower())),
        "has_negation": bool(re.search(r"\b(not|no|never|dont|don't|didnt|didn't|isnt|isn't|wasnt|wasn't|cant|can't|cannot)\b", text.lower())),
    }


def add_error_features(df: pd.DataFrame, confidence_bins: Sequence[float], length_bins: Sequence[int]) -> pd.DataFrame:
    features = pd.DataFrame([surface_features(t) for t in df["_text"].astype(str).tolist()])
    out = pd.concat([df.reset_index(drop=True), features], axis=1)
    out["correct"] = out["_label"] == out["prediction"]
    out["error_type"] = "Correct"
    out.loc[(out["_label"] == 0) & (out["prediction"] == 1), "error_type"] = "False Positive"
    out.loc[(out["_label"] == 1) & (out["prediction"] == 0), "error_type"] = "False Negative"
    out["confidence_bin"] = pd.cut(
        out["confidence"],
        bins=list(confidence_bins),
        labels=["<0.55", "0.55-0.70", "0.70-0.85", ">=0.85"],
        include_lowest=True,
        right=False,
    )
    out["length_bin"] = pd.cut(
        out["word_len"],
        bins=list(length_bins),
        labels=["0-20", "21-50", "51-100", ">100"],
        include_lowest=True,
        right=True,
    )
    return out


def safe_value_counts(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().items()}


def bool_error_summary(df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, dict]:
    rows = {}
    for col in feature_cols:
        sub = df.groupby([col, "correct"], dropna=False).size().unstack(fill_value=0)
        total_by_feature = df.groupby(col, dropna=False).size()
        col_summary = {}
        for val in total_by_feature.index:
            total = int(total_by_feature.loc[val])
            correct_count = int(sub.loc[val, True]) if True in sub.columns and val in sub.index else 0
            incorrect_count = int(sub.loc[val, False]) if False in sub.columns and val in sub.index else 0
            error_rate = incorrect_count / total if total else 0.0
            col_summary[str(val)] = {
                "total": total,
                "correct": correct_count,
                "incorrect": incorrect_count,
                "error_rate": round(error_rate, 6),
            }
        rows[col] = col_summary
    return rows


def save_bar_chart(series: pd.Series, title: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_grouped_chart(df: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    df.plot(kind="bar", ax=plt.gca())
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def truncate_text(text: str, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def write_markdown_report(
    path: Path,
    summary: dict,
    feature_error_summary: Dict[str, dict],
    config: dict,
    local_sample_paths: Dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    best = config["selected_experiment"]
    metrics = summary["metrics"]
    cm = summary["confusion_matrix"]
    error_counts = summary["error_counts"]

    lines = []
    lines.append("# Task 18: Error Analysis")
    lines.append("")
    lines.append("## Selected model")
    lines.append("")
    lines.append(f"- **Experiment:** {best['experiment_id']}")
    lines.append(f"- **Model:** {best['model_family']} (`{best['model_name']}`)")
    lines.append(f"- **Dataset version:** {best['dataset_version']} ({best['preprocessing']})")
    lines.append(f"- **Selection basis:** {best['selection_basis']}")
    lines.append("")
    lines.append("## Overall held-out test performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
        lines.append(f"| {key.replace('_', ' ').title()} | {metrics[key]:.4f} |")
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("")
    lines.append("Rows are actual labels and columns are predicted labels.")
    lines.append("")
    lines.append("| Actual \\ Predicted | Non-sarcastic (0) | Sarcastic (1) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Non-sarcastic (0) | {cm['tn']} | {cm['fp']} |")
    lines.append(f"| Sarcastic (1) | {cm['fn']} | {cm['tp']} |")
    lines.append("")
    lines.append("## Error counts")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    lines.append(f"| False Positive | {error_counts.get('False Positive', 0)} |")
    lines.append(f"| False Negative | {error_counts.get('False Negative', 0)} |")
    lines.append(f"| Correct | {error_counts.get('Correct', 0)} |")
    lines.append("")
    lines.append("## Main observations")
    lines.append("")
    observations = summary.get("observations", [])
    for obs in observations:
        lines.append(f"- {obs}")
    lines.append("")
    lines.append("## Error patterns analyzed")
    lines.append("")
    lines.append("The analysis summarizes errors by confidence level, text length, punctuation cues, quotation markers, all-caps words, laughter markers, and negation markers. These features are surface-level indicators only; they do not prove causality, but they help explain where the model struggles.")
    lines.append("")

    lines.append("### Confidence distribution by prediction correctness")
    lines.append("")
    for bucket, counts in summary["confidence_by_correctness"].items():
        lines.append(f"- **{bucket}:** {counts}")
    lines.append("")

    lines.append("### Length distribution by prediction correctness")
    lines.append("")
    for bucket, counts in summary["length_by_correctness"].items():
        lines.append(f"- **{bucket}:** {counts}")
    lines.append("")

    lines.append("## Local error examples")
    lines.append("")
    lines.append("Raw Reddit text examples are saved locally for internal team analysis only and are ignored by Git. They are not committed to the public repository to reduce privacy and responsible-use risks.")
    lines.append("")
    for label, sample_path in local_sample_paths.items():
        lines.append(f"- **{label}:** `{sample_path}`")
    lines.append("")

    lines.append("## Academic interpretation")
    lines.append("")
    lines.append("False positives usually mean the model detected sarcastic patterns in a non-sarcastic text. False negatives usually mean the model missed sarcasm, often because sarcasm can depend on context, implied meaning, or conversational background. This supports the project limitation that sarcasm detection should not be used for automatic moderation or punishment without human review.")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.append("- `reports/task18/figures/task18_error_type_counts.png`")
    lines.append("- `reports/task18/figures/task18_error_by_confidence_bin.png`")
    lines.append("- `reports/task18/figures/task18_error_by_length_bin.png`")
    lines.append("- `reports/task18/figures/task18_error_rate_by_surface_feature.png`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 18 error analysis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-test-samples", type=int, default=int(os.environ.get("TASK18_MAX_TEST_SAMPLES", "0")))
    args = parser.parse_args()

    config = load_json(args.config)
    selected = config["selected_experiment"]
    outputs = config["outputs"]

    max_samples = args.max_test_samples if args.max_test_samples > 0 else None
    device = choose_device(config["inference"].get("device_preference", "auto"))

    print("Task 18: Error Analysis")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config: {args.config}")
    print(f"Selected experiment: {selected['experiment_id']}")
    print(f"Device: {device}")
    print(f"Max test samples: {max_samples if max_samples else 'FULL TEST SET'}")

    dataset = load_dataset(PROJECT_ROOT / selected["test_split"], config, max_samples)
    print(f"Rows loaded: {len(dataset.df):,}")
    print(f"Label column: {dataset.label_col}")
    print(f"Text columns used: {dataset.text_cols}")
    print(f"Label distribution: {pd.Series(dataset.labels).value_counts().sort_index().to_dict()}")

    preds, probs, confidence = predict(
        checkpoint_dir=PROJECT_ROOT / selected["checkpoint_dir"],
        fallback_model_name=selected["model_name"],
        texts=dataset.texts,
        max_length=int(config["inference"]["max_length"]),
        batch_size=int(config["inference"]["batch_size"]),
        device=device,
    )

    work = dataset.df.copy()
    work["prediction"] = preds.astype(int)
    work["prob_non_sarcastic"] = probs[:, 0]
    work["prob_sarcastic"] = probs[:, 1]
    work["confidence"] = confidence
    work = add_error_features(work, config["analysis"]["confidence_bins"], config["analysis"]["length_bins"])

    labels = np.array(dataset.labels)
    accuracy = accuracy_score(labels, preds)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    report_text = classification_report(labels, preds, target_names=["non_sarcastic", "sarcastic"], digits=4, zero_division=0)

    error_counts = safe_value_counts(work["error_type"])
    confidence_by_correctness_df = work.groupby(["confidence_bin", "correct"], dropna=False).size().unstack(fill_value=0)
    length_by_correctness_df = work.groupby(["length_bin", "correct"], dropna=False).size().unstack(fill_value=0)

    confidence_by_correctness = {
        str(idx): {str(k): int(v) for k, v in row.items()} for idx, row in confidence_by_correctness_df.iterrows()
    }
    length_by_correctness = {
        str(idx): {str(k): int(v) for k, v in row.items()} for idx, row in length_by_correctness_df.iterrows()
    }

    bool_cols = [
        "has_exclamation",
        "has_question",
        "has_ellipsis",
        "has_quote",
        "has_all_caps_word",
        "has_laughter",
        "has_negation",
    ]
    feature_error_summary = bool_error_summary(work, bool_cols)

    # Local raw-text examples, ignored by Git.
    samples_dir = PROJECT_ROOT / outputs["local_error_samples_dir"]
    samples_dir.mkdir(parents=True, exist_ok=True)
    n_examples = int(config["analysis"]["num_examples_per_error_type"])

    fp_df = work[work["error_type"] == "False Positive"].sort_values("confidence", ascending=False)
    fn_df = work[work["error_type"] == "False Negative"].sort_values("confidence", ascending=False)

    keep_cols = [
        c for c in [
            "parent_comment",
            "comment",
            "_text",
            "_label",
            "prediction",
            "prob_non_sarcastic",
            "prob_sarcastic",
            "confidence",
            "word_len",
            "char_len",
            "has_exclamation",
            "has_question",
            "has_ellipsis",
            "has_quote",
            "has_all_caps_word",
            "has_laughter",
            "has_negation",
        ] if c in work.columns
    ]

    fp_path = samples_dir / "high_confidence_false_positives.local.csv"
    fn_path = samples_dir / "high_confidence_false_negatives.local.csv"
    fp_df[keep_cols].head(n_examples).to_csv(fp_path, index=False)
    fn_df[keep_cols].head(n_examples).to_csv(fn_path, index=False)

    # Save a local predictions sample without raw text? Keep local only.
    pred_dir = PROJECT_ROOT / outputs["local_predictions_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "task18_predictions.local.csv"
    slim = work[["_label", "prediction", "prob_non_sarcastic", "prob_sarcastic", "confidence", "error_type", "word_len", "char_len"]].copy()
    slim.to_csv(pred_path, index=False)

    # Figures.
    figures_dir = PROJECT_ROOT / outputs["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    error_type_series = work["error_type"].value_counts().reindex(["Correct", "False Positive", "False Negative"]).dropna()
    save_bar_chart(error_type_series, "Task 18 Error Type Counts", "Count", figures_dir / "task18_error_type_counts.png")

    save_grouped_chart(confidence_by_correctness_df, figures_dir / "task18_error_by_confidence_bin.png", "Prediction Correctness by Confidence Bin")
    save_grouped_chart(length_by_correctness_df, figures_dir / "task18_error_by_length_bin.png", "Prediction Correctness by Text Length Bin")

    feature_error_rates = []
    for col in bool_cols:
        for val, info in feature_error_summary[col].items():
            if val == "True":
                feature_error_rates.append({"feature": col.replace("has_", ""), "error_rate": info["error_rate"], "total": info["total"]})
    feature_error_df = pd.DataFrame(feature_error_rates)
    if not feature_error_df.empty:
        plt.figure(figsize=(10, 5))
        plot_series = feature_error_df.set_index("feature")["error_rate"].sort_values(ascending=False)
        plot_series.plot(kind="bar")
        plt.title("Error Rate for Texts Containing Surface Features")
        plt.ylabel("Error Rate")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(figures_dir / "task18_error_rate_by_surface_feature.png", dpi=160)
        plt.close()

    # Observations.
    observations: List[str] = []
    total_errors = int((~work["correct"]).sum())
    total_rows = int(len(work))
    error_rate = total_errors / total_rows if total_rows else 0.0
    observations.append(f"The selected model made {total_errors:,} errors out of {total_rows:,} evaluated test examples, giving an error rate of {error_rate:.4f}.")
    observations.append(f"False positives: {fp:,}; false negatives: {fn:,}. This shows whether the model is more likely to over-detect or miss sarcasm.")

    if fp > fn:
        observations.append("False positives are higher than false negatives, so the model is more likely to predict sarcasm when the true label is non-sarcastic.")
    elif fn > fp:
        observations.append("False negatives are higher than false positives, so the model is more likely to miss sarcastic comments.")
    else:
        observations.append("False positives and false negatives are balanced in count.")

    high_conf_errors = int(((work["error_type"] != "Correct") & (work["confidence"] >= 0.85)).sum())
    observations.append(f"There are {high_conf_errors:,} high-confidence errors with confidence >= 0.85. These are useful for qualitative review because the model was confident but wrong.")

    # Summaries.
    summary = {
        "task": "Task 18 - Error Analysis",
        "selected_experiment": selected,
        "rows_evaluated": total_rows,
        "label_column": dataset.label_col,
        "text_columns_used": dataset.text_cols,
        "metrics": {
            "accuracy": round(float(accuracy), 6),
            "macro_precision": round(float(macro_precision), 6),
            "macro_recall": round(float(macro_recall), 6),
            "macro_f1": round(float(macro_f1), 6),
            "weighted_f1": round(float(weighted_f1), 6),
        },
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "error_counts": error_counts,
        "confidence_by_correctness": confidence_by_correctness,
        "length_by_correctness": length_by_correctness,
        "surface_feature_error_summary": feature_error_summary,
        "observations": observations,
        "local_files_not_committed": {
            "false_positives": str(fp_path.relative_to(PROJECT_ROOT)),
            "false_negatives": str(fn_path.relative_to(PROJECT_ROOT)),
            "predictions": str(pred_path.relative_to(PROJECT_ROOT)),
        },
    }

    report_json = PROJECT_ROOT / outputs["report_json"]
    report_md = PROJECT_ROOT / outputs["report_markdown"]
    progress_note = PROJECT_ROOT / outputs["progress_note"]

    write_json(report_json, summary)
    write_markdown_report(
        report_md,
        summary,
        feature_error_summary,
        config,
        {
            "High-confidence false positives": str(fp_path.relative_to(PROJECT_ROOT)),
            "High-confidence false negatives": str(fn_path.relative_to(PROJECT_ROOT)),
            "Prediction table": str(pred_path.relative_to(PROJECT_ROOT)),
        },
    )

    # Classification report text is safe because it contains no raw text.
    cls_report_path = PROJECT_ROOT / "reports" / "task18" / "task18_classification_report.txt"
    cls_report_path.write_text(report_text, encoding="utf-8")

    progress = (
        "Error analysis completed for the selected best model, RoBERTa Version A. "
        f"The model was evaluated on {total_rows:,} held-out test examples, with accuracy {accuracy:.4f} "
        f"and macro-F1 {macro_f1:.4f}. False positives, false negatives, confidence-based errors, "
        "length-based errors, and surface-feature error patterns were analyzed. Summary reports and figures "
        "were saved under reports/task18, while raw text error examples were kept local and ignored by Git."
    )
    progress_note.write_text(progress + "\n", encoding="utf-8")

    print("\nTask 18 completed successfully.")
    print(f"Saved report: {report_md}")
    print(f"Saved JSON: {report_json}")
    print(f"Saved progress note: {progress_note}")
    print(f"Saved figures: {figures_dir}")
    print(f"Saved local raw-text error samples, not committed: {samples_dir}")


if __name__ == "__main__":
    main()
