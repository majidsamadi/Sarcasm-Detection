"""
Task 11: Train/Validation/Test Split

Academic goal:
- Create a fair 80/10/10 train/validation/test split.
- Preserve class balance using stratified splitting.
- Use the same row indices for Version A and Version B.
- Version A = stopwords kept.
- Version B = selective stopword removal.

Inputs:
    data/processed/A.csv
    data/processed/B.csv

Outputs:
    data/splits/versionA/train.csv
    data/splits/versionA/valid.csv
    data/splits/versionA/test.csv
    data/splits/versionB/train.csv
    data/splits/versionB/valid.csv
    data/splits/versionB/test.csv
    data/splits/split_summary.json
    reports/task11_split_summary.md
    reports/task11_progress_note.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CONTEXT_CANDIDATES,
    LABEL_CANDIDATES,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORTS_DIR,
    SPLITS_DIR,
    TEST_SIZE,
    TEXT_CANDIDATES,
    VALIDATION_SIZE,
    VERSION_A_FILE,
    VERSION_B_FILE,
)


def normalize_spaces(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str], column_type: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find {column_type} column. Tried: {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def infer_label_column(df: pd.DataFrame) -> str:
    return find_first_existing_column(df, LABEL_CANDIDATES, "label")


def infer_comment_column(df: pd.DataFrame) -> str:
    return find_first_existing_column(df, TEXT_CANDIDATES, "text/comment")


def infer_context_column(df: pd.DataFrame) -> str | None:
    for col in CONTEXT_CANDIDATES:
        if col in df.columns:
            return col
    return None


def add_model_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model_text column.

    If parent_comment exists, combine parent_comment + comment.
    This is still text-only input and is useful because sarcasm often depends on context.
    """
    df = df.copy()
    comment_col = infer_comment_column(df)
    context_col = infer_context_column(df)

    comment = df[comment_col].apply(normalize_spaces)

    if context_col is not None:
        context = df[context_col].apply(normalize_spaces)
        df["model_text"] = ("context: " + context + " response: " + comment).apply(normalize_spaces)
    else:
        df["model_text"] = comment.apply(normalize_spaces)

    return df


def load_versions(version_a_path: Path, version_b_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not version_a_path.exists():
        raise FileNotFoundError(f"Version A file not found: {version_a_path}")
    if not version_b_path.exists():
        raise FileNotFoundError(f"Version B file not found: {version_b_path}")

    print(f"Loading Version A: {version_a_path}")
    df_a = pd.read_csv(version_a_path)
    print(f"Version A shape: {df_a.shape}")

    print(f"Loading Version B: {version_b_path}")
    df_b = pd.read_csv(version_b_path)
    print(f"Version B shape: {df_b.shape}")

    if len(df_a) != len(df_b):
        raise ValueError(
            "Version A and Version B must have the same number of rows for fair comparison. "
            f"A={len(df_a)}, B={len(df_b)}"
        )

    label_col = infer_label_column(df_a)
    if label_col not in df_b.columns:
        raise ValueError(f"Label column '{label_col}' exists in Version A but not Version B.")

    df_a = add_model_text(df_a)
    df_b = add_model_text(df_b)

    valid_mask = (
        df_a["model_text"].astype(str).str.len().gt(0)
        & df_b["model_text"].astype(str).str.len().gt(0)
        & df_a[label_col].notna()
        & df_b[label_col].notna()
    )

    removed = int((~valid_mask).sum())
    if removed > 0:
        print(f"Removing {removed:,} rows with empty model_text or missing labels after final validation.")

    df_a = df_a.loc[valid_mask].reset_index(drop=True)
    df_b = df_b.loc[valid_mask].reset_index(drop=True)

    if not df_a[label_col].reset_index(drop=True).equals(df_b[label_col].reset_index(drop=True)):
        raise ValueError("Labels in Version A and Version B do not match by row after filtering.")

    df_a.insert(0, "row_id", np.arange(len(df_a)))
    df_b.insert(0, "row_id", np.arange(len(df_b)))

    print(f"Final aligned rows: {len(df_a):,}")
    print(f"Label column: {label_col}")
    return df_a, df_b, label_col


def create_stratified_indices(
    labels: pd.Series,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    y = labels.to_numpy()

    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    validation_fraction_of_train_valid = validation_size / (1.0 - test_size)
    y_train_valid = labels.iloc[train_valid_idx].to_numpy()

    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=validation_fraction_of_train_valid,
        random_state=seed,
        stratify=y_train_valid,
    )

    return train_idx, valid_idx, test_idx


def save_split_files(
    df: pd.DataFrame,
    label_col: str,
    split_dir: Path,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    split_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }

    summary: dict = {}

    for split_name, split_idx in split_map.items():
        split_df = df.iloc[split_idx].reset_index(drop=True)
        output_path = split_dir / f"{split_name}.csv"
        split_df.to_csv(output_path, index=False)

        counts = split_df[label_col].value_counts().sort_index()
        percentages = split_df[label_col].value_counts(normalize=True).sort_index() * 100

        summary[split_name] = {
            "rows": int(len(split_df)),
            "label_distribution": {
                str(label): {
                    "count": int(counts[label]),
                    "percentage": round(float(percentages[label]), 4),
                }
                for label in counts.index
            },
        }

        print(f"Saved {output_path} ({len(split_df):,} rows)")

    distribution_rows = []
    for split_name, details in summary.items():
        for label, label_info in details["label_distribution"].items():
            distribution_rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": label_info["count"],
                    "percentage": label_info["percentage"],
                }
            )
    pd.DataFrame(distribution_rows).to_csv(split_dir / "label_distribution.csv", index=False)

    return summary


def make_markdown_report(summary: dict) -> str:
    lines = []
    lines.append("# Task 11: Train/Validation/Test Split Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Task 11 creates a fair stratified train/validation/test split for both preprocessing versions. "
        "The same row indices are used for Version A and Version B so that later model comparison is academically fair."
    )
    lines.append("")
    lines.append("## Split Configuration")
    lines.append("")
    lines.append(f"- Random seed: `{summary['seed']}`")
    lines.append(f"- Label column: `{summary['label_column']}`")
    lines.append(f"- Train ratio: `{summary['split_ratio']['train']}`")
    lines.append(f"- Validation ratio: `{summary['split_ratio']['valid']}`")
    lines.append(f"- Test ratio: `{summary['split_ratio']['test']}`")
    lines.append("")
    lines.append("## Version A: Stopwords Kept")
    lines.append("")
    lines.append("| Split | Rows | Label Distribution |")
    lines.append("|---|---:|---|")
    for split_name in ["train", "valid", "test"]:
        details = summary["versionA"][split_name]
        dist = ", ".join(
            f"{label}: {info['count']} ({info['percentage']}%)"
            for label, info in details["label_distribution"].items()
        )
        lines.append(f"| {split_name} | {details['rows']} | {dist} |")
    lines.append("")
    lines.append("## Version B: Selective Stopword Removal")
    lines.append("")
    lines.append("| Split | Rows | Label Distribution |")
    lines.append("|---|---:|---|")
    for split_name in ["train", "valid", "test"]:
        details = summary["versionB"][split_name]
        dist = ", ".join(
            f"{label}: {info['count']} ({info['percentage']}%)"
            for label, info in details["label_distribution"].items()
        )
        lines.append(f"| {split_name} | {details['rows']} | {dist} |")
    lines.append("")
    lines.append("## Academic Note")
    lines.append("")
    lines.append(
        "The split is stratified, so the sarcastic and non-sarcastic label balance is preserved across train, validation, and test sets. "
        "This supports fair comparison between BERTweet and RoBERTa and fair comparison between stopword-kept and stopword-removed preprocessing."
    )
    lines.append("")
    return "\n".join(lines)


def write_reports(summary: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "task11_split_summary.md"
    report_path.write_text(make_markdown_report(summary), encoding="utf-8")
    print(f"Saved report: {report_path}")

    progress_note = (
        "Train/validation/test split completed for Version A and Version B using the same stratified 80/10/10 split "
        "with random seed 42. Class balance was preserved across splits, outputs were saved under data/splits/versionA "
        "and data/splits/versionB, and the split summary was documented in reports/task11_split_summary.md."
    )
    progress_path = REPORTS_DIR / "task11_progress_note.txt"
    progress_path.write_text(progress_note, encoding="utf-8")
    print(f"Saved progress note: {progress_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Task 11 train/validation/test splits.")
    parser.add_argument("--version-a", type=Path, default=VERSION_A_FILE)
    parser.add_argument("--version-b", type=Path, default=VERSION_B_FILE)
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=VALIDATION_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Task 11: Train/Validation/Test Split")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Random seed: {args.seed}")
    print(
        f"Split ratio: train={1 - args.test_size - args.validation_size:.2f}, "
        f"valid={args.validation_size:.2f}, test={args.test_size:.2f}"
    )

    df_a, df_b, label_col = load_versions(args.version_a, args.version_b)

    train_idx, valid_idx, test_idx = create_stratified_indices(
        labels=df_a[label_col],
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    summary = {
        "task": "Task 11 - Train/Validation/Test Split",
        "seed": args.seed,
        "label_column": label_col,
        "split_ratio": {
            "train": round(1 - args.test_size - args.validation_size, 4),
            "valid": args.validation_size,
            "test": args.test_size,
        },
        "versionA": {},
        "versionB": {},
    }

    print("\nSaving Version A splits...")
    summary["versionA"] = save_split_files(
        df_a, label_col, args.output_dir / "versionA", train_idx, valid_idx, test_idx
    )

    print("\nSaving Version B splits...")
    summary["versionB"] = save_split_files(
        df_b, label_col, args.output_dir / "versionB", train_idx, valid_idx, test_idx
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON: {summary_path}")

    write_reports(summary)
    print("\nDone. Task 11 completed successfully.")


if __name__ == "__main__":
    main()
