"""
Task 11: Train/Validation/Test Split

This script creates fair stratified train/validation/test splits for:
- Version A: stopwords kept
- Version B: selective stopword removal

Important academic rule:
The same row indices are used for Version A and Version B so that the model comparison is fair.

Expected input files:
    data/processed/A.csv
    data/processed/B.csv

Output:
    data/splits/versionA/train.csv
    data/splits/versionA/valid.csv
    data/splits/versionA/test.csv
    data/splits/versionB/train.csv
    data/splits/versionB/valid.csv
    data/splits/versionB/test.csv
    data/splits/split_summary.json
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
    SPLITS_DIR,
    TEST_SIZE,
    TEXT_CANDIDATES,
    VALIDATION_SIZE,
    VERSION_A_FILE,
    VERSION_B_FILE,
)


def normalize_spaces(text: object) -> str:
    """Convert to string and collapse repeated whitespace."""
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
    Add a model_text column.

    If parent_comment exists, we combine it with comment because sarcasm often depends on context.
    This still uses text-only input, not metadata.
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


def validate_same_rows(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    if len(df_a) != len(df_b):
        raise ValueError(
            f"Version A and Version B must have the same number of rows. "
            f"A={len(df_a)}, B={len(df_b)}"
        )


def load_versions(version_a_path: Path, version_b_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not version_a_path.exists():
        raise FileNotFoundError(f"Version A file not found: {version_a_path}")
    if not version_b_path.exists():
        raise FileNotFoundError(f"Version B file not found: {version_b_path}")

    print(f"Loading Version A: {version_a_path}")
    df_a = pd.read_csv(version_a_path)
    print(f"Loading Version B: {version_b_path}")
    df_b = pd.read_csv(version_b_path)

    validate_same_rows(df_a, df_b)

    label_col = infer_label_column(df_a)
    if label_col not in df_b.columns:
        raise ValueError(f"Label column '{label_col}' exists in Version A but not Version B.")

    df_a = add_model_text(df_a)
    df_b = add_model_text(df_b)

    # Keep only rows where both versions still have usable model text and label.
    valid_mask = (
        df_a["model_text"].astype(str).str.len().gt(0)
        & df_b["model_text"].astype(str).str.len().gt(0)
        & df_a[label_col].notna()
        & df_b[label_col].notna()
    )

    removed = int((~valid_mask).sum())
    if removed > 0:
        print(f"Removing {removed} rows with empty model_text or missing labels after final validation.")

    df_a = df_a.loc[valid_mask].reset_index(drop=True)
    df_b = df_b.loc[valid_mask].reset_index(drop=True)

    # Ensure labels match by row after filtering.
    if not df_a[label_col].reset_index(drop=True).equals(df_b[label_col].reset_index(drop=True)):
        raise ValueError("Labels in Version A and Version B do not match by row.")

    print(f"Final aligned rows: {len(df_a):,}")
    return df_a, df_b, label_col


def create_stratified_indices(
    labels: pd.Series,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/valid/test indices.

    validation_size and test_size are expressed as fractions of the full dataset.
    Example: validation_size=0.10 and test_size=0.10 gives 80/10/10 split.
    """
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
    split_dir: Path,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    split_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "valid": df.iloc[valid_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }

    summary = {}
    for split_name, split_df in splits.items():
        output_path = split_dir / f"{split_name}.csv"
        split_df.to_csv(output_path, index=False)
        summary[split_name] = {
            "rows": int(len(split_df)),
        }
        print(f"Saved {output_path} ({len(split_df):,} rows)")

    return summary


def add_distribution_to_summary(summary: dict, df: pd.DataFrame, label_col: str, split_dir: Path) -> dict:
    distribution_rows = []
    for split_name in ["train", "valid", "test"]:
        split_path = split_dir / f"{split_name}.csv"
        split_df = pd.read_csv(split_path)
        counts = split_df[label_col].value_counts().sort_index()
        percentages = split_df[label_col].value_counts(normalize=True).sort_index() * 100

        summary[split_name]["label_distribution"] = {
            str(label): {
                "count": int(counts[label]),
                "percentage": round(float(percentages[label]), 4),
            }
            for label in counts.index
        }

        for label in counts.index:
            distribution_rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(counts[label]),
                    "percentage": round(float(percentages[label]), 4),
                }
            )

    pd.DataFrame(distribution_rows).to_csv(split_dir / "label_distribution.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Task 11 train/validation/test splits.")
    parser.add_argument("--version-a", type=Path, default=VERSION_A_FILE, help="Path to Version A CSV")
    parser.add_argument("--version-b", type=Path, default=VERSION_B_FILE, help="Path to Version B CSV")
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR, help="Output directory for splits")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test size fraction")
    parser.add_argument("--validation-size", type=float, default=VALIDATION_SIZE, help="Validation size fraction")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Task 11: Train/Validation/Test Split")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Random seed: {args.seed}")
    print(f"Split ratio: train={1 - args.test_size - args.validation_size:.2f}, "
          f"valid={args.validation_size:.2f}, test={args.test_size:.2f}")

    df_a, df_b, label_col = load_versions(args.version_a, args.version_b)

    train_idx, valid_idx, test_idx = create_stratified_indices(
        labels=df_a[label_col],
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    version_a_dir = args.output_dir / "versionA"
    version_b_dir = args.output_dir / "versionB"

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
    summary["versionA"] = save_split_files(df_a, version_a_dir, train_idx, valid_idx, test_idx)
    summary["versionA"] = add_distribution_to_summary(summary["versionA"], df_a, label_col, version_a_dir)

    print("\nSaving Version B splits...")
    summary["versionB"] = save_split_files(df_b, version_b_dir, train_idx, valid_idx, test_idx)
    summary["versionB"] = add_distribution_to_summary(summary["versionB"], df_b, label_col, version_b_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")

    print("\nDone. Task 11 completed successfully.")


if __name__ == "__main__":
    main()
