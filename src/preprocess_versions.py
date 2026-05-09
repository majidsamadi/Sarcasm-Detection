"""
Task 09-10 preprocessing pipeline for the SARC sarcasm detection project.

This script reproduces the team member's notebook logic in reusable project code:

Version A: initial cleaned dataset with stopwords kept.
Version B: Version A plus selective stopword removal, while keeping negations.

Outputs:
    data/processed/A.csv
    data/processed/B.csv
    reports/task09_10_preprocessing_summary.md
    reports/task09_10_preprocessing_summary.json

Notes:
    - The dataset files are intentionally local and should not be pushed to GitHub.
    - The model input uses parent_comment + comment, because sarcasm often depends on context.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_CSV = DATA_RAW / "train-balanced-sarcasm.csv"
A_CSV = DATA_PROCESSED / "A.csv"
B_CSV = DATA_PROCESSED / "B.csv"
SUMMARY_JSON = REPORTS_DIR / "task09_10_preprocessing_summary.json"
SUMMARY_MD = REPORTS_DIR / "task09_10_preprocessing_summary.md"

TOKENIZER_NAME = "vinai/bertweet-base"
MAX_TOKENS = 128
RANDOM_SEED = 42

NEGATION_WORDS = {
    "no", "not", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "cannot", "without",
}


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def label_balance(df: pd.DataFrame) -> Dict[str, int]:
    if "label" not in df.columns:
        return {"label_0": 0, "label_1": 0}
    counts = df["label"].value_counts(dropna=False).to_dict()
    return {
        "label_0": int(counts.get(0, 0)),
        "label_1": int(counts.get(1, 0)),
    }


def add_step(steps: List[Dict[str, object]], name: str, before: int, df: pd.DataFrame) -> None:
    after = len(df)
    bal = label_balance(df)
    steps.append(
        {
            "step": name,
            "before": int(before),
            "after": int(after),
            "dropped": int(before - after),
            **bal,
        }
    )
    print(f"{name:<55} before={before:,} after={after:,} dropped={before-after:,}")


def find_local_raw_csv() -> Path | None:
    if RAW_CSV.exists():
        return RAW_CSV

    candidates = sorted(DATA_RAW.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]

    for path in candidates:
        lower = path.name.lower()
        if "sarcasm" in lower or "train-balanced" in lower:
            return path

    return None


def load_raw_dataset() -> pd.DataFrame:
    """Load SARC from local raw CSV if available; otherwise download via kagglehub."""
    local_csv = find_local_raw_csv()
    if local_csv is not None:
        print(f"Loading local raw CSV: {local_csv}")
        return pd.read_csv(local_csv)

    print("No local raw CSV found. Downloading public Kaggle SARC dataset using kagglehub...")
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is not installed. Run: pip install 'kagglehub[pandas-datasets]'"
        ) from exc

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "danofer/sarcasm",
        "train-balanced-sarcasm.csv",
    )

    print(f"Saving raw dataset locally to: {RAW_CSV}")
    df.to_csv(RAW_CSV, index=False)
    return df


def validate_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["parent_comment", "comment", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df = df[required].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    return df


def is_spam(text: object) -> bool:
    text = str(text).strip()
    words = text.split()

    # Drop single-character comments/parents as uninformative.
    if len(text) <= 1:
        return True

    # Drop repetitive spam: 10+ words but only 1-3 unique words.
    if len(words) >= 10:
        unique_words = {w.lower() for w in words}
        if len(unique_words) <= 3:
            return True

    return False


def remove_simple_markdown(text: object) -> str:
    """Remove lightweight Reddit/Markdown formatting while preserving text meaning."""
    x = str(text)

    # Decode HTML entities twice, matching the notebook's double-pass approach.
    x = html.unescape(html.unescape(x))

    # Remove bold/italic markers such as *word*, **word**, ***word***.
    # The negative lookarounds avoid censored words like f*ck.
    x = re.sub(r"(?<!\w)\*{1,3}([^*]+)\*{1,3}(?!\w)", r"\1", x)

    # Remove markdown heading markers at the beginning of words.
    x = re.sub(r"(?<!\w)#{1,6}\s", "", x)

    # Remove Reddit superscript marker when used as ^word.
    x = re.sub(r"(?<!\S)\^+(\S+)", r"\1", x)

    # Remove underscore italics such as _word_ or __word__.
    x = re.sub(r"(?<!\w)_{1,2}([^_]+)_{1,2}(?!\w)", r"\1", x)

    return x


def normalize_whitespace(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def compute_bertweet_lengths(texts: Iterable[str], batch_size: int = 1000) -> List[int]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is not installed. Run: pip install transformers") from exc

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)

    texts_list = list(texts)
    lengths: List[int] = []
    for start in tqdm(range(0, len(texts_list), batch_size), desc="Computing BERTweet token lengths"):
        batch = texts_list[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def create_version_a(skip_token_filter: bool = False) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    steps: List[Dict[str, object]] = []

    df = load_raw_dataset()
    print(f"Raw shape: {df.shape}")
    df = validate_raw_columns(df)
    print(f"Validated shape: {df.shape}")
    print(f"Initial label balance:\n{df['label'].value_counts().sort_index()}")

    before = len(df)
    df = df.dropna(subset=["comment"]).copy()
    df = df[df["comment"].astype(str).str.strip() != ""].copy()
    add_step(steps, "Drop null/empty comment rows", before, df)

    # Parent comments are context. If missing, keep row but use empty context.
    df["parent_comment"] = df["parent_comment"].fillna("").astype(str)
    df["comment"] = df["comment"].astype(str)

    before = len(df)
    df = df[
        ~df["comment"].str.contains(
            r"https?://\S+|\bwww\.\S+|\[[^\]]+\]\([^)]+\)",
            na=False,
            regex=True,
        )
    ].copy()
    add_step(steps, "Drop rows where comment contains URL/markdown hyperlink", before, df)

    before = len(df)
    df = df[~df["comment"].str.contains(r"\[deleted\]|\[removed\]", na=False, regex=True)].copy()
    add_step(steps, "Drop [deleted]/[removed] comments", before, df)

    # Clean lightweight markdown/HTML in both input columns.
    df["comment"] = df["comment"].apply(remove_simple_markdown)
    df["parent_comment"] = df["parent_comment"].apply(remove_simple_markdown)
    print("Applied lightweight markdown and HTML cleaning to comment and parent_comment")

    # Match the notebook: after cleaning, drop remaining underscore-italic patterns in comment.
    before = len(df)
    df = df[~df["comment"].str.contains(r"_{1,2}[^_]+_{1,2}", na=False, regex=True)].copy()
    add_step(steps, "Drop remaining underscore-italic rows in comment", before, df)

    before = len(df)
    df = df[~df["comment"].apply(is_spam)].copy()
    df = df[~df["parent_comment"].apply(is_spam)].copy()
    add_step(steps, "Drop spam/repetitive/single-character rows", before, df)

    before = len(df)
    df = df.drop_duplicates().copy()
    add_step(steps, "Drop duplicate rows", before, df)

    before = len(df)
    df = df[~df["comment"].str.contains(r"\[[^\]]+\]\([^)]+\)", na=False, regex=True)].copy()
    add_step(steps, "Drop remaining markdown hyperlink rows in comment", before, df)

    before = len(df)
    df = df[~df["comment"].str.contains(r"```[\s\S]+?```", na=False, regex=True)].copy()
    add_step(steps, "Drop code block rows in comment", before, df)

    df["comment"] = df["comment"].apply(normalize_whitespace)
    df["parent_comment"] = df["parent_comment"].apply(normalize_whitespace)
    print("Normalized whitespace in comment and parent_comment")

    before = len(df)
    df = df[(df["comment"] != "") & (df["parent_comment"] != "")].copy()
    add_step(steps, "Drop rows empty after final whitespace normalization", before, df)

    if skip_token_filter:
        print("Skipping BERTweet token length filter because --skip-token-filter was used.")
    else:
        combined = df["parent_comment"].astype(str) + " </s></s> " + df["comment"].astype(str)
        df["token_len"] = compute_bertweet_lengths(combined, batch_size=1000)
        print("Token length summary:")
        print(df["token_len"].describe())

        before = len(df)
        df = df[df["token_len"] <= MAX_TOKENS].copy()
        add_step(steps, f"Drop rows above {MAX_TOKENS} BERTweet tokens", before, df)

    df = df[["parent_comment", "comment", "label"]].copy()
    df.to_csv(A_CSV, index=False)
    print(f"Saved Version A: {A_CSV}")
    print(f"Version A shape: {df.shape}")
    print(f"Version A label balance:\n{df['label'].value_counts().sort_index()}")
    return df, steps


def get_stopwords() -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("nltk is not installed. Run: pip install nltk") from exc

    try:
        return set(stopwords.words("english"))
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")
        return set(stopwords.words("english"))


def remove_stopwords_selective(text: object, stop_words: set[str]) -> str:
    words = str(text).split()
    # Keep original casing/punctuation exactly like the notebook; only compare lowercase token.
    return " ".join([w for w in words if w.lower() not in stop_words])


def create_version_b() -> pd.DataFrame:
    if not A_CSV.exists():
        raise FileNotFoundError(f"Version A not found: {A_CSV}. Run Version A generation first.")

    df = pd.read_csv(A_CSV)
    print(f"Loaded Version A: {df.shape}")
    print(f"Version A label balance:\n{df['label'].value_counts().sort_index()}")

    base_stopwords = get_stopwords()
    stop_words = base_stopwords - NEGATION_WORDS
    print(f"Total NLTK stopwords: {len(base_stopwords)}")
    print(f"Stopwords after preserving negations: {len(stop_words)}")

    df["comment"] = df["comment"].astype(str).apply(lambda x: remove_stopwords_selective(x, stop_words))
    df["parent_comment"] = df["parent_comment"].astype(str).apply(lambda x: remove_stopwords_selective(x, stop_words))

    df.to_csv(B_CSV, index=False)
    print(f"Saved Version B: {B_CSV}")
    print(f"Version B shape: {df.shape}")
    print(f"Version B label balance:\n{df['label'].value_counts().sort_index()}")
    return df


def markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_summary(steps: List[Dict[str, object]], skip_token_filter: bool) -> None:
    a_shape = None
    b_shape = None
    a_balance = None
    b_balance = None

    if A_CSV.exists():
        df_a = pd.read_csv(A_CSV)
        a_shape = list(df_a.shape)
        a_balance = label_balance(df_a)
    if B_CSV.exists():
        df_b = pd.read_csv(B_CSV)
        b_shape = list(df_b.shape)
        b_balance = label_balance(df_b)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "Sarcasm Detection using SARC",
        "task": "Task 09-10 preprocessing versions A and B",
        "version_a": {
            "file": str(A_CSV.relative_to(PROJECT_ROOT)),
            "description": "Cleaned dataset with stopwords kept; punctuation, casing, and ALL CAPS preserved.",
            "shape": a_shape,
            "label_balance": a_balance,
            "token_filter": "skipped" if skip_token_filter else f"kept rows <= {MAX_TOKENS} BERTweet tokens",
        },
        "version_b": {
            "file": str(B_CSV.relative_to(PROJECT_ROOT)),
            "description": "Version A with selective stopword removal; negation words preserved.",
            "shape": b_shape,
            "label_balance": b_balance,
        },
        "cleaning_steps": steps,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Task 09-10 Preprocessing Summary")
    md.append("")
    md.append(f"Generated at: `{summary['generated_at']}`")
    md.append("")
    md.append("## Purpose")
    md.append("")
    md.append(
        "This document records the reproducible preprocessing pipeline used to create "
        "`A.csv` and `B.csv` for the SARC sarcasm detection project."
    )
    md.append("")
    md.append("## Version A")
    md.append("")
    md.append("- Stopwords are kept.")
    md.append("- Punctuation, casing, and ALL CAPS are kept because they may signal sarcasm.")
    md.append("- Null/empty comments, URL comments, `[deleted]`, `[removed]`, duplicate rows, simple Reddit/Markdown noise, spam-like rows, and overly long rows are handled.")
    md.append(f"- Output file: `{summary['version_a']['file']}`")
    md.append(f"- Shape: `{a_shape}`")
    md.append(f"- Label balance: `{a_balance}`")
    md.append("")
    md.append("## Version B")
    md.append("")
    md.append("- Starts from Version A.")
    md.append("- Removes common English stopwords.")
    md.append("- Preserves negation words such as `not`, `no`, `never`, `don't`, `isn't`, and `cannot`.")
    md.append(f"- Output file: `{summary['version_b']['file']}`")
    md.append(f"- Shape: `{b_shape}`")
    md.append(f"- Label balance: `{b_balance}`")
    md.append("")
    md.append("## Version A Cleaning Steps")
    md.append("")
    if steps:
        md.append(markdown_table(steps, ["step", "before", "after", "dropped", "label_0", "label_1"]))
    else:
        md.append("No Version A cleaning steps were recorded.")
    md.append("")
    md.append("## Academic Note")
    md.append("")
    md.append(
        "The two preprocessing versions support the experimental comparison requested by the lecturer: "
        "testing whether keeping or removing stopwords affects sarcasm detection performance. "
        "Both versions should use the same train/validation/test split in Task 11."
    )
    md.append("")
    md.append("## GitHub Note")
    md.append("")
    md.append("`A.csv`, `B.csv`, and split CSV files are local dataset files and should not be pushed to GitHub.")

    SUMMARY_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote summary: {SUMMARY_MD}")
    print(f"Wrote summary: {SUMMARY_JSON}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate preprocessing Version A and Version B datasets.")
    parser.add_argument(
        "--skip-token-filter",
        action="store_true",
        help="Skip BERTweet token length filtering. Use only if tokenizer download is unavailable.",
    )
    parser.add_argument(
        "--only-b",
        action="store_true",
        help="Only generate B.csv from an existing A.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    steps: List[Dict[str, object]] = []
    if args.only_b:
        create_version_b()
    else:
        _, steps = create_version_a(skip_token_filter=args.skip_token_filter)
        create_version_b()

    write_summary(steps=steps, skip_token_filter=args.skip_token_filter)
    print("Task 09-10 preprocessing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
