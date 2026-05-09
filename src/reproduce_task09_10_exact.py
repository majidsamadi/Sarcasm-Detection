"""
Reproduce Row 9 and Row 10 preprocessing exactly from the teammate notebooks.

Important:
- This is not an improved or redesigned preprocessing pipeline.
- It follows the same data-changing steps from row_9_versionA.ipynb and row_10_versionB.ipynb.
- Only file paths are changed from Google Drive to this project folder.

Outputs:
    data/processed/A.csv
    data/processed/B.csv
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import kagglehub
import nltk
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from nltk.corpus import stopwords
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_CSV = DATA_RAW / "train-balanced-sarcasm.csv"
A_CSV = DATA_PROCESSED / "A.csv"
B_CSV = DATA_PROCESSED / "B.csv"


def load_dataset_exact() -> pd.DataFrame:
    """Load the same Kaggle SARC file used in the notebook."""
    if RAW_CSV.exists():
        print(f"Loading local raw CSV: {RAW_CSV}")
        return pd.read_csv(RAW_CSV)

    print("Downloading SARC dataset from Kaggle using kagglehub...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "danofer/sarcasm",
        "train-balanced-sarcasm.csv",
    )
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved raw CSV locally: {RAW_CSV}")
    return df


def generate_A_exact() -> pd.DataFrame:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_dataset_exact()
    print(f"Initial shape: {df.shape}")

    # Same as notebook Cell 17: creates word_count before cleaning.
    df["word_count"] = df["comment"].astype(str).apply(lambda x: len(x.split()))

    # Same as notebook Cell 31.
    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip() != ""]
    print(f"After null/empty comment removal: {df.shape}")

    # Same as notebook Cell 35: drop comment URLs and markdown hyperlinks.
    before = df.shape[0]
    df = df[~df["comment"].str.contains(
        r"https?://\S+|\bwww\.\S+|\[[^\]]+\]\([^)]+\)",
        na=False,
    )]
    print(f"Dropped URL/hyperlink comments: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 36: drop [deleted]/[removed] in comment.
    before = df.shape[0]
    df = df[~df["comment"].str.contains(r"\[deleted\]|\[removed\]", na=False)]
    print(f"Dropped [deleted]/[removed] comments: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 40: clean * bold/italic markers in both columns.
    df["comment"] = df["comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\w)\*{1,3}([^*]+)\*{1,3}(?!\w)", r"\1", x)
    )
    df["parent_comment"] = df["parent_comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\w)\*{1,3}([^*]+)\*{1,3}(?!\w)", r"\1", x)
    )
    print("* cleaning applied")

    # Same as notebook Cell 43: clean markdown headers in both columns.
    df["comment"] = df["comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\w)#{1,6}\s", "", x)
    )
    df["parent_comment"] = df["parent_comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\w)#{1,6}\s", "", x)
    )
    print("# headers cleaned")

    # Same as notebook Cell 46: clean reddit superscript in both columns.
    df["comment"] = df["comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\S)\^+(\S+)", r"\1", x)
    )
    df["parent_comment"] = df["parent_comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\S)\^+(\S+)", r"\1", x)
    )
    print("^ superscript cleaned")

    # Same as notebook Cell 49: clean _italic_ in both columns.
    df["comment"] = df["comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\w)_{1,2}([^_]+)_{1,2}(?!\w)", r"\1", x)
    )
    df["parent_comment"] = df["parent_comment"].astype(str).apply(
        lambda x: re.sub(r"(?<!\w)_{1,2}([^_]+)_{1,2}(?!\w)", r"\1", x)
    )
    print("_italic_ cleaned")

    # Same as notebook Cell 50: drop remaining underscore italic rows in comment only.
    before = df.shape[0]
    df = df[~df["comment"].str.contains(r"_{1,2}[^_]+_{1,2}", na=False)]
    print(f"Dropped remaining underscore italic rows: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cells 53 and 54: one pass, then double pass = total three passes.
    df["comment"] = df["comment"].astype(str).apply(html.unescape)
    df["parent_comment"] = df["parent_comment"].astype(str).apply(html.unescape)
    df["comment"] = df["comment"].astype(str).apply(html.unescape).apply(html.unescape)
    df["parent_comment"] = df["parent_comment"].astype(str).apply(html.unescape).apply(html.unescape)
    print("HTML entities decoded with same passes as notebook")

    # Same as notebook Cell 56: create length columns before spam/duplicate handling.
    df["comment_len_num"] = df["comment"].astype(str).str.len()
    df["parent_length"] = df["parent_comment"].astype(str).str.len()

    # Same as notebook Cell 61.
    def is_spam(text):
        text = str(text).strip()
        words = text.split()

        # Drop single character comments only
        if len(text) <= 1:
            return True

        # Drop repetitive comments — 10+ words but only 1-3 unique words
        if len(words) >= 10:
            unique_words = set(w.lower() for w in words)
            if len(unique_words) <= 3:
                return True

        return False

    before = df.shape[0]
    df = df[~df["comment"].apply(is_spam)]
    df = df[~df["parent_comment"].apply(is_spam)]
    print(f"Dropped spam/single-char rows: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 62: drop_duplicates before final column selection.
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Dropped duplicate rows: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 63: drop markdown hyperlinks in comment.
    before = df.shape[0]
    df = df[~df["comment"].str.contains(r"\[[^\]]+\]\([^)]+\)", na=False)]
    print(f"Dropped remaining markdown hyperlink rows: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 64: drop code block rows in comment.
    before = df.shape[0]
    df = df[~df["comment"].str.contains(r"```[\s\S]+?```", na=False)]
    print(f"Dropped code block rows: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 66: normalize whitespace in both columns.
    df["comment"] = df["comment"].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())
    df["parent_comment"] = df["parent_comment"].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())
    print("Extra whitespace removed")

    # Same as notebook Cell 67: BERTweet tokenizer and combined parent + comment.
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    df["combined"] = df["parent_comment"] + " </s></s> " + df["comment"]
    df["token_len"] = df["combined"].apply(
        lambda x: len(tokenizer.encode(x, truncation=False))
    )

    print(df["token_len"].describe())
    print(f"Rows exceeding 128 tokens: {(df['token_len'] > 128).sum():,}")
    print(f"Rows exceeding 256 tokens: {(df['token_len'] > 256).sum():,}")
    print(f"Rows exceeding 512 tokens: {(df['token_len'] > 512).sum():,}")

    # Same as notebook Cell 68: keep <= 128.
    before = df.shape[0]
    df = df[df["token_len"] <= 128]
    print(f"Dropped rows above 128 tokens: {before - df.shape[0]:,}; shape={df.shape}")

    # Same as notebook Cell 69.
    df = df[["parent_comment", "comment", "label"]]
    print(f"Final A.csv shape: {df.shape}")
    print(f"Final A.csv label balance:\n{df['label'].value_counts()}")

    df.to_csv(A_CSV, index=False)
    print(f"Saved A.csv: {A_CSV}")
    return df


def generate_B_exact() -> pd.DataFrame:
    if not A_CSV.exists():
        raise FileNotFoundError(f"A.csv not found: {A_CSV}")

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    # Same as row_10_versionB.ipynb.
    df = pd.read_csv(A_CSV)
    print(f"Loaded A.csv: {df.shape}")
    print(f"Label balance:\n{df['label'].value_counts()}")

    negation_words = {
        "no", "not", "never", "neither", "nobody", "nothing", "nowhere",
        "nor", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
        "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't", "cannot", "without",
    }

    stop_words = set(stopwords.words("english")) - negation_words
    print(f"Total stopwords: {len(set(stopwords.words('english')))}")
    print(f"After removing negation: {len(stop_words)}")

    def remove_stopwords(text):
        words = str(text).split()
        return " ".join([w for w in words if w.lower() not in stop_words])

    df["comment"] = df["comment"].apply(remove_stopwords)
    df["parent_comment"] = df["parent_comment"].apply(remove_stopwords)
    print("Stopwords removed")

    df.to_csv(B_CSV, index=False)
    print(f"Saved B.csv: {B_CSV}")
    print(f"B.csv shape: {df.shape}")
    return df


def main() -> None:
    generate_A_exact()
    generate_B_exact()
    print("Task 09 and 10 exact reproduction completed.")


if __name__ == "__main__":
    main()
