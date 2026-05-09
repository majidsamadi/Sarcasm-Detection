# Task 11 Methodology: Train/Validation/Test Split

## Objective

Create a fair and reproducible split for the sarcasm detection experiments.

## Method

- Use Version A and Version B datasets.
- Use an 80/10/10 train/validation/test split.
- Use stratified splitting to preserve sarcastic/non-sarcastic label balance.
- Use the same row indices for Version A and Version B.
- Use random seed 42 for reproducibility.

## Why this is important

The project compares:

1. BERTweet vs RoBERTa
2. Stopwords kept vs selective stopword removal

To make this comparison fair, every experiment must use the same split.
