# Task 11: Train/Validation/Test Split Summary

## Purpose

Task 11 creates a fair stratified train/validation/test split for both preprocessing versions. The same row indices are used for Version A and Version B so that later model comparison is academically fair.

## Split Configuration

- Random seed: `42`
- Label column: `label`
- Train ratio: `0.8`
- Validation ratio: `0.1`
- Test ratio: `0.1`

## Version A: Stopwords Kept

| Split | Rows | Label Distribution |
|---|---:|---|
| train | 771609 | 0: 383657 (49.7217%), 1: 387952 (50.2783%) |
| valid | 96452 | 0: 47957 (49.7211%), 1: 48495 (50.2789%) |
| test | 96452 | 0: 47957 (49.7211%), 1: 48495 (50.2789%) |

## Version B: Selective Stopword Removal

| Split | Rows | Label Distribution |
|---|---:|---|
| train | 771609 | 0: 383657 (49.7217%), 1: 387952 (50.2783%) |
| valid | 96452 | 0: 47957 (49.7211%), 1: 48495 (50.2789%) |
| test | 96452 | 0: 47957 (49.7211%), 1: 48495 (50.2789%) |

## Academic Note

The split is stratified, so the sarcastic and non-sarcastic label balance is preserved across train, validation, and test sets. This supports fair comparison between BERTweet and RoBERTa and fair comparison between stopword-kept and stopword-removed preprocessing.
