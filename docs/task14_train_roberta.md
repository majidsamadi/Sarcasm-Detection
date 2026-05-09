# Task 14: Train RoBERTa

## Objective

Task 14 fine-tunes RoBERTa for binary sarcasm detection using the same controlled experimental setup as Task 13 BERTweet training.

## Experiments

| Experiment | Model | Dataset Version | Stopword Setting |
|---|---|---|---|
| E03 | RoBERTa | Version A | Stopwords kept |
| E04 | RoBERTa | Version B | Stopwords selectively removed |

## Methodological Control

To make the comparison fair, RoBERTa uses:

- the same train/validation/test split from Task 11,
- the same text-only input format,
- the same maximum sequence length of 128,
- the same practical training sample size as BERTweet,
- the same metrics: accuracy, precision, recall, macro-F1, weighted-F1, and confusion matrix.

## Text Input

The input is constructed as:

```text
Parent: <parent_comment> Reply: <comment>
```

This remains text-only input, but includes conversational context, which is useful for sarcasm detection.

## Final Use

The local RoBERTa checkpoints are saved under `models/roberta/`. These checkpoints are intentionally ignored by Git because they are large.

The best final model will be selected later after comparing Task 13 and Task 14 results.
