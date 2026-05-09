# Task 13: Train BERTweet

## Purpose

Task 13 trains BERTweet for binary sarcasm detection using the controlled experiment design from Task 12.

## Experiments

| Experiment | Model | Dataset version | Preprocessing |
|---|---|---|---|
| E01_BERTweet_VersionA | BERTweet | Version A | Stopwords kept |
| E02_BERTweet_VersionB | BERTweet | Version B | Selective stopword removal, negations kept |

## Input strategy

The model uses text-only input. If both `parent_comment` and `comment` columns exist, they are concatenated as context + reply. This is appropriate for sarcasm detection because sarcasm often depends on conversational context.

## Evaluation

Validation metrics include accuracy, precision, recall, F1-score, and confusion matrix. The main metric for model comparison is macro-F1 / weighted-F1 rather than accuracy alone.

## Local artifacts

Model checkpoints are saved locally under `models/bertweet/` and are ignored by Git. Metrics and summaries are saved under `reports/task13/` and can be committed.
