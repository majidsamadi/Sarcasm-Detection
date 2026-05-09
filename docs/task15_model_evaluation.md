# Task 15: Model Evaluation

## Purpose

Task 15 evaluates the four trained model experiments on the held-out test split created in Task 11.

## Evaluated Experiments

| Experiment | Model | Dataset Version | Preprocessing |
|---|---|---|---|
| E01 | BERTweet | Version A | Stopwords kept |
| E02 | BERTweet | Version B | Stopwords selectively removed |
| E03 | RoBERTa | Version A | Stopwords kept |
| E04 | RoBERTa | Version B | Stopwords selectively removed |

## Evaluation Methodology

All four experiments are evaluated using the same controlled setup:

- same held-out test split from Task 11;
- same maximum sequence length of 128;
- same text-only input construction;
- same labels: `0 = non-sarcastic`, `1 = sarcastic`;
- same evaluation metrics.

The input is constructed from `parent_comment + comment` when both fields are available, because sarcasm often depends on conversational context.

## Metrics

The evaluation reports:

- accuracy;
- macro precision;
- macro recall;
- macro-F1;
- weighted-F1;
- per-class precision, recall, and F1-score;
- confusion matrix.

Macro-F1 is used as the primary model-selection metric because it gives equal importance to both classes.

## Outputs

Task 15 creates:

- JSON metrics for each experiment;
- text classification reports;
- confusion matrix figures;
- a model-comparison CSV;
- a markdown summary report;
- a short progress note for the project tracking sheet.
