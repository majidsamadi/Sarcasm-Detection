# Task 15: Model Evaluation Summary
## Evaluation Setup
- Task: binary sarcasm classification.
- Labels: 0 = non-sarcastic, 1 = sarcastic.
- Input: text-only input built from `parent_comment + comment` when both are available.
- Test set: held-out split from Task 11.
- Evaluation scope: full held-out test set.
- Metrics: accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class metrics, and confusion matrix.

## Model Comparison

| Rank | Experiment | Model | Version | Accuracy | Macro-F1 | Weighted-F1 | Rows |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | E03_RoBERTa_VersionA | RoBERTa | versionA | 0.7223 | 0.7167 | 0.7165 | 96,509 |
| 2 | E04_RoBERTa_VersionB | RoBERTa | versionB | 0.6773 | 0.6648 | 0.6644 | 96,509 |
| 3 | E01_BERTweet_VersionA | BERTweet | versionA | 0.5092 | 0.3632 | 0.3615 | 96,509 |
| 4 | E02_BERTweet_VersionB | BERTweet | versionB | 0.5018 | 0.3452 | 0.3435 | 96,509 |

## Best Model

The best-performing experiment based on macro-F1 is **E03_RoBERTa_VersionA** with macro-F1 **0.7167** and accuracy **0.7223**.

## Stopword Impact

- BERTweet: Version A (stopwords kept) macro-F1 = 0.3632; Version B (stopwords removed) macro-F1 = 0.3452. Version A is better by 0.0179.
- RoBERTa: Version A (stopwords kept) macro-F1 = 0.7167; Version B (stopwords removed) macro-F1 = 0.6648. Version A is better by 0.0519.

## Outputs

- Per-model JSON metrics: `reports/task15/*_test_metrics.json`
- Confusion matrix figures: `reports/task15/figures/*_confusion_matrix.png`
- Classification reports: `reports/task15/*_classification_report.txt`
- Comparison CSV: `reports/task15/task15_model_evaluation_comparison.csv`
