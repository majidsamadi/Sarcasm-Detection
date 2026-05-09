# Task 16: Stopword Impact Analysis

## Purpose

This task analyzes whether keeping stopwords or selectively removing stopwords improves sarcasm detection performance. The comparison uses the Task 15 held-out test results for the four controlled experiments.

## Experimental Context

- Task: binary sarcasm classification.
- Label 0: non-sarcastic.
- Label 1: sarcastic.
- Version A: stopwords kept.
- Version B: stopwords selectively removed while preserving negations.
- Main selection metric: macro-F1.
- Evaluation source: Task 15 full held-out test split.

## Metrics Used

| Experiment | Model | Version | Stopword Setting | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Rows |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| E03_RoBERTa_VersionA | RoBERTa | versionA | stopwords kept | 0.7223 | 0.7430 | 0.7231 | 0.7167 | 0.7165 | 96,509 |
| E04_RoBERTa_VersionB | RoBERTa | versionB | stopwords selectively removed | 0.6773 | 0.7106 | 0.6784 | 0.6648 | 0.6644 | 96,509 |
| E01_BERTweet_VersionA | BERTweet | versionA | stopwords kept | 0.5092 | 0.6638 | 0.5118 | 0.3632 | 0.3615 | 96,509 |
| E02_BERTweet_VersionB | BERTweet | versionB | stopwords selectively removed | 0.5018 | 0.6403 | 0.5046 | 0.3452 | 0.3435 | 96,509 |

## Stopword Impact by Model

| Model | Stopwords Kept Macro-F1 | Stopwords Removed Macro-F1 | Delta Kept - Removed | Accuracy Delta | Conclusion |
|---|---:|---:|---:|---:|---|
| BERTweet | 0.3632 | 0.3452 | 0.0179 | 0.0073 | Keeping stopwords performed better. |
| RoBERTa | 0.7167 | 0.6648 | 0.0519 | 0.0450 | Keeping stopwords performed better. |

## Interpretation

For both BERTweet and RoBERTa, Version A achieved higher macro-F1 than Version B. This indicates that keeping stopwords is more effective for this sarcasm detection setup. This result is academically reasonable because sarcasm often depends on context, contrast, negation, short function words, and sentence structure.

## Final Recommendation from Task 16

Based on the held-out test macro-F1 results, the best-performing preprocessing setting is **versionA** for **RoBERTa**. The best experiment is **E03_RoBERTa_VersionA** with macro-F1 **0.7167** and accuracy **0.7223**.

For the final project report, the recommended conclusion is: **do not remove stopwords for the final model**, because Version A performs better under the controlled experiment design.

## Notes for Academic Reporting

- The same train/validation/test split was used for Version A and Version B.
- Both preprocessing versions were evaluated using the same held-out test set size per version.
- Macro-F1 is emphasized because it balances performance across sarcastic and non-sarcastic classes.
- The stopword decision is based on empirical evaluation rather than assumption.
