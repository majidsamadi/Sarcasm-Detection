# Task 16: Stopword Impact Analysis

## Purpose

Task 16 investigates whether stopword removal improves sarcasm detection performance. This is required because the project uses two preprocessing versions:

- **Version A:** stopwords kept.
- **Version B:** stopwords selectively removed while preserving negation words.

The goal is not to assume which version is better, but to use empirical results from the held-out test set.

## Academic Rationale

Stopword removal is common in traditional NLP pipelines, but it is not always suitable for transformer-based sarcasm detection. Sarcasm often depends on context, contrast, sentence structure, and small function words. For example, words such as “not”, “so”, “very”, or similar context-building terms may affect the intended meaning of a sarcastic sentence.

Therefore, the project compares the two preprocessing versions under a controlled setup.

## Controlled Comparison

The analysis compares:

| Model | Version A | Version B |
|---|---|---|
| BERTweet | Stopwords kept | Stopwords selectively removed |
| RoBERTa | Stopwords kept | Stopwords selectively removed |

The same split, task definition, input format, and evaluation metrics are used. This makes the comparison fair.

## Metrics

The main metric is **macro-F1**, supported by:

- accuracy
- macro precision
- macro recall
- weighted-F1
- confusion matrix outputs from Task 15

Macro-F1 is emphasized because it balances performance across the sarcastic and non-sarcastic classes.

## Expected Output

Task 16 generates:

- `reports/task16/task16_stopword_impact_summary.md`
- `reports/task16/task16_stopword_impact_summary.json`
- `reports/task16/task16_progress_note.txt`
- `reports/task16/figures/stopword_macro_f1_comparison.png`
- `reports/task16/figures/stopword_accuracy_comparison.png`

## Final Use in Report

The result of this task should be used in the final report to justify whether the final model should use Version A or Version B preprocessing.
