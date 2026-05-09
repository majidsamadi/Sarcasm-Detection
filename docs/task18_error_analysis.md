# Task 18: Error Analysis

## Purpose

Task 18 analyzes the prediction errors of the selected final candidate model after Task 17 model comparison.
The selected model is **RoBERTa Version A**, because Task 17 ranked it highest using held-out test macro-F1.

## Why error analysis is needed

Metrics such as accuracy and F1-score summarize performance, but they do not explain *where* the model fails.
For sarcasm detection, error analysis is important because sarcasm often depends on implied meaning, context,
punctuation, conversational structure, and cultural cues.

## Scope of analysis

This task focuses on:

1. False positives: non-sarcastic comments predicted as sarcastic.
2. False negatives: sarcastic comments predicted as non-sarcastic.
3. Error confidence: whether the model was uncertain or confidently wrong.
4. Text length: whether short or long comments are harder.
5. Surface indicators: punctuation, quotes, all-caps words, laughter markers, and negation markers.

## Responsible-use decision

Raw Reddit text error examples are saved locally under `reports/task18/error_samples`, but they are ignored by Git and not pushed to GitHub.
This avoids publishing unnecessary social-media text samples in the public repository.

## Main outputs

- `reports/task18/task18_error_analysis_summary.md`
- `reports/task18/task18_error_analysis_summary.json`
- `reports/task18/task18_classification_report.txt`
- `reports/task18/figures/task18_error_type_counts.png`
- `reports/task18/figures/task18_error_by_confidence_bin.png`
- `reports/task18/figures/task18_error_by_length_bin.png`
- `reports/task18/figures/task18_error_rate_by_surface_feature.png`
- `reports/task18/task18_progress_note.txt`

## Local-only outputs

- `reports/task18/error_samples/high_confidence_false_positives.local.csv`
- `reports/task18/error_samples/high_confidence_false_negatives.local.csv`
- `reports/task18/predictions/task18_predictions.local.csv`

These files are useful for internal team review but are intentionally ignored by Git.
