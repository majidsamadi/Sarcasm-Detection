# Task 18: Error Analysis

## Selected model

- **Experiment:** E03_RoBERTa_VersionA
- **Model:** RoBERTa (`roberta-base`)
- **Dataset version:** versionA (Stopwords kept)
- **Selection basis:** Task 17 ranked this model highest by held-out test macro-F1.

## Overall held-out test performance

| Metric | Value |
|---|---:|
| Accuracy | 0.7223 |
| Macro Precision | 0.7430 |
| Macro Recall | 0.7231 |
| Macro F1 | 0.7167 |
| Weighted F1 | 0.7165 |

## Confusion matrix

Rows are actual labels and columns are predicted labels.

| Actual \ Predicted | Non-sarcastic (0) | Sarcastic (1) |
|---|---:|---:|
| Non-sarcastic (0) | 41630 | 6357 |
| Sarcastic (1) | 20445 | 28077 |

## Error counts

| Type | Count |
|---|---:|
| False Positive | 6357 |
| False Negative | 20445 |
| Correct | 69707 |

## Main observations

- The selected model made 26,802 errors out of 96,509 evaluated test examples, giving an error rate of 0.2777.
- False positives: 6,357; false negatives: 20,445. This shows whether the model is more likely to over-detect or miss sarcasm.
- False negatives are higher than false positives, so the model is more likely to miss sarcastic comments.
- There are 2,569 high-confidence errors with confidence >= 0.85. These are useful for qualitative review because the model was confident but wrong.

## Error patterns analyzed

The analysis summarizes errors by confidence level, text length, punctuation cues, quotation markers, all-caps words, laughter markers, and negation markers. These features are surface-level indicators only; they do not prove causality, but they help explain where the model struggles.

### Confidence distribution by prediction correctness

- **<0.55:** {'False': 4305, 'True': 4834}
- **0.55-0.70:** {'False': 11490, 'True': 17639}
- **0.70-0.85:** {'False': 8438, 'True': 24586}
- **>=0.85:** {'False': 2569, 'True': 22648}

### Length distribution by prediction correctness

- **0-20:** {'False': 10482, 'True': 24524}
- **21-50:** {'False': 12927, 'True': 34442}
- **51-100:** {'False': 3293, 'True': 10433}
- **>100:** {'False': 100, 'True': 308}

## Local error examples

Raw Reddit text examples are saved locally for internal team analysis only and are ignored by Git. They are not committed to the public repository to reduce privacy and responsible-use risks.

- **High-confidence false positives:** `reports/task18/error_samples/high_confidence_false_positives.local.csv`
- **High-confidence false negatives:** `reports/task18/error_samples/high_confidence_false_negatives.local.csv`
- **Prediction table:** `reports/task18/predictions/task18_predictions.local.csv`

## Academic interpretation

False positives usually mean the model detected sarcastic patterns in a non-sarcastic text. False negatives usually mean the model missed sarcasm, often because sarcasm can depend on context, implied meaning, or conversational background. This supports the project limitation that sarcasm detection should not be used for automatic moderation or punishment without human review.

## Generated figures

- `reports/task18/figures/task18_error_type_counts.png`
- `reports/task18/figures/task18_error_by_confidence_bin.png`
- `reports/task18/figures/task18_error_by_length_bin.png`
- `reports/task18/figures/task18_error_rate_by_surface_feature.png`
