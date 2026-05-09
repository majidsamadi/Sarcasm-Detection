# Task 17: Model Comparison

## Purpose

This task compares the four controlled experiments defined in Task 12 and evaluated in Task 15. The comparison uses the held-out test-set results, not training or validation results. The final model should be selected mainly using **test Macro-F1**, with accuracy and error analysis used as supporting evidence.

## Ranked Model Comparison

| Rank | Experiment | Model | Version | Preprocessing | Accuracy | Macro-F1 | Macro Precision | Macro Recall | Weighted-F1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | E03_RoBERTa_VersionA | RoBERTa | Version A | Stopwords kept | 0.7223 | 0.7167 | 0.7430 | 0.7231 | 0.7165 |
| 2 | E04_RoBERTa_VersionB | RoBERTa | Version B | Selective stopword removal | 0.6773 | 0.6648 | 0.7106 | 0.6784 | 0.6644 |
| 3 | E01_BERTweet_VersionA | BERTweet | Version A | Stopwords kept | 0.5092 | 0.3632 | 0.6638 | 0.5118 | 0.3615 |
| 4 | E02_BERTweet_VersionB | BERTweet | Version B | Selective stopword removal | 0.5018 | 0.3452 | 0.6403 | 0.5046 | 0.3435 |

## Best Model

The best experiment based on test Macro-F1 is:

- **Experiment:** E03_RoBERTa_VersionA
- **Model:** RoBERTa
- **Dataset version:** Version A
- **Preprocessing:** Stopwords kept
- **Accuracy:** 0.7223
- **Macro-F1:** 0.7167
- **Macro Precision:** 0.7430
- **Macro Recall:** 0.7231

## Model-Family Summary

- **BERTweet** best run: E01_BERTweet_VersionA (Version A, Stopwords kept) with Macro-F1 = 0.3632 and Accuracy = 0.5092.
- **RoBERTa** best run: E03_RoBERTa_VersionA (Version A, Stopwords kept) with Macro-F1 = 0.7167 and Accuracy = 0.7223.

## Stopword / Preprocessing Comparison

- **BERTweet**: Version A Macro-F1 = 0.3632, Version B Macro-F1 = 0.3452. Version A is higher by 0.0180 Macro-F1.
- **RoBERTa**: Version A Macro-F1 = 0.7167, Version B Macro-F1 = 0.6648. Version A is higher by 0.0519 Macro-F1.

## Academic Interpretation

The comparison shows which model and preprocessing version performs best under the same experimental setup. Because all four experiments use the same split design, text-only input format, and maximum sequence length, the comparison is methodologically fair.

The current best-performing model is **E03_RoBERTa_VersionA**, selected mainly because it has the highest test Macro-F1 score. Macro-F1 is emphasized because it balances performance across both classes instead of focusing only on overall accuracy.

## Generated Outputs

- `reports/task17/task17_model_comparison_summary.md`
- `reports/task17/task17_model_comparison_summary.json`
- `reports/task17/task17_model_comparison_table.csv`
- `reports/task17/figures/task17_macro_f1_comparison.png`
- `reports/task17/figures/task17_accuracy_comparison.png`
- `reports/task17/figures/task17_accuracy_vs_macro_f1.png`
- `reports/task17/task17_progress_note.txt`

## Note for Next Task

Task 18 should perform error analysis on the best-performing model and compare false positives versus false negatives. This will help explain where the selected model still struggles with sarcasm detection.
