# Task 19: Final Model Selection

## Purpose

Task 19 selects **one final model** for the final implementation/demo. The selection is based mainly on **held-out test macro-F1**, with **accuracy** as a supporting metric and Task 18 error analysis as qualitative evidence.

## Selection Rule

The final model is selected using this rule:

1. Rank all experiments by **test macro-F1**.
2. If there is a tie, use **test accuracy** as the tie-breaker.
3. Use Task 18 error analysis to support the final decision and document limitations.

## Ranked Results

| Rank | Experiment | Model | Dataset Version | Preprocessing | Accuracy | Macro-F1 |
|---:|---|---|---|---|---:|---:|
| 1 | E03_RoBERTa_VersionA | RoBERTa | Version A | Stopwords kept | 0.7223 | 0.7167 |
| 2 | E04_RoBERTa_VersionB | RoBERTa | Version B | Selective stopword removal | 0.6773 | 0.6648 |
| 3 | E01_BERTweet_VersionA | BERTweet | Version A | Stopwords kept | 0.5092 | 0.3632 |
| 4 | E02_BERTweet_VersionB | BERTweet | Version B | Selective stopword removal | 0.5018 | 0.3452 |

## Selected Final Model

**Selected model:** `E03_RoBERTa_VersionA`  
**Model family:** RoBERTa  
**Dataset version:** Version A  
**Preprocessing:** Stopwords kept  
**Checkpoint path:** `models/roberta/versionA`  
**Tokenizer:** `roberta-base`  
**Input format:** `parent_comment + comment`  
**Max sequence length:** 128  

## Justification

`E03_RoBERTa_VersionA` is selected because it achieved the highest held-out test **macro-F1** among all four controlled experiments. Macro-F1 is prioritized because it evaluates both classes more fairly than accuracy alone, especially for classification tasks where the cost of false positives and false negatives should both be considered.

The selected model also supports the conclusion from the stopword impact analysis: **Version A, where stopwords are kept, performed better than Version B**. This is academically reasonable for sarcasm detection because sarcasm often depends on small function words, context, contrast, and sentence structure.

Task 18 error analysis was used as supporting evidence for the final selection. The selected model is not chosen only because it has the highest score; it is also the model for which false positives, false negatives, confidence behavior, and text-length effects were analyzed. The Task 18 error analysis file references: `{'experiment_id': 'E03_RoBERTa_VersionA', 'model_family': 'RoBERTa', 'model_name': 'roberta-base', 'checkpoint_dir': 'models/roberta/versionA', 'dataset_version': 'versionA', 'preprocessing': 'Stopwords kept', 'test_split': 'data/splits/versionA/test.csv', 'selection_basis': 'Task 17 ranked this model highest by held-out test macro-F1.'}`.

## Final Implementation Decision

The final demo should load only the selected model checkpoint:

```text
models/roberta/versionA
```

The other trained models are kept for comparison and reporting, but they should not be used in the final prediction interface unless the project scope changes.

## Limitations to Mention

- The model predicts sarcasm from text patterns and does not truly understand intent.
- Sarcasm is context-dependent and may require background knowledge beyond the available text.
- Reddit data may contain platform-specific language and bias.
- The model should not be used to automatically punish, moderate, or penalize users.
- Raw Reddit examples from error analysis should remain local and should not be pushed to GitHub.

## Outputs

- `configs/final_model_config.json`
- `reports/task19/task19_final_model_selection_summary.json`
- `reports/task19/task19_final_model_selection_table.csv`
- `reports/task19/task19_final_model_selection_summary.md`
- `reports/task19/figures/task19_final_model_macro_f1_ranking.png`
- `reports/task19/figures/task19_final_model_accuracy_ranking.png`
