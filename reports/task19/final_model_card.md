# Final Model Card: Sarcasm Detection

## Selected Model

- **Experiment ID:** E03_RoBERTa_VersionA
- **Model family:** RoBERTa
- **Checkpoint path:** `models/roberta/versionA`
- **Tokenizer:** `roberta-base`
- **Dataset version:** Version A
- **Preprocessing:** Stopwords kept
- **Input format:** `parent_comment + comment`
- **Task:** Binary sarcasm classification
- **Labels:** `0 = non-sarcastic`, `1 = sarcastic`

## Performance Summary

- **Accuracy:** 0.7223
- **Macro-F1:** 0.7167
- **Macro Precision:** 0.7430
- **Macro Recall:** 0.7231
- **Weighted-F1:** 0.7165

## Intended Use

This model is intended for an academic NLP demonstration of sarcasm detection in social media text.

## Not Intended For

This model should **not** be used to automatically remove, flag, punish, or penalize user-generated content.

## Key Limitation

Sarcasm is highly context-dependent. The model may fail when sarcasm requires cultural knowledge, longer conversation history, or real-world background information.
