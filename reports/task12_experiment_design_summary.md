# Task 12: Model Experiment Design Summary

## Purpose
Design a fair, reproducible comparison between BERTweet and RoBERTa under two preprocessing settings: stopwords kept and selective stopword removal.

## Task Definition
- Task type: Binary sarcasm classification
- Positive class: 1 = sarcastic
- Negative class: 0 = non-sarcastic
- Input strategy: text-only context pair using `parent_comment` and `comment`
- No user metadata is used

## Experiment Matrix
| Experiment ID | Model | Hugging Face Model | Data Version | Stopword Setting |
|---|---|---|---|---|
| E01_BERTWEET_A | BERTweet | vinai/bertweet-base | versionA | kept |
| E02_BERTWEET_B | BERTweet | vinai/bertweet-base | versionB | selectively_removed |
| E03_ROBERTA_A | RoBERTa | roberta-base | versionA | kept |
| E04_ROBERTA_B | RoBERTa | roberta-base | versionB | selectively_removed |

## Controlled Variables
- Random seed: 42
- Split design: stratified 80/10/10 split
- Max sequence length: 128
- Padding: max_length
- Truncation: True
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-05
- Weight decay: 0.01

## Evaluation Metrics
- accuracy
- precision
- recall
- f1_score
- confusion_matrix

## Split Validation Summary
| Version | Split | Rows | Label 0 | Label 1 |
|---|---:|---:|---:|---:|
| versionA | train | 772069 | 383897 | 388172 |
| versionA | valid | 96509 | 47987 | 48522 |
| versionA | test | 96509 | 47987 | 48522 |
| versionB | train | 772069 | 383897 | 388172 |
| versionB | valid | 96509 | 47987 | 48522 |
| versionB | test | 96509 | 47987 | 48522 |

## Validation Checks
- PASS: Experiment registry contains exactly four required experiments.
- PASS: Experiment matrix covers BERTweet and RoBERTa across Version A and Version B.
- PASS: Validated split files for versionA.
- PASS: Validated split files for versionB.
- PASS: Version A and Version B have matching row counts and label distributions for all splits.

## Model Selection Rule
The final model will be selected primarily based on F1-score, then checked using precision, recall, confusion matrix, and error analysis. Only one final model will be deployed in the demo.

## Academic Rationale
- Why two models: BERTweet is specialized for social media text, while RoBERTa is a strong general transformer baseline. Comparing them helps determine whether social-media pretraining improves sarcasm detection on SARC.
- Why two preprocessing versions: The lecturer requested checking whether removing stopwords helps or hurts. Sarcasm may depend on small words and sentence structure, so the effect must be tested rather than assumed.
- Why one final model: The final application should be simple and reproducible. After fair comparison, only the best-performing model will be selected for the final demo.

## Status
Task 12 is complete. The experiment design is ready for Task 13 and Task 14 model training.
