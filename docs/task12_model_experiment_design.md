# Task 12: Model Experiment Design

## Objective

Task 12 defines the controlled experimental design for the sarcasm detection project. The goal is to compare BERTweet and RoBERTa fairly under two preprocessing settings:

1. Version A: stopwords kept
2. Version B: selective stopword removal while keeping negations

This task does not train the models yet. Training will be done in the next tasks.

## Main Research Question

How do BERTweet and RoBERTa perform on binary sarcasm detection using the SARC dataset, and does selective stopword removal improve or reduce model performance?

## Classification Task

- Task type: Binary text classification
- Class 1: sarcastic
- Class 0: non-sarcastic
- Input: text only
- Text fields used: `parent_comment` and `comment`
- Metadata is not used

The `parent_comment` is treated as context, and the `comment` is treated as the target reply. This is academically suitable because sarcasm often depends on conversational context.

## Experiment Matrix

| Experiment ID | Model | Data Version | Stopword Setting |
|---|---|---|---|
| E01_BERTWEET_A | BERTweet | Version A | Stopwords kept |
| E02_BERTWEET_B | BERTweet | Version B | Stopwords selectively removed |
| E03_ROBERTA_A | RoBERTa | Version A | Stopwords kept |
| E04_ROBERTA_B | RoBERTa | Version B | Stopwords selectively removed |

## Controlled Conditions

To make the comparison fair, all experiments will use:

- Same train/validation/test split
- Same random seed: 42
- Same split ratio: 80/10/10
- Same maximum sequence length: 128
- Same evaluation metrics
- Same model selection rule

## Tokenization Plan

Each model will use its own official Hugging Face tokenizer:

- BERTweet: `vinai/bertweet-base`
- RoBERTa: `roberta-base`

Tokenization will handle padding and truncation. Punctuation will not be manually removed because punctuation can be useful for sarcasm detection.

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

F1-score is the main selection metric because it balances precision and recall.

## Final Model Selection

Although four experiments will be trained and compared, only one model will be selected for the final demo. The final model will be chosen based mainly on F1-score, then checked using confusion matrix and error analysis.

## Why This Design Is Academic

This design is academically appropriate because it uses a controlled comparison. Only two factors are varied:

1. Model architecture: BERTweet vs RoBERTa
2. Preprocessing version: stopwords kept vs stopwords selectively removed

All other factors are kept consistent so that the results can be interpreted fairly.
