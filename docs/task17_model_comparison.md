# Task 17: Model Comparison Methodology

## Objective

Task 17 compares the four completed model experiments using the held-out test results from Task 15:

1. BERTweet + Version A (stopwords kept)
2. BERTweet + Version B (selective stopword removal)
3. RoBERTa + Version A (stopwords kept)
4. RoBERTa + Version B (selective stopword removal)

## Methodology

The comparison uses a controlled experimental design. All models were trained and evaluated using the same train/validation/test split design, the same text-only input format, and the same maximum sequence length. This ensures that differences in performance are due mainly to model architecture and preprocessing strategy, rather than unfair data differences.

## Selection Criterion

The main selection criterion is **test Macro-F1**, because sarcasm detection is a binary classification task and Macro-F1 balances the performance across sarcastic and non-sarcastic classes. Accuracy is also reported, but it is not used as the only decision criterion.

## Outputs

Task 17 produces:

- ranked model comparison table
- model-family comparison
- preprocessing comparison
- best-model recommendation
- Macro-F1 comparison chart
- accuracy comparison chart
- accuracy vs Macro-F1 chart
- progress note for the project tracker

## Link to Future Work

After Task 17, Task 18 should analyze model errors. In particular, false positives and false negatives should be reviewed to understand why the selected model misclassifies some sarcastic or non-sarcastic comments.
