# Task 09-10: Preprocessing Methodology

## Project Context

The project uses the SARC dataset for binary sarcasm detection in social media text. The model input is text-only and uses both `parent_comment` and `comment`, because sarcasm often depends on conversational context.

## Version A: Stopwords Kept

Version A is the default cleaned dataset. It keeps stopwords, punctuation, casing, and ALL CAPS because these elements can be useful for sarcasm detection. The cleaning process removes invalid or noisy rows such as null comments, URL rows, `[deleted]`, `[removed]`, duplicates, simple Markdown noise, spam-like rows, and rows above the selected BERTweet token limit.

Output:

```text
data/processed/A.csv
```

## Version B: Selective Stopword Removal

Version B starts from Version A and removes common English stopwords while preserving important negation words such as `not`, `no`, `never`, `don't`, `isn't`, and `cannot`.

Output:

```text
data/processed/B.csv
```

## Academic Reason

The lecturer asked the team to test whether stopword removal improves or reduces performance. Therefore, Version A and Version B allow a fair controlled comparison:

- Version A: stopwords kept
- Version B: stopwords removed selectively, with negations preserved

Both versions must use the same train/validation/test split during Task 11 so the model comparison remains fair.
