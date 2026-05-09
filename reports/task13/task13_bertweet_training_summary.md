# Task 13: BERTweet Training Summary

This report summarizes BERTweet training for the two preprocessing versions.

| Experiment | Dataset | Train Rows | Valid Rows | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| E01_BERTweet_VersionA | versionA | 50,000 | 10,000 | 0.7634 | 0.7633 | 0.7633 |
| E02_BERTweet_VersionB | versionB | 50,000 | 10,000 | 0.7267 | 0.7265 | 0.7264 |

## Methodological note

Both BERTweet experiments use the same Task 11 stratified split, max sequence length 128, and identical training settings. The difference between E01 and E02 is only the preprocessing version: Version A keeps stopwords, while Version B removes stopwords selectively while preserving negation words.