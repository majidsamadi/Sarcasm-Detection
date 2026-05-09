# Task 14: Train RoBERTa

## Purpose
This task fine-tunes RoBERTa on Version A and Version B using the same controlled setup as BERTweet. The goal is to support a fair model comparison between BERTweet and RoBERTa.

## Results

| Experiment | Dataset Version | Stopword Setting | Accuracy | Macro-F1 | Precision Macro | Recall Macro |
|---|---|---|---:|---:|---:|---:|
| E03_RoBERTa_VersionA | versionA | stopwords_kept | 0.7475 | 0.7474 | 0.7483 | 0.7478 |
| E04_RoBERTa_VersionB | versionB | stopwords_selectively_removed | 0.7189 | 0.7189 | 0.7189 | 0.7190 |

## Notes
- Version A keeps stopwords.
- Version B selectively removes stopwords while preserving negation words.
- The best final model should be selected later after comparing Task 13 and Task 14 results.
- Local model checkpoints are saved under `models/roberta/` and are intentionally ignored by Git.
