# Task 13 Reports

This folder stores BERTweet training metrics and summaries.

Expected generated files after running:

```bash
bash run_task13_train_bertweet.sh
```

are:

```text
task13_bertweet_training_summary.md
task13_bertweet_training_summary.json
task13_bertweet_training_results.csv
E01_BERTweet_VersionA/best_valid_metrics.json
E02_BERTweet_VersionB/best_valid_metrics.json
```

The actual BERTweet model checkpoints are saved under `models/bertweet/` and are ignored by Git.
