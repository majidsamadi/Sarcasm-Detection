# Task 09-10 Exact Reproduction Note

This project includes `src/reproduce_task09_10_exact.py` to reproduce the preprocessing work from the Row 9 and Row 10 notebooks.

The purpose is not to redesign or improve the preprocessing. The purpose is to generate the same two local files needed for later tasks:

```text
data/processed/A.csv
data/processed/B.csv
```

Only file paths are changed from Google Drive to the local project folder.

- `A.csv`: Version A, stopwords kept.
- `B.csv`: Version B, selective stopword removal while keeping negation words.

The generated CSV files are local dataset files and should not be committed to GitHub.
