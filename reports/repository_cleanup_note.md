# Repository Cleanup Note

This repository keeps code, methodology notes, and reports under version control.
Large local dataset files are intentionally excluded from GitHub.

## Local files not pushed to GitHub

The following generated files should remain local:

- `data/raw/train-balanced-sarcasm.csv`
- `data/processed/A.csv`
- `data/processed/B.csv`
- `data/splits/versionA/train.csv`
- `data/splits/versionA/valid.csv`
- `data/splits/versionA/test.csv`
- `data/splits/versionB/train.csv`
- `data/splits/versionB/valid.csv`
- `data/splits/versionB/test.csv`
- `data/splits/split_summary.json`

## Correct preprocessing state

Tasks 9 and 10 are represented by the exact reproduction workflow:

- `src/reproduce_task09_10_exact.py`
- `run_task09_10_exact.sh`
- `docs/task09_10_exact_reproduction_note.md`

The earlier non-exact preprocessing pipeline files were removed to avoid confusion.

## Task 11 state

Task 11 uses:

- `src/prepare_splits.py`
- `run_task11_splits.sh`
- `run_task11_splits.bat`
- `reports/task11_split_summary.md`
- `reports/task11_progress_note.txt`

The split CSV files are generated locally and are not pushed to GitHub.
