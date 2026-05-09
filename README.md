# Sarcasm Detection from Social Media Text

This repository contains the implementation for the WQF7007 Natural Language Processing project.

## Project Scope

**Task:** Binary sarcasm detection  
**Dataset:** SARC - Self-Annotated Reddit Corpus  
**Models:** BERTweet and RoBERTa  
**Main comparison:**
1. BERTweet vs RoBERTa
2. Stopwords kept vs selective stopword removal

The final demo should use only the best-performing model after comparison.

## Project Structure

```text
Sarcasm-Detection/
├── app/                    # Demo app files
├── data/
│   ├── raw/                # Original raw dataset files - not committed
│   ├── processed/          # A.csv and B.csv - not committed
│   └── splits/             # Train/validation/test splits - not committed
├── docs/                   # Methodology notes and task list
├── models/                 # Saved model checkpoints - not committed
├── notebooks/              # Experiment notebooks
├── reports/figures/        # Figures for report/presentation
├── src/                    # Python source code
├── tests/                  # Test files if needed
└── requirements.txt
```

## Important Data Files

The dataset files are not pushed to GitHub because they may be large.

Each team member should place these files locally:

```text
data/processed/A.csv
data/processed/B.csv
```

- `A.csv`: Version A preprocessing, stopwords kept.
- `B.csv`: Version B preprocessing, selective stopword removal while keeping negations.

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Task 11: Train/Validation/Test Split

After placing `A.csv` and `B.csv` inside `data/processed/`, run:

```bash
python src/prepare_splits.py
```

This creates:

```text
data/splits/versionA/train.csv
data/splits/versionA/valid.csv
data/splits/versionA/test.csv
data/splits/versionB/train.csv
data/splits/versionB/valid.csv
data/splits/versionB/test.csv
data/splits/split_summary.json
```

## Academic Methodology

The project uses a controlled experimental design:

- same train/validation/test split for all experiments
- same metrics for all models
- comparison between BERTweet and RoBERTa
- comparison between stopwords kept and stopwords removed
- final model selected based mainly on F1-score and error analysis
