#!/usr/bin/env python3
"""Generate Task 22 final academic documentation for the NLP sarcasm detection project.

This script consolidates the repository outputs from Tasks 09-21 into a structured
academic final-report package. It is intentionally documentation-focused: it does
not retrain models or modify data splits.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "final_report"
TASK22_DIR = PROJECT_ROOT / "reports" / "task22"
DOCS_DIR = PROJECT_ROOT / "docs"

TEAM = [
    ("Faridah Albalawi", "22116678", "Project Manager; Data Engineer"),
    ("Dhanasree Seelam", "23077229", "Documentation Leader"),
    ("Gebriella Anthony", "17204189", "Evaluation and Ethics Specialist"),
    ("Majid Samadikuchaksaraee", "25068504", "Presenter; Modeling and ML Engineer; Evaluation and Ethics Specialist"),
    ("Rupika Selvaraja", "24082992", "Data Engineer; Modeling and ML Engineer"),
]

# Fallback values based on the verified terminal logs and generated reports.
FALLBACK_TEST_RESULTS = {
    "E01_BERTweet_VersionA": {"model": "BERTweet", "version": "Version A", "preprocessing": "Stopwords kept", "accuracy": 0.5092, "macro_f1": 0.3632},
    "E02_BERTweet_VersionB": {"model": "BERTweet", "version": "Version B", "preprocessing": "Selective stopword removal", "accuracy": 0.5018, "macro_f1": 0.3452},
    "E03_RoBERTa_VersionA": {"model": "RoBERTa", "version": "Version A", "preprocessing": "Stopwords kept", "accuracy": 0.7223, "macro_f1": 0.7167},
    "E04_RoBERTa_VersionB": {"model": "RoBERTa", "version": "Version B", "preprocessing": "Selective stopword removal", "accuracy": 0.6773, "macro_f1": 0.6648},
}

FALLBACK_VALID_RESULTS = {
    "E01_BERTweet_VersionA": {"accuracy": 0.7634, "macro_f1": 0.7633},
    "E02_BERTweet_VersionB": {"accuracy": 0.7267, "macro_f1": 0.7265},
    "E03_RoBERTa_VersionA": {"accuracy": 0.7475, "macro_f1": 0.7474},
    "E04_RoBERTa_VersionB": {"accuracy": 0.7189, "macro_f1": 0.7189},
}

TASKS = [
    ("09", "Preprocessing Version A", "Reproduced teammate preprocessing exactly and generated A.csv with stopwords kept."),
    ("10", "Preprocessing Version B", "Generated B.csv by selectively removing stopwords while preserving negation words."),
    ("11", "Train/Validation/Test Split", "Created stratified 80/10/10 splits for Version A and Version B using the same random seed."),
    ("12", "Model Experiment Design", "Defined four controlled experiments across two models and two preprocessing versions."),
    ("13", "Train BERTweet", "Fine-tuned BERTweet on Version A and Version B."),
    ("14", "Train RoBERTa", "Fine-tuned RoBERTa on Version A and Version B."),
    ("15", "Model Evaluation", "Evaluated all four experiments on the held-out test set with accuracy, precision, recall, F1, and confusion matrices."),
    ("16", "Stopword Impact Analysis", "Compared Version A and Version B to determine whether stopword removal helped or reduced performance."),
    ("17", "Model Comparison", "Ranked the four experiments mainly by held-out test macro-F1."),
    ("18", "Error Analysis", "Analyzed false positives, false negatives, confidence patterns, text length, and surface features for the best model."),
    ("19", "Final Model Selection", "Selected RoBERTa Version A as the final model based mainly on test macro-F1."),
    ("20", "Demo Interface", "Created a Streamlit prediction demo using the selected local checkpoint."),
    ("20B", "Enhanced Interactive Dashboard", "Created a fuller Streamlit dashboard showing workflow progress, reports, results, and demo."),
    ("21", "Ethics and Limitations", "Generated a risk register, responsible-use checklist, and ethics/limitations documentation."),
    ("22", "Final Report Writing", "Generated full academic documentation from the completed project outputs."),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def flatten_json(data: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            key = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_json(v, key))
    else:
        out[prefix] = data
    return out


def find_metric(data: Dict[str, Any], names: Iterable[str]) -> Optional[float]:
    flat = flatten_json(data)
    normalized_names = {n.lower().replace("-", "_") for n in names}
    for k, v in flat.items():
        key = k.lower().replace("-", "_")
        last = key.split(".")[-1]
        if last in normalized_names or any(key.endswith("." + name) for name in normalized_names):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    pass
    return None


def get_git_info() -> Dict[str, str]:
    def run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return ""
    return {
        "commit": run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "remote": run(["git", "config", "--get", "remote.origin.url"]),
        "latest_log": run(["git", "log", "--oneline", "-12"]),
    }


def detect_split_summary() -> Dict[str, Any]:
    # Prefer JSON if available, otherwise fallback to verified values.
    split = load_json(PROJECT_ROOT / "data" / "splits" / "split_summary.json")
    if split:
        return split
    return {
        "final_aligned_rows": 965087,
        "split_ratio": {"train": 0.80, "valid": 0.10, "test": 0.10},
        "random_seed": 42,
        "versionA": {"train_rows": 772069, "valid_rows": 96509, "test_rows": 96509},
        "versionB": {"train_rows": 772069, "valid_rows": 96509, "test_rows": 96509},
        "label_distribution": {
            "train": {"1": 388172, "0": 383897},
            "valid": {"1": 48522, "0": 47987},
            "test": {"1": 48522, "0": 47987},
        },
    }


def collect_results() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for exp, fallback in FALLBACK_TEST_RESULTS.items():
        metrics = load_json(PROJECT_ROOT / "reports" / "task15" / f"{exp}_test_metrics.json")
        accuracy = find_metric(metrics, ["accuracy", "test_accuracy"]) or fallback["accuracy"]
        macro_f1 = find_metric(metrics, ["macro_f1", "macro f1", "f1_macro", "test_macro_f1"]) or fallback["macro_f1"]
        macro_precision = find_metric(metrics, ["macro_precision", "precision_macro", "test_macro_precision"])
        macro_recall = find_metric(metrics, ["macro_recall", "recall_macro", "test_macro_recall"])
        weighted_f1 = find_metric(metrics, ["weighted_f1", "f1_weighted", "test_weighted_f1"])
        valid = FALLBACK_VALID_RESULTS.get(exp, {})
        results[exp] = {
            **fallback,
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_precision": None if macro_precision is None else round(float(macro_precision), 4),
            "macro_recall": None if macro_recall is None else round(float(macro_recall), 4),
            "weighted_f1": None if weighted_f1 is None else round(float(weighted_f1), 4),
            "validation_accuracy": valid.get("accuracy"),
            "validation_macro_f1": valid.get("macro_f1"),
        }
    return results


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def fmt(x: Any) -> str:
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(x) for x in row) + " |")
    return "\n".join(lines)


def file_exists(path: str) -> str:
    return "Available" if (PROJECT_ROOT / path).exists() else "Missing"


def source_index_rows() -> List[List[str]]:
    paths = [
        "docs/task09_10_exact_reproduction_note.md",
        "reports/task11_split_summary.md",
        "docs/task12_model_experiment_design.md",
        "reports/task13/task13_bertweet_training_summary.md",
        "reports/task14/task14_roberta_training_summary.md",
        "reports/task15/task15_model_evaluation_summary.md",
        "reports/task16/task16_stopword_impact_summary.md",
        "reports/task17/task17_model_comparison_summary.md",
        "reports/task18/task18_error_analysis_summary.md",
        "reports/task19/task19_final_model_selection_summary.md",
        "reports/task19/final_model_card.md",
        "docs/task20_demo_interface.md",
        "docs/task20b_enhanced_interactive_dashboard.md",
        "docs/task21_ethics_and_limitations.md",
        "reports/task21/task21_risk_register.csv",
        "reports/task21/task21_responsible_use_checklist.md",
    ]
    return [[p, file_exists(p)] for p in paths]


def build_task_completion_table() -> str:
    rows = []
    for code, name, desc in TASKS:
        status = "Completed" if code != "22" else "Completed by this script"
        rows.append([code, name, status, desc])
    return markdown_table(["Task", "Activity", "Status", "Summary"], rows)


def build_results_table(results: Dict[str, Dict[str, Any]]) -> str:
    ordered = sorted(results.items(), key=lambda item: item[1]["macro_f1"], reverse=True)
    rows = []
    for rank, (exp, r) in enumerate(ordered, start=1):
        rows.append([
            rank,
            exp,
            r["model"],
            r["version"],
            r["preprocessing"],
            r["accuracy"],
            r["macro_f1"],
            r.get("macro_precision"),
            r.get("macro_recall"),
        ])
    return markdown_table(["Rank", "Experiment", "Model", "Version", "Preprocessing", "Accuracy", "Macro-F1", "Macro Precision", "Macro Recall"], rows)


def build_validation_table(results: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for exp in ["E01_BERTweet_VersionA", "E02_BERTweet_VersionB", "E03_RoBERTa_VersionA", "E04_RoBERTa_VersionB"]:
        r = results[exp]
        rows.append([exp, r["model"], r["version"], r.get("validation_accuracy"), r.get("validation_macro_f1")])
    return markdown_table(["Experiment", "Model", "Version", "Validation Accuracy", "Validation Macro-F1"], rows)


def build_split_table(split: Dict[str, Any]) -> str:
    # Handle known fallback or common JSON formats.
    rows = []
    for version in ["versionA", "versionB"]:
        data = split.get(version, {}) if isinstance(split, dict) else {}
        train = data.get("train_rows") or data.get("train") or 772069
        valid = data.get("valid_rows") or data.get("valid") or 96509
        test = data.get("test_rows") or data.get("test") or 96509
        rows.append([version, train, valid, test])
    return markdown_table(["Dataset Version", "Train Rows", "Validation Rows", "Test Rows"], rows)


def build_full_report() -> str:
    git = get_git_info()
    split = detect_split_summary()
    results = collect_results()
    best_exp = max(results.items(), key=lambda item: item[1]["macro_f1"])[0]
    best = results[best_exp]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    team_table = markdown_table(["Name", "Student ID", "Contribution / Role"], [list(x) for x in TEAM])
    task_table = build_task_completion_table()
    result_table = build_results_table(results)
    valid_table = build_validation_table(results)
    split_table = build_split_table(split)
    sources_table = markdown_table(["Source File", "Status"], source_index_rows())

    final_rows = 965087
    try:
        final_rows = int(split.get("final_aligned_rows") or split.get("aligned_rows") or split.get("total_rows") or 965087)
    except Exception:
        pass

    return f"""# Final Academic Documentation: Sarcasm Detection in Social Media Text

**Course:** WQF7007 Natural Language Processing  
**Project:** NLP Application: Sarcasm Detection in Social Media Text using BERTweet and RoBERTa  
**Dataset:** SARC / Self-Annotated Reddit Corpus  
**Generated:** {now}  
**Repository:** `{git.get('remote', '')}`  
**Branch:** `{git.get('branch', '')}`  
**Current Commit:** `{git.get('commit', '')}`

---

## 1. Executive Summary

This project developed an NLP system for **binary sarcasm detection** in social-media text. The system classifies Reddit-style text as either **sarcastic** or **non-sarcastic**. The final project uses the **SARC dataset**, applies two controlled preprocessing versions, compares **BERTweet** and **RoBERTa**, evaluates all experiments on a held-out test split, performs stopword-impact analysis, completes error analysis, selects the best model, and builds an interactive Streamlit demonstration dashboard.

The final selected model is **RoBERTa Version A**, where **stopwords are kept**. It was selected because it achieved the highest held-out test Macro-F1 among the four controlled experiments.

**Final selected model:** `{best_exp}`  
**Model family:** {best['model']}  
**Preprocessing:** {best['preprocessing']}  
**Held-out test accuracy:** {best['accuracy']:.4f}  
**Held-out test Macro-F1:** {best['macro_f1']:.4f}  
**Input strategy:** `parent_comment + comment` when context is available  
**Maximum sequence length:** 128 tokens

---

## 2. Team Members and Contributions

{team_table}

The work was organized using the course-recommended structure: project coordination, data engineering, modeling, evaluation, ethics, documentation, and presentation. Some members contributed to more than one area, especially modeling/evaluation and data engineering.

---

## 3. Problem Statement

Social media users often express sarcasm in short, informal, and context-dependent text. Sarcasm is difficult for NLP models because the literal meaning of a sentence may differ from the intended meaning. For example, a user may write something positive on the surface while actually expressing criticism or frustration. This makes sarcasm detection challenging for sentiment analysis, online conversation analysis, moderation-support systems, and social-media understanding.

The project problem is therefore:

> How can transformer-based NLP models detect sarcasm in Reddit-style social-media text, and how does preprocessing, especially stopword removal, affect model performance?

The project does **not** claim to understand user intent perfectly. It is an academic demonstration of sarcasm classification using supervised NLP methods.

---

## 4. Objectives

The project objectives were:

1. Prepare a cleaned version of the SARC dataset for binary sarcasm classification.
2. Create two preprocessing versions:
   - Version A: stopwords kept.
   - Version B: selective stopword removal while preserving negations.
3. Build a controlled experiment design comparing BERTweet and RoBERTa.
4. Train and evaluate all four model/preprocessing combinations.
5. Analyze whether stopword removal improves or reduces sarcasm-detection performance.
6. Select one final model based mainly on held-out test Macro-F1.
7. Perform error analysis to understand model weaknesses.
8. Build a user-facing demo interface and enhanced interactive dashboard.
9. Document ethical risks, limitations, and responsible-use boundaries.

---

## 5. Dataset Description

The project uses the **SARC / Self-Annotated Reddit Corpus**, a Reddit-based sarcasm dataset. The dataset is suitable for this project because Reddit comments often contain conversational context, informal phrasing, and sarcastic expressions. The label is binary:

- `0`: non-sarcastic
- `1`: sarcastic

The original proposal described the dataset as approximately **1.01 million rows**, with a near-balanced sarcastic/non-sarcastic distribution. During implementation, the exact preprocessing reproduction produced a final aligned cleaned dataset of **{final_rows:,} rows**.

The main text fields used were:

- `parent_comment`: contextual parent text
- `comment`: target comment/reply
- `label`: sarcasm label

The model input uses **text-only input**, formed as:

```text
parent_comment + comment
```

when parent context exists. This is important because sarcasm often depends on conversational context.

---

## 6. Methodology Overview

The project followed a controlled academic NLP workflow:

1. Data preparation and exact preprocessing reproduction.
2. Controlled preprocessing-version creation.
3. Stratified train/validation/test splitting.
4. Experiment design with fixed model/preprocessing combinations.
5. Transformer fine-tuning.
6. Full held-out test evaluation.
7. Stopword-impact analysis.
8. Model comparison and final selection.
9. Error analysis.
10. Demo interface and enhanced dashboard.
11. Ethics and limitations documentation.
12. Final report generation.

This structure ensures that the project is not only a coding exercise, but a reproducible academic NLP pipeline.

---

## 7. Task Completion Matrix

{task_table}

---

## 8. Detailed Activity Log and Implementation Experience

### 8.1 Project Scope Finalization

The final project scope became **sarcasm detection from social-media text using the SARC dataset**. The task was defined as binary classification: sarcastic vs non-sarcastic. The team decided to compare BERTweet and RoBERTa, and to evaluate whether stopword removal helps or hurts the model.

### 8.2 Preprocessing Version A: Stopwords Kept

Version A was the main preprocessing version. The team intentionally kept stopwords, punctuation, casing, and other linguistic signals because sarcasm may depend on small function words, negations, contrastive phrasing, and punctuation.

Preprocessing activities included:

- Removing null or empty comments.
- Removing URL rows and obvious link noise.
- Removing `[deleted]` and `[removed]` comments.
- Handling markdown/code-like noise.
- Removing duplicate rows.
- Handling spam-like or repetitive text rows.
- Normalizing whitespace.
- Applying a token-length threshold based on BERTweet tokenization.
- Saving the final Version A file as `data/processed/A.csv`.

A key implementation experience was that the team first had a non-exact preprocessing reproduction that produced a different row count. This was corrected by reproducing the teammate's notebook logic exactly. The final correct aligned row count became **{final_rows:,} rows**.

### 8.3 Preprocessing Version B: Selective Stopword Removal

Version B was created from Version A by removing stopwords selectively. Negation words were preserved because words such as `not`, `no`, `never`, `don't`, and similar forms can change meaning and are especially important in sarcasm and sentiment-related tasks.

Version B was used as an ablation experiment to answer the lecturer's question about whether stopword removal improves performance.

### 8.4 Train/Validation/Test Split

Task 11 created a stratified split so that both classes remained balanced across train, validation, and test sets. The same split was used for Version A and Version B, ensuring fair comparison.

{split_table}

The split strategy was:

- Train: 80%
- Validation: 10%
- Test: 10%
- Random seed: 42
- Same row alignment across Version A and Version B

The use of the same split for all experiments is important because otherwise model performance could be affected by different data distributions.

### 8.5 Experiment Design

Task 12 defined four controlled experiments:

| Experiment | Model | Dataset Version | Purpose |
|---|---|---|---|
| E01 | BERTweet | Version A | BERTweet with stopwords kept |
| E02 | BERTweet | Version B | BERTweet with selective stopword removal |
| E03 | RoBERTa | Version A | RoBERTa with stopwords kept |
| E04 | RoBERTa | Version B | RoBERTa with selective stopword removal |

All experiments used:

- Text-only input.
- The same split strategy.
- Max length 128.
- Comparable metrics.
- Accuracy as supporting metric.
- Macro-F1 as the main ranking metric.

### 8.6 Model Training Experience

The team fine-tuned two transformer models:

- **BERTweet**, selected because it is trained for social-media/Twitter-style text.
- **RoBERTa**, selected as a strong general transformer baseline for text classification.

Due to local compute constraints, training used a practical sample of **50,000 training rows and 10,000 validation rows per version**, while final evaluation used the full held-out test split. This is important to state transparently in the academic report.

Validation results during training were:

{valid_table}

An important observation was that BERTweet looked strong on validation, but later underperformed on full held-out test evaluation. This difference became an important learning point: validation performance alone should not be treated as the final decision.

### 8.7 Model Evaluation

Task 15 evaluated all four experiments on the held-out test set. Evaluation metrics included:

- Accuracy
- Macro precision
- Macro recall
- Macro-F1
- Weighted-F1
- Per-class classification report
- Confusion matrix

Final held-out test ranking:

{result_table}

The strongest experiment was **RoBERTa Version A**.

### 8.8 Stopword Impact Analysis

Task 16 answered the preprocessing question directly. Keeping stopwords performed better than selective stopword removal for both BERTweet and RoBERTa.

Main finding:

> Stopword removal reduced performance for both model families. Therefore, keeping stopwords is more suitable for this sarcasm-detection task.

This makes sense because sarcasm often depends on function words, word order, contrast, and context. Removing too many words can weaken the information needed to detect sarcastic intent.

### 8.9 Model Comparison

Task 17 ranked the four experiments mainly by held-out test Macro-F1. Accuracy was used as a supporting metric, but Macro-F1 was prioritized because it gives a more balanced view of class-level performance.

The final ranking showed:

1. RoBERTa Version A
2. RoBERTa Version B
3. BERTweet Version A
4. BERTweet Version B

This result means that RoBERTa generalized better than BERTweet in the final test evaluation, even though BERTweet had looked stronger earlier on validation.

### 8.10 Error Analysis

Task 18 analyzed the selected best model, RoBERTa Version A. The analysis considered:

- False positives: non-sarcastic comments predicted as sarcastic.
- False negatives: sarcastic comments predicted as non-sarcastic.
- Confidence patterns.
- Text-length error behavior.
- Surface-feature error rates.

Raw text examples were stored locally only and ignored by Git to avoid publishing raw Reddit examples unnecessarily.

The major lesson from error analysis is that sarcasm remains difficult because it often depends on missing context, cultural knowledge, tone, conversation history, and implicit meaning.

### 8.11 Final Model Selection

Task 19 selected **RoBERTa Version A** as the final model.

Selection justification:

- It achieved the highest held-out test Macro-F1.
- It achieved the highest accuracy among the four experiments.
- Version A preprocessing kept context-rich linguistic signals.
- Stopword removal was shown to reduce performance.
- It was supported by model comparison and error analysis.

Final model details:

| Item | Value |
|---|---|
| Selected experiment | E03_RoBERTa_VersionA |
| Model family | RoBERTa |
| Preprocessing | Version A, stopwords kept |
| Input | parent_comment + comment |
| Max length | 128 |
| Test accuracy | {best['accuracy']:.4f} |
| Test Macro-F1 | {best['macro_f1']:.4f} |
| Local checkpoint | `models/roberta/versionA` |

### 8.12 Demo Interface

Task 20 created a Streamlit demo interface. The app allows the user to enter:

- optional parent comment/context
- main comment to classify

The app returns:

- predicted class: sarcastic or non-sarcastic
- confidence score
- class probabilities
- combined model input
- technical details
- ethical warning

The demo uses the local model checkpoint at:

```text
models/roberta/versionA
```

A smoke test was completed successfully.

### 8.13 Enhanced Interactive Dashboard

Task 20B improved the UI into an interactive NLP workflow dashboard. The dashboard includes:

- Project overview.
- Workflow progress from preprocessing to demo.
- Pipeline runner for local scripts.
- Live model demo.
- Results dashboard.
- Reports explorer.
- Error analysis viewer.
- Hosting readiness section.

This dashboard was designed to show the full AI/NLP process, not only the final prediction.

### 8.14 Ethics and Limitations

Task 21 produced a responsible-use analysis. Main risks included:

- Misclassification risk.
- Context loss.
- Reddit dataset bias.
- Privacy concerns.
- Over-reliance on confidence scores.
- Deployment misuse.
- Compute and reproducibility limitations.
- Cultural and linguistic limitations.

The final system should **not** be used to automatically remove, flag, penalize, or judge user content. It is an academic demonstration and should be interpreted cautiously.

---

## 9. Technical Architecture

The final system has the following structure:

```text
SARC Dataset
  -> Version A preprocessing: stopwords kept
  -> Version B preprocessing: selective stopword removal
  -> stratified train/validation/test split
  -> BERTweet and RoBERTa fine-tuning
  -> full test evaluation
  -> stopword impact analysis
  -> model comparison
  -> final model selection
  -> Streamlit demo and enhanced dashboard
```

Repository-level structure:

```text
configs/        experiment and final model configuration files
docs/           methodology and task documentation
src/            preprocessing, training, evaluation, reporting, prediction utilities
app/            Streamlit simple demo and enhanced dashboard
reports/        generated reports, metrics, figures, summaries
models/         local-only model checkpoints, ignored by Git
data/           local-only raw/processed/split data, ignored by Git
```

---

## 10. Reproducibility Notes

The repository was organized to keep source code and reports in GitHub, while large local files remain ignored:

- raw dataset CSV files
- processed A/B CSV files
- split CSV files
- local model checkpoints
- raw Reddit error samples

This is good practice because GitHub is not ideal for large model/data artifacts. For future hosting, the final model should be uploaded to a Hugging Face model repository, and the app should load the model from there.

---

## 11. Results Discussion

The strongest final result was achieved by RoBERTa Version A. The key pattern was that **keeping stopwords improved performance** for both model families. This suggests that sarcasm detection benefits from retaining sentence structure and small words that contribute to tone, contrast, and meaning.

The BERTweet result was unexpected. Although BERTweet is designed for social-media text, it underperformed on full test evaluation compared with RoBERTa. Possible reasons include:

- SARC is Reddit-based rather than Twitter-based.
- Reddit comments may differ in style from tweets.
- Training sample size and local compute limits may affect generalization.
- RoBERTa's general language representations may better capture Reddit comment structure.

---

## 12. Challenges and Lessons Learned

### 12.1 Exact Reproduction Matters

The team initially saw a mismatch between non-exact preprocessing and the teammate's notebook output. This was corrected by reproducing the notebook logic exactly. The lesson is that preprocessing details can change dataset size and therefore affect all downstream results.

### 12.2 GitHub Should Not Store Large Data and Model Files

The project correctly kept large datasets and checkpoints local. This prevented accidental upload of large files and made the repository cleaner.

### 12.3 Validation and Test Results Can Differ

BERTweet had strong validation results but much weaker full test results. This showed why final selection should be based on held-out test evaluation rather than validation performance alone.

### 12.4 Stopword Removal Is Not Always Good

The experiment showed that removing stopwords reduced performance. This directly supports the course concept that preprocessing should be task-dependent.

### 12.5 UI Adds Communication Value

The enhanced dashboard makes the project easier to explain because users can see not only the prediction result but also the full NLP pipeline, reports, metrics, error analysis, and ethics documentation.

---

## 13. Ethical Considerations

Sarcasm detection can be sensitive because sarcastic language depends on context, culture, speaker intent, and social norms. A model may misclassify sincere comments as sarcastic or sarcastic comments as sincere. Therefore:

- The system should be used only as a research or educational demo.
- The model should not automatically moderate users.
- Confidence scores should not be interpreted as certainty about intent.
- Raw text examples should be handled responsibly.
- The limitations of Reddit-based data should be clearly disclosed.

---

## 14. Final Demo and Dashboard

Two user interfaces are available:

| Interface | File | Purpose |
|---|---|---|
| Simple prediction demo | `app/streamlit_app.py` | Clean single-purpose sarcasm prediction demo |
| Enhanced dashboard | `app/enhanced_dashboard.py` | Full workflow dashboard with progress, reports, results, and demo |

Run the enhanced dashboard locally:

```bash
bash run_enhanced_dashboard.sh
```

Run the simple demo locally:

```bash
bash run_task20_demo.sh
```

---

## 15. Hosting Recommendation

For future public deployment, the recommended architecture is:

```text
GitHub repository
  -> source code, reports, scripts, documentation

Hugging Face Model Hub
  -> final RoBERTa Version A model checkpoint

Hugging Face Space
  -> Streamlit app or Docker-based Streamlit app
  -> loads final model from Hugging Face Model Hub
```

The hosted public version should mainly support inference and report viewing. It should not allow public users to trigger heavy training tasks because training is resource-intensive and outputs may not persist reliably on free hosting.

---

## 16. Conclusion

This project successfully completed an end-to-end academic NLP workflow for sarcasm detection. The team prepared the dataset, designed controlled experiments, trained transformer models, evaluated all models, analyzed preprocessing impact, selected a final model, created a demo interface, built an enhanced dashboard, and documented ethical limitations.

The final conclusion is:

> RoBERTa with Version A preprocessing, where stopwords are kept, is the best-performing model in this project. Stopword removal reduced performance, indicating that sarcasm detection benefits from preserving context and sentence structure.

---

## 17. References

- SARC / Self-Annotated Reddit Corpus for sarcasm research.
- RoBERTa: Robustly Optimized BERT Pretraining Approach.
- BERTweet: A pretrained language model for English Tweets.
- Hugging Face Transformers documentation.
- Streamlit documentation.
- Scikit-learn evaluation metrics documentation.
- WQF7007 NLP course materials and project guidelines.

---

## 18. Source Files Used for This Documentation

{sources_table}

---

## 19. Latest Git Log Snapshot

```text
{git.get('latest_log', '')}
```
"""


def build_technical_log() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# Technical Implementation Log

Generated: {now}

This document records the implementation experience across the project tasks.

## Key Implementation Events

1. The project scope was finalized as sarcasm detection using SARC.
2. Version A and Version B preprocessing were defined to test the impact of stopword handling.
3. A non-exact preprocessing script initially produced a different row count; it was replaced by an exact reproduction of the teammate's notebook workflow.
4. Final exact preprocessing produced 965,087 aligned rows.
5. Task 11 created identical stratified splits for Version A and Version B.
6. Task 12 created a four-experiment matrix.
7. Task 13 initially had label-column loading issues, which were fixed by rewriting robust label handling in the BERTweet training script.
8. Task 14 trained RoBERTa under the same controlled setup.
9. Task 15 initially had an evaluation function parameter bug and was fixed by rewriting the model evaluation script.
10. Task 16 showed that stopword removal reduced performance.
11. Task 17 selected RoBERTa Version A as the best model by test Macro-F1.
12. Task 18 analyzed the selected model's errors while keeping raw text samples local only.
13. Task 19 formalized final model selection and created a model card.
14. Task 20 created the Streamlit demo.
15. Task 20B created the enhanced workflow dashboard.
16. Task 21 documented ethics and limitations.
17. Task 22 generated final academic documentation.

## Important Practical Lessons

- Reproducibility requires exact preprocessing, not approximate preprocessing.
- Large files should remain local or be stored in a model/data platform rather than GitHub.
- Validation performance may not match held-out test performance.
- Stopword removal should be treated as an experiment, not a default rule.
- UI should communicate the full workflow, not only the final prediction.
- Responsible-use warnings are essential for language understanding tasks.
"""


def build_sources_index() -> str:
    rows = source_index_rows()
    return "# Final Report Source Index\n\n" + markdown_table(["Source File", "Status"], rows) + "\n"


def build_task22_doc() -> str:
    return """# Task 22: Final Report Writing

## Purpose

Task 22 consolidates all completed project activities into final academic documentation. The generated documentation is intended to support the final report, final presentation, project demonstration, and Q&A preparation.

## Outputs

Task 22 generates:

- `reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md`
- `reports/final_report/technical_implementation_log.md`
- `reports/final_report/final_report_sources_index.md`
- `reports/final_report/task_completion_matrix.md`
- `reports/final_report/presentation_key_points.md`
- `reports/task22/task22_progress_note.txt`

## Documentation Coverage

The documentation covers:

1. Project background and problem statement.
2. Dataset and preprocessing.
3. Experiment design.
4. Model training.
5. Evaluation results.
6. Stopword impact analysis.
7. Model comparison.
8. Error analysis.
9. Final model selection.
10. Demo interface and enhanced dashboard.
11. Ethics and limitations.
12. Implementation challenges and lessons learned.

## Note

This task does not retrain models and does not modify datasets. It only generates documentation from the completed task outputs.
"""


def build_presentation_key_points(results: Dict[str, Dict[str, Any]]) -> str:
    best_exp = max(results.items(), key=lambda item: item[1]["macro_f1"])[0]
    best = results[best_exp]
    return f"""# Presentation Key Points

## One-Sentence Project Summary

We built an NLP sarcasm detection system using the SARC Reddit dataset, compared BERTweet and RoBERTa under two preprocessing settings, and selected RoBERTa Version A as the final model.

## Best Result

- Final model: {best_exp}
- Accuracy: {best['accuracy']:.4f}
- Macro-F1: {best['macro_f1']:.4f}
- Preprocessing: {best['preprocessing']}

## What We Learned

1. Keeping stopwords worked better than removing them.
2. RoBERTa generalized better than BERTweet on the held-out test set.
3. Sarcasm detection is highly context-dependent.
4. Error analysis showed that misclassification can occur even with high confidence.
5. The demo should be used for academic purposes only, not automatic moderation.

## Expected Q&A

### Why did we keep stopwords?
Because Task 16 showed that stopword removal reduced performance for both models. Sarcasm often depends on sentence structure, contrast, and small function words.

### Why did we choose RoBERTa instead of BERTweet?
RoBERTa Version A achieved the best held-out test Macro-F1 and accuracy among the four controlled experiments.

### Why use parent_comment + comment?
Sarcasm often requires conversational context. Combining parent comment and reply gives the model more context than the reply alone.

### Can the model be used for moderation?
No. The model is a research demo and should not automatically remove, flag, or penalize content.

### What is the biggest limitation?
Sarcasm depends on culture, tone, context, speaker intent, and conversation history, which may be missing from text-only input.
"""


def build_task_completion_matrix_file() -> str:
    return "# Task Completion Matrix\n\n" + build_task_completion_table() + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TASK22_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    results = collect_results()
    final_report = build_full_report()

    outputs = {
        REPORT_DIR / "UM_WQF7007_Sarcasm_Detection_Final_Report.md": final_report,
        REPORT_DIR / "technical_implementation_log.md": build_technical_log(),
        REPORT_DIR / "final_report_sources_index.md": build_sources_index(),
        REPORT_DIR / "task_completion_matrix.md": build_task_completion_matrix_file(),
        REPORT_DIR / "presentation_key_points.md": build_presentation_key_points(results),
        DOCS_DIR / "task22_final_report_writing.md": build_task22_doc(),
        TASK22_DIR / "README.md": "# Task 22 Reports\n\nThis folder stores progress notes and metadata for final report writing.\n",
        TASK22_DIR / "task22_progress_note.txt": (
            "Final report documentation completed. A structured academic final report was generated covering the full NLP workflow: "
            "project scope, dataset, preprocessing Version A and Version B, stratified splitting, experiment design, BERTweet and RoBERTa training, "
            "full test evaluation, stopword impact analysis, model comparison, error analysis, final model selection, demo interface, enhanced dashboard, "
            "ethics and limitations, implementation challenges, and lessons learned. Documentation files were saved under reports/final_report and docs/task22_final_report_writing.md."
        ),
    }

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {rel(path)}")

    # Machine-readable summary
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "final_model": "E03_RoBERTa_VersionA",
        "final_model_accuracy": FALLBACK_TEST_RESULTS["E03_RoBERTa_VersionA"]["accuracy"],
        "final_model_macro_f1": FALLBACK_TEST_RESULTS["E03_RoBERTa_VersionA"]["macro_f1"],
        "tasks_documented": [code for code, _, _ in TASKS],
        "outputs": [rel(p) for p in outputs],
    }
    (TASK22_DIR / "task22_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {rel(TASK22_DIR / 'task22_summary.json')}")

    # CSV task matrix for easy spreadsheet/report use.
    csv_path = REPORT_DIR / "task_completion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Activity", "Status", "Summary"])
        for code, name, desc in TASKS:
            writer.writerow([code, name, "Completed" if code != "22" else "Completed by this script", desc])
    print(f"Wrote {rel(csv_path)}")

    print("Done. Task 22 final documentation generated successfully.")


if __name__ == "__main__":
    main()
