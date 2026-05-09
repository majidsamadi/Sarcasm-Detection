#!/usr/bin/env python3
"""Generate final academic documentation and GitHub README for the sarcasm detection project.

This Task 22 generator intentionally excludes the team member / role section because
that part will be completed separately by the group leader.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "final_report"
TASK22_DIR = PROJECT_ROOT / "reports" / "task22"
DOCS_DIR = PROJECT_ROOT / "docs"


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def read_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {path.relative_to(PROJECT_ROOT)}")


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join([h, sep] + body)


FINAL_METRICS = [
    ["1", "E03", "RoBERTa", "Version A", "Stopwords kept", "0.7223", "0.7167", "Selected final model"],
    ["2", "E04", "RoBERTa", "Version B", "Selective stopword removal", "0.6773", "0.6648", "Second best"],
    ["3", "E01", "BERTweet", "Version A", "Stopwords kept", "0.5092", "0.3632", "Lower test generalization"],
    ["4", "E02", "BERTweet", "Version B", "Selective stopword removal", "0.5018", "0.3452", "Lowest ranked"],
]

TASK_MATRIX = [
    ["9", "Preprocessing Version A", "Completed", "Created `A.csv` using exact reproduced preprocessing. Stopwords, punctuation, casing, and ALL CAPS were kept."],
    ["10", "Preprocessing Version B", "Completed", "Created `B.csv` from Version A using selective stopword removal while preserving negations."],
    ["11", "Train/Validation/Test Split", "Completed", "Created stratified 80/10/10 splits for Version A and Version B using the same indices."],
    ["12", "Model Experiment Design", "Completed", "Defined four controlled experiments: BERTweet A/B and RoBERTa A/B."],
    ["13", "Train BERTweet", "Completed", "Fine-tuned BERTweet on Version A and Version B with the same training setup."],
    ["14", "Train RoBERTa", "Completed", "Fine-tuned RoBERTa on Version A and Version B with the same training setup."],
    ["15", "Model Evaluation", "Completed", "Evaluated all four models on the full held-out test set and generated metrics/confusion matrices."],
    ["16", "Stopword Impact Analysis", "Completed", "Compared Version A vs Version B and found that keeping stopwords performed better for both models."],
    ["17", "Model Comparison", "Completed", "Ranked all four experiments using test macro-F1 as the main metric."],
    ["18", "Error Analysis", "Completed", "Analyzed false positives, false negatives, confidence patterns, text length, and surface features for the selected model."],
    ["19", "Final Model Selection", "Completed", "Selected RoBERTa Version A as the final model based on test macro-F1."],
    ["20", "Demo Interface", "Completed", "Built a local Streamlit prediction demo."],
    ["20B", "Enhanced Dashboard", "Completed", "Built an interactive dashboard showing the full AI/NLP workflow, reports, progress, and model demo."],
    ["21", "Ethics and Limitations", "Completed", "Generated risk register, responsible-use checklist, and ethics/limitations summary."],
    ["22", "Final Documentation", "Completed", "Generated final report package and full GitHub README."],
]


def build_final_report() -> str:
    report = f"""
# Final Report: Sarcasm Detection in Social Media Text using RoBERTa and BERTweet

**Course:** WQF7007 Natural Language Processing  
**Project theme:** NLP application for sarcasm detection in social media text  
**Dataset:** SARC / Self-Annotated Reddit Corpus  
**Final selected model:** RoBERTa Version A  
**Final test accuracy:** 0.7223  
**Final test macro-F1:** 0.7167  
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

This project developed a sarcasm detection system for social media text using the SARC dataset. The task was framed as a binary classification problem: predicting whether a Reddit comment is sarcastic or non-sarcastic. The project followed a complete academic NLP workflow: preprocessing, dataset splitting, experiment design, transformer model training, full test evaluation, stopword impact analysis, model comparison, error analysis, final model selection, demonstration interface creation, enhanced dashboard development, and ethics/limitations documentation.

The final selected model is **RoBERTa Version A**, trained using the preprocessing version where stopwords are kept. It achieved the highest held-out test performance among the four controlled experiments, with **accuracy = 0.7223** and **macro-F1 = 0.7167**. The project also found that removing stopwords reduced performance for both BERTweet and RoBERTa, which supports the linguistic intuition that sarcasm often depends on small function words, contrastive structure, and context.

---

## 2. Problem Statement

Sarcasm is difficult for NLP systems because the literal meaning of a sentence may differ from the intended meaning. Social media comments are especially challenging because they are short, informal, context-dependent, and often rely on hidden assumptions, prior conversation, punctuation, or contrast between a parent comment and a reply.

The main problem addressed in this project is:

> How effectively can transformer-based NLP models detect sarcasm in Reddit-style social media text, and how does preprocessing, especially stopword removal, affect performance?

---

## 3. Project Objectives

The project objectives were:

1. Build a sarcasm detection system that classifies social media text as sarcastic or non-sarcastic.
2. Use the SARC dataset as the main data source.
3. Reproduce the preprocessing workflow into the project repository so the dataset can be regenerated locally.
4. Compare two transformer-based models: BERTweet and RoBERTa.
5. Test two preprocessing versions: stopwords kept and selective stopword removal.
6. Evaluate all experiments using accuracy, precision, recall, macro-F1, weighted-F1, classification reports, and confusion matrices.
7. Conduct stopword impact analysis, model comparison, and error analysis.
8. Select one final model for implementation.
9. Build a Streamlit demo and an enhanced interactive NLP dashboard.
10. Document ethical risks, limitations, and responsible-use guidance.

---

## 4. Dataset Description

The project uses the **SARC dataset**, also known as the Self-Annotated Reddit Corpus. The dataset contains Reddit comments labeled as sarcastic or non-sarcastic. The original dataset used in this project contained approximately **1,010,798+ rows** and was nearly balanced between sarcastic and non-sarcastic labels.

The selected text input uses two fields:

- `parent_comment`: the previous/context comment
- `comment`: the reply/comment to classify

The final model input is constructed as:

```text
parent_comment + comment
```

This choice is important because sarcasm often depends on context. For example, a reply may sound normal in isolation but sarcastic when read after the parent comment.

---

## 5. Preprocessing Workflow

Two preprocessing versions were created.

### 5.1 Version A: Stopwords Kept

Version A was designed to preserve sarcasm-related linguistic signals. The preprocessing workflow handled:

- null and empty comments
- URL rows
- `[deleted]` and `[removed]` comments
- markdown/code noise
- duplicate rows
- spam-like or repetitive rows
- extra whitespace
- very long rows above the configured BERTweet token threshold

Version A intentionally kept:

- stopwords
- punctuation
- casing
- ALL CAPS
- negation words

This was important because sarcasm may depend on punctuation, contrastive wording, negation, sentence structure, and small function words.

### 5.2 Version B: Selective Stopword Removal

Version B was created from Version A by applying selective stopword removal. The preprocessing removed common stopwords but preserved negation terms such as:

- `not`
- `no`
- `never`
- `don't`
- `can't`
- similar negation expressions

The purpose of Version B was to test the lecturer's question: whether removing stopwords improves or reduces model performance.

### 5.3 Important Experience: Exact Reproduction Correction

During the implementation, an earlier clean project version of preprocessing produced a slightly different row count. This was identified as a methodological risk because Tasks 9 and 10 were supposed to reproduce the teammate's preprocessing exactly, not redesign it.

The project was corrected by:

1. Reproducing the teammate's notebook logic exactly.
2. Fixing only a local pandas compatibility issue where null handling caused a local execution error.
3. Regenerating `A.csv` and `B.csv` with the expected final aligned row count of **965,087 rows**.
4. Removing the earlier non-exact preprocessing files to avoid confusion.

This correction improved the reproducibility and academic integrity of the project.

---

## 6. Train/Validation/Test Split

Task 11 created a stratified 80/10/10 split for both Version A and Version B.

Final split sizes:

| Split | Rows |
|---|---:|
| Train | 772,069 |
| Validation | 96,509 |
| Test | 96,509 |

The same split indices were used for both preprocessing versions. This was necessary for fair comparison because Version A and Version B needed to differ only in preprocessing, not in which records appeared in each split.

---

## 7. Experiment Design

Task 12 defined four controlled experiments:

{md_table(["Experiment", "Model", "Preprocessing Version", "Stopword Setting"], [
    ["E01", "BERTweet", "Version A", "Stopwords kept"],
    ["E02", "BERTweet", "Version B", "Selective stopword removal"],
    ["E03", "RoBERTa", "Version A", "Stopwords kept"],
    ["E04", "RoBERTa", "Version B", "Selective stopword removal"],
])}

All experiments used:

- text-only input
- `parent_comment + comment`
- max sequence length of 128
- the same train/validation/test split
- the same evaluation metrics
- macro-F1 as the main selection metric

---

## 8. Model Training

### 8.1 BERTweet Training

Task 13 trained BERTweet on Version A and Version B. BERTweet was included because it is designed for social-media-style language. During implementation, a label-column handling issue occurred, where the training script expected a hardcoded column name. This was fixed by rewriting the label loading logic to robustly detect the correct label column.

### 8.2 RoBERTa Training

Task 14 trained RoBERTa on Version A and Version B. RoBERTa was included as a strong general-purpose transformer baseline. The training process saved local model checkpoints under `models/roberta/`, while code and reports were pushed to GitHub.

### 8.3 Training Practicality

For practical local computation, model training used a controlled subset for training and validation in the practical run:

- 50,000 training rows per version
- 10,000 validation rows per version

This was transparently documented. The final evaluation, however, was performed on the full held-out test set.

---

## 9. Model Evaluation

Task 15 evaluated all four trained models on the full held-out test set. The evaluation generated:

- accuracy
- macro precision
- macro recall
- macro-F1
- weighted-F1
- per-class classification reports
- confusion matrices
- comparison summaries

During implementation, an evaluation function parameter mismatch occurred (`max_test_samples` issue). This was fixed by rewriting the evaluation script and rerunning full evaluation successfully.

---

## 10. Final Evaluation Results

{md_table(["Rank", "Experiment", "Model", "Version", "Stopword Setting", "Accuracy", "Macro-F1", "Decision"], FINAL_METRICS)}

The best-performing experiment was **E03 RoBERTa Version A**, with:

- Accuracy: **0.7223**
- Macro-F1: **0.7167**

---

## 11. Stopword Impact Analysis

Task 16 compared Version A and Version B for both models.

Results showed that keeping stopwords performed better:

| Model | Stopwords Kept Macro-F1 | Stopwords Removed Macro-F1 | Conclusion |
|---|---:|---:|---|
| BERTweet | 0.3632 | 0.3452 | Keeping stopwords helped |
| RoBERTa | 0.7167 | 0.6648 | Keeping stopwords helped |

This result is academically meaningful. Sarcasm often depends on sentence structure, contrast, negation, and subtle function words. Removing stopwords may remove useful context that the transformer could otherwise use.

---

## 12. Model Comparison

Task 17 ranked all experiments using macro-F1 as the main criterion. RoBERTa Version A ranked first. This result also revealed a key lesson: validation performance alone can be misleading. Earlier, BERTweet appeared stronger during validation, but the full held-out test evaluation showed RoBERTa Version A generalized better.

The final model comparison demonstrated the value of:

- using a held-out test set
- using macro-F1 rather than accuracy only
- keeping preprocessing conditions controlled
- comparing models fairly under the same split

---

## 13. Error Analysis

Task 18 performed error analysis for the selected best model: RoBERTa Version A. The analysis examined:

- false positives
- false negatives
- confidence patterns
- text-length behavior
- surface features such as punctuation and casing
- high-confidence errors

Raw error examples containing Reddit text were saved locally only and ignored by Git. Aggregate reports and figures were committed to GitHub.

The main insight is that sarcasm remains difficult when:

- the comment lacks enough conversational context
- sarcasm is implicit
- humor depends on cultural knowledge
- the model over-relies on surface wording
- the text is short or ambiguous

---

## 14. Final Model Selection

Task 19 selected **RoBERTa Version A** as the final model because it had the best test macro-F1.

Final model configuration:

| Item | Value |
|---|---|
| Selected model | RoBERTa Version A |
| Experiment ID | E03 |
| Preprocessing | Version A |
| Stopwords | Kept |
| Input | `parent_comment + comment` |
| Max length | 128 |
| Test accuracy | 0.7223 |
| Test macro-F1 | 0.7167 |
| Local checkpoint | `models/roberta/versionA` |

---

## 15. Demo Interface

Task 20 created a local Streamlit demo. The app allows a user to enter:

- optional parent comment/context
- main comment to classify

The app outputs:

- predicted class: sarcastic or non-sarcastic
- confidence score
- class probabilities
- ethical warning

The local demo loads the final checkpoint from:

```text
models/roberta/versionA
```

A smoke test was completed successfully.

---

## 16. Enhanced Interactive Dashboard

Task 20B extended the UI into a full interactive NLP dashboard. The enhanced dashboard includes:

- workflow overview
- task progress tracking
- pipeline runner controls
- final model prediction demo
- results dashboard
- reports explorer
- error analysis section
- hosting readiness section

This dashboard is useful for presentation because it shows the full AI/NLP lifecycle rather than only a prediction box.

Important design decision:

- Local dashboard can trigger project scripts.
- Public hosted version should not trigger heavy training jobs.

---

## 17. Ethics and Limitations

Task 21 documented responsible-use concerns. Key risks include:

1. **Misclassification risk**: sarcastic and non-sarcastic comments may be confused.
2. **Context limitation**: sarcasm often requires broader conversation history.
3. **Dataset bias**: Reddit language may not represent all users or platforms.
4. **Privacy concerns**: social media data can contain sensitive expressions.
5. **Over-reliance on confidence**: high model confidence does not guarantee correctness.
6. **Deployment misuse**: the model should not be used to automatically punish, remove, or flag user content.
7. **Compute limitation**: full-scale transformer training requires significant resources.
8. **Cultural and linguistic limitation**: sarcasm differs across cultures, communities, and languages.

The responsible-use position is:

> This model is suitable for coursework demonstration and research analysis, but not for automated moderation or punitive decision-making.

---

## 18. Technical Challenges Experienced

Several important implementation challenges were encountered and resolved:

### 18.1 Exact Preprocessing Reproduction

The first preprocessing pipeline produced a different dataset size. The team identified that this was not aligned with the requirement to reproduce the teammate's exact Task 9 and Task 10 workflow. The pipeline was corrected and regenerated.

### 18.2 Local Pandas Compatibility

A local pandas issue occurred when null comment values were processed before being dropped. The fix preserved the final logic while avoiding local execution failure.

### 18.3 Label Column Handling

The BERTweet training script initially expected a hardcoded label column. This caused a `KeyError`. The script was rewritten to handle label columns robustly.

### 18.4 Evaluation Function Bug

The evaluation script initially had a mismatch between function parameters and function call. This was fixed and Task 15 was rerun successfully.

### 18.5 Validation vs Test Difference

BERTweet appeared promising during validation, but full test evaluation showed RoBERTa Version A was stronger. This reinforced the importance of held-out test evaluation.

### 18.6 Local vs Hosted Model Storage

The model checkpoint is local-only and ignored by Git because it is large. For Hugging Face hosting, the checkpoint should be uploaded to a Hugging Face Model Hub repository and then loaded by the app.

---

## 19. What We Learned

The project provided several lessons:

1. Sarcasm detection is more than sentiment classification; it requires context and subtle language understanding.
2. Preprocessing must be task-aware. Removing stopwords can hurt performance when function words carry meaning.
3. Fair comparison requires the same dataset split and the same evaluation metrics.
4. Validation results are useful but not final; held-out test evaluation is essential.
5. Transformer models still make errors on implicit, cultural, or context-heavy sarcasm.
6. Reproducibility requires exact scripts, fixed random seeds, documented assumptions, and clear separation of local artifacts from Git-tracked code.
7. UI/UX can strengthen an academic project by making the NLP pipeline explainable and interactive.
8. Ethics and limitations are not optional; they are part of responsible NLP development.

---

## 20. Repository and Reproducibility

The project repository contains:

- source code in `src/`
- Streamlit apps in `app/`
- task documentation in `docs/`
- report outputs in `reports/`
- configs in `configs/`
- runner scripts in the root folder

Large local artifacts are intentionally ignored by Git:

- raw dataset CSV files
- processed dataset CSV files
- split CSV files
- local model checkpoints
- raw Reddit error examples

This keeps the GitHub repository clean while preserving reproducibility through scripts.

---

## 21. Hugging Face Hosting Recommendation

For hosting, the recommended architecture is:

| Component | Recommendation |
|---|---|
| Source code | GitHub repository |
| Model checkpoint | Hugging Face Model Hub |
| Hosted demo | Hugging Face Spaces |
| UI framework | Streamlit or Docker-based Streamlit Space |
| Public behavior | Inference and reports only |
| Heavy training | Local only, not public hosted execution |

The public hosted version should avoid triggering heavy training jobs. It should focus on:

- final model inference
- project explanation
- reports
- model comparison
- limitations and ethics

---

## 22. Conclusion

This project successfully implemented a complete NLP workflow for sarcasm detection using the SARC dataset. The workflow included preprocessing, controlled experiment design, transformer training, evaluation, stopword analysis, model comparison, error analysis, final model selection, demo development, enhanced dashboard creation, and ethics documentation.

The final selected model is **RoBERTa Version A**, which achieved the best held-out test macro-F1. The project also found that keeping stopwords improved model performance, supporting the idea that sarcasm depends heavily on context and sentence structure.

The final system is suitable for academic demonstration and research exploration, but it should not be used for automated moderation or punitive decisions.
"""
    return report


def build_technical_log() -> str:
    return f"""
# Technical Implementation Log

This document records what was implemented during the sarcasm detection project and the major technical experiences encountered.

## Completed Workflow

{md_table(["Task", "Activity", "Status", "Implementation Note"], TASK_MATRIX)}

## Key Implementation Decisions

| Decision | Reason |
|---|---|
| Use `parent_comment + comment` | Sarcasm often depends on conversational context. |
| Keep punctuation and casing | Punctuation and ALL CAPS can signal sarcasm. |
| Compare stopwords kept vs removed | Lecturer asked whether stopword removal improves performance. |
| Use macro-F1 as main metric | Macro-F1 balances performance across classes better than accuracy alone. |
| Select only one final model | The project compares multiple models but implements the best one for the final demo. |
| Keep data/model files local | Large artifacts should not be pushed to GitHub. |
| Use Streamlit for UI | Streamlit is fast, explainable, and suitable for coursework demos. |
| Use enhanced dashboard locally | Full pipeline triggering is appropriate locally but not for public hosting. |

## Issues Encountered and Fixes

| Issue | Impact | Fix |
|---|---|---|
| Non-exact preprocessing generated different row count | Risked mismatch with teammate work | Reproduced teammate workflow exactly and removed non-exact files. |
| Local pandas null handling error | Exact preprocessing failed locally | Adjusted null handling without changing final logic. |
| Hardcoded label column in training | BERTweet training failed | Rewrote loader with robust label detection. |
| Evaluation parameter mismatch | Task 15 failed | Rewrote evaluation script and reran full test evaluation. |
| Local checkpoint not in GitHub | Hosted demo cannot load local model | Recommend Hugging Face Model Hub for deployment. |

## Final Model

- Experiment: E03
- Model: RoBERTa Version A
- Stopwords: kept
- Test accuracy: 0.7223
- Test macro-F1: 0.7167
- Local checkpoint: `models/roberta/versionA`
"""


def build_sources_index() -> str:
    rows = [
        ["Task 9/10", "docs/task09_10_exact_reproduction_note.md", "Exact preprocessing reproduction note"],
        ["Task 11", "reports/task11_split_summary.md", "Train/validation/test split summary"],
        ["Task 12", "reports/task12_experiment_design_summary.md", "Experiment design summary"],
        ["Task 13", "reports/task13/task13_bertweet_training_summary.md", "BERTweet training summary"],
        ["Task 14", "reports/task14/task14_roberta_training_summary.md", "RoBERTa training summary"],
        ["Task 15", "reports/task15/task15_model_evaluation_summary.md", "Full model evaluation summary"],
        ["Task 16", "reports/task16/task16_stopword_impact_summary.md", "Stopword impact analysis"],
        ["Task 17", "reports/task17/task17_model_comparison_summary.md", "Model comparison and ranking"],
        ["Task 18", "reports/task18/task18_error_analysis_summary.md", "Error analysis report"],
        ["Task 19", "reports/task19/task19_final_model_selection_summary.md", "Final model selection"],
        ["Task 19", "reports/task19/final_model_card.md", "Final model card"],
        ["Task 20", "docs/task20_demo_interface.md", "Demo interface documentation"],
        ["Task 20B", "docs/task20b_enhanced_interactive_dashboard.md", "Enhanced dashboard documentation"],
        ["Task 21", "reports/task21/task21_ethics_and_limitations_summary.md", "Ethics and limitations summary"],
        ["Task 22", "reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md", "Final academic documentation"],
    ]
    return "# Final Report Sources Index\n\n" + md_table(["Task", "File", "Purpose"], rows)


def build_task_matrix_md() -> str:
    return "# Task Completion Matrix\n\n" + md_table(["Task", "Activity", "Status", "Implementation Note"], TASK_MATRIX)


def write_task_matrix_csv() -> None:
    path = REPORT_DIR / "task_completion_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Activity", "Status", "Implementation Note"])
        writer.writerows(TASK_MATRIX)
    print(f"Wrote: {path.relative_to(PROJECT_ROOT)}")


def build_presentation_key_points() -> str:
    return """
# Presentation Key Points

## One-Sentence Project Summary

This project built a sarcasm detection system using the SARC Reddit dataset and compared BERTweet and RoBERTa under two preprocessing settings.

## Main Methodology

- Dataset: SARC / Reddit comments
- Task: binary classification, sarcastic vs non-sarcastic
- Input: `parent_comment + comment`
- Models: BERTweet and RoBERTa
- Preprocessing versions: stopwords kept vs selective stopword removal
- Metrics: accuracy, precision, recall, macro-F1, weighted-F1, confusion matrix

## Final Result

- Final model: RoBERTa Version A
- Stopwords: kept
- Test accuracy: 0.7223
- Test macro-F1: 0.7167

## Main Finding

Stopword removal reduced performance for both BERTweet and RoBERTa. Keeping stopwords helped because sarcasm depends on context, contrast, and sentence structure.

## Important Limitation

The model should not be used for automatic moderation or punitive decisions because sarcasm is context-dependent and can be misclassified.

## Demo Talking Point

The enhanced dashboard shows the full AI/NLP workflow, including preprocessing, model comparison, reports, final prediction demo, and ethics/limitations.
"""


def build_readme() -> str:
    return """
# Sarcasm Detection in Social Media Text

A complete Natural Language Processing project for detecting sarcasm in Reddit-style social media text using the SARC dataset, BERTweet, and RoBERTa.

This repository contains the full academic workflow: preprocessing, controlled experiment design, transformer training, model evaluation, stopword impact analysis, model comparison, error analysis, final model selection, Streamlit demo interface, enhanced interactive dashboard, ethics documentation, and final report materials.

---

## Project Overview

Sarcasm is challenging for NLP because the literal meaning of a comment may differ from the intended meaning. In social media, sarcasm often depends on context, short informal language, punctuation, contrast, and the relationship between a parent comment and a reply.

This project builds a binary classifier that predicts whether a Reddit comment is:

- **Sarcastic**
- **Non-sarcastic**

The final system uses the context-aware text input:

```text
parent_comment + comment
```

This allows the model to use the surrounding Reddit conversation context when available.

---

## Final Result

The final selected model is:

```text
RoBERTa Version A
```

where Version A means **stopwords are kept**.

| Metric | Value |
|---|---:|
| Test Accuracy | 0.7223 |
| Test Macro-F1 | 0.7167 |
| Input Format | parent_comment + comment |
| Max Length | 128 tokens |
| Final Checkpoint | models/roberta/versionA |

The final model was selected because it achieved the highest held-out test macro-F1 among all four controlled experiments.

---

## Experiment Summary

The project compared two transformer models and two preprocessing versions:

| Experiment | Model | Preprocessing | Stopword Setting |
|---|---|---|---|
| E01 | BERTweet | Version A | Stopwords kept |
| E02 | BERTweet | Version B | Selective stopword removal |
| E03 | RoBERTa | Version A | Stopwords kept |
| E04 | RoBERTa | Version B | Selective stopword removal |

Final held-out test ranking:

| Rank | Experiment | Model | Version | Accuracy | Macro-F1 |
|---:|---|---|---|---:|---:|
| 1 | E03 | RoBERTa | Version A | 0.7223 | 0.7167 |
| 2 | E04 | RoBERTa | Version B | 0.6773 | 0.6648 |
| 3 | E01 | BERTweet | Version A | 0.5092 | 0.3632 |
| 4 | E02 | BERTweet | Version B | 0.5018 | 0.3452 |

---

## Main Finding

The project found that **keeping stopwords performed better than removing stopwords** for both BERTweet and RoBERTa.

This makes sense for sarcasm detection because sarcasm often depends on:

- contrast between words
- sentence structure
- negation
- small function words
- punctuation
- context from the parent comment

Therefore, Version A became the final preprocessing choice.

---

## Repository Structure

```text
Sarcasm-Detection/
├── app/
│   ├── streamlit_app.py              # Simple prediction demo
│   └── enhanced_dashboard.py         # Full interactive NLP dashboard
├── configs/                          # Task and model configuration files
├── data/                             # Local-only data folders; CSV files ignored by Git
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/                             # Task methodology documentation
├── models/                           # Local-only model checkpoints; ignored by Git
├── reports/                          # Generated reports, metrics, figures, summaries
├── src/                              # Python source code
├── run_task*.sh                      # macOS/Linux task runners
├── run_task*.bat                     # Windows task runners
├── run_task20_demo.sh                # Simple demo runner
├── run_enhanced_dashboard.sh         # Enhanced dashboard runner
├── requirements.txt
└── README.md
```

---

## Local Files Not Pushed to GitHub

The following files are intentionally ignored by Git because they are large, private, or runtime-only:

```text
data/raw/*.csv
data/processed/*.csv
data/splits/**/*.csv
data/splits/*.json
models/**
reports/task18/error_samples/*.csv
reports/task18/local_only/**
reports/task20/local_only/**
```

This means the repository stores the **code and reports**, while raw data and model checkpoints remain local.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/majidsamadi/Sarcasm-Detection.git
cd Sarcasm-Detection
```

### 2. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Reproduce the Workflow

The project includes runner scripts for each task. On macOS/Linux, use `.sh` files. On Windows, use `.bat` files.

### Reproduce preprocessing and splits

```bash
bash run_task09_10_exact.sh
bash run_task11_splits.sh
```

### Validate experiment design

```bash
bash run_task12_experiment_design.sh
```

### Train models

```bash
bash run_task13_train_bertweet.sh
bash run_task14_train_roberta.sh
```

### Evaluate and analyze

```bash
bash run_task15_model_evaluation.sh
bash run_task16_stopword_impact_analysis.sh
bash run_task17_model_comparison.sh
bash run_task18_error_analysis.sh
bash run_task19_final_model_selection.sh
```

### Generate ethics and final documentation

```bash
bash run_task21_ethics_limitations.sh
bash run_task22_final_report.sh
```

---

## Running the Simple Demo

The simple Streamlit demo loads the final local checkpoint from:

```text
models/roberta/versionA
```

Run:

```bash
bash run_task20_demo.sh
```

Open the Streamlit URL, usually:

```text
http://localhost:8501
```

The simple demo provides:

- optional parent-comment/context input
- comment input
- sarcastic/non-sarcastic prediction
- confidence score
- class probabilities
- responsible-use warning

---

## Running the Enhanced Interactive Dashboard

The enhanced dashboard provides a more complete view of the entire NLP workflow.

Run:

```bash
bash run_enhanced_dashboard.sh
```

The dashboard includes:

- full workflow overview
- task progress tracker
- pipeline runner controls
- final model prediction demo
- results dashboard
- reports explorer
- error analysis section
- hosting readiness section

This dashboard is useful for presenting the complete AI/NLP process, not just the final prediction.

---

## What We Did

The project was implemented through a structured set of tasks:

| Task | Description | Status |
|---|---|---|
| 9 | Preprocessing Version A | Completed |
| 10 | Preprocessing Version B | Completed |
| 11 | Train/validation/test split | Completed |
| 12 | Model experiment design | Completed |
| 13 | BERTweet training | Completed |
| 14 | RoBERTa training | Completed |
| 15 | Full model evaluation | Completed |
| 16 | Stopword impact analysis | Completed |
| 17 | Model comparison | Completed |
| 18 | Error analysis | Completed |
| 19 | Final model selection | Completed |
| 20 | Demo interface | Completed |
| 20B | Enhanced dashboard | Completed |
| 21 | Ethics and limitations | Completed |
| 22 | Final documentation | Completed |

---

## What We Experienced and Fixed

Several technical issues were encountered and resolved:

### 1. Exact preprocessing reproduction

An earlier preprocessing script produced a different row count. Since the goal was to reproduce the teammate's preprocessing exactly, the pipeline was corrected and the non-exact version was removed.

### 2. Local pandas compatibility issue

The exact preprocessing script initially failed locally due to null values during word-count calculation. This was fixed without changing the final preprocessing logic.

### 3. Label column issue during training

The BERTweet training script initially expected a hardcoded `label` column and failed. The loader was rewritten to robustly detect the correct label column.

### 4. Evaluation function mismatch

The evaluation script initially failed because a function call passed an unexpected parameter. The evaluation script was rewritten and rerun successfully.

### 5. Validation vs test difference

BERTweet looked stronger during validation, but full test evaluation showed RoBERTa Version A generalized better. This reinforced the importance of full held-out test evaluation.

### 6. Local vs hosted model storage

The final model checkpoint is local-only and ignored by GitHub. For Hugging Face hosting, the model should be uploaded to Hugging Face Model Hub and the app should load it from there.

---

## Reports Guide

Important report files:

| File | Purpose |
|---|---|
| `reports/task11_split_summary.md` | Train/validation/test split summary |
| `reports/task12_experiment_design_summary.md` | Four-experiment design summary |
| `reports/task13/task13_bertweet_training_summary.md` | BERTweet training report |
| `reports/task14/task14_roberta_training_summary.md` | RoBERTa training report |
| `reports/task15/task15_model_evaluation_summary.md` | Full test evaluation summary |
| `reports/task16/task16_stopword_impact_summary.md` | Stopword impact analysis |
| `reports/task17/task17_model_comparison_summary.md` | Model comparison and ranking |
| `reports/task18/task18_error_analysis_summary.md` | Error analysis |
| `reports/task19/task19_final_model_selection_summary.md` | Final model selection |
| `reports/task19/final_model_card.md` | Final model card |
| `reports/task21/task21_ethics_and_limitations_summary.md` | Ethics and limitations |
| `reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md` | Full academic report |

---

## How the Final Decision Was Made

The final model was selected using the following evidence:

1. All four experiments were evaluated on the same held-out test split.
2. Macro-F1 was used as the main ranking metric.
3. RoBERTa Version A achieved the best macro-F1.
4. Version A outperformed Version B for both models, showing that keeping stopwords was better.
5. Error analysis was performed on the selected best model.
6. Ethics and limitations were documented before considering deployment.

Final decision:

```text
RoBERTa Version A was selected as the final model.
```

---

## Ethics and Responsible Use

This model is for academic demonstration and research exploration only.

It should **not** be used to automatically:

- remove user content
- flag users for punishment
- moderate posts without human review
- make high-stakes decisions

Key limitations:

- sarcasm depends on context
- Reddit data may contain platform/community bias
- model confidence does not guarantee correctness
- cultural and linguistic differences affect sarcasm
- some comments are ambiguous even for humans

---

## Hugging Face Hosting Recommendation

Recommended deployment architecture:

| Component | Recommendation |
|---|---|
| Code repository | GitHub |
| Model checkpoint | Hugging Face Model Hub |
| Hosted demo | Hugging Face Spaces |
| UI | Streamlit or Docker-based Streamlit Space |
| Heavy training | Local only |
| Public hosted app | Inference + reports only |

The public hosted version should not allow users to trigger heavy training tasks. It should focus on:

- final model inference
- project explanation
- model comparison
- reports
- ethics and limitations

---

## Quick Demo Inputs

Likely sarcastic example:

```text
Parent: The deadline moved to tomorrow.
Comment: Perfect, I love surprise deadlines.
```

Likely non-sarcastic example:

```text
Parent: How was the seminar?
Comment: It was useful and I learned several new methods.
```

---

## Conclusion

This project demonstrates a full NLP development lifecycle for sarcasm detection. It shows that preprocessing choices matter, fair model comparison is essential, held-out test evaluation can change the final decision, and ethical limitations must be documented before deployment.

The final selected system uses **RoBERTa Version A** with stopwords kept and context-aware input from `parent_comment + comment`.
"""


def build_task22_doc() -> str:
    return """
# Task 22: Final Report Writing and README Generation

Task 22 generates the final academic documentation package and the full GitHub `README.md`.

## Scope

This task documents the complete NLP workflow from preprocessing through demo interface and ethics. The team member and role section is intentionally excluded because it will be completed separately by the group leader.

## Generated Files

- `README.md`
- `reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md`
- `reports/final_report/technical_implementation_log.md`
- `reports/final_report/final_report_sources_index.md`
- `reports/final_report/task_completion_matrix.md`
- `reports/final_report/task_completion_matrix.csv`
- `reports/final_report/presentation_key_points.md`
- `reports/task22/task22_progress_note.txt`
- `reports/task22/task22_summary.json`

## Purpose

The generated documentation explains what was done, how each task contributed to the project, what issues were experienced, how the final model was selected, and how to run the project locally.
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TASK22_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    write(REPORT_DIR / "UM_WQF7007_Sarcasm_Detection_Final_Report.md", build_final_report())
    write(REPORT_DIR / "technical_implementation_log.md", build_technical_log())
    write(REPORT_DIR / "final_report_sources_index.md", build_sources_index())
    write(REPORT_DIR / "task_completion_matrix.md", build_task_matrix_md())
    write(REPORT_DIR / "presentation_key_points.md", build_presentation_key_points())
    write_task_matrix_csv()

    write(PROJECT_ROOT / "README.md", build_readme())
    write(DOCS_DIR / "task22_final_report_writing.md", build_task22_doc())

    progress = (
        "Final documentation and README completed. The generated documentation covers the complete NLP workflow, "
        "including dataset preparation, preprocessing, split design, experiment design, BERTweet/RoBERTa training, "
        "evaluation, stopword impact analysis, model comparison, error analysis, final model selection, demo interface, "
        "enhanced dashboard, ethics and limitations, lessons learned, and run instructions. The team members and roles "
        "section was intentionally excluded for the group leader to complete separately."
    )
    write(TASK22_DIR / "task22_progress_note.txt", progress)

    summary = {
        "task": "Task 22 Final Documentation and README",
        "status": "completed",
        "team_section_included": False,
        "readme_generated": True,
        "final_report_generated": True,
        "final_model": "RoBERTa Version A",
        "accuracy": 0.7223,
        "macro_f1": 0.7167,
        "generated_files": [
            "README.md",
            "reports/final_report/UM_WQF7007_Sarcasm_Detection_Final_Report.md",
            "reports/final_report/technical_implementation_log.md",
            "reports/final_report/final_report_sources_index.md",
            "reports/final_report/task_completion_matrix.md",
            "reports/final_report/task_completion_matrix.csv",
            "reports/final_report/presentation_key_points.md",
            "docs/task22_final_report_writing.md",
            "reports/task22/task22_progress_note.txt",
        ],
    }
    write(TASK22_DIR / "task22_summary.json", json.dumps(summary, indent=2))

    print("\nTask 22 documentation generation completed successfully.")


if __name__ == "__main__":
    main()
