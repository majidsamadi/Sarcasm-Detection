
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


## What We Learned From This Project

This project gave us a strong learning experience because we followed the full AI/NLP workflow from dataset preparation to final user interface. The main learning outcomes were not only about training a model, but also about understanding how each stage affects the quality, reliability, and explainability of the final system.

### 1. We learned that sarcasm detection is context-dependent

Sarcasm is difficult because the literal meaning of a comment may be different from the intended meaning. This helped us understand why the parent comment matters. By using `parent_comment + comment`, we learned that conversational context is important for detecting sarcasm in Reddit-style text.

### 2. We learned that preprocessing must match the NLP task

Before this project, stopword removal might look like a normal preprocessing step. However, our experiments showed that removing stopwords reduced model performance. This taught us that preprocessing is not one-size-fits-all. For sarcasm detection, small words, negations, and sentence structure can help the model understand contrast and intention.

### 3. We learned how to design fair model experiments

The project helped us understand the importance of controlled comparison. BERTweet and RoBERTa were evaluated using the same split, same input format, same maximum sequence length, and same metrics. This made the result more academically valid because the model comparison was fair.

### 4. We learned to evaluate models using more than accuracy

We used accuracy, precision, recall, macro-F1, weighted-F1, classification reports, and confusion matrices. This helped us understand that accuracy alone is not enough, especially when explaining model behavior academically. Macro-F1 became our main decision metric because it gives a more balanced view across classes.

### 5. We learned how to make a final model decision using evidence

The final model was not selected based on preference. It was selected based on test performance, stopword impact analysis, model comparison, and error analysis. This taught us how to justify a model selection decision using evidence instead of assumptions.

### 6. We learned how error analysis improves model understanding

Error analysis helped us look beyond the final score. By studying false positives, false negatives, confidence patterns, text-length behavior, and surface features, we learned how to explain what the model does well and where sarcasm remains difficult.

### 7. We learned how to build an end-to-end NLP application

This project connected many parts of the NLP pipeline: preprocessing, data splitting, experiment design, model training, evaluation, final selection, demo interface, enhanced dashboard, and responsible-use documentation. This gave us practical experience in building a complete NLP system rather than only running a model notebook.

### 8. We learned the value of clear documentation and reproducibility

The project taught us the importance of keeping scripts, configs, reports, figures, and documentation organized. This makes the work easier to explain, easier to rerun, and easier for teammates or lecturers to review.

### 9. We learned responsible AI thinking

The ethics and limitations work helped us understand that an NLP model should be presented with caution. Sarcasm is subjective and context-sensitive, so the model should be used for research and demonstration, not for automatic moderation or punishment.

Overall, this project improved our technical skills, research methodology, teamwork discipline, and ability to communicate AI/NLP results in an academic way.

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
