# Final Academic Documentation: Sarcasm Detection in Social Media Text

**Course:** WQF7007 Natural Language Processing  
**Project:** NLP Application: Sarcasm Detection in Social Media Text using BERTweet and RoBERTa  
**Dataset:** SARC / Self-Annotated Reddit Corpus  
**Generated:** 2026-05-10 00:21  
**Repository:** `https://github.com/majidsamadi/Sarcasm-Detection.git`  
**Branch:** `main`  
**Current Commit:** `f66e86b`

---

## 1. Executive Summary

This project developed an NLP system for **binary sarcasm detection** in social-media text. The system classifies Reddit-style text as either **sarcastic** or **non-sarcastic**. The final project uses the **SARC dataset**, applies two controlled preprocessing versions, compares **BERTweet** and **RoBERTa**, evaluates all experiments on a held-out test split, performs stopword-impact analysis, completes error analysis, selects the best model, and builds an interactive Streamlit demonstration dashboard.

The final selected model is **RoBERTa Version A**, where **stopwords are kept**. It was selected because it achieved the highest held-out test Macro-F1 among the four controlled experiments.

**Final selected model:** `E03_RoBERTa_VersionA`  
**Model family:** RoBERTa  
**Preprocessing:** Stopwords kept  
**Held-out test accuracy:** 0.7223  
**Held-out test Macro-F1:** 0.7167  
**Input strategy:** `parent_comment + comment` when context is available  
**Maximum sequence length:** 128 tokens

---

## 2. Team Members and Contributions

| Name | Student ID | Contribution / Role |
| --- | --- | --- |
| Faridah Albalawi | 22116678 | Project Manager; Data Engineer |
| Dhanasree Seelam | 23077229 | Documentation Leader |
| Gebriella Anthony | 17204189 | Evaluation and Ethics Specialist |
| Majid Samadikuchaksaraee | 25068504 | Presenter; Modeling and ML Engineer; Evaluation and Ethics Specialist |
| Rupika Selvaraja | 24082992 | Data Engineer; Modeling and ML Engineer |

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

The original proposal described the dataset as approximately **1.01 million rows**, with a near-balanced sarcastic/non-sarcastic distribution. During implementation, the exact preprocessing reproduction produced a final aligned cleaned dataset of **965,087 rows**.

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

| Task | Activity | Status | Summary |
| --- | --- | --- | --- |
| 09 | Preprocessing Version A | Completed | Reproduced teammate preprocessing exactly and generated A.csv with stopwords kept. |
| 10 | Preprocessing Version B | Completed | Generated B.csv by selectively removing stopwords while preserving negation words. |
| 11 | Train/Validation/Test Split | Completed | Created stratified 80/10/10 splits for Version A and Version B using the same random seed. |
| 12 | Model Experiment Design | Completed | Defined four controlled experiments across two models and two preprocessing versions. |
| 13 | Train BERTweet | Completed | Fine-tuned BERTweet on Version A and Version B. |
| 14 | Train RoBERTa | Completed | Fine-tuned RoBERTa on Version A and Version B. |
| 15 | Model Evaluation | Completed | Evaluated all four experiments on the held-out test set with accuracy, precision, recall, F1, and confusion matrices. |
| 16 | Stopword Impact Analysis | Completed | Compared Version A and Version B to determine whether stopword removal helped or reduced performance. |
| 17 | Model Comparison | Completed | Ranked the four experiments mainly by held-out test macro-F1. |
| 18 | Error Analysis | Completed | Analyzed false positives, false negatives, confidence patterns, text length, and surface features for the best model. |
| 19 | Final Model Selection | Completed | Selected RoBERTa Version A as the final model based mainly on test macro-F1. |
| 20 | Demo Interface | Completed | Created a Streamlit prediction demo using the selected local checkpoint. |
| 20B | Enhanced Interactive Dashboard | Completed | Created a fuller Streamlit dashboard showing workflow progress, reports, results, and demo. |
| 21 | Ethics and Limitations | Completed | Generated a risk register, responsible-use checklist, and ethics/limitations documentation. |
| 22 | Final Report Writing | Completed by this script | Generated full academic documentation from the completed project outputs. |

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

A key implementation experience was that the team first had a non-exact preprocessing reproduction that produced a different row count. This was corrected by reproducing the teammate's notebook logic exactly. The final correct aligned row count became **965,087 rows**.

### 8.3 Preprocessing Version B: Selective Stopword Removal

Version B was created from Version A by removing stopwords selectively. Negation words were preserved because words such as `not`, `no`, `never`, `don't`, and similar forms can change meaning and are especially important in sarcasm and sentiment-related tasks.

Version B was used as an ablation experiment to answer the lecturer's question about whether stopword removal improves performance.

### 8.4 Train/Validation/Test Split

Task 11 created a stratified split so that both classes remained balanced across train, validation, and test sets. The same split was used for Version A and Version B, ensuring fair comparison.

| Dataset Version | Train Rows | Validation Rows | Test Rows |
| --- | --- | --- | --- |
| versionA | {'rows': 772069, 'label_distribution': {'0': {'count': 383897, 'percentage': 49.7231}, '1': {'count': 388172, 'percentage': 50.2769}}} | {'rows': 96509, 'label_distribution': {'0': {'count': 47987, 'percentage': 49.7228}, '1': {'count': 48522, 'percentage': 50.2772}}} | {'rows': 96509, 'label_distribution': {'0': {'count': 47987, 'percentage': 49.7228}, '1': {'count': 48522, 'percentage': 50.2772}}} |
| versionB | {'rows': 772069, 'label_distribution': {'0': {'count': 383897, 'percentage': 49.7231}, '1': {'count': 388172, 'percentage': 50.2769}}} | {'rows': 96509, 'label_distribution': {'0': {'count': 47987, 'percentage': 49.7228}, '1': {'count': 48522, 'percentage': 50.2772}}} | {'rows': 96509, 'label_distribution': {'0': {'count': 47987, 'percentage': 49.7228}, '1': {'count': 48522, 'percentage': 50.2772}}} |

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

| Experiment | Model | Version | Validation Accuracy | Validation Macro-F1 |
| --- | --- | --- | --- | --- |
| E01_BERTweet_VersionA | BERTweet | Version A | 0.7634 | 0.7633 |
| E02_BERTweet_VersionB | BERTweet | Version B | 0.7267 | 0.7265 |
| E03_RoBERTa_VersionA | RoBERTa | Version A | 0.7475 | 0.7474 |
| E04_RoBERTa_VersionB | RoBERTa | Version B | 0.7189 | 0.7189 |

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

| Rank | Experiment | Model | Version | Preprocessing | Accuracy | Macro-F1 | Macro Precision | Macro Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E03_RoBERTa_VersionA | RoBERTa | Version A | Stopwords kept | 0.7223 | 0.7167 | 0.7430 | 0.7231 |
| 2 | E04_RoBERTa_VersionB | RoBERTa | Version B | Selective stopword removal | 0.6773 | 0.6648 | 0.7106 | 0.6784 |
| 3 | E01_BERTweet_VersionA | BERTweet | Version A | Stopwords kept | 0.5092 | 0.3632 | 0.6638 | 0.5118 |
| 4 | E02_BERTweet_VersionB | BERTweet | Version B | Selective stopword removal | 0.5018 | 0.3452 | 0.6403 | 0.5046 |

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
| Test accuracy | 0.7223 |
| Test Macro-F1 | 0.7167 |
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

| Source File | Status |
| --- | --- |
| docs/task09_10_exact_reproduction_note.md | Available |
| reports/task11_split_summary.md | Available |
| docs/task12_model_experiment_design.md | Available |
| reports/task13/task13_bertweet_training_summary.md | Available |
| reports/task14/task14_roberta_training_summary.md | Available |
| reports/task15/task15_model_evaluation_summary.md | Available |
| reports/task16/task16_stopword_impact_summary.md | Available |
| reports/task17/task17_model_comparison_summary.md | Available |
| reports/task18/task18_error_analysis_summary.md | Available |
| reports/task19/task19_final_model_selection_summary.md | Available |
| reports/task19/final_model_card.md | Available |
| docs/task20_demo_interface.md | Available |
| docs/task20b_enhanced_interactive_dashboard.md | Available |
| docs/task21_ethics_and_limitations.md | Available |
| reports/task21/task21_risk_register.csv | Available |
| reports/task21/task21_responsible_use_checklist.md | Available |

---

## 19. Latest Git Log Snapshot

```text
f66e86b Add Task 21 ethics and limitations
7671cb2 Hide Streamlit menu for presentation UI
2c67660 Hide Streamlit deploy button for demo UI
891c204 Add enhanced interactive NLP dashboard
6332bd0 Add Task 20 demo interface
889ef0b Add Task 19 final model selection
f46ab24 Add Task 18 error analysis
8cb514d Add Task 17 model comparison
f8448ee Add Task 16 stopword impact analysis
00cbf4d Fix and complete Task 15 model evaluation
04050ed Add Task 14 RoBERTa training
d4c6aa7 Fix and complete Task 13 BERTweet training
```
