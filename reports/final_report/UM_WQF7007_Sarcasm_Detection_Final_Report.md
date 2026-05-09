
# Final Report: Sarcasm Detection in Social Media Text using RoBERTa and BERTweet

**Course:** WQF7007 Natural Language Processing  
**Project theme:** NLP application for sarcasm detection in social media text  
**Dataset:** SARC / Self-Annotated Reddit Corpus  
**Final selected model:** RoBERTa Version A  
**Final test accuracy:** 0.7223  
**Final test macro-F1:** 0.7167  
**Generated on:** 2026-05-10 00:39:06

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

| Experiment | Model | Preprocessing Version | Stopword Setting |
| --- | --- | --- | --- |
| E01 | BERTweet | Version A | Stopwords kept |
| E02 | BERTweet | Version B | Selective stopword removal |
| E03 | RoBERTa | Version A | Stopwords kept |
| E04 | RoBERTa | Version B | Selective stopword removal |

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

| Rank | Experiment | Model | Version | Stopword Setting | Accuracy | Macro-F1 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | E03 | RoBERTa | Version A | Stopwords kept | 0.7223 | 0.7167 | Selected final model |
| 2 | E04 | RoBERTa | Version B | Selective stopword removal | 0.6773 | 0.6648 | Second best |
| 3 | E01 | BERTweet | Version A | Stopwords kept | 0.5092 | 0.3632 | Lower test generalization |
| 4 | E02 | BERTweet | Version B | Selective stopword removal | 0.5018 | 0.3452 | Lowest ranked |

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
