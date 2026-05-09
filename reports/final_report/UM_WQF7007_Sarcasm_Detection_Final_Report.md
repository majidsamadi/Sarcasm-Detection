
# Final Report: Sarcasm Detection in Social Media Text using RoBERTa and BERTweet

**Course:** WQF7007 Natural Language Processing  
**Project theme:** NLP application for sarcasm detection in social media text  
**Dataset:** SARC / Self-Annotated Reddit Corpus  
**Final selected model:** RoBERTa Version A  
**Final test accuracy:** 0.7223  
**Final test macro-F1:** 0.7167  
**Generated on:** 2026-05-10 00:54:54

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
2. Keeping local execution compatible while preserving the intended preprocessing logic.
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

Task 13 trained BERTweet on Version A and Version B. BERTweet was included because it is designed for social-media-style language. During implementation, the team learned the importance of robust data-loading design. The training workflow was written to detect the correct label column reliably and support reproducible model training.

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

The evaluation stage helped the team learn the importance of consistent function design and controlled test evaluation. The final evaluation workflow was structured to run all four experiments on the same held-out test set.

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


## 18. Learning Experience and Skills Gained

This project gave the group a complete practical learning experience in building an NLP system from data preparation to final demonstration. Instead of only applying a model, the team experienced how different stages of an AI project connect together and affect the final result.

### 18.1 Understanding Sarcasm as a Context-Dependent NLP Problem

One important learning outcome was that sarcasm detection is not the same as simple sentiment analysis. A sarcastic comment may use positive words while expressing a negative or ironic intention. This helped us understand why the project needed conversational context through `parent_comment + comment`, rather than relying only on the reply text.

### 18.2 Learning the Importance of Task-Aware Preprocessing

The project showed that preprocessing decisions should depend on the NLP task. For example, stopword removal is useful in some text-mining tasks, but in sarcasm detection, small words and sentence structure can carry meaning. By comparing Version A and Version B, we learned that keeping stopwords gave better performance for both RoBERTa and BERTweet.

### 18.3 Learning How to Design a Fair Experiment

The team learned that model comparison must be controlled and fair. All four experiments used the same dataset split, same input format, same maximum sequence length, and same evaluation metrics. This made the comparison between BERTweet and RoBERTa academically defensible.

### 18.4 Learning to Evaluate Beyond Accuracy

The project helped us understand why accuracy alone is not enough. We used macro-F1, precision, recall, classification reports, and confusion matrices to evaluate the models more deeply. Macro-F1 became the main metric because it gives a balanced view of model performance across both classes.

### 18.5 Learning the Difference Between Validation and Test Results

Another important lesson was that validation performance is useful during model development, but final decisions should be based on the held-out test set. This helped us make a stronger final model decision using full test evaluation instead of relying only on earlier training results.

### 18.6 Learning How to Interpret Model Behavior

Through error analysis, we learned how to look beyond numbers and study the types of examples that models find difficult. This improved our understanding of false positives, false negatives, confidence patterns, text length effects, and surface features such as punctuation or capitalization.

### 18.7 Learning to Build an End-to-End NLP Application

The project gave us hands-on experience in connecting the full NLP lifecycle: dataset preparation, preprocessing, splitting, model training, evaluation, model selection, demo interface, and dashboard visualization. The enhanced dashboard helped us present the workflow as a complete AI system rather than only a single prediction model.

### 18.8 Learning Responsible AI Practice

The ethics and limitations section helped us understand that NLP models can be useful but should be deployed carefully. We learned to include responsible-use statements, limitations, risk awareness, and guidance against using the model for automatic moderation or punitive decisions.

### 18.9 Learning Reproducible Project Organization

The team also gained experience organizing an academic AI project in a structured repository. Scripts, reports, configurations, figures, model outputs, and documentation were separated clearly. This made the project easier to explain, rerun, and present.

Overall, the project strengthened our understanding of both the technical and academic sides of NLP. We gained practical experience in transformer-based classification, evaluation methodology, report writing, UI demonstration, and responsible AI communication.

---

## 19. Key Academic Takeaways

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
