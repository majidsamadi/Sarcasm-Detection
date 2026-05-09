# Task 20: Demo Interface

## Purpose

Task 20 implements a simple demonstration interface for the final sarcasm detection model. The interface allows a user to enter social media text and receive a binary prediction: sarcastic or non-sarcastic.

## Final Model Used

The demo uses the final model selected in Task 19:

- **Selected experiment:** E03_RoBERTa_VersionA
- **Model family:** RoBERTa
- **Preprocessing version:** Version A, where stopwords are kept
- **Input type:** text-only input
- **Input construction:** `parent_comment + comment` when parent context is provided
- **Maximum token length:** 128
- **Local checkpoint:** `models/roberta/versionA`

## Interface Design

The Streamlit app provides:

1. An optional parent-comment/context text box.
2. A comment text box.
3. A prediction button.
4. The predicted class label.
5. Confidence score.
6. Probabilities for both classes.
7. A short ethical warning.

## Why Streamlit?

Streamlit is lightweight and suitable for an academic demonstration because it allows the model to be demonstrated interactively without building a full production web system.

## Ethical Limitation

The demo is for research and coursework only. It should not be used to automatically remove, flag, or penalize user content. Sarcasm is highly context-dependent and may be misclassified, especially when cultural context, conversation history, or speaker intent is missing.

## How to Run

From the project root:

```bash
bash run_task20_demo.sh
```

Or manually:

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

## Expected Outcome

The demo should load the selected RoBERTa Version A checkpoint and return a sarcasm/non-sarcasm prediction for user-entered text.
