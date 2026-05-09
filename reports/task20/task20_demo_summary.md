# Task 20 Demo Interface Summary

Task 20 implemented a Streamlit demo interface for the final sarcasm detection model.

## Final Model

- Selected model: RoBERTa Version A
- Checkpoint path: `models/roberta/versionA`
- Preprocessing: Version A, stopwords kept
- Input: text-only input using `parent_comment + comment` when context is available
- Max token length: 128

## Demo Features

- Optional parent-comment input
- Main comment input
- Sarcastic/non-sarcastic prediction
- Confidence score
- Class probabilities
- Ethical warning about non-production use

## Academic Value

The interface demonstrates the final NLP pipeline in an accessible way while preserving the academic methodology from previous tasks: preprocessing, controlled model comparison, evaluation, error analysis, and final model selection.

## Limitation

The demo should not be used for automated moderation. Sarcasm detection remains context-dependent and imperfect.
