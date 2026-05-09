# Task 20 Demo Interface

This folder contains the Streamlit demo for the sarcasm detection project.

The demo loads the selected final model from Task 19:

- Model: RoBERTa Version A
- Local checkpoint: `models/roberta/versionA`
- Preprocessing setting: Version A, stopwords kept
- Input format: text-only input using `parent_comment + comment` when context is provided
- Output: sarcastic or non-sarcastic prediction with confidence scores

Run locally:

```bash
bash run_task20_demo.sh
```

Important: model checkpoints are local-only and are not pushed to GitHub.
