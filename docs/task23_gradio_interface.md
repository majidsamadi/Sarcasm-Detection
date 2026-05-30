# Task 23: Gradio User Interface

## Purpose

Stakeholders requested that the project interface use Gradio. Task 23 rebuilds the user interface using Gradio Blocks and makes Gradio the primary interactive front end for the sarcasm detection project.

## What the Gradio UI Provides

The Gradio interface includes:

1. **Live Sarcasm Detector** using the selected final RoBERTa Version A model.
2. **Project Workflow Overview** showing completion status from preprocessing to documentation.
3. **Results Dashboard** showing model comparison, stopword impact, and confusion matrices.
4. **Reports Explorer** for browsing generated Markdown, JSON, CSV, and text reports.
5. **Local Pipeline Runner** for executing task scripts from the interface.
6. **Error Analysis and Ethics** sections for responsible interpretation.
7. **Hosting Readiness** notes for Hugging Face Spaces deployment.

## Final Model Used

The interface uses the selected final model from Task 19:

- Model: RoBERTa Version A
- Preprocessing: Version A, stopwords kept
- Input format: optional `parent_comment` plus main `comment`
- Local checkpoint: `models/roberta/versionA`

## Running Locally

```bash
bash run_gradio_dashboard.sh
```

The app opens at:

```text
http://localhost:7860
```

## Hugging Face Spaces

The repository now includes a root `app.py`, which can be used as the Hugging Face Spaces entry point. For public hosting, task execution should be disabled:

```bash
GRADIO_ALLOW_TASK_RUNS=0
```

The model should be uploaded to the Hugging Face Model Hub or made available to the Space so the Gradio app can load it without relying only on the local `models/` folder.

## Notes

The previous Streamlit interface is kept as a legacy/local reference, but the stakeholder-facing interface is now the Gradio dashboard.
