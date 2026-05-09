# Demo Interfaces

This folder contains two Streamlit interfaces for the sarcasm detection project.

## 1. Basic Demo

File:

```text
app/streamlit_app.py
```

Run:

```bash
bash run_task20_demo.sh
```

Purpose: simple prediction demo using the selected final model.

## 2. Enhanced Interactive Dashboard

File:

```text
app/enhanced_dashboard.py
```

Run:

```bash
bash run_enhanced_dashboard.sh
```

Purpose: complete AI/NLP workflow dashboard showing:

- pipeline status
- local task runner
- live sarcasm prediction
- model comparison
- stopword impact analysis
- reports explorer
- error analysis
- Hugging Face hosting readiness

Important: the task-runner section is intended for local use. For public Hugging Face hosting, heavy training execution should be disabled or avoided.
