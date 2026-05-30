# Application Interfaces

This project now uses **Gradio** as the primary stakeholder-facing interface.

## Primary Interface: Gradio Dashboard

Run locally:

```bash
bash run_gradio_dashboard.sh
```

Main file:

```text
app/gradio_app.py
```

Hugging Face Spaces entry point:

```text
app.py
```

The Gradio dashboard includes live prediction, project workflow status, model comparison, stopword impact charts, report exploration, error analysis, ethics notes, and optional local task execution.

## Legacy Interface: Streamlit

The earlier Streamlit files are kept as reference/local backup files:

```text
app/streamlit_app.py
app/enhanced_dashboard.py
```

For stakeholder presentation and hosting, use the Gradio dashboard.
