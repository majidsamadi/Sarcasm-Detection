# Application Interfaces

This project now uses **Gradio** as the primary stakeholder-facing interface.

## Main Interface

Run the redesigned Gradio dashboard:

```bash
bash run_gradio_dashboard.sh
```

Then open:

```text
http://localhost:7860
```

## What the Gradio Dashboard Includes

- Live sarcasm prediction using the final RoBERTa Version A model.
- Optional parent-comment context plus comment input.
- Class probabilities and confidence score.
- Model comparison dashboard.
- Stopword impact and confusion matrix visualisations.
- Workflow status from preprocessing to documentation.
- Reports explorer.
- Local task runner.
- Ethics and hosting guidance.

## Legacy Interfaces

Earlier Streamlit interfaces may still exist in this folder for reference, but Gradio is now the primary interface because of the stakeholder requirement.
