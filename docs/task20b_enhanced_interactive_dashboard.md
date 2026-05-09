# Task 20B: Enhanced Interactive NLP Dashboard

## Purpose

Task 20B extends the basic demo into a complete interactive dashboard for the full AI/NLP workflow. The goal is to make the project easier to demonstrate academically by showing not only predictions, but also the full methodology flow.

## Dashboard Sections

1. **Overview**  
   Shows the complete task pipeline, completion status, final model, and key test metrics.

2. **Pipeline Runner**  
   Allows local execution of project scripts from the UI. This is intended for local use only because training and evaluation tasks are computationally heavy.

3. **Live Demo**  
   Uses the final selected RoBERTa Version A model to predict whether a text is sarcastic or non-sarcastic.

4. **Results Dashboard**  
   Shows model comparison, accuracy, macro-F1, and stopword impact results.

5. **Reports Explorer**  
   Allows browsing of generated reports, methodology notes, JSON files, and CSV result tables.

6. **Error Analysis**  
   Presents the error-analysis report, classification report, and Task 18 figures.

7. **Hosting Readiness**  
   Explains what is ready for Hugging Face deployment and what still needs to be done.

## Best Practice Note

The dashboard can trigger scripts locally. However, the public hosted version on Hugging Face should not run heavy training jobs unless dedicated compute and persistence are configured. For Hugging Face, the recommended version is a report + inference dashboard.

## Run Locally

```bash
bash run_enhanced_dashboard.sh
```

## Recommended Hosting Architecture

- GitHub: code, reports, documentation
- Hugging Face Model Hub: final model checkpoint
- Hugging Face Spaces: hosted Streamlit/Docker app

## Ethical Note

The UI includes reminders that sarcasm detection can be wrong and should not be used for automatic moderation or penalties.
