# Task 21: Ethics and Limitations Summary

## Purpose

Task 21 documents the ethical considerations, limitations, and responsible-use requirements for the sarcasm detection project. This is necessary because sarcasm detection can be misinterpreted if treated as a fully reliable judgment of user intent.

## Final System Context

- **Project:** Sarcasm Detection in Social Media Text using RoBERTa
- **Selected model:** {'experiment_id': 'E03_RoBERTa_VersionA', 'model_family': 'RoBERTa', 'dataset_version': 'Version A', 'preprocessing': 'Stopwords kept', 'checkpoint_path': 'models/roberta/versionA', 'tokenizer_name': 'roberta-base', 'accuracy': 0.722284968241304, 'macro_precision': 0.7430131550757031, 'macro_recall': 0.7230856801312752, 'macro_f1': 0.7166978860458602, 'weighted_f1': 0.7164773379034147, 'source_metrics_file': 'reports/task15/E03_RoBERTa_VersionA_test_metrics.json'}
- **Model family:** RoBERTa
- **Preprocessing:** Version A - stopwords kept
- **Input strategy:** Text-only input using parent_comment + comment when context is available
- **Test accuracy:** 0.7223
- **Test macro-F1:** 0.7167
- **Best model from comparison:** {'experiment_id': 'E03_RoBERTa_VersionA', 'short_id': 'E03', 'model': 'RoBERTa', 'model_family': 'RoBERTa', 'dataset_version': 'Version A', 'preprocessing': 'Stopwords kept', 'accuracy': 0.7223, 'macro_precision': 0.743, 'macro_recall': 0.7231, 'macro_f1': 0.7167, 'weighted_f1': 0.7165, 'rows_evaluated': 96509, 'metrics_path': 'reports/task15/E03_RoBERTa_VersionA_test_metrics.json'}

## Main Ethical Position

The system is a research and coursework demonstration. It should not be used to automatically remove, flag, penalize, or judge user content.

The system should be presented as a research and coursework demonstration, not as a production moderation or user-judgment system.

## Key Ethical Risks

| Risk ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Misclassification of sarcasm | High | Show confidence score, communicate uncertainty, avoid automated punitive use, and require human interpretation. |
| R2 | Context loss | High | Use optional parent comment context, state that predictions are approximate, and avoid using the model for moderation decisions. |
| R3 | Dataset bias from Reddit data | High | Limit claims to Reddit-style/social-media text and disclose that performance may not generalize to other platforms or cultures. |
| R4 | Privacy and sensitive text handling | Medium | Do not commit raw error samples to GitHub, keep local examples ignored, and use aggregate reporting instead of exposing raw user text. |
| R5 | Over-reliance on confidence score | Medium | Present confidence as model confidence only, not truth certainty. Include clear wording in the UI. |
| R6 | Deployment misuse | High | Add a visible research-demo warning and disable heavy training/execution features on public hosting. |
| R7 | Compute and reproducibility limitations | Medium | Report the training setup transparently and keep scripts/configs reproducible in GitHub. |
| R8 | Cultural and linguistic limitations | Medium | Avoid claiming universal sarcasm detection and recommend further testing on other communities or languages. |


## High-Priority Risks

The project identifies **4 high-severity risks**. The most important risks are misclassification, context loss, dataset bias, and deployment misuse.

## Limitations

1. **Sarcasm is context-dependent.** Text alone may not capture tone, speaker intent, shared background, or full conversation history.
2. **Dataset limitation.** The model is trained and evaluated on Reddit-style data, so results may not generalize to other platforms or languages.
3. **Model uncertainty.** A confidence score reflects model probability, not guaranteed truth.
4. **Training-resource limitation.** The project used a practical training sample due to local compute constraints, while evaluation was conducted on the full held-out test split.
5. **Public deployment limitation.** Public hosting should focus on inference and report presentation. Heavy training execution should remain local.

## Responsible Use Rules

- Use the model for academic demonstration and exploratory analysis only.
- Do not use the model to automatically remove, flag, penalize, or judge users.
- Show the prediction confidence and communicate uncertainty.
- Avoid exposing raw Reddit text samples in public reports or repositories.
- Keep human review in the loop for any interpretation of model outputs.
- Disclose dataset, preprocessing, evaluation, and model limitations clearly.


## Academic Conclusion

The project includes ethics and limitations as part of the NLP pipeline because responsible interpretation is essential for social-media text classification. The model can demonstrate how transformer-based models detect sarcasm patterns, but it should not be treated as a definitive detector of human intention.
