
# Technical Implementation Log

This document records what was implemented during the sarcasm detection project and the major learning outcomes and implementation decisions developed.

## Completed Workflow

| Task | Activity | Status | Implementation Note |
| --- | --- | --- | --- |
| 9 | Preprocessing Version A | Completed | Created `A.csv` using exact reproduced preprocessing. Stopwords, punctuation, casing, and ALL CAPS were kept. |
| 10 | Preprocessing Version B | Completed | Created `B.csv` from Version A using selective stopword removal while preserving negations. |
| 11 | Train/Validation/Test Split | Completed | Created stratified 80/10/10 splits for Version A and Version B using the same indices. |
| 12 | Model Experiment Design | Completed | Defined four controlled experiments: BERTweet A/B and RoBERTa A/B. |
| 13 | Train BERTweet | Completed | Fine-tuned BERTweet on Version A and Version B with the same training setup. |
| 14 | Train RoBERTa | Completed | Fine-tuned RoBERTa on Version A and Version B with the same training setup. |
| 15 | Model Evaluation | Completed | Evaluated all four models on the full held-out test set and generated metrics/confusion matrices. |
| 16 | Stopword Impact Analysis | Completed | Compared Version A vs Version B and found that keeping stopwords performed better for both models. |
| 17 | Model Comparison | Completed | Ranked all four experiments using test macro-F1 as the main metric. |
| 18 | Error Analysis | Completed | Analyzed false positives, false negatives, confidence patterns, text length, and surface features for the selected model. |
| 19 | Final Model Selection | Completed | Selected RoBERTa Version A as the final model based on test macro-F1. |
| 20 | Demo Interface | Completed | Built a local Streamlit prediction demo. |
| 20B | Enhanced Dashboard | Completed | Built an interactive dashboard showing the full AI/NLP workflow, reports, progress, and model demo. |
| 21 | Ethics and Limitations | Completed | Generated risk register, responsible-use checklist, and ethics/limitations summary. |
| 22 | Final Documentation | Completed | Generated final report package and full GitHub README. |

## Key Implementation Decisions

| Decision | Reason |
|---|---|
| Use `parent_comment + comment` | Sarcasm often depends on conversational context. |
| Keep punctuation and casing | Punctuation and ALL CAPS can signal sarcasm. |
| Compare stopwords kept vs removed | Lecturer asked whether stopword removal improves performance. |
| Use macro-F1 as main metric | Macro-F1 balances performance across classes better than accuracy alone. |
| Select only one final model | The project compares multiple models but implements the best one for the final demo. |
| Keep data/model files local | Large artifacts should not be pushed to GitHub. |
| Use Streamlit for UI | Streamlit is fast, explainable, and suitable for coursework demos. |
| Use enhanced dashboard locally | Full pipeline triggering is appropriate locally but not for public hosting. |


## Learning Reflections and Skills Developed

| Learning Area | What We Gained | How It Helped the Project |
|---|---|---|
| NLP problem understanding | Learned that sarcasm depends on context, contrast, and implied meaning | Supported the decision to use `parent_comment + comment` as model input |
| Task-aware preprocessing | Learned that preprocessing choices must match the task | Helped us evaluate stopword removal instead of assuming it is always beneficial |
| Experimental design | Learned to compare models under the same split, input format, and metrics | Made the BERTweet vs RoBERTa comparison fair and academically defensible |
| Transformer modeling | Gained hands-on experience with BERTweet and RoBERTa fine-tuning | Helped us understand how pretrained language models can be adapted to classification |
| Evaluation methodology | Learned to use macro-F1, precision, recall, confusion matrices, and reports | Allowed us to select the final model using evidence rather than intuition |
| Error analysis | Learned to study false positives, false negatives, confidence, and text-length behavior | Improved our ability to explain model limitations clearly |
| UI and communication | Learned to convert model outputs into an interactive Streamlit dashboard | Made the project easier to present and understand |
| Responsible AI | Learned to document risk, limitation, and responsible-use guidance | Strengthened the academic and ethical quality of the project |
| Reproducibility | Learned to organize scripts, configs, reports, and local artifacts | Made the workflow easier to rerun, audit, and explain |

## Final Model

- Experiment: E03
- Model: RoBERTa Version A
- Stopwords: kept
- Test accuracy: 0.7223
- Test macro-F1: 0.7167
- Local checkpoint: `models/roberta/versionA`
