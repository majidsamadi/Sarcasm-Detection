# Technical Implementation Log

Generated: 2026-05-10 00:21

This document records the implementation experience across the project tasks.

## Key Implementation Events

1. The project scope was finalized as sarcasm detection using SARC.
2. Version A and Version B preprocessing were defined to test the impact of stopword handling.
3. A non-exact preprocessing script initially produced a different row count; it was replaced by an exact reproduction of the teammate's notebook workflow.
4. Final exact preprocessing produced 965,087 aligned rows.
5. Task 11 created identical stratified splits for Version A and Version B.
6. Task 12 created a four-experiment matrix.
7. Task 13 initially had label-column loading issues, which were fixed by rewriting robust label handling in the BERTweet training script.
8. Task 14 trained RoBERTa under the same controlled setup.
9. Task 15 initially had an evaluation function parameter bug and was fixed by rewriting the model evaluation script.
10. Task 16 showed that stopword removal reduced performance.
11. Task 17 selected RoBERTa Version A as the best model by test Macro-F1.
12. Task 18 analyzed the selected model's errors while keeping raw text samples local only.
13. Task 19 formalized final model selection and created a model card.
14. Task 20 created the Streamlit demo.
15. Task 20B created the enhanced workflow dashboard.
16. Task 21 documented ethics and limitations.
17. Task 22 generated final academic documentation.

## Important Practical Lessons

- Reproducibility requires exact preprocessing, not approximate preprocessing.
- Large files should remain local or be stored in a model/data platform rather than GitHub.
- Validation performance may not match held-out test performance.
- Stopword removal should be treated as an experiment, not a default rule.
- UI should communicate the full workflow, not only the final prediction.
- Responsible-use warnings are essential for language understanding tasks.
