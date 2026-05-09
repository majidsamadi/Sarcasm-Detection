# Task 21: Ethics and Limitations

## Objective

Task 21 documents the ethical risks, limitations, and responsible-use requirements for the sarcasm detection system.

This task is important because the model predicts whether text is sarcastic or non-sarcastic, but sarcasm is highly context-dependent. A model prediction should not be treated as a final judgment of the writer's intention.

## Final Model Context

The final system uses:

- Selected model: **RoBERTa Version A**
- Preprocessing: **Version A, stopwords kept**
- Input format: **text-only input**, using `parent_comment + comment` when parent context is available
- Output: **sarcastic** or **non-sarcastic**
- Main evaluation basis: **held-out test macro-F1**

## Core Ethical Position

The system is a coursework and research demonstration. It should not be used to automatically remove, flag, penalize, or judge user content.

## Why This Matters

Sarcasm can depend on:

- conversation history
- social context
- humor style
- culture
- speaker intent
- community norms
- tone that is not visible in text

Therefore, the model may make confident but incorrect predictions.

## Responsible Design Choices

The project includes several safeguards:

1. The UI displays an ethical warning.
2. The model shows confidence and probabilities instead of only a hard label.
3. Raw Reddit error examples are kept local and are not committed to GitHub.
4. The final report includes limitations and bias discussion.
5. Heavy training execution should remain local, especially for public hosting.

## Task 21 Outputs

Task 21 produces:

- ethics and limitations summary
- risk register
- responsible-use checklist
- JSON summary
- risk severity figure
- progress note for the work breakdown tracker

## Academic Value

This task shows that the project does not only focus on model accuracy. It also considers social impact, misuse risk, privacy, transparency, and limitations of NLP systems.
