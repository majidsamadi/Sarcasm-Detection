# Task 19: Final Model Selection

## Objective

The purpose of Task 19 is to select **one final model** for the final sarcasm detection implementation/demo.

## Academic Methodology

The final model is not selected subjectively. It is selected using the controlled experimental results from previous tasks:

1. **Task 15:** Full held-out test evaluation for all four experiments.
2. **Task 16:** Stopword impact analysis comparing Version A and Version B.
3. **Task 17:** Overall model comparison and ranking.
4. **Task 18:** Error analysis of the best candidate model.

## Selection Rule

The final model is selected using the following rule:

1. Rank all four experiments by **held-out test macro-F1**.
2. Use **accuracy** as a tie-breaker only if macro-F1 is tied.
3. Use Task 18 error analysis as supporting evidence to explain limitations and expected failure cases.

Macro-F1 is prioritized because it considers performance across both classes: sarcastic and non-sarcastic.

## Candidate Experiments

| Experiment | Model | Dataset Version | Preprocessing |
|---|---|---|---|
| E01 | BERTweet | Version A | Stopwords kept |
| E02 | BERTweet | Version B | Selective stopword removal |
| E03 | RoBERTa | Version A | Stopwords kept |
| E04 | RoBERTa | Version B | Selective stopword removal |

## Final Implementation

Only the selected final model should be used in the final demo interface. The other models remain useful for academic comparison but should not be deployed in the final app.

The final model configuration is saved at:

```text
configs/final_model_config.json
```

## Ethical Note

The final model is for academic demonstration only. It should not be used to automatically remove, flag, punish, or moderate user-generated content.
