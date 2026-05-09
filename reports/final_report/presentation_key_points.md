# Presentation Key Points

## One-Sentence Project Summary

We built an NLP sarcasm detection system using the SARC Reddit dataset, compared BERTweet and RoBERTa under two preprocessing settings, and selected RoBERTa Version A as the final model.

## Best Result

- Final model: E03_RoBERTa_VersionA
- Accuracy: 0.7223
- Macro-F1: 0.7167
- Preprocessing: Stopwords kept

## What We Learned

1. Keeping stopwords worked better than removing them.
2. RoBERTa generalized better than BERTweet on the held-out test set.
3. Sarcasm detection is highly context-dependent.
4. Error analysis showed that misclassification can occur even with high confidence.
5. The demo should be used for academic purposes only, not automatic moderation.

## Expected Q&A

### Why did we keep stopwords?
Because Task 16 showed that stopword removal reduced performance for both models. Sarcasm often depends on sentence structure, contrast, and small function words.

### Why did we choose RoBERTa instead of BERTweet?
RoBERTa Version A achieved the best held-out test Macro-F1 and accuracy among the four controlled experiments.

### Why use parent_comment + comment?
Sarcasm often requires conversational context. Combining parent comment and reply gives the model more context than the reply alone.

### Can the model be used for moderation?
No. The model is a research demo and should not automatically remove, flag, or penalize content.

### What is the biggest limitation?
Sarcasm depends on culture, tone, context, speaker intent, and conversation history, which may be missing from text-only input.
