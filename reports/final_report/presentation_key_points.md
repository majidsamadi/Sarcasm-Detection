
# Presentation Key Points

## One-Sentence Project Summary

This project built a sarcasm detection system using the SARC Reddit dataset and compared BERTweet and RoBERTa under two preprocessing settings.

## Main Methodology

- Dataset: SARC / Reddit comments
- Task: binary classification, sarcastic vs non-sarcastic
- Input: `parent_comment + comment`
- Models: BERTweet and RoBERTa
- Preprocessing versions: stopwords kept vs selective stopword removal
- Metrics: accuracy, precision, recall, macro-F1, weighted-F1, confusion matrix

## Final Result

- Final model: RoBERTa Version A
- Stopwords: kept
- Test accuracy: 0.7223
- Test macro-F1: 0.7167

## Main Finding

Stopword removal reduced performance for both BERTweet and RoBERTa. Keeping stopwords helped because sarcasm depends on context, contrast, and sentence structure.

## Important Limitation

The model should not be used for automatic moderation or punitive decisions because sarcasm is context-dependent and can be misclassified.

## Demo Talking Point

The enhanced dashboard shows the full AI/NLP workflow, including preprocessing, model comparison, reports, final prediction demo, and ethics/limitations.
