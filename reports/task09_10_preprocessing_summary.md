# Task 09-10 Preprocessing Summary

Generated at: `2026-05-09T10:25:12`

## Purpose

This document records the reproducible preprocessing pipeline used to create `A.csv` and `B.csv` for the SARC sarcasm detection project.

## Version A

- Stopwords are kept.
- Punctuation, casing, and ALL CAPS are kept because they may signal sarcasm.
- Null/empty comments, URL comments, `[deleted]`, `[removed]`, duplicate rows, simple Reddit/Markdown noise, spam-like rows, and overly long rows are handled.
- Output file: `data/processed/A.csv`
- Shape: `[964513, 3]`
- Label balance: `{'label_0': 479571, 'label_1': 484942}`

## Version B

- Starts from Version A.
- Removes common English stopwords.
- Preserves negation words such as `not`, `no`, `never`, `don't`, `isn't`, and `cannot`.
- Output file: `data/processed/B.csv`
- Shape: `[964513, 3]`
- Label balance: `{'label_0': 479571, 'label_1': 484942}`

## Version A Cleaning Steps

| step | before | after | dropped | label_0 | label_1 |
| --- | --- | --- | --- | --- | --- |
| Drop null/empty comment rows | 1010826 | 1010771 | 55 | 505403 | 505368 |
| Drop rows where comment contains URL/markdown hyperlink | 1010771 | 1010564 | 207 | 505226 | 505338 |
| Drop [deleted]/[removed] comments | 1010564 | 1010545 | 19 | 505211 | 505334 |
| Drop remaining underscore-italic rows in comment | 1010545 | 1010312 | 233 | 505059 | 505253 |
| Drop spam/repetitive/single-character rows | 1010312 | 1008680 | 1632 | 503619 | 505061 |
| Drop duplicate rows | 1008680 | 1008071 | 609 | 503320 | 504751 |
| Drop remaining markdown hyperlink rows in comment | 1008071 | 1008071 | 0 | 503320 | 504751 |
| Drop code block rows in comment | 1008071 | 1008069 | 2 | 503319 | 504750 |
| Drop rows empty after final whitespace normalization | 1008069 | 1008069 | 0 | 503319 | 504750 |
| Drop rows above 128 BERTweet tokens | 1008069 | 964513 | 43556 | 479571 | 484942 |

## Academic Note

The two preprocessing versions support the experimental comparison requested by the lecturer: testing whether keeping or removing stopwords affects sarcasm detection performance. Both versions should use the same train/validation/test split in Task 11.

## GitHub Note

`A.csv`, `B.csv`, and split CSV files are local dataset files and should not be pushed to GitHub.