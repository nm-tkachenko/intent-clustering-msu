# Translation folder structure

This folder contains code and data relevant to the 1st project subtask: **automated data translation**.

## Code files
Code used directly for data translation (from English to Russian):
* `translation_omniling_clinc.py`: code used for translation of CLINC150 dataset with OmniLing-V1-8b (8-bit) model and subsequent artifact deletion.
* `translation_omniling_banking.py`: code used for translation of banking-77 dataset with OmniLing-V1-8b (8-bit) model and subsequent artifact deletion.
* `translation_omniling_natural_questions.py`: code used for translation of 2700 examples of natural_questions_clean dataset. These examples are later included in banking-77 translated version as out-of-scope examples.
* `translation_qwen.py`: code used for regeneration of OmniLing-V1-8b translations with low COMET scores (< 0.2) by implementing RuadaptQwen3-8B-Hybrid model.

Code used for the translation quality evaluation (by COMET metric):
* `comet.py`: code for evaluating first translation version by OmniLing-V1-8b (8-bit) model and choosing candidates for regeneration (low COMET score).
* `comet_qwen.py`: code for merging the final translated dataset via comparing original OmniLing translations with RuadaptQwen3-8B-Hybrid alternative translations and choosing the best variant according to the metric score.

## Translation data files
Two following files contain the final dataset translations. Each file contains six splits (`train`, `val`, `test`, `oos_train`, `oos_val`, `oos_test`): every element consists of original dataset entry in English (`text` field),
best translation variant in Russian (`translation` field), original dataset label in English (`label` field) and COMET metric score for translation (`score` field).
* `clinc_qwen2.json`: translation of CLINC150 dataset into Russian.
* `banking_qwen2.json`: translation of banking-77 dataset into Russian augmented by 2700 translated examples from Natural Questions datset.
