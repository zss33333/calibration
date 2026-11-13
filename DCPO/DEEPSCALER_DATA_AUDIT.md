# DCPO DeepScaleR training data audit

Audited the files published in [`icip-cas/DCPO/data`](https://github.com/icip-cas/DCPO/tree/master/data). This is a read-only audit; no training rows were changed or removed.

## Provenance and structure

| File | Rows | Git blob SHA |
| --- | ---: | --- |
| `data/deepscaler_uniform_train.parquet` | 10,000 | `ce78a01254afc74827f85ef0278195476cfe3b0e` |
| `data/deepscaler_uniform_train_with_confidence.parquet` | 10,000 | `0e5c987e51446016bf9e0abfa0db040ed26bf68e` |

Both local files match these published Git blob SHAs. Removing the exact directive defined in `scripts/add_confidence_prompt_instruction.py` from the published `with_confidence` file yields all 10,000 rows of the published original file, with every column value and row order equal. The restored file is `data/deepscaler_uniform_train_official_restored.parquet`.

The original columns are `data_source`, `prompt`, `ability`, `reward_model`, and `extra_info`. All prompts are single `user` messages and already end with an instruction to put the final answer in `\boxed{}`. None contains the appended `CONFIDENCE:` directive. Every row has a nonempty prompt and `reward_model.ground_truth`.

## Repeated questions and labels

After stripping the shared answer-format suffix and normalizing case and whitespace, 10,000 rows contain **7,598 distinct question texts**. There are **2,402 rows beyond the first occurrence of their question**. `extra_info.index` has 7,662 distinct values, so it should not be used alone as a unique question key.

Seven repeated-question groups have more than one distinct ground-truth string:

| `extra_info.index` values | Ground-truth strings | Assessment |
| --- | --- | --- |
| 15891, 12786, 12597 | `65 \frac{4}{5}`; `65\frac{4}{5}` | Formatting difference |
| 7289, 1093 | `17.67`; `\frac{53}{3}` | Rounding difference; review |
| 14108, 11208, 1575, 13535 | `24`; `47` | Conflicting answers; review |
| 1148, 8606, 7438, 7438, 10532 | `24`; `25`; `26` | Conflicting answers; review |
| 3172, 4289 | `400`; `500` | Conflicting answers; review |
| 868, 5976 | `-151`; `51` | Conflicting answers; review |
| 11254, 1522 | `\dfrac{1}{8}`; `\frac{5}{32}` | Conflicting answers; review |

Do not split repeats of one question across training, model-selection, and probability-calibration partitions. Review or exclude unresolved conflicting-label groups before generating labels for the calibrator.

## Overlap with the five DCPO test sets

The test files were obtained from the same public repository. We used the first 45 AMC23 rows, as specified by the local DCPO evaluation script (the underlying AMC23 file contains 46 rows). For exact matching, we removed the common answer-format suffix and normalized Unicode, case, and whitespace. For candidate near matches, we used character 3–5-gram TF-IDF cosine similarity; this score is a screening tool, not proof of semantic equivalence.

| Test set | Evaluated questions | Exact training-question matches | Nearest pairs with similarity ≥ 0.80 |
| --- | ---: | ---: | ---: |
| MATH-500 | 500 | 0 | 1 |
| AIME24 | 30 | 0 | 0 |
| AIME25 | 30 | 0 | 0 |
| AMC23 | 45 | 0 | 0 |
| AMC24 | 45 | 0 | 0 |

The one high-similarity pair is MATH-500 test row 123 and training row 2851 (cosine similarity 0.9425). They share the same diagram and coordinates, but the training question asks for the sum of the coordinates of point F, while the test question asks for the area of quadrilateral DBEF. At a lower threshold of 0.70, 11 MATH-500 questions have candidate matches, mostly analogous questions or altered numbers. For strict isolation, review these pairs before deciding whether to exclude related training questions. No test set labels were used to tune any model.

## Status

The **source-data audit is complete**. Data cleaning, question-level split creation, rollout generation, hidden-state extraction, and calibrator training have **not** begun.
