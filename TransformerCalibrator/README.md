# Transformer Correctness Calibrator

This project is separate from DCPO. It uses DCPO's original DeepScaleR questions and Qwen3-8B model by path; it does not copy their data or weights.

- Source questions: `../DCPO/data/deepscaler_uniform_train.parquet`
- Data audit: `DEEPSCALER_DATA_AUDIT.md`
- Natural post-answer token specification: `POST_ANSWER_TOKEN_SPEC.md`
- Token locator: `scripts/post_answer_token.py`
- Boundary tests: `tests/test_post_answer_token.py`

From this directory, run `python -m unittest tests.test_post_answer_token -v` to check the completed token locator. Model weights are configured in the environment where the pilot runs; no local model path has been verified here.

Current status: source-data audit and token locator are complete. Pilot generation, hidden-state extraction, and calibrator training have not started.
Step 3 pilot generation and token-coverage audit: see STEP3_PILOT.md. Run from this directory with python -m scripts.pilot_generate.

