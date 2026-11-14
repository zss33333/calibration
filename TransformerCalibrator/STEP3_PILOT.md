# Step 3: Qwen3 thinking-mode token-location pilot

This pilot checks the Step 2 locator on real generated Qwen3-8B token IDs. It
does **not** score answers, extract hidden states, train the calibrator, or
alter the original DeepScaleR file.

## Requirements on the GPU machine

- The `TransformerCalibrator` project, the original DCPO
  `data/deepscaler_uniform_train.parquet`, and a local Qwen3-8B model directory.
- Python with `torch`, `transformers >= 4.51`, `accelerate`, `pandas`, and a
  Parquet engine such as `pyarrow`.
- Enough GPU memory for Qwen3-8B and its KV cache at the context length used.

Run commands **from the TransformerCalibrator directory**. The data and model
paths below are examples; pass the actual paths on the GPU machine.

```bash
python -m unittest tests.test_post_answer_token tests.test_pilot_generate -v
python -m scripts.pilot_generate \
  --data ../DCPO/data/deepscaler_uniform_train.parquet \
  --model /path/to/Qwen3-8B \
  --num-questions 400 --samples-per-question 2 \
  --output-dir pilot_output --check-input-only
python -m scripts.pilot_generate \
  --data ../DCPO/data/deepscaler_uniform_train.parquet \
  --model /path/to/Qwen3-8B \
  --num-questions 400 --samples-per-question 2 \
  --output-dir pilot_output
```

Start with `--num-questions 5 --samples-per-question 1 --output-dir pilot_smoke`
if the machine or model installation is new. Once the five real generations
and their raw token IDs look correct, launch the 400-question pilot above.

The script uses Qwen3's thinking template, temperature 0.6, top-p 0.95, and
top-k 20. It sets no short fixed output cap: each request may generate until
EOS or until the configured context is exhausted. By default, context length
comes from the model config. Use `--max-context` only to match a different
verified service limit. Context exhaustion is recorded as `context_limit`.
No synthetic answer-adjacent token is inserted.

## Files and resuming

- `pilot_output/manifest.json`: fixed selection and generation settings.
- `pilot_output/rollouts.jsonl`: one append-only record per completed sample,
  including prompt/generated token IDs, raw decoded output, stop reason, and
  Step 2 locator result.
- `pilot_output/summary.json`: coverage, status counts, target-token types,
  output-length distribution, and up to five examples per locator status.

Rerun the same command after interruption to skip completed sample IDs. A
different dataset path, model argument, sample plan, or sampling setting must
use a new output directory. If the last JSONL line is malformed after a crash,
inspect and repair it before resuming; the script will not silently skip it.

## Review before Step 4

Inspect the counts and representative response tails for `ok`,
`missing_think_end`, `no_box`, `boundary_straddled`, and `no_following_token`.
For `ok`, compare `target_index` with the saved `generated_ids` and confirm the
selected token is strictly after the closing box. Keep `eos` and `newline`
counts separate: only a separately generated newline matches the paper's PANL
token type. Review `context_limit` records and long-output quantiles. Do not
infer answer correctness from this pilot; that belongs to a later step.
