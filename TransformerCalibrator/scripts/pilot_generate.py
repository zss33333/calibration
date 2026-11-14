"""Step 3: audit natural post-answer token coverage on real Qwen3 rollouts.

Run from the TransformerCalibrator project root with ``python -m
scripts.pilot_generate``. This step does not score answers or extract activations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

from scripts.post_answer_token import locate_post_answer_token


def normalized_prompt(prompt: Any) -> str:
    if not isinstance(prompt, (str, list)) and hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        content = prompt
    elif isinstance(prompt, list):
        user = [part.get("content") for part in prompt
                if isinstance(part, dict) and part.get("role") == "user"]
        if len(user) != 1 or not isinstance(user[0], str):
            raise ValueError("Expected exactly one user message with string content")
        content = user[0]
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")
    return " ".join(content.casefold().split())


def selected_rows(frame: Any, count: int, seed: int) -> list[tuple[int, Any]]:
    unique: dict[str, tuple[int, Any]] = {}
    for row_index, prompt in enumerate(frame["prompt"]):
        key = normalized_prompt(prompt)
        unique.setdefault(key, (row_index, prompt))
    if count > len(unique):
        raise ValueError(f"Requested {count} questions, only {len(unique)} unique prompts")
    chosen = random.Random(seed).sample(list(unique.values()), count)
    return sorted(chosen, key=lambda item: item[0])


def messages_for(prompt: Any) -> list[dict[str, str]]:
    if not isinstance(prompt, (str, list)) and hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def sample_seed(base_seed: int, row_index: int, repeat: int) -> int:
    value = f"{base_seed}:{row_index}:{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def summarize(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    status = Counter(record["locator"]["status"] for record in records)
    types = Counter(record["locator"].get("token_type") for record in records
                    if record["locator"]["status"] == "ok")
    finishes = Counter(record["finish_reason"] for record in records)
    lengths = [record["generated_tokens"] for record in records]
    examples: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = record["locator"]["status"]
        bucket = examples.setdefault(key, [])
        if len(bucket) < 5:
            bucket.append({"sample_id": record["sample_id"],
                           "row_index": record["row_index"],
                           "response_tail": record["response_text"][-500:],
                           "target_text": record["locator"].get("target_text")})
    ordered = sorted(lengths)
    def percentile(q: float) -> int | None:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))] if ordered else None
    return {
        "expected_samples": expected,
        "completed_samples": len(records),
        "coverage": (status["ok"] / len(records)) if records else None,
        "status_counts": dict(sorted(status.items())),
        "token_type_counts": dict(sorted(types.items())),
        "finish_reason_counts": dict(sorted(finishes.items())),
        "generated_length": {"min": min(lengths) if lengths else None,
                             "median": statistics.median(lengths) if lengths else None,
                             "p90": percentile(.9), "p99": percentile(.99),
                             "max": max(lengths) if lengths else None},
        "examples_by_status": examples,
    }


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}; inspect before resuming") from exc
            key = record["sample_id"]
            if key in records:
                raise ValueError(f"Duplicate sample ID {key} at {path}:{line_no}")
            records[key] = record
    return records


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data", type=Path,
                         default=Path("../DCPO/data/deepscaler_uniform_train.parquet"))
    command.add_argument("--model", required=True,
                         help="Local Qwen3-8B model directory or Hugging Face model ID")
    command.add_argument("--output-dir", type=Path, default=Path("pilot_output"))
    command.add_argument("--num-questions", type=int, default=400)
    command.add_argument("--samples-per-question", type=int, default=2)
    command.add_argument("--seed", type=int, default=20261114)
    command.add_argument("--temperature", type=float, default=0.6)
    command.add_argument("--top-p", type=float, default=0.95)
    command.add_argument("--top-k", type=int, default=20)
    command.add_argument("--max-context", type=int, default=None,
                         help="Use model config max_position_embeddings when omitted")
    command.add_argument("--check-input-only", action="store_true",
                         help="Inspect and select data without loading Qwen")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.num_questions < 1 or args.samples_per_question < 1:
        raise ValueError("Question and sample counts must be positive")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    import pandas as pd
    frame = pd.read_parquet(args.data)
    if "prompt" not in frame.columns:
        raise KeyError(f"Missing prompt column; available: {list(frame.columns)}")
    chosen = selected_rows(frame, args.num_questions, args.seed)
    plan = [(row, repeat, sample_seed(args.seed, row, repeat), prompt)
            for row, prompt in chosen for repeat in range(args.samples_per_question)]
    print(json.dumps({"source_rows": len(frame), "selected_unique_questions": len(chosen),
                      "planned_samples": len(plan), "first_row_indices": [r for r, _ in chosen[:10]]},
                     ensure_ascii=False), flush=True)
    if args.check_input_only:
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {"data": str(args.data.resolve()), "model": args.model,
                "num_questions": args.num_questions,
                "samples_per_question": args.samples_per_question,
                "seed": args.seed, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k,
                "max_context_override": args.max_context,
                "selected_row_indices": [row for row, _ in chosen]}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing output has a different manifest; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result_path = args.output_dir / "rollouts.jsonl"
    existing = read_existing(result_path)
    expected_ids = {f"row{row}_sample{repeat}" for row, repeat, _, _ in plan}
    if not set(existing).issubset(expected_ids):
        raise ValueError("Existing rollouts contain IDs outside this plan")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=False
    ).eval()
    max_context = args.max_context or int(model.config.max_position_embeddings)
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    if think_end is None or think_end == tokenizer.unk_token_id:
        raise ValueError("Tokenizer lacks </think> token")
    eos_raw = model.generation_config.eos_token_id
    eos_ids = {int(eos_raw)} if isinstance(eos_raw, int) else set(eos_raw or [])
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    if not eos_ids:
        raise ValueError("No EOS token IDs configured")

    interrupted = False
    try:
        input_device = model.get_input_embeddings().weight.device
        with result_path.open("a", encoding="utf-8") as stream:
            for position, (row, repeat, seed, prompt) in enumerate(plan, 1):
                sample_id = f"row{row}_sample{repeat}"
                if sample_id in existing:
                    continue
                encoded = tokenizer.apply_chat_template(
                    messages_for(prompt), tokenize=True, add_generation_prompt=True,
                    enable_thinking=True, return_dict=True, return_tensors="pt"
                )
                prompt_ids = encoded["input_ids"][0].tolist()
                remaining = max_context - len(prompt_ids)
                if remaining < 1:
                    raise ValueError(f"Prompt row {row} already exceeds context {max_context}")
                encoded = {key: value.to(input_device) for key, value in encoded.items()}
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    sequence = model.generate(
                        **encoded, do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k,
                        max_new_tokens=remaining, eos_token_id=sorted(eos_ids),
                        pad_token_id=tokenizer.eos_token_id,
                    )[0]
                generated_ids = sequence[len(prompt_ids):].tolist()
                finish_reason = "eos" if generated_ids and generated_ids[-1] in eos_ids else "context_limit"
                locator = locate_post_answer_token(
                    generated_ids, tokenizer, think_end_token_id=int(think_end),
                    eos_token_ids=eos_ids
                )
                record = {
                    "sample_id": sample_id, "row_index": row, "repeat": repeat,
                    "seed": seed, "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(generated_ids), "finish_reason": finish_reason,
                    "prompt_ids": prompt_ids, "generated_ids": generated_ids,
                    "response_text": tokenizer.decode(generated_ids, skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False),
                    "locator": locator.to_dict(),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                existing[sample_id] = record
                print(f"{position}/{len(plan)} {sample_id}: {locator.status} "
                      f"{locator.token_type or ''} {finish_reason} ({len(generated_ids)} tokens)",
                      flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; completed JSONL records remain resumable", file=sys.stderr)
    finally:
        report = summarize(list(read_existing(result_path).values()), len(plan))
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: report[key] for key in
                          ("completed_samples", "expected_samples", "coverage",
                           "status_counts", "token_type_counts", "finish_reason_counts")},
                         ensure_ascii=False), flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
