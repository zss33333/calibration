"""Reproduce the DCPO verbal-confidence row of the math evaluation table.

Run this against a vLLM OpenAI-compatible server. Each dataset/repetition has
its own append-only JSONL file so an interrupted job can resume safely.
"""

import argparse
import csv
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import roc_auc_score


DATASETS = {
    "MATH-500": ("MATH-500/test.jsonl", 500),
    "AIME24": ("aime24.parquet", 30),
    "AIME25": ("aime25/aime2025.jsonl", 30),
    "AMC23": ("AMC/data/amc23-00000-of-00001.parquet", 45),
    "AMC24": ("AMC/data/amc24-00000-of-00001.parquet", 45),
}
PAPER_DIRECTIVE = (
    "\n\nPlease put your final answer within \\boxed{}. "
    "Also output your confidence for this answer after <conf> as a floating-point "
    "number between 0 (complete uncertainty) and 1 (full confidence). "
    "Example: <conf> Confidence: 0.83 Ensure that the confidence output appears "
    "on a single line and strictly follows the specified format."
)
CONF_RE = re.compile(r"<conf>\s*Confidence:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.I)
TRAINED_DIRECTIVE = (
    "\n\nPlease put your final answer within \\boxed{}.\n"
    "Also output a single line at the end of the answer: CONFIDENCE: "
    "<float number between 0 and 1>\nExample: CONFIDENCE: 0.83\n"
)
TRAINED_CONF_RE = re.compile(r"CONFIDENCE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.I)


def load_dataset(name, root):
    relative_path, expected = DATASETS[name]
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    if name in ("MATH-500", "AIME25"):
        with path.open(encoding="utf-8") as stream:
            source = [json.loads(line) for line in stream if line.strip()]
    else:
        source = pd.read_parquet(path).to_dict("records")
    if len(source) < expected:
        raise ValueError(f"{name}: expected at least {expected} rows, found {len(source)}")
    source = source[:expected]
    items = []
    for index, row in enumerate(source):
        if name == "MATH-500":
            problem, reference = row["problem"], row.get("solution") or row["answer"]
        elif name == "AIME24":
            problem = row["prompt"][0]["content"]
            reference = str(row["reward_model"]["ground_truth"])
        elif name == "AIME25":
            problem, reference = row["question"], str(row["answer"])
        else:
            problem, reference = row["problem"], str(row["answer"])
        items.append({"index": index, "problem": str(problem), "reference": str(reference)})
    return items


def read_records(path):
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                index = record["index"]
                if index in records:
                    raise ValueError(f"Duplicate index {index} in {path}")
                records[index] = record
    return records


def audit_confidence(output_dir):
    """Inspect saved responses without launching a model or changing records."""
    paths = sorted(output_dir.glob("*_run*.jsonl"))
    if not paths:
        raise ValueError(f"No run JSONL files found in {output_dir}")
    report = []
    for path in paths:
        records = read_records(path)
        bad = []
        for record in records.values():
            if record.get("confidence") is not None:
                continue
            raw = record.get("response") or ""
            alternate = TRAINED_CONF_RE.findall(raw)
            bad.append({
                "index": record["index"],
                "alternate_confidence": alternate[-1] if alternate else None,
                "has_conf_tag": "<conf>" in raw.lower(),
                "finish_reason": record.get("finish_reason"),
                "completion_tokens": record.get("completion_tokens"),
                "response_chars": len(raw),
                "response_tail": raw[-180:],
            })
        report.append({"file": path.name, "records": len(records),
                       "missing_confidence": len(bad), "examples": bad})
        print(f"{path.name}: {len(records)} records, {len(bad)} invalid confidence", flush=True)
        for item in bad:
            print(f"  index={item['index']} alternate={item['alternate_confidence']} "
                  f"finish={item['finish_reason']} tokens={item['completion_tokens']} "
                  f"tail={item['response_tail']!r}", flush=True)
    audit_path = output_dir / "confidence_audit.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit saved: {audit_path}", flush=True)


def generate(item, args, seed):
    directive = PAPER_DIRECTIVE if args.prompt_format == "paper" else TRAINED_DIRECTIVE
    pattern = CONF_RE if args.prompt_format == "paper" else TRAINED_CONF_RE
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": item["problem"] + directive}],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 3000,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(args.url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    result = response.json()
    choice = result["choices"][0]
    raw = choice["message"]["content"] or ""
    matches = list(pattern.finditer(raw))
    confidence = float(matches[-1].group(1)) if matches else None
    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
        confidence = None
    answer = raw[:matches[-1].start()].strip() if matches else raw.strip()
    return {
        "index": item["index"],
        "seed": seed,
        "problem": item["problem"],
        "reference": item["reference"],
        "response": raw,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
        "answer": answer,
        "confidence": confidence,
    }


def metrics(records, bins, invalid_policy="strict"):
    if not records:
        raise ValueError("No records to summarize")
    bad = [r["index"] for r in records if r["confidence"] is None]
    if bad and invalid_policy == "strict":
        raise ValueError(f"Missing/invalid confidence at indices {bad[:20]} ({len(bad)} total)")
    # The released calibration_main.py assigns -0.1 when no verbal confidence
    # is found. Its ECE/PCE bins exclude that value but still divide by all N;
    # AUROC includes it. Keep the raw JSONL confidence as null for auditing.
    conf = [-0.1 if r["confidence"] is None else r["confidence"] for r in records]
    correct = [r["correct"] for r in records]
    n = len(records)
    ece = pce = 0.0
    for bin_index in range(bins):
        group = [i for i, c in enumerate(conf) if (c >= 0 if bin_index == 0 else c > bin_index / bins)
                 and c <= (bin_index + 1) / bins]
        if not group:
            continue
        mean_c = sum(conf[i] for i in group) / len(group)
        mean_a = sum(correct[i] for i in group) / len(group)
        ece += len(group) / n * abs(mean_c - mean_a)
        pce += len(group) / n * max(mean_c - mean_a, 0)
    auroc = roc_auc_score(correct, conf) if len(set(correct)) == 2 else None
    return {"n": n, "invalid_confidence": len(bad),
            "acc": sum(correct) / n, "ece": ece, "pce": pce,
            "auroc": auroc}


def summarize_existing(args, datasets):
    """Summarize complete runs already on disk without contacting the model."""
    rows = []
    for name, items in datasets.items():
        expected = {item["index"] for item in items}
        for repeat in range(1, args.repeats + 1):
            path = args.output_dir / f"{name}_run{repeat}.jsonl"
            records = read_records(path)
            if not records:
                continue
            if set(records) != expected:
                print(f"Skipping incomplete {path}: {len(records)}/{len(expected)} records", flush=True)
                continue
            if any(record.get("model") != args.model or
                   record.get("prompt_format") != args.prompt_format for record in records.values()):
                raise ValueError(f"Model or prompt format mismatch: {path}")
            rows.append({"dataset": name, "run": repeat,
                         **metrics(list(records.values()), args.bins, args.invalid_policy)})
    if not rows:
        raise ValueError("No complete runs found; check JSONL line counts first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "partial_per_run.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "run", "n", "invalid_confidence", "acc", "ece", "pce", "auroc"])
        writer.writeheader()
        writer.writerows(rows)
    partial = []
    for name in DATASETS:
        current = [row for row in rows if row["dataset"] == name]
        if current:
            partial.append({"dataset": name, "runs_completed": len(current),
                            "n_per_run": current[0]["n"],
                            "invalid_confidence_total": sum(row["invalid_confidence"] for row in current),
                            **{key: sum(row[key] for row in current) / len(current)
                               if all(row[key] is not None for row in current) else None
                               for key in ("acc", "ece", "pce", "auroc")}})
    with (args.output_dir / "partial_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "runs_completed", "n_per_run", "invalid_confidence_total",
                                                    "acc", "ece", "pce", "auroc"])
        writer.writeheader()
        writer.writerows(partial)
    print(f"Summarized {len(rows)} complete runs; partial results: "
          f"{args.output_dir / 'partial_summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("math_table_results"))
    parser.add_argument("--model", default="dcpo-step120")
    parser.add_argument("--prompt-format", choices=("paper", "trained"), default="paper")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--invalid-policy", choices=("strict", "author"), default="strict",
                        help="strict rejects missing confidence; author uses the released evaluator's -0.1 sentinel")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260310,
                        help="Base seed; each dataset, run and question gets a distinct seed")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="Recompute metrics from complete JSONL files without calling the model")
    parser.add_argument("--audit-confidence", action="store_true",
                        help="Inspect missing confidence in saved JSONL without using GPU")
    parser.add_argument("--summarize-existing", action="store_true",
                        help="Write partial summaries for complete runs already on disk, without GPU")
    args = parser.parse_args()
    if args.repeats < 1 or args.bins < 1 or args.workers < 1:
        parser.error("repeats, bins and workers must be positive")
    if args.audit_confidence:
        audit_confidence(args.output_dir)
        return
    datasets = {name: load_dataset(name, args.data_root) for name in DATASETS}
    for name, items in datasets.items():
        print(f"{name}: {len(items)} questions", flush=True)
    if args.check_data_only:
        print("EVAL_DATA_CHECK_PASSED", flush=True)
        return
    if args.summarize_existing:
        summarize_existing(args, datasets)
        return
    # OpenCompass registers SIGALRM while importing datasets. Import and score
    # on the main thread; worker threads only make HTTP generation requests.
    evaluator = None
    if not args.summary_only:
        from opencompass.evaluator import MATHVerifyEvaluator
        evaluator = MATHVerifyEvaluator()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run = []
    for dataset_number, (name, items) in enumerate(datasets.items()):
        for repeat in range(1, args.repeats + 1):
            path = args.output_dir / f"{name}_run{repeat}.jsonl"
            records = read_records(path)
            if any(record.get("prompt_format") != args.prompt_format or
                   record.get("model") != args.model for record in records.values()):
                raise ValueError(f"Existing records use another model/prompt format: {path}")
            if any(record.get("seed") != args.seed + dataset_number * 1000000 + repeat * 10000 + index
                   for index, record in records.items()):
                raise ValueError(f"Existing records use another seed scheme: {path}")
            pending = [item for item in items if item["index"] not in records]
            print(f"{name} run {repeat}: {len(pending)} pending", flush=True)
            if args.summary_only and pending:
                raise ValueError(f"Cannot summarize incomplete result file: {path}")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(generate, item, args,
                                    args.seed + dataset_number * 1000000 + repeat * 10000 + item["index"]): item
                    for item in pending
                }
                with path.open("a", encoding="utf-8") as stream:
                    for future in as_completed(futures):
                        record = future.result()
                        score = evaluator.score([record["answer"]], [record["reference"]])
                        record["correct"] = int(round(score["accuracy"] / 100))
                        record.update({"dataset": name, "run": repeat, "model": args.model,
                                       "prompt_format": args.prompt_format})
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
            records = read_records(path)
            if set(records) != {item["index"] for item in items}:
                raise ValueError(f"Incomplete result file: {path}")
            row = {"dataset": name, "run": repeat,
                   **metrics(list(records.values()), args.bins, args.invalid_policy)}
            per_run.append(row)
            print(row, flush=True)
    with (args.output_dir / "per_run.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "run", "n", "invalid_confidence", "acc", "ece", "pce", "auroc"])
        writer.writeheader()
        writer.writerows(per_run)
    summary = []
    for name in DATASETS:
        rows = [row for row in per_run if row["dataset"] == name]
        summary.append({"dataset": name, "n_per_run": rows[0]["n"],
                        "invalid_confidence_total": sum(row["invalid_confidence"] for row in rows),
                        **{key: sum(row[key] for row in rows) / len(rows)
                           if all(row[key] is not None for row in rows) else None
                           for key in ("acc", "ece", "pce", "auroc")}})
    (args.output_dir / "summary.json").write_text(
        json.dumps({"model": args.model, "prompt_format": args.prompt_format,
                    "repeats": args.repeats, "bins": args.bins, "base_seed": args.seed,
                    "invalid_policy": args.invalid_policy,
                    "aggregation": "mean of per-run metrics", "datasets": summary},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "n_per_run", "invalid_confidence_total", "acc", "ece", "pce", "auroc"])
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)
    print("MATH_TABLE_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
