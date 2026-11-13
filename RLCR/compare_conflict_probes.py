"""Compare completed legacy and reused-gradient probe records by update step."""

import argparse
import json
import math
from pathlib import Path


def read_records(directory):
    path = directory / "conflict_gradients.jsonl"
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["global_step"])] = record
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_dir", type=Path)
    parser.add_argument("reused_dir", type=Path)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()
    legacy = read_records(args.legacy_dir)
    reused = read_records(args.reused_dir)
    if not legacy or set(legacy) != set(reused):
        raise SystemExit(f"Step sets differ: legacy={sorted(legacy)}, reused={sorted(reused)}")
    for step in sorted(legacy):
        a, b = legacy[step], reused[step]
        if a["status"] != b["status"] or a["active_tokens"] != b["active_tokens"]:
            raise SystemExit(f"Step {step}: status or active token count differs")
        for key in ("D_full_on_no_brier", "D_brier_contribution_on_no_brier", "cosine"):
            left, right = a.get(key), b.get(key)
            if left is None and right is None:
                continue
            if left is None or right is None or not math.isclose(left, right, rel_tol=args.rtol, abs_tol=args.atol):
                raise SystemExit(f"Step {step}: {key} differs: {left} vs {right}")
        print(f"Step {step}: matched, D={b.get('D_full_on_no_brier')}, cosine={b.get('cosine')}")
    print("REUSED_GRADIENT_PROBE_COMPARISON_PASSED")


if __name__ == "__main__":
    main()
