"""Summarize every logged optimizer update from an online RLCR diagnosis."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    log_path = args.output_dir / "conflict_gradients.jsonl"
    state_path = args.output_dir / "trainer_state.json"
    records = {}
    duplicates = 0
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("reward_scope") != "weighted_without_brier_vs_weighted_full":
                raise ValueError("Diagnostic log contains a different gradient definition")
            step = int(record["global_step"])
            if step in records:
                duplicates += 1
            records[step] = record  # Last completed run wins after a resume.

    expected_steps = None
    if state_path.exists():
        with state_path.open(encoding="utf-8") as stream:
            expected_steps = int(json.load(stream)["global_step"])

    valid = [r for r in records.values() if r["status"] == "ok"]
    active = [r for r in valid if r["brier_weight"] > 0]
    negative = sum(r["D_brier_contribution_on_no_brier"] < 0 for r in active)
    full_reversal = sum(r["D_full_on_no_brier"] < 0 for r in valid)
    attenuated = sum(0 < r["D_full_on_no_brier"] < 1 for r in valid)
    status_counts = {}
    for record in records.values():
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    result = {
        "gradient_pair": "weighted_without_brier_vs_weighted_full",
        "expected_optimizer_steps": expected_steps,
        "logged_unique_steps": len(records),
        "complete_coverage": (
            set(records) == set(range(1, expected_steps + 1))
            if expected_steps is not None else None
        ),
        "duplicate_log_rows": duplicates,
        "missing_steps": (
            sorted(set(range(1, expected_steps + 1)) - records.keys())
            if expected_steps is not None else None
        ),
        "status_counts": status_counts,
        "valid_steps": len(valid),
        "brier_active_steps": len(active),
        "brier_conflict_steps": negative,
        "brier_conflict_frequency": negative / len(active) if active else None,
        "full_reversal_steps": full_reversal,
        "full_reversal_frequency": full_reversal / len(valid) if valid else None,
        "attenuated_projection_steps": attenuated,
    }
    if expected_steps:
        phases = {}
        for phase_index, phase_name in enumerate(("early", "middle", "late")):
            phase_records = [
                r for step, r in records.items()
                if min(3 * (step - 1) // expected_steps, 2) == phase_index
            ]
            phase_valid = [r for r in phase_records if r["status"] == "ok"]
            phase_active = [r for r in phase_valid if r["brier_weight"] > 0]
            phase_conflicts = sum(
                r["D_brier_contribution_on_no_brier"] < 0 for r in phase_active
            )
            phases[phase_name] = {
                "logged_steps": len(phase_records),
                "valid_steps": len(phase_valid),
                "conflict_steps": phase_conflicts,
                "brier_conflict_frequency": (
                    phase_conflicts / len(phase_active) if phase_active else None
                ),
                "median_D_full_on_no_brier": (
                    statistics.median(r["D_full_on_no_brier"] for r in phase_valid)
                    if phase_valid else None
                ),
            }
        result["phases"] = phases
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
