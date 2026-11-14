"""CPU-only tests for pilot selection, resumability, and audit summary."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.pilot_generate import (messages_for, normalized_prompt, read_existing,
                                    sample_seed, selected_rows, summarize)


class PilotGenerateTests(unittest.TestCase):
    def test_prompt_normalization_and_unique_question_selection(self):
        prompts = [
            [{"role": "user", "content": "Find X.  Put answer in \\boxed{}."}],
            [{"role": "user", "content": "find x. put answer in \\boxed{}."}],
            [{"role": "user", "content": "Find Y."}],
        ]
        self.assertEqual(normalized_prompt(prompts[0]), normalized_prompt(prompts[1]))
        chosen = selected_rows({"prompt": prompts}, 2, seed=7)
        self.assertEqual({row for row, _ in chosen}, {0, 2})
        self.assertEqual(messages_for(prompts[0]), prompts[0])

    def test_sample_seeds_are_stable_and_distinct(self):
        self.assertEqual(sample_seed(8, 10, 0), sample_seed(8, 10, 0))
        self.assertNotEqual(sample_seed(8, 10, 0), sample_seed(8, 10, 1))

    def test_summary_counts_coverage_and_token_types(self):
        records = [
            {"sample_id": "a", "row_index": 0, "response_text": "\\boxed{1}\n",
             "generated_tokens": 10, "finish_reason": "eos",
             "locator": {"status": "ok", "token_type": "newline", "target_text": "\n"}},
            {"sample_id": "b", "row_index": 1, "response_text": "\\boxed{2}",
             "generated_tokens": 20, "finish_reason": "context_limit",
             "locator": {"status": "no_following_token", "token_type": None}},
        ]
        report = summarize(records, expected=4)
        self.assertEqual(report["completed_samples"], 2)
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["token_type_counts"], {"newline": 1})
        self.assertEqual(report["finish_reason_counts"]["context_limit"], 1)

    def test_resume_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rollouts.jsonl"
            row = json.dumps({"sample_id": "a"})
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
                read_existing(path)


if __name__ == "__main__":
    unittest.main()
