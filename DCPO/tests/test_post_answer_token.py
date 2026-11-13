"""Boundary tests for the natural post-answer token locator."""

import unittest

from scripts.post_answer_token import locate_post_answer_token


THINK_END = 900
EOS = 901


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(self.pieces[i] for i in ids)


class PostAnswerTokenTests(unittest.TestCase):
    def locate(self, pieces, ids, **kwargs):
        return locate_post_answer_token(
            ids, PieceTokenizer(pieces), think_end_token_id=THINK_END,
            eos_token_ids={EOS}, **kwargs
        )

    def test_nested_math_and_natural_newline(self):
        pieces = {1: "thinking \\boxed{bad}", THINK_END: "</think>",
                  2: "The answer is \\boxed{\\frac{1}{2}}", 3: "\n", EOS: "<|im_end|>"}
        result = self.locate(pieces, [1, THINK_END, 2, 3, EOS])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.answer, "\\frac{1}{2}")
        self.assertEqual(result.target_index, 3)
        self.assertEqual(result.token_type, "newline")
        self.assertEqual(result.box_count, 1)

    def test_natural_eos_is_selected(self):
        pieces = {THINK_END: "</think>", 1: "\\boxed{42}", EOS: "<|im_end|>"}
        result = self.locate(pieces, [THINK_END, 1, EOS])
        self.assertEqual((result.target_index, result.target_id, result.token_type),
                         (2, EOS, "eos"))

    def test_multiple_boxes_choose_last(self):
        pieces = {THINK_END: "</think>", 1: "\\boxed{3} then \\boxed{4}", 2: "."}
        result = self.locate(pieces, [THINK_END, 1, 2])
        self.assertEqual(result.answer, "4")
        self.assertEqual(result.box_count, 2)
        self.assertEqual(result.token_type, "punctuation")

    def test_boundary_straddled_token_is_not_substituted(self):
        pieces = {THINK_END: "</think>", 1: "\\boxed{42}\n", 2: "next"}
        result = self.locate(pieces, [THINK_END, 1, 2])
        self.assertEqual(result.status, "boundary_straddled")
        self.assertIsNone(result.target_index)

    def test_no_following_token(self):
        pieces = {THINK_END: "</think>", 1: "\\boxed{42}"}
        result = self.locate(pieces, [THINK_END, 1])
        self.assertEqual(result.status, "no_following_token")

    def test_missing_think_end_and_box(self):
        pieces = {1: "\\boxed{42}", THINK_END: "</think>", 2: "42", EOS: "<|im_end|>"}
        self.assertEqual(self.locate(pieces, [1, EOS]).status, "missing_think_end")
        self.assertEqual(self.locate(pieces, [THINK_END, 2, EOS]).status, "no_box")

    def test_unbalanced_box(self):
        pieces = {THINK_END: "</think>", 1: "\\boxed{\\frac{1}{2}"}
        self.assertEqual(self.locate(pieces, [THINK_END, 1]).status, "unbalanced_box")

    def test_escaped_braces_do_not_close_box(self):
        pieces = {THINK_END: "</think>", 1: r"\boxed{\{x\}}", 2: " "}
        result = self.locate(pieces, [THINK_END, 1, 2])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.answer, r"\{x\}")
        self.assertEqual(result.token_type, "whitespace")

    def test_incomplete_last_box_does_not_fall_back_to_earlier_answer(self):
        pieces = {THINK_END: "</think>", 1: r"\boxed{3} then \boxed{4"}
        result = self.locate(pieces, [THINK_END, 1])
        self.assertEqual(result.status, "unbalanced_final_box")

    def test_box_marker_split_across_tokens(self):
        pieces = {THINK_END: "</think>", 1: r"\box", 2: "ed{7}", 3: "\n"}
        result = self.locate(pieces, [THINK_END, 1, 2, 3])
        self.assertEqual((result.status, result.target_index, result.answer),
                         ("ok", 3, "7"))

    def test_nested_box_chooses_completed_outer_box(self):
        pieces = {THINK_END: "</think>", 1: r"\boxed{\boxed{4}}", 2: "\n"}
        result = self.locate(pieces, [THINK_END, 1, 2])
        self.assertEqual((result.status, result.answer, result.target_index),
                         ("ok", r"\boxed{4}", 2))


if __name__ == "__main__":
    unittest.main()
