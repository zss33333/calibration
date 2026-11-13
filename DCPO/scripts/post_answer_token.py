"""Locate the first *generated* token strictly after the final boxed answer.

This module does not tokenize text, append a sentinel, or run a model. Pass the
raw generated token IDs, including an emitted EOS token when available. Token
indices in the result are zero-based indices into that generated-ID sequence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import re
import unicodedata
from typing import Any, Iterable


BOX_START = re.compile(r"\\boxed\s*\{")


@dataclass(frozen=True)
class PostAnswerToken:
    status: str
    target_index: int | None = None
    target_id: int | None = None
    target_text: str | None = None
    token_type: str | None = None
    answer: str | None = None
    box_count: int = 0
    box_start: int | None = None
    box_end: int | None = None
    trailing_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def _escaped(text: str, at: int) -> bool:
    backslashes = 0
    while at - backslashes - 1 >= 0 and text[at - backslashes - 1] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _boxes(text: str) -> tuple[list[tuple[int, int, str]], list[int]]:
    boxes: list[tuple[int, int, str]] = []
    malformed_starts: list[int] = []
    for match in BOX_START.finditer(text):
        if _escaped(text, match.start()):
            continue
        depth = 1
        content_start = match.end()
        for index in range(content_start, len(text)):
            char = text[index]
            if char == "{" and not _escaped(text, index):
                depth += 1
            elif char == "}" and not _escaped(text, index):
                depth -= 1
                if depth == 0:
                    boxes.append((match.start(), index + 1, text[content_start:index]))
                    break
        else:
            malformed_starts.append(match.start())
    return boxes, malformed_starts


def _token_type(token_text: str, token_id: int, eos_ids: set[int]) -> str:
    if token_id in eos_ids:
        return "eos"
    if token_text in ("\n", "\r\n"):
        return "newline"
    if token_text.isspace():
        return "whitespace"
    if token_text and all(unicodedata.category(c).startswith("P") for c in token_text):
        return "punctuation"
    return "text"


def locate_post_answer_token(
    generated_ids: Iterable[int],
    tokenizer: Any,
    *,
    think_end_token_id: int,
    eos_token_ids: Iterable[int] = (),
) -> PostAnswerToken:
    """Find a strict post-box token in a Qwen3 thinking-mode completion.

    A box must occur after the last emitted ``</think>`` token. Its closing
    brace must end exactly on a token boundary. If the closing brace and later
    text share one token, the result is ``boundary_straddled``. The function
    never chooses a later token as a substitute in that case.
    """
    ids = [int(token_id) for token_id in generated_ids]
    try:
        final_start = len(ids) - 1 - ids[::-1].index(think_end_token_id) + 1
    except ValueError:
        return PostAnswerToken(status="missing_think_end")

    final_ids = ids[final_start:]
    final_text = _decode(tokenizer, final_ids)
    boxes, malformed_starts = _boxes(final_text)
    if not boxes:
        return PostAnswerToken(status="unbalanced_box" if malformed_starts else "no_box")
    # Prefer the last *completed* box. This also chooses the outer box when
    # a box is nested inside another box.
    start, end, answer = max(boxes, key=lambda box: box[1])
    if malformed_starts and malformed_starts[-1] > start:
        return PostAnswerToken(status="unbalanced_final_box", box_count=len(boxes))
    fields = dict(
        answer=answer,
        box_count=len(boxes),
        box_start=start,
        box_end=end,
        trailing_text=final_text[end:],
    )

    @lru_cache(maxsize=None)
    def prefix(count: int) -> str:
        return _decode(tokenizer, final_ids[:count])

    # Decode only O(log n) prefixes. Retokenizing the decoded response would
    # silently change token boundaries, especially around special tokens.
    lo, hi = 0, len(final_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(prefix(mid)) < end:
            lo = mid + 1
        else:
            hi = mid
    count = lo
    if count == 0 or count > len(final_ids):
        return PostAnswerToken(status="decode_alignment_error", **fields)
    before, through = prefix(count - 1), prefix(count)
    if not final_text.startswith(before) or not final_text.startswith(through):
        return PostAnswerToken(status="decode_alignment_error", **fields)
    if len(before) >= end:
        return PostAnswerToken(status="decode_alignment_error", **fields)
    if len(through) > end:
        return PostAnswerToken(status="boundary_straddled", **fields)
    if count == len(final_ids):
        return PostAnswerToken(status="no_following_token", **fields)

    target_prefix = prefix(count + 1)
    if not final_text.startswith(target_prefix) or len(target_prefix) <= end:
        return PostAnswerToken(status="decode_alignment_error", **fields)
    target_text = target_prefix[end:]
    target_index = final_start + count
    target_id = ids[target_index]
    return PostAnswerToken(
        status="ok",
        target_index=target_index,
        target_id=target_id,
        target_text=target_text,
        token_type=_token_type(target_text, target_id, set(eos_token_ids)),
        **fields,
    )
