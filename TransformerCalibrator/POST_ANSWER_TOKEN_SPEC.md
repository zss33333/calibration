# Step 2: natural post-answer token

This step defines a position in a **generated Qwen3 thinking-mode token-ID
sequence**. It does not generate answers, score correctness, append a sentinel,
or extract hidden states. The source questions remain the audited official DCPO
DeepScaleR prompts without the appended confidence directive.

## Input contract

- Pass raw generated token IDs in their original order. Preserve an emitted
  `<|im_end|>` ID if the generation backend returns one. Text-only responses
  cannot establish exact token boundaries.
- Pass the matching tokenizer, the `</think>` token ID, and emitted EOS/turn-end
  token IDs. Do not retokenize decoded response text.
- Use `skip_special_tokens=False` and
  `clean_up_tokenization_spaces=False` throughout alignment.
- The returned `target_index` is zero-based **within generated IDs**, excluding
  prompt IDs. Add the actual prompt length before indexing a full-sequence
  hidden-state tensor.

## Selection rule

1. Search only after the **last emitted `</think>` token**. A missing thinking
   boundary is reported; a box appearing in the reasoning trace is not used.
2. Parse every `\boxed{...}` in the final response, balancing nested braces and
   ignoring escaped `\{` and `\}`. Select the last completed box by its end
   offset. If a later box is incomplete, report `unbalanced_final_box` rather
   than falling back to an earlier answer.
3. Let `box_end` be the character offset immediately after the selected box's
   closing `}`. The target is the first actually generated token whose decoded
   span starts **exactly at** `box_end`. If the closing `}` and later text share
   one token, report `boundary_straddled`; do not use that token or its successor.
4. A naturally emitted EOS/turn-end token can be the target. It is classified
   as `eos`, never as PANL. Only a separate generated newline token is `newline`
   and comparable to the paper's post-answer-newline position.
5. Preserve text following the selected box in `trailing_text` for auditing.
   This locator does not decide whether trailing prose makes the answer invalid.

## Result statuses

| Status | Meaning | `target_index` |
| --- | --- | --- |
| `ok` | Strictly aligned natural post-box token found | Present |
| `missing_think_end` | No generated `</think>` ID | Absent |
| `no_box` | No complete `\boxed{...}` after thinking | Absent |
| `unbalanced_box` | Box opened but not closed | Absent |
| `unbalanced_final_box` | A later box is incomplete | Absent |
| `boundary_straddled` | Closing brace and post-answer text share one token | Absent |
| `no_following_token` | Box closes at the end of supplied IDs | Absent |
| `decode_alignment_error` | Tokenizer prefix decoding cannot establish the boundary | Absent |

The parser returns the selected answer text, box count, box span, target token
ID/text/type, and trailing text when available. Offsets in `box_start` and
`box_end` are Python character offsets within the decoded final-response
segment, not within the entire generated sequence.

## Hand-checked cases

| Generated final response | Expected outcome |
| --- | --- |
| `\boxed{\frac{1}{2}}` then a separate `\n` | Target is the newline token; answer is `\frac{1}{2}` |
| `\boxed{42}` then emitted `<|im_end|>` | Target is EOS, type `eos` |
| `\boxed{42}\n` inside one token | `boundary_straddled` |
| `\boxed{3} ... \boxed{4}` then `.` | Last completed box chosen; target is `.` |
| `\boxed{3} ... \boxed{4` | `unbalanced_final_box` |
| `\boxed{42}` at end of supplied IDs | `no_following_token` |
| `\boxed{\{x\}}` then a space token | Answer is `\{x\}`; target is whitespace |

`python -m unittest tests.test_post_answer_token -v` runs the synthetic
boundary tests. A later pilot with **real Qwen3 generated IDs** must check
coverage and tokenizer alignment before large-scale data generation; these
synthetic tests cannot establish actual token-type frequencies.
