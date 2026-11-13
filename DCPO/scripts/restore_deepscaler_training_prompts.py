#!/usr/bin/env python3
"""Remove DCPO's exact appended confidence directive from a training parquet.

The source parquet is never modified. An optional original parquet can be
provided to verify every restored row, including its ground truth and order.
"""

import argparse
import copy
import os
import tempfile
from pathlib import Path

from add_confidence_prompt_instruction import CONFIDENCE_INSTRUCTION


REPO_ROOT = Path(__file__).resolve().parents[1]


def restore_prompt(prompt):
    """Undo one application of add_confidence_prompt_instruction.py."""
    if isinstance(prompt, str):
        if not prompt.endswith(CONFIDENCE_INSTRUCTION):
            raise ValueError("Prompt does not end with the exact DCPO directive")
        return prompt[: -len(CONFIDENCE_INSTRUCTION)]

    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")

    restored = copy.deepcopy(prompt)
    target = next(
        (
            i
            for i in range(len(restored) - 1, -1, -1)
            if isinstance(restored[i], dict)
            and restored[i].get("role") == "user"
            and isinstance(restored[i].get("content"), str)
        ),
        None,
    )
    if target is None:
        target = next(
            (
                i
                for i in range(len(restored) - 1, -1, -1)
                if isinstance(restored[i], dict)
                and isinstance(restored[i].get("content"), str)
            ),
            None,
        )
    if target is None:
        raise ValueError("Prompt contains no text message")

    content = restored[target]["content"]
    if not content.endswith(CONFIDENCE_INSTRUCTION):
        raise ValueError("Prompt does not end with the exact DCPO directive")
    restored[target]["content"] = content[: -len(CONFIDENCE_INSTRUCTION)]
    return restored


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "deepscaler_uniform_train_with_confidence.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "deepscaler_uniform_train_restored.parquet",
    )
    parser.add_argument("--reference", type=Path, help="Original parquet for row-by-row verification")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow to read and write parquet files") from exc

    args = parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if source == output:
        raise ValueError("Input and output must be different files")
    if args.reference is not None and output == args.reference.resolve(strict=True):
        raise ValueError("Output and reference must be different files")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    table = pq.read_table(source)
    if "prompt" not in table.column_names:
        raise KeyError("Input parquet has no 'prompt' column")
    original_prompts = table.column("prompt").to_pylist()
    restored_prompts = []
    for row_number, prompt in enumerate(original_prompts):
        try:
            restored_prompts.append(restore_prompt(prompt))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_number}: {exc}") from exc

    position = table.schema.get_field_index("prompt")
    restored = table.set_column(
        position,
        table.schema.field(position),
        pa.array(restored_prompts, type=table.schema.field(position).type),
    )

    if args.reference is not None:
        reference = pq.read_table(args.reference.resolve(strict=True))
        if restored.column_names != reference.column_names or restored.num_rows != reference.num_rows:
            raise ValueError("Restored data differs from reference in columns or row count")
        for name in restored.column_names:
            if restored.column(name).to_pylist() != reference.column(name).to_pylist():
                raise ValueError(f"Restored column {name!r} differs from reference")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        pq.write_table(restored, temporary)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    print(f"Restored {restored.num_rows} prompts: {output}")
    if args.reference is not None:
        print(f"All columns and row order match reference: {args.reference}")


if __name__ == "__main__":
    main()
