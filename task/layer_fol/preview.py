"""Preview helpers for layered FOL task samples."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .eval_inputs import extract_prompt_info_from_row_tokens, split_prompt_row_segments
from .task import FOLLayerTask


def _single_completion_text(tokens: np.ndarray, *, tokenizer) -> str:
    statements = tokenizer.decode_completion_texts(tokens.tolist())
    if len(statements) != 1:
        raise ValueError("Expected a single completion statement.")
    return statements[0]


def _format_single_completion_record(
    task: FOLLayerTask,
    record: dict,
    *,
    role: str,
) -> list[str]:
    tokenizer = task.tokenizer
    if tokenizer is None:
        raise RuntimeError("Task tokenizer is not initialized.")

    prompt = np.asarray(record["prompt"], dtype=np.int32)
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    demo_segments, main_segment = split_prompt_row_segments(prompt, tokenizer=tokenizer)
    sequent = tokenizer.decode_prompt(main_segment.tolist())
    completion_text = _single_completion_text(completion, tokenizer=tokenizer)

    lines = [
        (
            f"[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
            f"prompt_len={prompt.size} completion_len={completion.size} n_demos={len(demo_segments)}"
        ),
        f"  sequent: {sequent.text}",
        f"  completion: {completion_text}",
    ]
    for idx, demo in enumerate(demo_segments):
        demo_text = _single_completion_text(
            np.asarray(list(demo) + [int(tokenizer.eot_token_id)], dtype=np.int32),
            tokenizer=tokenizer,
        )
        lines.append(f"  demo[{idx}]: {demo_text}")
    return lines


def _format_full_completion_record(
    task: FOLLayerTask,
    record: dict,
    *,
    role: str,
) -> list[str]:
    tokenizer = task.tokenizer
    if tokenizer is None:
        raise RuntimeError("Task tokenizer is not initialized.")

    prompt = np.asarray(record["prompt"], dtype=np.int32)
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    _, sequent, _, _ = extract_prompt_info_from_row_tokens(prompt, tokenizer=tokenizer)
    completion_texts = tokenizer.decode_completion_texts(completion.tolist())
    n_demos = len((record.get("rule_context") or {}).get("demo_schema_texts", []))

    lines = [
        (
            f"[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
            f"prompt_len={prompt.size} completion_len={completion.size} "
            f"n_steps={len(completion_texts)} n_demos={n_demos}"
        ),
        f"  prompt: {sequent.text}",
    ]
    for idx, text in enumerate(completion_texts):
        lines.append(f"  completion[{idx}]: {text}")
    return lines


def format_preview_record(task: FOLLayerTask, record: dict, *, role: str) -> str:
    """Return a human-readable preview string for one sampled record."""
    if task.mode != "online":
        raise ValueError("Preview formatting only supports online FOLLayerTask instances.")

    completion_format = str(task.completion_format)
    if completion_format == "single":
        lines = _format_single_completion_record(task, record, role=role)
    elif completion_format == "full":
        lines = _format_full_completion_record(task, record, role=role)
    else:
        raise ValueError(f"Unsupported completion_format={completion_format!r}")
    return "\n".join(lines)


def print_task_preview(
    task: FOLLayerTask,
    *,
    role: str,
    n_examples: int = 3,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Print sampled records from an online task without consuming its iterator."""
    if task.mode != "online":
        raise ValueError("Preview printing only supports online FOLLayerTask instances.")

    total = int(n_examples)
    if total < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}")

    print_fn(f"{role.upper()} DATA PREVIEW ({total} examples)")
    for idx in range(total):
        print_fn("-" * 80)
        print_fn(f"example[{idx}]")
        print_fn(format_preview_record(task, task._sample_online_record(), role=role))
