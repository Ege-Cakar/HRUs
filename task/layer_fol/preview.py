"""Preview helpers for layered FOL task samples."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .eval_inputs import extract_prompt_info_from_row_tokens
from .task import FOLLayerTask


def _split_prompt_segments(prompt_tokens: np.ndarray, sep_token_id: int) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for tok in prompt_tokens.tolist():
        tok = int(tok)
        if tok == int(sep_token_id):
            segments.append(current)
            current = []
            continue
        current.append(tok)
    if current:
        segments.append(current)
    return segments


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
    segments = _split_prompt_segments(prompt, tokenizer.sep_token_id)
    if not segments:
        raise ValueError("Preview record prompt did not contain a sequent segment.")

    demo_segments = segments[:-1]
    main_segment = segments[-1] + [int(tokenizer.sep_token_id)]
    sequent = tokenizer.decode_prompt(main_segment)
    completion_text = tokenizer.decode_completion_text(completion.tolist())

    lines = [
        (
            f"[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
            f"prompt_len={prompt.size} completion_len={completion.size} n_demos={len(demo_segments)}"
        ),
        f"  sequent: {sequent.text}",
        f"  completion: {completion_text}",
    ]
    for idx, demo in enumerate(demo_segments):
        demo_text = tokenizer.decode_completion_text(list(demo) + [int(tokenizer.eot_token_id)])
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
    completion_texts = tokenizer.decode_completion_sequence_texts(completion.tolist())
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
