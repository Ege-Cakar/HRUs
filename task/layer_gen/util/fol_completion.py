"""Shared helpers for layered-FOL completion formatting and sampling."""

from __future__ import annotations


def sampled_completion_texts(
    *,
    sampled,
    step_idx: int,
    completion_format: str,
) -> list[str]:
    step_idx = int(step_idx)
    completion_format = str(completion_format)
    if completion_format == "single":
        return [str(sampled.step_rules[step_idx].statement_text)]
    if completion_format == "full":
        return [str(rule.statement_text) for rule in sampled.step_rules[step_idx:]]
    raise ValueError(
        f"completion_format must be 'single' or 'full', got {completion_format!r}"
    )
