"""Shared construction and shape helpers for layered FOL tasks."""

from __future__ import annotations

from typing import Iterable

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLDepth3ICLSplitBundle, FOLRuleBank


def _build_tokenizer_for_split_bundle(
    bundle: FOLDepth3ICLSplitBundle,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    max_vars = max(
        int(bundle.train_bank.vars_per_rule_max),
        int(bundle.eval_bank.vars_per_rule_max),
    )
    identifiers: set[str] = set(bundle.train_bank.constants) | set(bundle.eval_bank.constants)
    identifiers.update(bundle.train_bank.predicate_arities)
    identifiers.update(bundle.eval_bank.predicate_arities)
    identifiers.update(f"x{idx}" for idx in range(1, int(max_vars) + 1))
    predicate_identifiers = set(bundle.train_bank.predicate_arities)
    predicate_identifiers.update(bundle.eval_bank.predicate_arities)
    return tokenize_layer_fol.build_tokenizer_from_identifiers(
        sorted(identifiers),
        predicate_identifiers=sorted(predicate_identifiers),
    )


def _fresh_predicate_sentinels(*, name_len: int = 1) -> list[str]:
    """Generate sentinel predicates covering all chars in the fresh predicate charset."""
    from task.layer_gen.util.fol_rule_bank import _FRESH_PREDICATE_CHARSET

    charset = _FRESH_PREDICATE_CHARSET
    sentinels: list[str] = []
    for i in range(0, len(charset), name_len):
        chunk = charset[i : i + name_len].ljust(name_len, charset[0])
        sentinels.append(f"r_{chunk}")
    return sentinels


def _build_tokenizer_for_fresh_icl(
    *,
    base_bank: FOLRuleBank,
    predicate_name_len: int = 1,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    """Build a tokenizer for fresh-ICL that covers all possible fresh predicate chars."""
    identifiers: set[str] = set(base_bank.constants)
    identifiers.update(base_bank.predicate_arities)
    identifiers.update(f"x{idx}" for idx in range(1, int(base_bank.vars_per_rule_max) + 1))

    predicate_identifiers = set(base_bank.predicate_arities)
    sentinels = _fresh_predicate_sentinels(name_len=predicate_name_len)
    identifiers.update(sentinels)
    predicate_identifiers.update(sentinels)

    return tokenize_layer_fol.build_tokenizer_from_identifiers(
        sorted(identifiers),
        predicate_identifiers=sorted(predicate_identifiers),
    )


def compute_fol_dims(
    *,
    rule_banks: Iterable[FOLRuleBank],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    initial_ant_max: int,
    max_n_demos: int = 0,
    completion_format: str = "single",
    completion_steps_max: int = 1,
    extra_predicate_arities: dict[str, int] | None = None,
    fresh_k_in_max: int | None = None,
    fresh_k_out_max: int | None = None,
) -> dict:
    """Compute max sequence-length dimensions from one or more FOL rule banks."""
    all_rules = []
    merged_predicate_arities: dict[str, int] = {}
    first_const: str | None = None
    for bank in rule_banks:
        for rules in bank.transitions.values():
            all_rules.extend(rules)
        merged_predicate_arities.update(bank.predicate_arities)
        if first_const is None and bank.constants:
            first_const = str(bank.constants[0])

    if not all_rules:
        raise ValueError("Rule banks have no rules.")
    if first_const is None:
        raise ValueError("Rule banks have no constants.")

    if extra_predicate_arities:
        for pred, arity in extra_predicate_arities.items():
            if pred not in merged_predicate_arities:
                merged_predicate_arities[pred] = int(arity)

    max_rhs_atoms = max(len(rule.rhs) for rule in all_rules)
    max_prompt_facts = max(int(initial_ant_max), int(max_rhs_atoms))

    max_atom_len = 1
    for predicate, arity in merged_predicate_arities.items():
        atom = FOLAtom(
            predicate=str(predicate),
            args=tuple(first_const for _ in range(int(arity))),
        )
        max_atom_len = max(
            int(max_atom_len),
            len(tokenizer.encode_completion(atom.text)) - 1,
        )

    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1
        + max_atom_len
        + 1
    )

    if int(max_n_demos) > 0:
        max_demo_clause_len = max(
            len(tokenizer.encode_completion(rule.statement_text)) - 1
            for rule in all_rules
        )
        if fresh_k_in_max is not None and fresh_k_out_max is not None:
            fresh_clause_estimate = max_atom_len * (int(fresh_k_in_max) + int(fresh_k_out_max)) + 5
            max_demo_clause_len = max(max_demo_clause_len, fresh_clause_estimate)
        max_prompt_len += int(max_n_demos) * (int(max_demo_clause_len) + 1)

    max_completion_len = max(
        len(tokenizer.encode_completion(rule.statement_text))
        for rule in all_rules
    )
    if fresh_k_in_max is not None and fresh_k_out_max is not None:
        fresh_completion_estimate = max_atom_len * (int(fresh_k_in_max) + int(fresh_k_out_max)) + 5
        max_completion_len = max(max_completion_len, fresh_completion_estimate)

    completion_format = str(completion_format)
    if completion_format not in {"single", "full"}:
        raise ValueError(
            f"completion_format must be 'single' or 'full', got {completion_format!r}"
        )
    completion_steps_max = int(completion_steps_max)
    if completion_steps_max < 1:
        raise ValueError(
            f"completion_steps_max must be >= 1, got {completion_steps_max}"
        )
    if completion_format == "full":
        max_completion_body_len = max(0, int(max_completion_len) - 1)
        max_completion_len = (
            max_completion_body_len * completion_steps_max
            + max(0, completion_steps_max - 1)
            + 1
        )

    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_vocab = max(512, int(tokenizer.vocab_size))

    return {
        "n_vocab": int(n_vocab),
        "max_prompt_len": int(max_prompt_len),
        "max_completion_len": int(max_completion_len),
        "n_seq_ar": int(n_seq_ar),
        "max_atom_len": int(max_atom_len),
    }
