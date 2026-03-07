"""Evaluation and rollout utilities for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.decode_result import DecodeAttempt
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    FOLSequent,
    parse_atom_text,
    parse_clause_text,
    sample_fol_problem,
)

def completion_is_valid_for_layer_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    completion_format: str = "single",
    prompt_tokens: list[int] | np.ndarray | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> bool:
    completion_format = str(completion_format)
    if completion_format == "single":
        result = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion_tokens,
            tokenizer=tokenizer,
            active_rules=active_rules,
            fixed_rules=fixed_rules,
            demo_rules=demo_rules,
        )
        return result.is_valid_rule
    if completion_format != "full":
        raise ValueError(
            f"completion_format must be 'single' or 'full', got {completion_format!r}"
        )
    if prompt_tokens is None:
        raise ValueError("prompt_tokens are required for completion_format='full'.")
    result = validate_completion_path_fol(
        rule_bank=rule_bank,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokenizer=tokenizer,
        active_rules=active_rules,
        fixed_rules=fixed_rules,
        demo_rules=demo_rules,
    )
    return bool(result.success)


completion_is_valid_for_layer = completion_is_valid_for_layer_fol


_LAYERED_PREDICATE_RE = re.compile(r"r(\d+)_(\d+)$")
_FRESH_PREDICATE_RE = re.compile(r"r_[a-z0-9]+$")


def infer_fol_predicate_layer(predicate: str) -> int:
    """Infer logical layer from a layered or fresh predicate identifier."""
    token = str(predicate)
    match = _LAYERED_PREDICATE_RE.fullmatch(token)
    if match is not None:
        return int(match.group(1))
    if _FRESH_PREDICATE_RE.fullmatch(token):
        return 0
    raise ValueError(f"Unsupported layered predicate name: {predicate}")


def _decode_single_completion_statement(
    *,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    completion_tokens: list[int] | np.ndarray,
) -> DecodeAttempt[str]:
    decoded = tokenizer.try_decode_completion_texts(
        [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    )
    if not decoded.ok or decoded.value is None:
        return DecodeAttempt.failure(decoded.error or "Unknown decode error.")
    statements = decoded.value
    if len(statements) != 1:
        return DecodeAttempt.failure("Expected a single completion statement.")
    return DecodeAttempt.success(statements[0])


class FOLLayerPredictionAdapter(Protocol):
    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens: list[int] | np.ndarray,
        tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FOLRuleMatchResult:
    src_layer: int
    decoded_statement: str | None
    expected_statement_text: str | None
    matched_rule: FOLLayerRule | None
    decode_error: bool
    unknown_rule_error: bool
    wrong_rule_error: bool
    is_valid_rule: bool
    is_correct: bool
    match_source: str | None = None


@dataclass(frozen=True)
class FOLRuleMatchMetrics:
    n_examples: int
    n_correct: int
    n_decode_error: int
    n_unknown_rule_error: int
    n_wrong_rule_error: int
    accuracy: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    wrong_rule_error_rate: float
    results: tuple[FOLRuleMatchResult, ...]


@dataclass(frozen=True)
class FOLCompletionPathStep:
    step_idx: int
    src_layer: int
    decoded_statement: str | None
    matched_rule_statement: str | None
    decode_error: bool
    unknown_rule_error: bool
    inapplicable_rule_error: bool
    goal_reached: bool


@dataclass(frozen=True)
class FOLCompletionPathResult:
    success: bool
    failure_reason: str | None
    n_steps: int
    goal_reached: bool
    final_layer: int
    final_facts: tuple[str, ...]
    steps: tuple[FOLCompletionPathStep, ...]


@dataclass(frozen=True)
class FOLCompletionPathMetrics:
    n_examples: int
    n_success: int
    n_failure_decode_error: int
    n_failure_unknown_rule_error: int
    n_failure_inapplicable_rule_error: int
    n_failure_goal_not_reached: int
    success_rate: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    inapplicable_rule_error_rate: float
    goal_not_reached_rate: float
    avg_steps: float
    results: tuple[FOLCompletionPathResult, ...]


@dataclass(frozen=True)
class FOLLayerRolloutExample:
    distance: int
    start_layer: int
    goal_atom: str
    initial_ants: tuple[str, ...]
    max_steps: int
    oracle_rule_statements: tuple[str, ...] = ()


@dataclass(frozen=True)
class FOLLayerRolloutTraceStep:
    step_idx: int
    src_layer: int
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    decoded_statement: str | None
    matched_rule_statement: str | None
    decode_error: bool
    unknown_rule_error: bool
    inapplicable_rule_error: bool
    goal_reached: bool


@dataclass(frozen=True)
class FOLLayerRolloutResult:
    success: bool
    failure_reason: str | None
    n_steps: int
    goal_reached: bool
    final_layer: int
    final_facts: tuple[str, ...]
    steps: tuple[FOLLayerRolloutTraceStep, ...]
    example: FOLLayerRolloutExample


@dataclass(frozen=True)
class FOLLayerRolloutMetrics:
    n_examples: int
    n_success: int
    n_failure_decode_error: int
    n_failure_unknown_rule_error: int
    n_failure_wrong_rule_error: int
    n_failure_inapplicable_rule_error: int
    n_failure_goal_not_reached: int
    success_rate: float
    decode_error_rate: float
    unknown_rule_error_rate: float
    wrong_rule_error_rate: float
    inapplicable_rule_error_rate: float
    goal_not_reached_rate: float
    avg_steps: float
    results: tuple[FOLLayerRolloutResult, ...]


FAILURE_DECODE_ERROR = "decode_error"
FAILURE_UNKNOWN_RULE_ERROR = "unknown_rule_error"
FAILURE_WRONG_RULE_ERROR = "wrong_rule_error"
FAILURE_INAPPLICABLE_RULE_ERROR = "inapplicable_rule_error"
FAILURE_GOAL_NOT_REACHED = "goal_not_reached"


def _is_variable(token: str) -> bool:
    return token.startswith("x")


def _unify_template_atom_with_ground(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
) -> dict[str, str] | None:
    if template.predicate != ground.predicate:
        return None
    if len(template.args) != len(ground.args):
        return None

    out = dict(subst)
    for templ_term, ground_term in zip(template.args, ground.args):
        if _is_variable(templ_term):
            bound = out.get(templ_term)
            if bound is None:
                out[templ_term] = ground_term
            elif bound != ground_term:
                return None
        elif templ_term != ground_term:
            return None
    return out


def _find_multiset_matches(
    *,
    templates: tuple[FOLAtom, ...],
    grounds: tuple[FOLAtom, ...],
    seed_subst: dict[str, str],
    max_solutions: int,
) -> list[dict[str, str]]:
    if len(templates) != len(grounds):
        return []

    solutions: list[dict[str, str]] = []
    used = [False] * len(grounds)

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(templates):
            solutions.append(dict(subst))
            return

        template = templates[idx]
        for ground_idx, ground in enumerate(grounds):
            if used[ground_idx]:
                continue
            maybe = _unify_template_atom_with_ground(template, ground, subst)
            if maybe is None:
                continue
            used[ground_idx] = True
            _search(idx + 1, maybe)
            used[ground_idx] = False
            if len(solutions) >= max_solutions:
                return

    _search(0, dict(seed_subst))
    return solutions


def _find_instantiation_for_rule(
    *,
    template: FOLLayerRule,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    max_solutions: int = 64,
) -> dict[str, str] | None:
    lhs_subs = _find_multiset_matches(
        templates=template.lhs,
        grounds=lhs_ground,
        seed_subst={},
        max_solutions=max_solutions,
    )
    if not lhs_subs:
        return None

    for lhs_sub in lhs_subs:
        rhs_subs = _find_multiset_matches(
            templates=template.rhs,
            grounds=rhs_ground,
            seed_subst=lhs_sub,
            max_solutions=1,
        )
        if rhs_subs:
            return rhs_subs[0]
    return None


def _unify_template_atom_with_ground_schema(
    template: FOLAtom,
    ground: FOLAtom,
    subst: dict[str, str],
    pred_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Like ``_unify_template_atom_with_ground`` but matches structurally.

    Instead of requiring exact predicate name equality, maintains an injective
    mapping *pred_map* from template predicate names to ground predicate names.
    Arity must still match, and variable bindings are consistent as before.

    Returns ``(updated_subst, updated_pred_map)`` on success, ``None`` on failure.
    """
    if len(template.args) != len(ground.args):
        return None

    # Check / extend predicate mapping
    new_pred_map = dict(pred_map)
    bound_pred = new_pred_map.get(template.predicate)
    if bound_pred is None:
        # Injective: ground predicate must not already be a target of another template pred
        if ground.predicate in new_pred_map.values():
            return None
        new_pred_map[template.predicate] = ground.predicate
    elif bound_pred != ground.predicate:
        return None

    # Unify variable / constant arguments (identical to the exact version)
    out = dict(subst)
    for templ_term, ground_term in zip(template.args, ground.args):
        if _is_variable(templ_term):
            bound = out.get(templ_term)
            if bound is None:
                out[templ_term] = ground_term
            elif bound != ground_term:
                return None
        elif templ_term != ground_term:
            return None
    return out, new_pred_map


def _find_multiset_matches_schema(
    *,
    templates: tuple[FOLAtom, ...],
    grounds: tuple[FOLAtom, ...],
    seed_subst: dict[str, str],
    seed_pred_map: dict[str, str],
    max_solutions: int,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Like ``_find_multiset_matches`` but threads *pred_map* alongside *subst*."""
    if len(templates) != len(grounds):
        return []

    solutions: list[tuple[dict[str, str], dict[str, str]]] = []
    used = [False] * len(grounds)

    def _search(idx: int, subst: dict[str, str], pred_map: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(templates):
            solutions.append((dict(subst), dict(pred_map)))
            return

        template = templates[idx]
        for ground_idx, ground in enumerate(grounds):
            if used[ground_idx]:
                continue
            maybe = _unify_template_atom_with_ground_schema(
                template, ground, subst, pred_map,
            )
            if maybe is None:
                continue
            new_subst, new_pred_map = maybe
            used[ground_idx] = True
            _search(idx + 1, new_subst, new_pred_map)
            used[ground_idx] = False
            if len(solutions) >= max_solutions:
                return

    _search(0, dict(seed_subst), dict(seed_pred_map))
    return solutions


def _find_schema_match_for_rule(
    *,
    template: FOLLayerRule,
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    max_solutions: int = 64,
) -> bool:
    """Like ``_find_instantiation_for_rule`` but uses schema unification.

    Returns ``True`` if the structural pattern (atom counts, arities,
    variable-binding pattern) of *template* can be matched against the ground
    atoms, allowing arbitrary predicate-name renaming.
    """
    lhs_sols = _find_multiset_matches_schema(
        templates=template.lhs,
        grounds=lhs_ground,
        seed_subst={},
        seed_pred_map={},
        max_solutions=max_solutions,
    )
    if not lhs_sols:
        return False

    for lhs_subst, lhs_pred_map in lhs_sols:
        rhs_sols = _find_multiset_matches_schema(
            templates=template.rhs,
            grounds=rhs_ground,
            seed_subst=lhs_subst,
            seed_pred_map=lhs_pred_map,
            max_solutions=1,
        )
        if rhs_sols:
            return True
    return False


def _any_rule_schema_matches(
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
    candidates: Iterable[FOLLayerRule],
) -> bool:
    """Return ``True`` if any candidate rule's schema matches the ground atoms."""
    for rule in candidates:
        if _find_schema_match_for_rule(
            template=rule,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        ):
            return True
    return False


def _match_instantiated_rule_from_candidates(
    *,
    candidates: Iterable[FOLLayerRule],
    lhs_ground: tuple[FOLAtom, ...],
    rhs_ground: tuple[FOLAtom, ...],
) -> FOLLayerRule | None:
    for rule in candidates:
        subst = _find_instantiation_for_rule(
            template=rule,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if subst is not None:
            return rule.instantiate(subst)
    return None


def match_rule_completion_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    completion_tokens: list[int] | np.ndarray,
    expected_statement_text: str | None = None,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLRuleMatchResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    src_layer = int(src_layer)
    active_rules_list = (
        list(active_rules)
        if active_rules is not None
        else list(rule_bank.transition_rules(src_layer))
    )
    fixed_rules_list = list(fixed_rules or ())
    demo_rules_list = list(demo_rules or ())

    try:
        completion = [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError):
        completion = []

    decoded = _decode_single_completion_statement(
        tokenizer=tokenizer,
        completion_tokens=completion,
    )
    if not decoded.ok or decoded.value is None:
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=None,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=True,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
            match_source=None,
        )
    decoded_statement = decoded.value
    try:
        lhs_ground, rhs_ground = parse_clause_text(decoded_statement)
    except (ValueError, TypeError):
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=None,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=True,
            unknown_rule_error=False,
            wrong_rule_error=False,
            is_valid_rule=False,
            is_correct=False,
            match_source=None,
        )

    matched_rule = _match_instantiated_rule_from_candidates(
        candidates=active_rules_list,
        lhs_ground=lhs_ground,
        rhs_ground=rhs_ground,
    )
    match_source: str | None = None
    if matched_rule is not None:
        match_source = "active"
    if matched_rule is None and fixed_rules_list:
        matched_rule = _match_instantiated_rule_from_candidates(
            candidates=fixed_rules_list,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if matched_rule is not None:
            match_source = "fixed"
    if matched_rule is None and demo_rules_list:
        matched_rule = _match_instantiated_rule_from_candidates(
            candidates=demo_rules_list,
            lhs_ground=lhs_ground,
            rhs_ground=rhs_ground,
        )
        if matched_rule is not None:
            match_source = "demo"
    if matched_rule is None:
        candidates = [
            *active_rules_list,
            *fixed_rules_list,
            *demo_rules_list,
        ]
        schema_matches = _any_rule_schema_matches(
            lhs_ground, rhs_ground, candidates,
        )
        return FOLRuleMatchResult(
            src_layer=src_layer,
            decoded_statement=decoded_statement,
            expected_statement_text=expected_statement_text,
            matched_rule=None,
            decode_error=False,
            unknown_rule_error=not schema_matches,
            wrong_rule_error=schema_matches,
            is_valid_rule=False,
            is_correct=False,
            match_source=None,
        )

    wrong_rule_error = (
        expected_statement_text is not None
        and decoded_statement != str(expected_statement_text)
    )
    is_correct = not wrong_rule_error
    return FOLRuleMatchResult(
        src_layer=src_layer,
        decoded_statement=decoded_statement,
        expected_statement_text=expected_statement_text,
        matched_rule=matched_rule,
        decode_error=False,
        unknown_rule_error=False,
        wrong_rule_error=bool(wrong_rule_error),
        is_valid_rule=True,
        is_correct=is_correct,
        match_source=match_source,
    )


match_rule_completion = match_rule_completion_fol


def evaluate_rule_matches_fol(
    *,
    rule_bank: FOLRuleBank,
    src_layers: Iterable[int],
    completion_tokens: Iterable[list[int] | np.ndarray],
    expected_statement_texts: Iterable[str | None] | None = None,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
    active_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    fixed_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    demo_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
) -> FOLRuleMatchMetrics:
    src_layers = [int(layer) for layer in src_layers]
    completion_tokens = list(completion_tokens)
    if len(src_layers) != len(completion_tokens):
        raise ValueError(
            f"src_layers and completion_tokens must have same length, got "
            f"{len(src_layers)} and {len(completion_tokens)}"
        )

    if expected_statement_texts is None:
        expected_statement_texts = [None] * len(src_layers)
    else:
        expected_statement_texts = list(expected_statement_texts)
        if len(expected_statement_texts) != len(src_layers):
            raise ValueError(
                "expected_statement_texts must match src_layers length, got "
                f"{len(expected_statement_texts)} and {len(src_layers)}"
            )

    active_rules_list = list(active_rules) if active_rules is not None else None
    fixed_rules_list = list(fixed_rules) if fixed_rules is not None else None
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
    if active_rules_by_example is None:
        active_rules_per_example: list[list[FOLLayerRule] | None] = [None] * len(src_layers)
    else:
        active_rules_per_example = [
            None if rules is None else list(rules)
            for rules in active_rules_by_example
        ]
        if len(active_rules_per_example) != len(src_layers):
            raise ValueError(
                "active_rules_by_example must match src_layers length, got "
                f"{len(active_rules_per_example)} and {len(src_layers)}"
            )
    if fixed_rules_by_example is None:
        fixed_rules_per_example: list[list[FOLLayerRule] | None] = [None] * len(src_layers)
    else:
        fixed_rules_per_example = [
            None if rules is None else list(rules)
            for rules in fixed_rules_by_example
        ]
        if len(fixed_rules_per_example) != len(src_layers):
            raise ValueError(
                "fixed_rules_by_example must match src_layers length, got "
                f"{len(fixed_rules_per_example)} and {len(src_layers)}"
            )
    if demo_rules_by_example is None:
        demo_rules_per_example: list[list[FOLLayerRule] | None] = [None] * len(src_layers)
    else:
        demo_rules_per_example = [
            None if rules is None else list(rules)
            for rules in demo_rules_by_example
        ]
        if len(demo_rules_per_example) != len(src_layers):
            raise ValueError(
                "demo_rules_by_example must match src_layers length, got "
                f"{len(demo_rules_per_example)} and {len(src_layers)}"
            )

    results = tuple(
        match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            expected_statement_text=expected_statement,
            tokenizer=tokenizer,
            active_rules=(
                [
                    *(active_rules_list or ()),
                    *(example_active_rules or ()),
                ]
                if active_rules_list is not None or example_active_rules is not None
                else None
            ),
            fixed_rules=(
                [
                    *(fixed_rules_list or ()),
                    *(example_fixed_rules or ()),
                ]
                if fixed_rules_list is not None or example_fixed_rules is not None
                else None
            ),
            demo_rules=(
                [
                    *(demo_rules_list or ()),
                    *(example_demo_rules or ()),
                ]
                if demo_rules_list is not None or example_demo_rules is not None
                else None
            ),
        )
        for (
            src_layer,
            completion,
            expected_statement,
            example_active_rules,
            example_fixed_rules,
            example_demo_rules,
        ) in zip(
            src_layers,
            completion_tokens,
            expected_statement_texts,
            active_rules_per_example,
            fixed_rules_per_example,
            demo_rules_per_example,
        )
    )

    n_examples = len(results)
    n_correct = sum(int(result.is_correct) for result in results)
    n_decode_error = sum(int(result.decode_error) for result in results)
    n_unknown_rule_error = sum(int(result.unknown_rule_error) for result in results)
    n_wrong_rule_error = sum(int(result.wrong_rule_error) for result in results)

    return FOLRuleMatchMetrics(
        n_examples=n_examples,
        n_correct=n_correct,
        n_decode_error=n_decode_error,
        n_unknown_rule_error=n_unknown_rule_error,
        n_wrong_rule_error=n_wrong_rule_error,
        accuracy=_safe_rate(n_correct, n_examples),
        decode_error_rate=_safe_rate(n_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_unknown_rule_error, n_examples),
        wrong_rule_error_rate=_safe_rate(n_wrong_rule_error, n_examples),
        results=results,
    )


def _rule_match_unknown_for_path(result: FOLRuleMatchResult) -> bool:
    return bool(result.unknown_rule_error or result.wrong_rule_error or result.matched_rule is None)


def _decode_prompt_context(
    *,
    prompt_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
) -> FOLSequent:
    try:
        prompt = [int(tok) for tok in np.asarray(prompt_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError) as err:
        raise ValueError("Prompt tokens must be a 1D integer sequence.") from err
    return tokenizer.decode_prompt(prompt)


def validate_completion_path_fol(
    *,
    rule_bank: FOLRuleBank,
    prompt_tokens: list[int] | np.ndarray,
    completion_tokens: list[int] | np.ndarray,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLCompletionPathResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    sequent = _decode_prompt_context(prompt_tokens=prompt_tokens, tokenizer=tokenizer)
    if len(sequent.ants) == 0:
        raise ValueError("Prompt antecedents are empty; cannot validate completion path.")

    initial_layer = int(min(infer_fol_predicate_layer(atom.predicate) for atom in sequent.ants))
    facts = set(sequent.ants)
    goal = sequent.cons
    steps: list[FOLCompletionPathStep] = []

    try:
        completion = [int(tok) for tok in np.asarray(completion_tokens, dtype=np.int32).tolist()]
    except (TypeError, ValueError):
        completion = []

    decoded = tokenizer.try_decode_completion_texts(completion)
    if not decoded.ok or decoded.value is None:
        return FOLCompletionPathResult(
            success=False,
            failure_reason=FAILURE_DECODE_ERROR,
            n_steps=0,
            goal_reached=False,
            final_layer=initial_layer,
            final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
            steps=(),
        )
    decoded_statements = decoded.value

    current_layer = int(initial_layer)
    active_rules_list = list(active_rules) if active_rules is not None else None
    fixed_rules_list = list(fixed_rules) if fixed_rules is not None else None
    demo_rules_list = list(demo_rules) if demo_rules is not None else None

    for step_idx, statement_text in enumerate(decoded_statements):
        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=current_layer,
            completion_tokens=tokenizer.encode_completion_texts([statement_text]),
            tokenizer=tokenizer,
            active_rules=active_rules_list,
            fixed_rules=fixed_rules_list,
            demo_rules=demo_rules_list,
        )

        if matched.decode_error:
            steps.append(
                FOLCompletionPathStep(
                    step_idx=int(step_idx),
                    src_layer=int(current_layer),
                    decoded_statement=None,
                    matched_rule_statement=None,
                    decode_error=True,
                    unknown_rule_error=False,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLCompletionPathResult(
                success=False,
                failure_reason=FAILURE_DECODE_ERROR,
                n_steps=len(steps),
                goal_reached=False,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

        if _rule_match_unknown_for_path(matched):
            steps.append(
                FOLCompletionPathStep(
                    step_idx=int(step_idx),
                    src_layer=int(current_layer),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=None,
                    decode_error=False,
                    unknown_rule_error=True,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLCompletionPathResult(
                success=False,
                failure_reason=FAILURE_UNKNOWN_RULE_ERROR,
                n_steps=len(steps),
                goal_reached=False,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

        rule = matched.matched_rule
        assert rule is not None
        if not set(rule.lhs).issubset(facts):
            steps.append(
                FOLCompletionPathStep(
                    step_idx=int(step_idx),
                    src_layer=int(current_layer),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=rule.statement_text,
                    decode_error=False,
                    unknown_rule_error=False,
                    inapplicable_rule_error=True,
                    goal_reached=False,
                )
            )
            return FOLCompletionPathResult(
                success=False,
                failure_reason=FAILURE_INAPPLICABLE_RULE_ERROR,
                n_steps=len(steps),
                goal_reached=False,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

        facts = set(rule.rhs)
        goal_reached = goal in facts
        steps.append(
            FOLCompletionPathStep(
                step_idx=int(step_idx),
                src_layer=int(current_layer),
                decoded_statement=matched.decoded_statement,
                matched_rule_statement=rule.statement_text,
                decode_error=False,
                unknown_rule_error=False,
                inapplicable_rule_error=False,
                goal_reached=bool(goal_reached),
            )
        )
        current_layer = int(rule.dst_layer)

        if goal_reached:
            return FOLCompletionPathResult(
                success=True,
                failure_reason=None,
                n_steps=len(steps),
                goal_reached=True,
                final_layer=int(current_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(steps),
            )

    return FOLCompletionPathResult(
        success=False,
        failure_reason=FAILURE_GOAL_NOT_REACHED,
        n_steps=len(steps),
        goal_reached=False,
        final_layer=int(current_layer),
        final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
        steps=tuple(steps),
    )


def evaluate_completion_paths_fol(
    *,
    rule_bank: FOLRuleBank,
    prompt_tokens: Iterable[list[int] | np.ndarray],
    completion_tokens: Iterable[list[int] | np.ndarray],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    active_rules: Iterable[FOLLayerRule] | None = None,
    fixed_rules: Iterable[FOLLayerRule] | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
    active_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    fixed_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
    demo_rules_by_example: Iterable[Iterable[FOLLayerRule] | None] | None = None,
) -> FOLCompletionPathMetrics:
    prompt_tokens = list(prompt_tokens)
    completion_tokens = list(completion_tokens)
    if len(prompt_tokens) != len(completion_tokens):
        raise ValueError(
            "prompt_tokens and completion_tokens must have same length, got "
            f"{len(prompt_tokens)} and {len(completion_tokens)}"
        )

    active_rules_list = list(active_rules) if active_rules is not None else None
    fixed_rules_list = list(fixed_rules) if fixed_rules is not None else None
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
    if active_rules_by_example is None:
        active_rules_per_example: list[list[FOLLayerRule] | None] = [None] * len(prompt_tokens)
    else:
        active_rules_per_example = [
            None if rules is None else list(rules)
            for rules in active_rules_by_example
        ]
        if len(active_rules_per_example) != len(prompt_tokens):
            raise ValueError(
                "active_rules_by_example must match prompt_tokens length, got "
                f"{len(active_rules_per_example)} and {len(prompt_tokens)}"
            )
    if fixed_rules_by_example is None:
        fixed_rules_per_example: list[list[FOLLayerRule] | None] = [None] * len(prompt_tokens)
    else:
        fixed_rules_per_example = [
            None if rules is None else list(rules)
            for rules in fixed_rules_by_example
        ]
        if len(fixed_rules_per_example) != len(prompt_tokens):
            raise ValueError(
                "fixed_rules_by_example must match prompt_tokens length, got "
                f"{len(fixed_rules_per_example)} and {len(prompt_tokens)}"
            )
    if demo_rules_by_example is None:
        demo_rules_per_example: list[list[FOLLayerRule] | None] = [None] * len(prompt_tokens)
    else:
        demo_rules_per_example = [
            None if rules is None else list(rules)
            for rules in demo_rules_by_example
        ]
        if len(demo_rules_per_example) != len(prompt_tokens):
            raise ValueError(
                "demo_rules_by_example must match prompt_tokens length, got "
                f"{len(demo_rules_per_example)} and {len(prompt_tokens)}"
            )

    results = tuple(
        validate_completion_path_fol(
            rule_bank=rule_bank,
            prompt_tokens=prompt,
            completion_tokens=completion,
            tokenizer=tokenizer,
            active_rules=(
                active_rules_list if active_rules_per_example[idx] is None
                else active_rules_per_example[idx]
            ),
            fixed_rules=(
                fixed_rules_list if fixed_rules_per_example[idx] is None
                else fixed_rules_per_example[idx]
            ),
            demo_rules=(
                demo_rules_list if demo_rules_per_example[idx] is None
                else demo_rules_per_example[idx]
            ),
        )
        for idx, (prompt, completion) in enumerate(zip(prompt_tokens, completion_tokens))
    )

    n_examples = len(results)
    n_success = sum(int(result.success) for result in results)
    n_failure_decode_error = sum(
        int(result.failure_reason == FAILURE_DECODE_ERROR)
        for result in results
    )
    n_failure_unknown_rule_error = sum(
        int(result.failure_reason == FAILURE_UNKNOWN_RULE_ERROR)
        for result in results
    )
    n_failure_inapplicable_rule_error = sum(
        int(result.failure_reason == FAILURE_INAPPLICABLE_RULE_ERROR)
        for result in results
    )
    n_failure_goal_not_reached = sum(
        int(result.failure_reason == FAILURE_GOAL_NOT_REACHED)
        for result in results
    )
    avg_steps = float(np.mean([result.n_steps for result in results])) if results else 0.0

    return FOLCompletionPathMetrics(
        n_examples=n_examples,
        n_success=n_success,
        n_failure_decode_error=n_failure_decode_error,
        n_failure_unknown_rule_error=n_failure_unknown_rule_error,
        n_failure_inapplicable_rule_error=n_failure_inapplicable_rule_error,
        n_failure_goal_not_reached=n_failure_goal_not_reached,
        success_rate=_safe_rate(n_success, n_examples),
        decode_error_rate=_safe_rate(n_failure_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_failure_unknown_rule_error, n_examples),
        inapplicable_rule_error_rate=_safe_rate(n_failure_inapplicable_rule_error, n_examples),
        goal_not_reached_rate=_safe_rate(n_failure_goal_not_reached, n_examples),
        avg_steps=avg_steps,
        results=results,
    )


def sample_rollout_examples_fol(
    *,
    rule_bank: FOLRuleBank,
    distance: int,
    n_examples: int,
    initial_ant_max: int,
    max_steps: int | None = None,
    max_unify_solutions: int = 128,
    rng: np.random.Generator | None = None,
) -> list[FOLLayerRolloutExample]:
    if n_examples < 1:
        raise ValueError(f"n_examples must be >= 1, got {n_examples}")
    if rng is None:
        rng = np.random.default_rng()

    out: list[FOLLayerRolloutExample] = []
    for _ in range(int(n_examples)):
        sampled = sample_fol_problem(
            bank=rule_bank,
            distance=int(distance),
            initial_ant_max=int(initial_ant_max),
            rng=rng,
            max_unify_solutions=int(max_unify_solutions),
        )
        if not sampled.step_ants:
            raise RuntimeError("Sampled problem contained no steps.")
        out.append(
            FOLLayerRolloutExample(
                distance=int(sampled.distance),
                start_layer=int(sampled.start_layer),
                goal_atom=str(sampled.goal_atom.text),
                initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
                max_steps=int(sampled.distance if max_steps is None else max_steps),
                oracle_rule_statements=tuple(
                    str(rule.statement_text) for rule in sampled.step_rules
                ),
            )
        )
    return out


def _adapter_last_demo_rules(
    adapter: FOLLayerPredictionAdapter,
) -> list[FOLLayerRule] | None:
    getter = getattr(adapter, "get_last_demo_rules", None)
    if not callable(getter):
        return None
    try:
        raw = getter()
    except Exception:
        return None
    if raw is None:
        return None
    out = [
        rule
        for rule in raw
        if isinstance(rule, FOLLayerRule)
    ]
    return out


def _adapter_set_oracle_rule(
    adapter: FOLLayerPredictionAdapter,
    oracle_rule: FOLLayerRule | None,
) -> None:
    setter = getattr(adapter, "set_oracle_rule", None)
    if not callable(setter):
        return
    setter(oracle_rule)


def run_layer_rollout_fol(
    *,
    rule_bank: FOLRuleBank,
    example: FOLLayerRolloutExample,
    model: Callable[[np.ndarray], Any],
    adapter: FOLLayerPredictionAdapter,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLLayerRolloutResult:
    tokenizer = _resolve_fol_tokenizer(rule_bank=rule_bank, tokenizer=tokenizer)
    demo_rules_list = list(demo_rules) if demo_rules is not None else None
    if rng is None:
        rng = np.random.default_rng()

    facts = {parse_atom_text(atom_text) for atom_text in example.initial_ants}
    goal = parse_atom_text(example.goal_atom)
    traces: list[FOLLayerRolloutTraceStep] = []

    for step_idx in range(int(example.max_steps)):
        src_layer = int(example.start_layer) + step_idx
        oracle_rule = None
        if step_idx < len(example.oracle_rule_statements):
            lhs, rhs = parse_clause_text(str(example.oracle_rule_statements[step_idx]))
            oracle_rule = FOLLayerRule(
                src_layer=src_layer,
                dst_layer=src_layer + 1,
                lhs=lhs,
                rhs=rhs,
            )
        _adapter_set_oracle_rule(adapter, oracle_rule)
        prompt = tokenizer.tokenize_prompt(
            FOLSequent(
                ants=_sorted_fol_atoms(facts),
                cons=goal,
            )
        )
        completion = adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )
        step_demo_rules = _adapter_last_demo_rules(adapter)
        combined_demo_rules = (
            [
                *(demo_rules_list or ()),
                *(step_demo_rules or ()),
            ]
            if demo_rules_list is not None or step_demo_rules is not None
            else None
        )

        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=completion,
            tokenizer=tokenizer,
            demo_rules=combined_demo_rules,
        )

        if matched.decode_error:
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=None,
                    matched_rule_statement=None,
                    decode_error=True,
                    unknown_rule_error=False,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_DECODE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        if matched.unknown_rule_error or matched.matched_rule is None:
            failure = (
                FAILURE_UNKNOWN_RULE_ERROR if matched.unknown_rule_error
                else FAILURE_WRONG_RULE_ERROR
            )
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=None,
                    decode_error=False,
                    unknown_rule_error=matched.unknown_rule_error,
                    inapplicable_rule_error=False,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=failure,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        rule = matched.matched_rule
        if not set(rule.lhs).issubset(facts):
            traces.append(
                FOLLayerRolloutTraceStep(
                    step_idx=step_idx,
                    src_layer=src_layer,
                    prompt_tokens=tuple(int(tok) for tok in prompt),
                    completion_tokens=tuple(int(tok) for tok in completion),
                    decoded_statement=matched.decoded_statement,
                    matched_rule_statement=rule.statement_text,
                    decode_error=False,
                    unknown_rule_error=False,
                    inapplicable_rule_error=True,
                    goal_reached=False,
                )
            )
            return FOLLayerRolloutResult(
                success=False,
                failure_reason=FAILURE_INAPPLICABLE_RULE_ERROR,
                n_steps=len(traces),
                goal_reached=False,
                final_layer=src_layer,
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

        facts = set(rule.rhs)
        goal_reached = goal in facts
        traces.append(
            FOLLayerRolloutTraceStep(
                step_idx=step_idx,
                src_layer=src_layer,
                prompt_tokens=tuple(int(tok) for tok in prompt),
                completion_tokens=tuple(int(tok) for tok in completion),
                decoded_statement=matched.decoded_statement,
                matched_rule_statement=rule.statement_text,
                decode_error=False,
                unknown_rule_error=False,
                inapplicable_rule_error=False,
                goal_reached=goal_reached,
            )
        )

        if goal_reached:
            return FOLLayerRolloutResult(
                success=True,
                failure_reason=None,
                n_steps=len(traces),
                goal_reached=True,
                final_layer=int(rule.dst_layer),
                final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
                steps=tuple(traces),
                example=example,
            )

    return FOLLayerRolloutResult(
        success=False,
        failure_reason=FAILURE_GOAL_NOT_REACHED,
        n_steps=len(traces),
        goal_reached=False,
        final_layer=int(example.start_layer) + len(traces),
        final_facts=tuple(atom.text for atom in _sorted_fol_atoms(facts)),
        steps=tuple(traces),
        example=example,
    )


def evaluate_layer_rollouts_fol(
    *,
    rule_bank: FOLRuleBank,
    examples: Iterable[FOLLayerRolloutExample],
    model: Callable[[np.ndarray], Any],
    adapter: FOLLayerPredictionAdapter,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None = None,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    demo_rules: Iterable[FOLLayerRule] | None = None,
) -> FOLLayerRolloutMetrics:
    if rng is None:
        rng = np.random.default_rng()

    demo_rules_list = list(demo_rules) if demo_rules is not None else None

    results = tuple(
        run_layer_rollout_fol(
            rule_bank=rule_bank,
            example=example,
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
            demo_rules=demo_rules_list,
        )
        for example in examples
    )

    n_examples = len(results)
    n_success = sum(int(result.success) for result in results)
    n_failure_decode_error = sum(
        int(result.failure_reason == FAILURE_DECODE_ERROR)
        for result in results
    )
    n_failure_unknown_rule_error = sum(
        int(result.failure_reason == FAILURE_UNKNOWN_RULE_ERROR)
        for result in results
    )
    n_failure_wrong_rule_error = sum(
        int(result.failure_reason == FAILURE_WRONG_RULE_ERROR)
        for result in results
    )
    n_failure_inapplicable_rule_error = sum(
        int(result.failure_reason == FAILURE_INAPPLICABLE_RULE_ERROR)
        for result in results
    )
    n_failure_goal_not_reached = sum(
        int(result.failure_reason == FAILURE_GOAL_NOT_REACHED)
        for result in results
    )
    avg_steps = float(np.mean([result.n_steps for result in results])) if results else 0.0

    return FOLLayerRolloutMetrics(
        n_examples=n_examples,
        n_success=n_success,
        n_failure_decode_error=n_failure_decode_error,
        n_failure_unknown_rule_error=n_failure_unknown_rule_error,
        n_failure_wrong_rule_error=n_failure_wrong_rule_error,
        n_failure_inapplicable_rule_error=n_failure_inapplicable_rule_error,
        n_failure_goal_not_reached=n_failure_goal_not_reached,
        success_rate=_safe_rate(n_success, n_examples),
        decode_error_rate=_safe_rate(n_failure_decode_error, n_examples),
        unknown_rule_error_rate=_safe_rate(n_failure_unknown_rule_error, n_examples),
        wrong_rule_error_rate=_safe_rate(n_failure_wrong_rule_error, n_examples),
        inapplicable_rule_error_rate=_safe_rate(n_failure_inapplicable_rule_error, n_examples),
        goal_not_reached_rate=_safe_rate(n_failure_goal_not_reached, n_examples),
        avg_steps=avg_steps,
        results=results,
    )


def _facts_key(facts: tuple[FOLAtom, ...]) -> tuple[str, ...]:
    return tuple(sorted(atom.text for atom in facts))


def reachable_goal_exact_steps(
    *,
    rule_bank: FOLRuleBank,
    layer: int,
    facts: tuple[FOLAtom, ...],
    goal: FOLAtom,
    steps_remaining: int,
    max_unify_solutions: int,
    memo: dict[tuple[int, int, tuple[str, ...]], bool],
) -> bool:
    from task.layer_fol.demos import (
        _find_lhs_substitutions_for_facts,
        _subst_binds_rhs_variables,
    )

    key = (int(layer), int(steps_remaining), _facts_key(facts))
    cached = memo.get(key)
    if cached is not None:
        return bool(cached)

    if int(steps_remaining) == 0:
        out = bool(goal in set(facts))
        memo[key] = out
        return out

    if int(layer) >= int(rule_bank.n_layers) - 1:
        memo[key] = False
        return False

    facts_tuple = tuple(facts)
    for rule in rule_bank.transition_rules(int(layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=facts_tuple,
            max_solutions=int(max_unify_solutions),
        )
        if not substitutions:
            continue

        for subst in substitutions:
            if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                continue
            next_rule = rule.instantiate(subst)
            next_facts = tuple(next_rule.rhs)
            if reachable_goal_exact_steps(
                rule_bank=rule_bank,
                layer=int(layer) + 1,
                facts=next_facts,
                goal=goal,
                steps_remaining=int(steps_remaining) - 1,
                max_unify_solutions=int(max_unify_solutions),
                memo=memo,
            ):
                memo[key] = True
                return True

    memo[key] = False
    return False


def predicted_rule_reaches_goal(
    *,
    rule_bank: FOLRuleBank,
    matched_rule: FOLLayerRule,
    goal: FOLAtom,
    goal_layer: int,
    max_unify_solutions: int,
) -> bool:
    dst_layer = int(matched_rule.dst_layer)
    remaining = int(goal_layer) - dst_layer
    if remaining < 0:
        return False

    memo: dict[tuple[int, int, tuple[str, ...]], bool] = {}
    return reachable_goal_exact_steps(
        rule_bank=rule_bank,
        layer=dst_layer,
        facts=tuple(matched_rule.rhs),
        goal=goal,
        steps_remaining=remaining,
        max_unify_solutions=int(max_unify_solutions),
        memo=memo,
    )


def parse_rule_statements_for_layer(
    *,
    src_layer: int,
    statement_texts: Iterable[str],
) -> list[FOLLayerRule]:
    out: list[FOLLayerRule] = []
    src_layer = int(src_layer)
    for text in statement_texts:
        try:
            lhs, rhs = parse_clause_text(str(text))
        except ValueError:
            continue
        out.append(
            FOLLayerRule(
                src_layer=src_layer,
                dst_layer=src_layer + 1,
                lhs=lhs,
                rhs=rhs,
            )
        )
    return out


def resolve_rule_sets_from_context(
    *,
    src_layer: int,
    rule_context: dict | None,
) -> tuple[list[FOLLayerRule] | None, list[FOLLayerRule] | None, list[FOLLayerRule] | None]:
    if not isinstance(rule_context, dict):
        return None, None, None
    active = (
        parse_rule_statements_for_layer(
            src_layer=src_layer,
            statement_texts=rule_context.get("active_rule_texts", []),
        )
        if "active_rule_texts" in rule_context
        else None
    )
    fixed = (
        parse_rule_statements_for_layer(
            src_layer=src_layer,
            statement_texts=rule_context.get("fixed_rule_texts", []),
        )
        if "fixed_rule_texts" in rule_context
        else None
    )
    demo = (
        parse_rule_statements_for_layer(
            src_layer=src_layer,
            statement_texts=rule_context.get("demo_schema_texts", []),
        )
        if "demo_schema_texts" in rule_context
        else None
    )
    return active, fixed, demo


def _resolve_fol_tokenizer(
    *,
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer | None,
) -> tokenize_layer_fol.FOLLayerTokenizer:
    if tokenizer is not None:
        return tokenizer
    return tokenize_layer_fol.build_tokenizer_from_rule_bank(rule_bank)


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _sorted_fol_atoms(atoms: Iterable[FOLAtom]) -> tuple[FOLAtom, ...]:
    return tuple(sorted((atom for atom in atoms), key=lambda atom: atom.text))


evaluate_rule_matches = evaluate_rule_matches_fol
sample_rollout_examples = sample_rollout_examples_fol
run_layer_rollout = run_layer_rollout_fol
evaluate_layer_rollouts = evaluate_layer_rollouts_fol
