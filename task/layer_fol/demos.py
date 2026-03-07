"""Demo augmentation utilities for layered FOL tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank
from .eval_inputs import extract_prompt_info_from_row_tokens


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


@dataclass(frozen=True)
class FOLDemoAugmentationResult:
    prompt_tokens: list[int]
    demo_schemas: tuple[FOLLayerRule, ...]
    demo_instances: tuple[str, ...]


def augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
) -> FOLDemoAugmentationResult:
    max_n_demos = int(max_n_demos)
    min_n_demos = int(min_n_demos)
    if min_n_demos > max_n_demos:
        raise ValueError(
            f"min_n_demos must be <= max_n_demos, got {min_n_demos} > {max_n_demos}"
        )
    if max_n_demos <= 0:
        return FOLDemoAugmentationResult(
            prompt_tokens=list(int(tok) for tok in prompt_tokens),
            demo_schemas=(),
            demo_instances=(),
        )

    n_demos = int(rng.integers(min_n_demos, max_n_demos + 1))
    if n_demos <= 0:
        return FOLDemoAugmentationResult(
            prompt_tokens=list(int(tok) for tok in prompt_tokens),
            demo_schemas=(),
            demo_instances=(),
        )
    schemas = _collect_applicable_demo_schemas(
        rule_bank=rule_bank,
        src_layer=int(src_layer),
        ants=ants,
        max_unify_solutions=int(max_unify_solutions),
    )
    if not schemas:
        return FOLDemoAugmentationResult(
            prompt_tokens=list(int(tok) for tok in prompt_tokens),
            demo_schemas=(),
            demo_instances=(),
        )

    sampled_schemas = _sample_demo_schemas_with_replacement(
        rng=rng,
        schemas=schemas,
        n_demos=n_demos,
    )
    if not sampled_schemas:
        return FOLDemoAugmentationResult(
            prompt_tokens=list(int(tok) for tok in prompt_tokens),
            demo_schemas=(),
            demo_instances=(),
        )

    demo_statements = [
        _instantiate_demo_schema_with_random_constants(
            rule=schema,
            constants=rule_bank.constants,
            rng=rng,
        )
        for schema in sampled_schemas
    ]

    augmented_prompt = _prepend_demo_statements_to_prompt(
        prompt_tokens=prompt_tokens,
        demo_statements=demo_statements,
        tokenizer=tokenizer,
    )
    return FOLDemoAugmentationResult(
        prompt_tokens=augmented_prompt,
        demo_schemas=tuple(sampled_schemas),
        demo_instances=tuple(str(statement) for statement in demo_statements),
    )


def _augment_prompt_with_demos(
    *,
    prompt_tokens: list[int],
    rule_bank: FOLRuleBank,
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
    rng: np.random.Generator,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_n_demos: int,
    min_n_demos: int,
    max_unify_solutions: int,
) -> list[int]:
    return augment_prompt_with_demos(
        prompt_tokens=prompt_tokens,
        rule_bank=rule_bank,
        tokenizer=tokenizer,
        rng=rng,
        src_layer=src_layer,
        ants=ants,
        max_n_demos=max_n_demos,
        min_n_demos=min_n_demos,
        max_unify_solutions=max_unify_solutions,
    ).prompt_tokens


def _collect_applicable_demo_schemas(
    *,
    rule_bank: FOLRuleBank,
    src_layer: int,
    ants: tuple[FOLAtom, ...],
    max_unify_solutions: int,
) -> list[FOLLayerRule]:
    schemas: list[FOLLayerRule] = []
    seen_schema_keys: set[str] = set()
    ground_ants = tuple(ants)
    for rule in rule_bank.transition_rules(int(src_layer)):
        substitutions = _find_lhs_substitutions_for_facts(
            lhs=rule.lhs,
            facts=ground_ants,
            max_solutions=int(max_unify_solutions),
        )
        if not any(_subst_binds_rhs_variables(rule=rule, subst=subst) for subst in substitutions):
            continue

        schema_key = str(rule.statement_text)
        if schema_key in seen_schema_keys:
            continue
        seen_schema_keys.add(schema_key)
        schemas.append(rule)
    return schemas


def _subst_binds_rhs_variables(*, rule: FOLLayerRule, subst: dict[str, str]) -> bool:
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


def _find_lhs_substitutions_for_facts(
    *,
    lhs: tuple[FOLAtom, ...],
    facts: tuple[FOLAtom, ...],
    max_solutions: int,
) -> list[dict[str, str]]:
    solutions: list[dict[str, str]] = []

    def _search(idx: int, subst: dict[str, str]) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx >= len(lhs):
            solutions.append(dict(subst))
            return

        templ = lhs[idx]
        for fact in facts:
            maybe = _unify_template_atom_with_ground(templ, fact, subst)
            if maybe is None:
                continue
            _search(idx + 1, maybe)
            if len(solutions) >= max_solutions:
                return

    _search(0, {})
    return solutions


def _sample_demo_schemas_with_replacement(
    *,
    rng: np.random.Generator,
    schemas: list[FOLLayerRule],
    n_demos: int,
) -> list[FOLLayerRule]:
    if n_demos < 1 or not schemas:
        return []

    picks = rng.integers(0, len(schemas), size=int(n_demos))
    return [schemas[int(idx)] for idx in picks]


def _instantiate_demo_schema_with_random_constants(
    *,
    rule: FOLLayerRule,
    constants: tuple[str, ...],
    rng: np.random.Generator,
) -> str:
    if not constants:
        raise ValueError("Cannot instantiate demo rule schema without constants.")

    variables = tuple(sorted(rule.variables()))
    if not variables:
        return str(rule.statement_text)

    substitution = {
        var: str(constants[int(rng.integers(0, len(constants)))])
        for var in variables
    }
    return str(rule.instantiate(substitution).statement_text)


def _prepend_demo_statements_to_prompt(
    *,
    prompt_tokens: list[int],
    demo_statements: list[str],
    tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
) -> list[int]:
    out: list[int] = []
    for statement in demo_statements:
        demo_completion = tokenizer.encode_completion_texts([statement])
        out.extend(int(tok) for tok in demo_completion[:-1])
        out.append(int(tokenizer.sep_token_id))
    out.extend(int(tok) for tok in prompt_tokens)
    return out


class FOLDemoAugmentedAdapter:
    """Adapter wrapper that prepends sampled demonstration rules to prompts."""

    def __init__(
        self,
        *,
        base_adapter,
        rule_bank: FOLRuleBank,
        tokenizer,
        min_n_demos: int,
        max_n_demos: int,
        max_unify_solutions: int,
    ) -> None:
        self.base_adapter = base_adapter
        self.rule_bank = rule_bank
        self.tokenizer = tokenizer
        self.min_n_demos = int(min_n_demos)
        self.max_n_demos = int(max_n_demos)
        self.max_unify_solutions = int(max_unify_solutions)
        self._last_demo_rules: list[FOLLayerRule] = []

    def get_last_demo_rules(self) -> list[FOLLayerRule]:
        return list(self._last_demo_rules)

    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens,
        tokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        self._last_demo_rules = []
        if self.max_n_demos <= 0:
            return self.base_adapter.predict_completion(
                model=model,
                prompt_tokens=prompt_tokens,
                tokenizer=tokenizer,
                temperature=temperature,
                rng=rng,
            )

        if rng is None:
            rng = np.random.default_rng()

        prompt = np.asarray(prompt_tokens, dtype=np.int32).tolist()
        try:
            _, sequent, src_layer, _ = extract_prompt_info_from_row_tokens(
                np.asarray(prompt, dtype=np.int32),
                tokenizer=self.tokenizer,
            )
            augmented = augment_prompt_with_demos(
                prompt_tokens=prompt,
                rule_bank=self.rule_bank,
                tokenizer=self.tokenizer,
                rng=rng,
                src_layer=int(src_layer),
                ants=tuple(sequent.ants),
                min_n_demos=self.min_n_demos,
                max_n_demos=self.max_n_demos,
                max_unify_solutions=self.max_unify_solutions,
            )
            prompt = augmented.prompt_tokens
            self._last_demo_rules = list(augmented.demo_schemas)
        except ValueError:
            self._last_demo_rules = []

        return self.base_adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )
