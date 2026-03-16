"""Unification and matching primitives for FOL demo schemas."""

from __future__ import annotations

from task.layer_gen.util.fol_rule_bank import FOLAtom, FOLLayerRule, FOLRuleBank


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


def _subst_binds_rhs_variables(*, rule: FOLLayerRule, subst: dict[str, str]) -> bool:
    rhs_vars = {
        term
        for atom in rule.rhs
        for term in atom.args
        if _is_variable(term)
    }
    return rhs_vars.issubset(set(subst))


def _find_demo_schema_instantiation(
    *,
    schema: FOLLayerRule,
    ground_rule: FOLLayerRule,
) -> dict[str, str] | None:
    if int(schema.src_layer) != int(ground_rule.src_layer):
        return None
    if int(schema.dst_layer) != int(ground_rule.dst_layer):
        return None
    if len(schema.lhs) != len(ground_rule.lhs):
        return None
    if len(schema.rhs) != len(ground_rule.rhs):
        return None

    subst: dict[str, str] = {}
    for templ_atom, ground_atom in zip(schema.lhs, ground_rule.lhs):
        maybe = _unify_template_atom_with_ground(templ_atom, ground_atom, subst)
        if maybe is None:
            return None
        subst = maybe
    for templ_atom, ground_atom in zip(schema.rhs, ground_rule.rhs):
        maybe = _unify_template_atom_with_ground(templ_atom, ground_atom, subst)
        if maybe is None:
            return None
        subst = maybe
    return subst


def _find_matching_demo_schema_for_rule(
    *,
    schemas: list[FOLLayerRule],
    oracle_rule: FOLLayerRule,
) -> FOLLayerRule | None:
    for schema in schemas:
        if _find_demo_schema_instantiation(
            schema=schema,
            ground_rule=oracle_rule,
        ) is not None:
            return schema
    return None


def _find_oracle_schema_or_raise(
    ranked_rules: dict[int, list[FOLLayerRule]],
    oracle_rule: FOLLayerRule,
) -> FOLLayerRule:
    """Search rank-1 first, fallback to all ranks, raise if not found."""
    oracle_schema = _find_matching_demo_schema_for_rule(
        schemas=ranked_rules.get(1, []),
        oracle_rule=oracle_rule,
    )
    if oracle_schema is None:
        all_schemas = [r for rules in ranked_rules.values() for r in rules]
        oracle_schema = _find_matching_demo_schema_for_rule(
            schemas=all_schemas,
            oracle_rule=oracle_rule,
        )
    if oracle_schema is None:
        raise RuntimeError(
            "Oracle rule schema was not found among rules in any rank."
        )
    return oracle_schema


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
