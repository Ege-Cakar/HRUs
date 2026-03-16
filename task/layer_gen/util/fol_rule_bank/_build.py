"""Bank and split construction, rule sampling."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ._types import (
    FOLAtom,
    FOLDepth3ICLSplitBundle,
    FOLLayerRule,
    FOLRuleBank,
    _is_identifier,
    _is_variable,
    _normalize_count_spec,
    _rule_sort_key,
    _scalarize_count_spec,
    _sorted_atoms,
)


# -- Shared validation helpers (Phase 0 dedup) -----------------------------

def _validate_bank_params(
    *,
    arity_max: int,
    arity_min: int,
    vars_per_rule_max: int,
    k_in_min: int,
    k_in_max: int,
    k_out_min: int,
    k_out_max: int,
) -> None:
    """Validate the numeric parameters shared by bank construction functions."""
    if arity_max < 0:
        raise ValueError(f"arity_max must be >= 0, got {arity_max}")
    if arity_min < 0:
        raise ValueError(f"arity_min must be >= 0, got {arity_min}")
    if arity_min > arity_max:
        raise ValueError(
            f"arity_min must be <= arity_max, got arity_min={arity_min}, arity_max={arity_max}"
        )
    if vars_per_rule_max < 1:
        raise ValueError(f"vars_per_rule_max must be >= 1, got {vars_per_rule_max}")
    if k_in_min < 1 or k_out_min < 1:
        raise ValueError("k_in_min and k_out_min must be >= 1")
    if k_in_max < 1 or k_out_max < 1:
        raise ValueError("k_in_max and k_out_max must be >= 1")
    if k_in_min > k_in_max:
        raise ValueError(
            f"k_in_min must be <= k_in_max, got k_in_min={k_in_min}, k_in_max={k_in_max}"
        )
    if k_out_min > k_out_max:
        raise ValueError(
            f"k_out_min must be <= k_out_max, got k_out_min={k_out_min}, k_out_max={k_out_max}"
        )


def _validate_constants(constants: Iterable[str]) -> tuple[str, ...]:
    """Normalize and validate constant symbols, returning a tuple."""
    constants = tuple(str(tok) for tok in constants)
    if not constants:
        raise ValueError("constants must contain at least one symbol")
    for token in constants:
        if not _is_identifier(token):
            raise ValueError(f"Invalid constant symbol: {token!r}")
    return constants


# -- Internal helpers -------------------------------------------------------

def _canonicalize_rule(
    *,
    src_layer: int,
    lhs: Iterable[FOLAtom],
    rhs: Iterable[FOLAtom],
) -> FOLLayerRule:
    return FOLLayerRule(
        src_layer=int(src_layer),
        dst_layer=int(src_layer) + 1,
        lhs=_sorted_atoms(lhs),
        rhs=_sorted_atoms(rhs),
    )


def _layer_predicates(layer: int, predicates_per_layer: int) -> tuple[str, ...]:
    return tuple(f"r{int(layer)}_{idx}" for idx in range(1, predicates_per_layer + 1))


def _build_random_atom(
    *,
    predicate: str,
    predicate_arities: dict[str, int],
    term_pool: tuple[str, ...],
    rng: np.random.Generator,
) -> FOLAtom:
    arity = int(predicate_arities[predicate])
    args = tuple(
        str(term_pool[int(rng.integers(0, len(term_pool)))]) for _ in range(arity)
    )
    return FOLAtom(predicate=predicate, args=args)


def _sample_transition_rules(
    *,
    src_layer: int,
    lhs_predicates: tuple[str, ...],
    rhs_predicates: tuple[str, ...],
    rules_per_transition: int,
    k_in_min: int,
    k_in_max: int,
    k_out_min: int,
    k_out_max: int,
    predicate_arities: dict[str, int],
    var_pool: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[FOLLayerRule, ...]:
    lhs_predicates_arr = np.asarray(lhs_predicates, dtype=object)
    rhs_predicates_arr = np.asarray(rhs_predicates, dtype=object)
    if lhs_predicates_arr.size < 1 or rhs_predicates_arr.size < 1:
        raise ValueError(
            f"Transition {src_layer}->{src_layer + 1} requires non-empty predicate pools."
        )

    min_lhs = int(k_in_min)
    max_lhs = min(int(k_in_max), int(lhs_predicates_arr.size))
    min_rhs = int(k_out_min)
    max_rhs = min(int(k_out_max), int(rhs_predicates_arr.size))
    if min_lhs > max_lhs or min_rhs > max_rhs:
        raise ValueError(
            "k_in_min/k_out_min exceed available predicates or configured maxima for "
            f"transition {src_layer}->{src_layer + 1}: "
            f"k_in_min={min_lhs}, k_in_max={int(k_in_max)}, lhs_pool={int(lhs_predicates_arr.size)}, "
            f"k_out_min={min_rhs}, k_out_max={int(k_out_max)}, rhs_pool={int(rhs_predicates_arr.size)}."
        )

    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    rules: list[FOLLayerRule] = []
    attempts = 0
    max_attempts = max(1_000, int(rules_per_transition) * 250)

    while len(rules) < int(rules_per_transition) and attempts < max_attempts:
        attempts += 1
        lhs_size = int(rng.integers(min_lhs, max_lhs + 1))
        rhs_size = int(rng.integers(min_rhs, max_rhs + 1))

        lhs_chosen = [
            str(tok)
            for tok in rng.choice(lhs_predicates_arr, size=lhs_size, replace=False)
        ]
        rhs_chosen = [
            str(tok)
            for tok in rng.choice(rhs_predicates_arr, size=rhs_size, replace=False)
        ]

        lhs_atoms = tuple(
            _build_random_atom(
                predicate=pred,
                predicate_arities=predicate_arities,
                term_pool=var_pool,
                rng=rng,
            )
            for pred in lhs_chosen
        )
        lhs_vars = {
            term
            for atom in lhs_atoms
            for term in atom.args
            if _is_variable(term)
        }
        # When all LHS atoms are arity-0 there are no variables.
        # Allow the rule only if all RHS predicates also have arity 0.
        if not lhs_vars:
            all_rhs_zero = all(
                int(predicate_arities.get(pred, 1)) == 0 for pred in rhs_chosen
            )
            if not all_rhs_zero:
                continue

        rhs_atoms = tuple(
            _build_random_atom(
                predicate=pred,
                predicate_arities=predicate_arities,
                term_pool=tuple(sorted(lhs_vars)) if lhs_vars else (),
                rng=rng,
            )
            for pred in rhs_chosen
        )

        lhs = _sorted_atoms(lhs_atoms)
        rhs = _sorted_atoms(rhs_atoms)

        rhs_vars = {
            term
            for atom in rhs
            for term in atom.args
            if _is_variable(term)
        }
        if not rhs_vars.issubset(lhs_vars):
            continue

        sig = (
            tuple(atom.text for atom in lhs),
            tuple(atom.text for atom in rhs),
        )
        if sig in seen:
            continue
        seen.add(sig)

        rules.append(
            _canonicalize_rule(
                src_layer=src_layer,
                lhs=lhs,
                rhs=rhs,
            )
        )

    if len(rules) < int(rules_per_transition):
        raise RuntimeError(
            "Could not sample enough unique FOL rules for transition "
            f"{src_layer}->{src_layer + 1}."
        )

    return tuple(sorted(rules, key=_rule_sort_key))


# -- Public construction functions ------------------------------------------

def build_random_fol_rule_bank(
    *,
    n_layers: int,
    predicates_per_layer: int | Sequence[int],
    rules_per_transition: int | Sequence[int],
    arity_max: int,
    vars_per_rule_max: int,
    k_in_max: int,
    k_out_max: int,
    constants: Iterable[str],
    rng: np.random.Generator,
    k_in_min: int = 1,
    k_out_min: int = 1,
    arity_min: int = 1,
) -> FOLRuleBank:
    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2, got {n_layers}")
    predicates_per_layer_counts = _normalize_count_spec(
        predicates_per_layer,
        expected_len=int(n_layers),
        name="predicates_per_layer",
    )
    rules_per_transition_counts = _normalize_count_spec(
        rules_per_transition,
        expected_len=int(n_layers) - 1,
        name="rules_per_transition",
    )
    _validate_bank_params(
        arity_max=arity_max,
        arity_min=arity_min,
        vars_per_rule_max=vars_per_rule_max,
        k_in_min=k_in_min,
        k_in_max=k_in_max,
        k_out_min=k_out_min,
        k_out_max=k_out_max,
    )
    constants = _validate_constants(constants)

    predicate_arities: dict[str, int] = {}
    transitions: dict[int, tuple[FOLLayerRule, ...]] = {}
    layer_predicates = {
        int(layer): _layer_predicates(layer, int(predicates_per_layer_counts[layer]))
        for layer in range(int(n_layers))
    }
    var_pool = tuple(f"x{idx}" for idx in range(1, vars_per_rule_max + 1))

    for layer in range(n_layers):
        for predicate in layer_predicates[int(layer)]:
            predicate_arities[predicate] = int(rng.integers(int(arity_min), arity_max + 1))

    for src_layer in range(n_layers - 1):
        src_width = int(predicates_per_layer_counts[src_layer])
        dst_width = int(predicates_per_layer_counts[src_layer + 1])
        if int(k_in_min) > src_width or int(k_out_min) > dst_width:
            raise ValueError(
                "k_in_min and k_out_min cannot exceed the source/destination predicate "
                f"counts for transition {src_layer}->{src_layer + 1}; "
                f"got src_width={src_width}, dst_width={dst_width}, "
                f"k_in_min={k_in_min}, k_out_min={k_out_min}."
            )
        transitions[src_layer] = _sample_transition_rules(
            src_layer=int(src_layer),
            lhs_predicates=tuple(layer_predicates[int(src_layer)]),
            rhs_predicates=tuple(layer_predicates[int(src_layer) + 1]),
            rules_per_transition=int(rules_per_transition_counts[src_layer]),
            k_in_min=int(k_in_min),
            k_in_max=int(k_in_max),
            k_out_min=int(k_out_min),
            k_out_max=int(k_out_max),
            predicate_arities=predicate_arities,
            var_pool=var_pool,
            rng=rng,
        )

    return FOLRuleBank(
        n_layers=int(n_layers),
        predicates_per_layer=_scalarize_count_spec(predicates_per_layer_counts),
        arity_max=int(arity_max),
        arity_min=int(arity_min),
        constants=constants,
        vars_per_rule_max=int(vars_per_rule_max),
        predicate_arities=predicate_arities,
        transitions=transitions,
        layer_predicates=layer_predicates,
    )


# -- Fresh predicate generation --------------------------------------------

_FRESH_PREDICATE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"


def generate_fresh_predicate_names(
    n: int,
    rng: np.random.Generator,
    *,
    name_len: int = 1,
) -> tuple[str, ...]:
    """Generate ``n`` unique ``r_XX..`` predicate names with *name_len* random alphanumeric chars."""
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    name_len = int(name_len)
    if name_len < 1:
        raise ValueError(f"name_len must be >= 1, got {name_len}")
    charset = np.array(list(_FRESH_PREDICATE_CHARSET), dtype="U1")
    seen: set[str] = set()
    names: list[str] = []
    max_attempts = max(1_000, n * 50)
    for _ in range(max_attempts):
        suffix = "".join(rng.choice(charset, size=name_len))
        name = f"r_{suffix}"
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= n:
            break
    if len(names) < n:
        raise RuntimeError(
            f"Could not generate {n} unique fresh predicate names after {max_attempts} attempts."
        )
    return tuple(names)


def build_fresh_layer0_bank(
    *,
    base_bank: FOLRuleBank,
    fresh_predicates: tuple[str, ...],
    rules_per_transition: int,
    k_in_min: int,
    k_in_max: int,
    k_out_min: int,
    k_out_max: int,
    rng: np.random.Generator,
) -> FOLRuleBank:
    """Build a temporary 3-layer bank with fresh layer-0 predicates and base layer-1->2 content."""
    if int(base_bank.n_layers) != 3:
        raise ValueError("base_bank must have n_layers=3.")
    if not fresh_predicates:
        raise ValueError("fresh_predicates must be non-empty.")

    base_l1 = base_bank.predicates_for_layer(1)
    base_l2 = base_bank.predicates_for_layer(2)

    # Assign random arities to fresh predicates.
    predicate_arities: dict[str, int] = dict(base_bank.predicate_arities)
    for pred in fresh_predicates:
        predicate_arities[pred] = int(rng.integers(int(base_bank.arity_min), int(base_bank.arity_max) + 1))

    var_pool = tuple(f"x{idx}" for idx in range(1, int(base_bank.vars_per_rule_max) + 1))

    fresh_rules_01 = _sample_transition_rules(
        src_layer=0,
        lhs_predicates=tuple(fresh_predicates),
        rhs_predicates=tuple(base_l1),
        rules_per_transition=int(rules_per_transition),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )

    return FOLRuleBank(
        n_layers=3,
        predicates_per_layer=_scalarize_count_spec(
            (
                len(tuple(fresh_predicates)),
                len(tuple(base_l1)),
                len(tuple(base_l2)),
            )
        ),
        arity_max=int(base_bank.arity_max),
        arity_min=int(base_bank.arity_min),
        constants=tuple(base_bank.constants),
        vars_per_rule_max=int(base_bank.vars_per_rule_max),
        predicate_arities=predicate_arities,
        transitions={
            0: tuple(fresh_rules_01),
            1: base_bank.transitions.get(1, ()),
        },
        layer_predicates={
            0: tuple(fresh_predicates),
            1: tuple(base_l1),
            2: tuple(base_l2),
        },
    )


# -- Depth-3 ICL split bundle construction ---------------------------------

def build_depth3_icl_split_bundle(
    *,
    predicates_per_layer: int,
    rules_01_train: int,
    rules_01_eval: int,
    rules_12_shared: int,
    arity_max: int,
    vars_per_rule_max: int,
    k_in_max: int,
    k_out_max: int,
    constants: Iterable[str],
    rng: np.random.Generator,
    k_in_min: int = 1,
    k_out_min: int = 1,
    arity_min: int = 1,
) -> FOLDepth3ICLSplitBundle:
    if int(predicates_per_layer) < 1:
        raise ValueError(
            f"predicates_per_layer must be >= 1, got {predicates_per_layer}"
        )
    if int(rules_01_train) < 1 or int(rules_01_eval) < 1 or int(rules_12_shared) < 1:
        raise ValueError("All split transition rule counts must be >= 1.")
    _validate_bank_params(
        arity_max=int(arity_max),
        arity_min=int(arity_min),
        vars_per_rule_max=int(vars_per_rule_max),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
    )
    if int(k_in_min) > int(predicates_per_layer) or int(k_out_min) > int(predicates_per_layer):
        raise ValueError(
            "k_in_min and k_out_min cannot exceed predicates_per_layer for depth-3 split "
            f"generation; got predicates_per_layer={predicates_per_layer}, "
            f"k_in_min={k_in_min}, k_out_min={k_out_min}."
        )
    constants = _validate_constants(constants)

    predicates_per_layer = int(predicates_per_layer)
    arity_max = int(arity_max)
    arity_min = int(arity_min)
    vars_per_rule_max = int(vars_per_rule_max)
    k_in_min = int(k_in_min)
    k_in_max = int(k_in_max)
    k_out_min = int(k_out_min)
    k_out_max = int(k_out_max)

    all_layer0_indices = np.arange(1, 2 * predicates_per_layer + 1, dtype=np.int32)
    shuffled_layer0_indices = rng.permutation(all_layer0_indices)
    train_layer0_indices = tuple(
        sorted(int(idx) for idx in shuffled_layer0_indices[:predicates_per_layer].tolist())
    )
    eval_layer0_indices = tuple(
        sorted(int(idx) for idx in shuffled_layer0_indices[predicates_per_layer:].tolist())
    )
    train_layer0 = tuple(f"r0_{idx}" for idx in train_layer0_indices)
    eval_layer0 = tuple(f"r0_{idx}" for idx in eval_layer0_indices)
    shared_layer1 = _layer_predicates(1, predicates_per_layer)
    shared_layer2 = _layer_predicates(2, predicates_per_layer)

    predicate_arities_all: dict[str, int] = {}
    for predicate in train_layer0 + eval_layer0 + shared_layer1 + shared_layer2:
        if predicate in predicate_arities_all:
            continue
        predicate_arities_all[predicate] = int(rng.integers(arity_min, arity_max + 1))

    train_predicate_arities = {
        predicate: int(predicate_arities_all[predicate])
        for predicate in train_layer0 + shared_layer1 + shared_layer2
    }
    eval_predicate_arities = {
        predicate: int(predicate_arities_all[predicate])
        for predicate in eval_layer0 + shared_layer1 + shared_layer2
    }

    var_pool = tuple(f"x{idx}" for idx in range(1, vars_per_rule_max + 1))
    shared_rules_12 = _sample_transition_rules(
        src_layer=1,
        lhs_predicates=tuple(shared_layer1),
        rhs_predicates=tuple(shared_layer2),
        rules_per_transition=int(rules_12_shared),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=train_predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )
    train_rules_01 = _sample_transition_rules(
        src_layer=0,
        lhs_predicates=tuple(train_layer0),
        rhs_predicates=tuple(shared_layer1),
        rules_per_transition=int(rules_01_train),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=train_predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )
    eval_rules_01 = _sample_transition_rules(
        src_layer=0,
        lhs_predicates=tuple(eval_layer0),
        rhs_predicates=tuple(shared_layer1),
        rules_per_transition=int(rules_01_eval),
        k_in_min=int(k_in_min),
        k_in_max=int(k_in_max),
        k_out_min=int(k_out_min),
        k_out_max=int(k_out_max),
        predicate_arities=eval_predicate_arities,
        var_pool=var_pool,
        rng=rng,
    )

    train_bank = FOLRuleBank(
        n_layers=3,
        predicates_per_layer=predicates_per_layer,
        arity_max=arity_max,
        arity_min=arity_min,
        constants=constants,
        vars_per_rule_max=vars_per_rule_max,
        predicate_arities=train_predicate_arities,
        transitions={
            0: tuple(train_rules_01),
            1: tuple(shared_rules_12),
        },
        layer_predicates={
            0: tuple(train_layer0),
            1: tuple(shared_layer1),
            2: tuple(shared_layer2),
        },
    )
    eval_bank = FOLRuleBank(
        n_layers=3,
        predicates_per_layer=predicates_per_layer,
        arity_max=arity_max,
        arity_min=arity_min,
        constants=constants,
        vars_per_rule_max=vars_per_rule_max,
        predicate_arities=eval_predicate_arities,
        transitions={
            0: tuple(eval_rules_01),
            1: tuple(shared_rules_12),
        },
        layer_predicates={
            0: tuple(eval_layer0),
            1: tuple(shared_layer1),
            2: tuple(shared_layer2),
        },
    )
    return FOLDepth3ICLSplitBundle(
        train_bank=train_bank,
        eval_bank=eval_bank,
        train_layer0_indices=tuple(train_layer0_indices),
        eval_layer0_indices=tuple(eval_layer0_indices),
        train_layer0_predicates=tuple(train_layer0),
        eval_layer0_predicates=tuple(eval_layer0),
        shared_layer1_predicates=tuple(shared_layer1),
        shared_layer2_predicates=tuple(shared_layer2),
    )
