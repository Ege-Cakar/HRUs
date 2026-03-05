"""Tests for experiment 10 fresh-rule split utility helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLRuleBank,
    FOLSequent,
    build_random_fol_rule_bank,
    sample_fol_problem,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiment" / "remote" / "10_fresh_rule_split" / "utils.py"

_SPEC = importlib.util.spec_from_file_location("fresh_rule_split_utils", MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_layer_from_predicate = _MODULE._layer_from_predicate
extract_prompt_info_from_row_tokens = _MODULE.extract_prompt_info_from_row_tokens
first_transition_mask = _MODULE.first_transition_mask
predicted_rule_reaches_goal = _MODULE.predicted_rule_reaches_goal


def _sampled_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    bank = build_random_fol_rule_bank(
        n_layers=3,
        predicates_per_layer=4,
        rules_per_transition=8,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )
    tokenizer = tok.build_tokenizer_from_rule_bank(bank)
    sampled = sample_fol_problem(
        bank=bank,
        distance=2,
        initial_ant_max=3,
        rng=rng,
    )
    return tokenizer, sampled


def test_layer_from_predicate_standard() -> None:
    assert _layer_from_predicate("r0_1") == 0
    assert _layer_from_predicate("r1_5") == 1
    assert _layer_from_predicate("r2_0") == 2


def test_layer_from_predicate_fresh() -> None:
    assert _layer_from_predicate("r_a") == 0
    assert _layer_from_predicate("r_ab") == 0
    assert _layer_from_predicate("r_a1b2") == 0
    assert _layer_from_predicate("r_zzzz") == 0
    assert _layer_from_predicate("r_0000") == 0
    assert _layer_from_predicate("r_ab12") == 0
    assert _layer_from_predicate("r_abcde") == 0


def test_layer_from_predicate_invalid() -> None:
    import pytest

    with pytest.raises(ValueError):
        _layer_from_predicate("foo")
    with pytest.raises(ValueError):
        _layer_from_predicate("r_ABC")  # uppercase not allowed


def test_extract_prompt_info_from_demo_prefixed_row_tokens() -> None:
    tokenizer, sampled = _sampled_case(seed=7)

    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    demo_completion = tokenizer.encode_completion(sampled.step_rules[0].statement_text)

    row_tokens = np.array(
        demo_completion[:-1] + [int(tokenizer.sep_token_id)] + prompt + [0, 0],
        dtype=np.int32,
    )

    prompt_prefix, sequent, src_layer, goal_layer = extract_prompt_info_from_row_tokens(
        row_tokens,
        tokenizer=tokenizer,
    )

    assert prompt_prefix[-1] == int(tokenizer.sep_token_id)
    assert sequent.text == FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom).text
    assert src_layer == 0
    assert goal_layer == 2


def test_first_transition_mask_filters_nonzero_src_layers() -> None:
    mask = first_transition_mask([0, 1, 0, 2, 0])
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, True, False, True]


def test_predicted_rule_reaches_goal_on_tiny_bank() -> None:
    rule01 = FOLLayerRule(
        src_layer=0,
        dst_layer=1,
        lhs=(FOLAtom("r0_1", ("x1",)),),
        rhs=(FOLAtom("r1_1", ("x1",)),),
    )
    rule12 = FOLLayerRule(
        src_layer=1,
        dst_layer=2,
        lhs=(FOLAtom("r1_1", ("x1",)),),
        rhs=(FOLAtom("r2_1", ("x1",)),),
    )
    bank = FOLRuleBank(
        n_layers=3,
        predicates_per_layer=1,
        arity_max=1,
        constants=("a",),
        vars_per_rule_max=1,
        predicate_arities={"r0_1": 1, "r1_1": 1, "r2_1": 1},
        transitions={0: (rule01,), 1: (rule12,)},
        layer_predicates={0: ("r0_1",), 1: ("r1_1",), 2: ("r2_1",)},
    )

    matched_rule = rule01.instantiate({"x1": "a"})

    assert predicted_rule_reaches_goal(
        rule_bank=bank,
        matched_rule=matched_rule,
        goal=FOLAtom("r2_1", ("a",)),
        goal_layer=2,
        max_unify_solutions=16,
    )
    assert not predicted_rule_reaches_goal(
        rule_bank=bank,
        matched_rule=matched_rule,
        goal=FOLAtom("r2_1", ("b",)),
        goal_layer=2,
        max_unify_solutions=16,
    )
