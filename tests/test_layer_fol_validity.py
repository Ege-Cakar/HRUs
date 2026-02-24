from __future__ import annotations

import numpy as np

from task.layer_fol import completion_is_valid_for_layer_fol
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import build_random_fol_rule_bank, sample_fol_problem


def test_completion_lookup_validity_fol() -> None:
    rng = np.random.default_rng(7)
    bank = build_random_fol_rule_bank(
        n_layers=5,
        predicates_per_layer=4,
        rules_per_transition=6,
        arity_max=3,
        vars_per_rule_max=4,
        constants=("a", "b", "c"),
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )
    tokenizer = tok.build_tokenizer_from_rule_bank(bank)

    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer = sampled.step_layers[0]
    valid_completion = tokenizer.encode_completion(sampled.step_rules[0].statement_text)

    assert completion_is_valid_for_layer_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=valid_completion,
        tokenizer=tokenizer,
    )

    other_layer = 0 if src_layer != 0 else 2
    assert not completion_is_valid_for_layer_fol(
        rule_bank=bank,
        src_layer=other_layer,
        completion_tokens=valid_completion,
        tokenizer=tokenizer,
    )

    bad_completion = valid_completion[:-1]
    assert not completion_is_valid_for_layer_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=bad_completion,
        tokenizer=tokenizer,
    )
