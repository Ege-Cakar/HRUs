from __future__ import annotations

import numpy as np

from task.layer_axiom import completion_is_valid_for_layer
from task.layer_gen.util import tokenize_layer_axiom as tok
from task.layer_gen.util.rule_bank import build_random_rule_bank


def test_completion_lookup_validity() -> None:
    rng = np.random.default_rng(7)
    bank = build_random_rule_bank(
        n_layers=5,
        props_per_layer=4,
        rules_per_transition=6,
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )

    src_layer = 1
    rule = bank.transition_rules(src_layer)[0]
    valid_completion = tok.encode_completion(rule.statement_text)
    assert completion_is_valid_for_layer(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=valid_completion,
    )

    # Same statement should generally be invalid for a different transition.
    other_layer = 0 if src_layer != 0 else 2
    assert not completion_is_valid_for_layer(
        rule_bank=bank,
        src_layer=other_layer,
        completion_tokens=valid_completion,
    )

    bad_completion = valid_completion[:-1]
    assert not completion_is_valid_for_layer(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=bad_completion,
    )
