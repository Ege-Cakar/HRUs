from __future__ import annotations

import numpy as np
import pytest

from task.layer_fol import (
    AutoregressiveLogitsAdapter,
    CompletionLogitsAdapter,
    FOLLayerRolloutExample,
    evaluate_layer_rollouts_fol,
    evaluate_rule_matches_fol,
    match_rule_completion_fol,
    run_layer_rollout_fol,
)
from task.layer_gen.util import tokenize_layer_fol as tok
from task.layer_gen.util.fol_rule_bank import build_random_fol_rule_bank, sample_fol_problem


class _ScriptedAdapter:
    def __init__(self, completions):
        self._completions = [np.asarray(completion, dtype=np.int32) for completion in completions]
        self._idx = 0

    def predict_completion(self, *, model, prompt_tokens, tokenizer, temperature=0.0, rng=None):
        _ = model, prompt_tokens, tokenizer, temperature, rng
        if self._idx >= len(self._completions):
            raise RuntimeError("No scripted completion left.")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


def _sampled_bank(seed: int = 0):
    rng = np.random.default_rng(seed)
    bank = build_random_fol_rule_bank(
        n_layers=6,
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
    return bank, tokenizer, rng


def _make_unknown_statement(bank, src_layer: int) -> str:
    src_pred = bank.predicates_for_layer(src_layer)[0]
    dst_pred = bank.predicates_for_layer(src_layer + 1)[0]
    src_arity = bank.predicate_arities[src_pred] + 1
    dst_arity = bank.predicate_arities[dst_pred] + 1

    consts = list(bank.constants)
    lhs_args = ",".join(consts[idx % len(consts)] for idx in range(src_arity))
    rhs_args = ",".join(consts[idx % len(consts)] for idx in range(dst_arity))
    return f"{src_pred}({lhs_args}) → {dst_pred}({rhs_args})"


def _find_alternative_valid_completion(bank, tokenizer, src_layer: int, expected: str, rng) -> str | None:
    for _ in range(128):
        sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
        if sampled.step_layers[0] != src_layer:
            continue
        statement = sampled.step_rules[0].statement_text
        if statement != expected:
            return statement
    return None


def test_match_rule_completion_categorizes_errors_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=7)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer = sampled.step_layers[0]
    expected_statement = sampled.step_rules[0].statement_text

    ok = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(expected_statement),
        expected_statement_text=expected_statement,
        tokenizer=tokenizer,
    )
    assert ok.is_correct
    assert not ok.decode_error
    assert not ok.unknown_rule_error
    assert not ok.wrong_rule_error

    unknown_statement = _make_unknown_statement(bank, src_layer)
    unknown = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(unknown_statement),
        tokenizer=tokenizer,
    )
    assert unknown.unknown_rule_error
    assert not unknown.decode_error
    assert not unknown.is_valid_rule

    decode_err = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(expected_statement)[:-1],
        tokenizer=tokenizer,
    )
    assert decode_err.decode_error
    assert not decode_err.is_valid_rule


def test_evaluate_rule_matches_aggregates_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=11)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text
    wrong = _find_alternative_valid_completion(bank, tokenizer, src_layer, expected, rng)
    if wrong is None:
        pytest.skip("Could not find alternative valid completion for wrong-rule bucket test.")

    unknown = _make_unknown_statement(bank, src_layer)

    metrics = evaluate_rule_matches_fol(
        rule_bank=bank,
        src_layers=[src_layer, src_layer, src_layer, src_layer],
        completion_tokens=[
            tokenizer.encode_completion(expected),
            tokenizer.encode_completion(wrong),
            tokenizer.encode_completion(unknown),
            tokenizer.encode_completion(expected)[:-1],
        ],
        expected_statement_texts=[expected, expected, expected, expected],
        tokenizer=tokenizer,
    )
    assert metrics.n_examples == 4
    assert metrics.n_correct == 1
    assert metrics.n_wrong_rule_error == 1
    assert metrics.n_unknown_rule_error == 1
    assert metrics.n_decode_error == 1


def test_run_layer_rollout_success_with_scripted_rules_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=21)
    sampled = sample_fol_problem(bank=bank, distance=3, initial_ant_max=3, rng=rng)

    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=sampled.distance,
    )
    adapter = _ScriptedAdapter(
        [tokenizer.encode_completion(rule.statement_text) for rule in sampled.step_rules]
    )

    result = run_layer_rollout_fol(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert result.success
    assert result.goal_reached
    assert result.failure_reason is None
    assert result.n_steps == sampled.distance


def test_run_layer_rollout_unknown_rule_fails_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=22)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    unknown_statement = _make_unknown_statement(bank, sampled.start_layer)
    adapter = _ScriptedAdapter([tokenizer.encode_completion(unknown_statement)])
    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=sampled.distance,
    )

    result = run_layer_rollout_fol(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert not result.success
    assert result.failure_reason == "unknown_rule_error"


def test_run_layer_rollout_inapplicable_rule_fails_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=23)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=1, rng=rng)
    src_layer = sampled.start_layer
    initial_facts = set(sampled.step_ants[0])

    present_terms = {term for atom in initial_facts for term in atom.args}
    missing_constants = [c for c in bank.constants if c not in present_terms]
    if not missing_constants:
        pytest.skip("No missing constant available for robust inapplicable-rule construction.")

    bad_rule = None
    chosen_const = missing_constants[0]
    for rule in bank.transition_rules(src_layer):
        vars_in_rule = sorted(
            {
                term
                for atom in rule.lhs + rule.rhs
                for term in atom.args
                if term.startswith("x")
            }
        )
        instantiated = rule.instantiate({var: chosen_const for var in vars_in_rule})
        if not set(instantiated.lhs).issubset(initial_facts):
            bad_rule = instantiated
            break

    if bad_rule is None:
        pytest.skip("Could not construct an inapplicable but valid instantiated rule.")

    adapter = _ScriptedAdapter([tokenizer.encode_completion(bad_rule.statement_text)])
    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=1,
    )

    result = run_layer_rollout_fol(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert not result.success
    assert result.failure_reason == "inapplicable_rule_error"


def test_run_layer_rollout_goal_not_reached_fails_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=24)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    adapter = _ScriptedAdapter([tokenizer.encode_completion(sampled.step_rules[0].statement_text)])
    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=1,
    )

    result = run_layer_rollout_fol(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert not result.success
    assert result.failure_reason == "goal_not_reached"


def test_evaluate_layer_rollouts_aggregates_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=31)
    sampled_ok = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    sampled_bad = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    unknown_statement = _make_unknown_statement(bank, sampled_bad.start_layer)

    examples = [
        FOLLayerRolloutExample(
            distance=1,
            start_layer=sampled_ok.start_layer,
            goal_atom=sampled_ok.goal_atom.text,
            initial_ants=tuple(atom.text for atom in sampled_ok.step_ants[0]),
            max_steps=1,
        ),
        FOLLayerRolloutExample(
            distance=1,
            start_layer=sampled_bad.start_layer,
            goal_atom=sampled_bad.goal_atom.text,
            initial_ants=tuple(atom.text for atom in sampled_bad.step_ants[0]),
            max_steps=1,
        ),
    ]
    adapter = _ScriptedAdapter(
        [
            tokenizer.encode_completion(sampled_ok.step_rules[0].statement_text),
            tokenizer.encode_completion(unknown_statement),
        ]
    )

    metrics = evaluate_layer_rollouts_fol(
        rule_bank=bank,
        examples=examples,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert metrics.n_examples == 2
    assert metrics.n_success == 1
    assert metrics.n_failure_unknown_rule_error == 1


def test_completion_logits_adapter_decodes_greedy_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=41)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    completion = tokenizer.encode_completion(sampled.step_rules[0].statement_text)
    vocab = tokenizer.vocab_size
    out_len = len(completion)

    def model(_):
        logits = np.zeros((1, out_len, vocab), dtype=np.float32)
        for idx, tok_id in enumerate(completion):
            logits[0, idx, tok_id] = 10.0
        return logits

    adapter = CompletionLogitsAdapter(n_seq=64)
    prompt = tokenizer.tokenize_prompt(
        tok.parse_sequent_text(f"{sampled.step_ants[0][0].text}⊢{sampled.goal_atom.text}")
    )

    decoded = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == completion


def test_autoregressive_logits_adapter_decodes_greedy_fol() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "a", "b", "x1"])
    target = [tokenizer.char_to_id("r0_1"), tokenizer.eot_token_id]
    vocab = tokenizer.vocab_size
    call_idx = {"value": 0}

    def model(xs):
        logits = np.zeros((xs.shape[0], xs.shape[1], vocab), dtype=np.float32)
        tok_id = target[min(call_idx["value"], len(target) - 1)]
        call_idx["value"] += 1
        last_idx = int(np.max(np.where(xs[0] != 0)[0]))
        logits[0, last_idx, int(tok_id)] = 8.0
        return logits

    prompt = np.asarray([tokenizer.char_to_id("a"), tokenizer.sep_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(n_seq=16, max_completion_len=4)
    decoded = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == target
