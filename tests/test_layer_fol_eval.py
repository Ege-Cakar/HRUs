from __future__ import annotations

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from task.layer_fol import (
    AutoregressiveLogitsAdapter,
    CompletionLogitsAdapter,
    FAILURE_WRONG_RULE_ERROR,
    FOLDemoAugmentedAdapter,
    FOLLayerRolloutExample,
    _find_instantiation_for_rule,
    evaluate_completion_paths_fol,
    evaluate_layer_rollouts_fol,
    evaluate_rule_matches_fol,
    match_rule_completion_fol,
    run_layer_rollout_fol,
    validate_completion_path_fol,
)
from task.layer_fol.demos import _find_lhs_substitutions_for_facts, _subst_binds_rhs_variables
from task.layer_gen.util.fol_rule_bank import (
    FOLAtom,
    FOLLayerRule,
    FOLSequent,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    sample_fol_problem,
)
from task.layer_gen.util import tokenize_layer_fol as tok


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


class _ScriptedAdapterWithDemoRules(_ScriptedAdapter):
    def __init__(self, completions, demo_rules):
        super().__init__(completions)
        self._demo_rules = list(demo_rules)

    def get_last_demo_rules(self):
        return list(self._demo_rules)


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


def _enc1(tokenizer, statement_text: str) -> list[int]:
    return tokenizer.encode_completion_texts([statement_text])


def _encs(tokenizer, statement_texts) -> list[int]:
    return tokenizer.encode_completion_texts(statement_texts)


def _dec1(tokenizer, completion_tokens) -> str:
    statements = tokenizer.decode_completion_texts(completion_tokens)
    assert len(statements) == 1
    return statements[0]


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


def _find_alternative_valid_path(bank, sampled):
    goal = sampled.goal_atom
    oracle_texts = tuple(rule.statement_text for rule in sampled.step_rules)

    def _search(layer: int, facts: tuple[FOLAtom, ...], steps_remaining: int):
        if steps_remaining == 0:
            return [] if goal in set(facts) else None

        for rule in bank.transition_rules(layer):
            substitutions = _find_lhs_substitutions_for_facts(
                lhs=rule.lhs,
                facts=facts,
                max_solutions=64,
            )
            for subst in substitutions:
                if not _subst_binds_rhs_variables(rule=rule, subst=subst):
                    continue
                instantiated = rule.instantiate(subst)
                suffix = _search(
                    int(instantiated.dst_layer),
                    tuple(instantiated.rhs),
                    int(steps_remaining) - 1,
                )
                if suffix is not None:
                    return [instantiated, *suffix]
        return None

    path = _search(
        int(sampled.step_layers[0]),
        tuple(sampled.step_ants[0]),
        int(sampled.distance),
    )
    if path is None:
        return None
    if tuple(rule.statement_text for rule in path) == oracle_texts:
        return None
    return path


def test_match_rule_completion_categorizes_errors_fol() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=7)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer = sampled.step_layers[0]
    expected_statement = sampled.step_rules[0].statement_text

    ok = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=_enc1(tokenizer, expected_statement),
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
        completion_tokens=_enc1(tokenizer, unknown_statement),
        tokenizer=tokenizer,
    )
    assert unknown.unknown_rule_error
    assert not unknown.decode_error
    assert not unknown.is_valid_rule

    decode_err = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=_enc1(tokenizer, expected_statement)[:-1],
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
            _enc1(tokenizer, expected),
            _enc1(tokenizer, wrong),
            _enc1(tokenizer, unknown),
            _enc1(tokenizer, expected)[:-1],
        ],
        expected_statement_texts=[expected, expected, expected, expected],
        tokenizer=tokenizer,
    )
    assert metrics.n_examples == 4
    assert metrics.n_correct == 1
    assert metrics.n_wrong_rule_error == 1
    assert metrics.n_unknown_rule_error == 1
    assert metrics.n_decode_error == 1


def test_evaluate_rule_matches_uses_demo_rules_by_example() -> None:
    base_bank, temp_bank, fresh_preds, tokenizer, rng = _fresh_demo_setup(seed=73)
    _ = fresh_preds

    sampled = sample_fol_problem(bank=temp_bank, distance=1, initial_ant_max=3, rng=rng)
    while sampled.start_layer != 0:
        sampled = sample_fol_problem(bank=temp_bank, distance=1, initial_ant_max=3, rng=rng)

    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text

    demo_rule = None
    for rule in temp_bank.transition_rules(src_layer):
        subst = _find_instantiation_for_rule(
            template=rule,
            lhs_ground=sampled.step_rules[0].lhs,
            rhs_ground=sampled.step_rules[0].rhs,
        )
        if subst is not None:
            demo_rule = rule
            break
    assert demo_rule is not None

    metrics_without_demo = evaluate_rule_matches_fol(
        rule_bank=base_bank,
        src_layers=[src_layer],
        completion_tokens=[_enc1(tokenizer, expected)],
        expected_statement_texts=[expected],
        tokenizer=tokenizer,
    )
    assert metrics_without_demo.n_correct == 0

    metrics_with_demo = evaluate_rule_matches_fol(
        rule_bank=base_bank,
        src_layers=[src_layer],
        completion_tokens=[_enc1(tokenizer, expected)],
        expected_statement_texts=[expected],
        tokenizer=tokenizer,
        demo_rules_by_example=[[demo_rule]],
    )
    assert metrics_with_demo.n_correct == 1
    assert metrics_with_demo.n_wrong_rule_error == 0
    assert metrics_with_demo.n_unknown_rule_error == 0
    assert metrics_with_demo.n_decode_error == 0


def test_evaluate_rule_matches_rejects_demo_rules_by_example_len_mismatch() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=74)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text

    with pytest.raises(ValueError, match="demo_rules_by_example"):
        evaluate_rule_matches_fol(
            rule_bank=bank,
            src_layers=[src_layer, src_layer],
            completion_tokens=[
                _enc1(tokenizer, expected),
                _enc1(tokenizer, expected),
            ],
            expected_statement_texts=[expected, expected],
            tokenizer=tokenizer,
            demo_rules_by_example=[[]],
        )


def test_validate_completion_path_accepts_gold_full_sequence() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=81)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    completion = _encs(
        tokenizer,
        [rule.statement_text for rule in sampled.step_rules]
    )

    result = validate_completion_path_fol(
        rule_bank=bank,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tokenizer=tokenizer,
    )

    assert result.success
    assert result.goal_reached
    assert result.n_steps == 2


def test_validate_completion_path_accepts_demo_prefixed_prompt_row() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=181)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    completion = _encs(
        tokenizer,
        [rule.statement_text for rule in sampled.step_rules]
    )
    demo = _enc1(tokenizer, sampled.step_rules[0].statement_text)
    prompt_row = np.array(
        demo[:-1] + [int(tokenizer.sep_token_id)] + prompt + [0, 0],
        dtype=np.int32,
    )

    result = validate_completion_path_fol(
        rule_bank=bank,
        prompt_tokens=prompt_row,
        completion_tokens=completion,
        tokenizer=tokenizer,
    )

    assert result.success
    assert result.goal_reached
    assert result.n_steps == len(sampled.step_rules)


def test_validate_completion_path_accepts_alternative_goal_reaching_sequence() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=82)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    alternative = _find_alternative_valid_path(bank, sampled)
    if alternative is None:
        pytest.skip("Could not find alternative valid first step for full-path test.")

    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    completion = _encs(
        tokenizer,
        [rule.statement_text for rule in alternative]
    )

    result = validate_completion_path_fol(
        rule_bank=bank,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tokenizer=tokenizer,
    )

    assert result.success
    assert result.goal_reached
    assert result.n_steps == len(alternative)


def test_evaluate_completion_paths_aggregates_failures() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=83)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    gold = _encs(tokenizer, [rule.statement_text for rule in sampled.step_rules])
    bad = gold[:-1]

    metrics = evaluate_completion_paths_fol(
        rule_bank=bank,
        prompt_tokens=[prompt, prompt],
        completion_tokens=[gold, bad],
        tokenizer=tokenizer,
    )

    assert metrics.n_examples == 2
    assert metrics.n_success == 1
    assert metrics.n_failure_decode_error == 1


def test_evaluate_completion_paths_accepts_demo_prefixed_prompt_rows() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=182)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    demo = _enc1(tokenizer, sampled.step_rules[0].statement_text)
    prompt_row = np.array(
        demo[:-1] + [int(tokenizer.sep_token_id)] + prompt + [0, 0],
        dtype=np.int32,
    )
    completion = _encs(
        tokenizer,
        [rule.statement_text for rule in sampled.step_rules]
    )

    metrics = evaluate_completion_paths_fol(
        rule_bank=bank,
        prompt_tokens=[prompt_row],
        completion_tokens=[completion],
        tokenizer=tokenizer,
    )

    assert metrics.n_examples == 1
    assert metrics.n_success == 1
    assert metrics.n_failure_decode_error == 0


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
        [_enc1(tokenizer, rule.statement_text) for rule in sampled.step_rules]
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
    adapter = _ScriptedAdapter([_enc1(tokenizer, unknown_statement)])
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

    adapter = _ScriptedAdapter([_enc1(tokenizer, bad_rule.statement_text)])
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
    adapter = _ScriptedAdapter([_enc1(tokenizer, sampled.step_rules[0].statement_text)])
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
            _enc1(tokenizer, sampled_ok.step_rules[0].statement_text),
            _enc1(tokenizer, unknown_statement),
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
    completion = _enc1(tokenizer, sampled.step_rules[0].statement_text)
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
    target = [tokenizer.char_to_id("r"), tokenizer.eot_token_id]
    vocab = tokenizer.vocab_size
    call_idx = {"value": 0}

    def model(xs):
        logits = np.zeros((xs.shape[0], xs.shape[1], vocab), dtype=np.float32)
        tok_id = target[min(call_idx["value"], len(target) - 1)]
        call_idx["value"] += 1
        last_idx = int(np.max(np.where(xs[0] != 0)[0]))
        logits[0, last_idx, int(tok_id)] = 8.0
        return logits

    prompt = np.asarray([tokenizer.char_to_id("a"), tokenizer.start_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(n_seq=16, max_completion_len=4)
    decoded = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == target


def test_autoregressive_logits_adapter_uses_cache_when_supported_fol() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "a", "b", "x1"])
    target = [tokenizer.char_to_id("r"), tokenizer.eot_token_id]
    vocab = tokenizer.vocab_size
    calls: list[tuple[tuple[int, ...], int | None, bool]] = []

    def model(xs, *, cache=None, return_cache=False):
        calls.append((xs.shape, cache, return_cache))
        assert return_cache
        step = 0 if cache is None else int(cache)
        logits = np.zeros((xs.shape[0], xs.shape[1], vocab), dtype=np.float32)
        tok_id = target[min(step, len(target) - 1)]
        logits[0, -1, int(tok_id)] = 9.0
        return logits, step + 1

    prompt = np.asarray([tokenizer.char_to_id("a"), tokenizer.start_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(n_seq=16, max_completion_len=4)
    decoded = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )

    assert decoded.tolist() == target
    assert calls[0] == ((1, prompt.shape[0]), None, True)
    assert calls[1][0] == (1, 1)
    assert calls[1][1] == 1
    assert calls[1][2] is True


def test_autoregressive_logits_adapter_jit_step_uses_cache_when_supported_fol() -> None:
    tokenizer = tok.build_tokenizer_from_identifiers(["r0_1", "a", "b", "x1"])
    first_tok = tokenizer.char_to_id("r")
    target = [int(first_tok), int(tokenizer.eot_token_id)]
    vocab = int(tokenizer.vocab_size)

    class _CacheAwareModel(nnx.Module):
        def __init__(self, *, vocab_size: int):
            self.vocab_size = int(vocab_size)

        def __call__(self, xs, *, cache=None, return_cache=False):
            if not return_cache:
                raise RuntimeError("jit_step cache path should call return_cache=True.")
            step = jnp.asarray(0, dtype=jnp.int32) if cache is None else jnp.asarray(cache, dtype=jnp.int32)
            logits = jnp.zeros((xs.shape[0], xs.shape[1], self.vocab_size), dtype=jnp.float32)
            tok = jnp.where(step == 0, first_tok, tokenizer.eot_token_id)
            logits = logits.at[0, -1, tok].set(8.0)
            return logits, step + 1

    prompt = np.asarray([tokenizer.char_to_id("a"), tokenizer.start_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(
        n_seq=16,
        max_completion_len=4,
        jit_step=True,
    )
    decoded = adapter.predict_completion(
        model=_CacheAwareModel(vocab_size=vocab),
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == target


# ── Schema-matching tests ──────────────────────────────────────────────────────


def _rename_predicates(rule: FOLLayerRule, pred_map: dict[str, str]) -> FOLLayerRule:
    """Return a new rule with predicate names renamed according to *pred_map*."""
    def _rename_atom(atom: FOLAtom) -> FOLAtom:
        return FOLAtom(pred_map.get(atom.predicate, atom.predicate), atom.args)

    return FOLLayerRule(
        src_layer=rule.src_layer,
        dst_layer=rule.dst_layer,
        lhs=tuple(_rename_atom(a) for a in rule.lhs),
        rhs=tuple(_rename_atom(a) for a in rule.rhs),
    )


def _bank_identifiers(bank):
    """Collect identifiers the tokenizer needs for *bank*."""
    ids: set[str] = set(bank.constants)
    ids.update(bank.predicate_arities)
    ids.update(f"x{idx}" for idx in range(1, int(bank.vars_per_rule_max) + 1))
    return sorted(ids)


def _extended_tokenizer(bank, extra_predicates):
    """Build a tokenizer that knows both *bank* identifiers and *extra_predicates*."""
    ids = _bank_identifiers(bank)
    for name in sorted(extra_predicates):
        if name not in ids:
            ids.append(name)
    return tok.build_tokenizer_from_identifiers(ids)


def test_schema_match_renamed_predicates_is_wrong_rule() -> None:
    """A rule with the right structure but renamed predicates should be
    classified as wrong_rule_error (not unknown_rule_error)."""
    bank, tokenizer, rng = _sampled_bank(seed=50)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer = sampled.step_layers[0]
    rule = sampled.step_rules[0]

    # Rename to fresh predicate names that match _PREDICATE_RE (r\d+_\d+)
    # but don't exist in the bank (which uses layers 0-5)
    src_preds = list(bank.predicates_for_layer(src_layer))
    dst_preds = list(bank.predicates_for_layer(src_layer + 1))
    pred_map = {}
    for idx, pred in enumerate(src_preds):
        pred_map[pred] = f"r90_{idx + 1}"
    for idx, pred in enumerate(dst_preds):
        pred_map[pred] = f"r91_{idx + 1}"

    renamed = _rename_predicates(rule, pred_map)
    # Instantiate the renamed rule so it's ground
    subst = {var: list(bank.constants)[i % len(bank.constants)]
             for i, var in enumerate(sorted(renamed.variables()))}
    instantiated = renamed.instantiate(subst)

    fresh_tokenizer = _extended_tokenizer(bank, set(pred_map.values()))

    result = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=fresh_tokenizer.encode_completion_texts([instantiated.statement_text]),
        tokenizer=fresh_tokenizer,
    )
    # Schema should match (same arity pattern, same variable binding structure)
    assert not result.unknown_rule_error, "Expected schema match, got unknown_rule_error"
    assert result.wrong_rule_error, "Expected wrong_rule_error for renamed predicates"
    assert result.matched_rule is None
    assert not result.is_valid_rule


def test_schema_mismatch_wrong_arity_is_unknown_rule() -> None:
    """A rule with wrong arities should remain unknown_rule_error."""
    bank, tokenizer, rng = _sampled_bank(seed=51)
    src_layer = 0
    unknown_statement = _make_unknown_statement(bank, src_layer)

    result = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=_enc1(tokenizer, unknown_statement),
        tokenizer=tokenizer,
    )
    # _make_unknown_statement inflates arities, so schema should NOT match
    assert result.unknown_rule_error
    assert not result.wrong_rule_error


def test_schema_match_with_demo_rules() -> None:
    """When demo_rules are provided, schema matching should consider them."""
    bank, tokenizer, rng = _sampled_bank(seed=52)
    src_layer = 0
    # Pick a bank rule to use as a structural template
    template_rule = bank.transition_rules(src_layer)[0]

    # Create a fresh rule with renamed predicates (using r\d+_\d+ format)
    src_preds = bank.predicates_for_layer(src_layer)
    dst_preds = bank.predicates_for_layer(src_layer + 1)
    pred_map = {}
    for idx, pred in enumerate(src_preds):
        pred_map[pred] = f"r80_{idx + 1}"
    for idx, pred in enumerate(dst_preds):
        pred_map[pred] = f"r81_{idx + 1}"

    demo_rule = _rename_predicates(template_rule, pred_map)

    # Instantiate the demo rule
    subst = {var: list(bank.constants)[i % len(bank.constants)]
             for i, var in enumerate(sorted(demo_rule.variables()))}
    instantiated = demo_rule.instantiate(subst)

    demo_tokenizer = _extended_tokenizer(bank, set(pred_map.values()))

    # Without demo_rules: schema still matches bank rules (same arity pattern)
    result_no_demo = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=demo_tokenizer.encode_completion_texts([instantiated.statement_text]),
        tokenizer=demo_tokenizer,
    )
    # Schema should match against bank rules since the structural pattern is the same
    assert not result_no_demo.unknown_rule_error

    # With demo_rules: should now exact-match the demo rule
    result_with_demo = match_rule_completion_fol(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=demo_tokenizer.encode_completion_texts([instantiated.statement_text]),
        tokenizer=demo_tokenizer,
        demo_rules=[demo_rule],
    )
    assert not result_with_demo.unknown_rule_error
    assert result_with_demo.matched_rule is not None
    # No expected_statement_text provided, so is_correct=True
    assert result_with_demo.is_correct


def test_rollout_wrong_rule_failure_reason() -> None:
    """A rollout where the model predicts a schema-matched but wrong rule
    should fail with FAILURE_WRONG_RULE_ERROR."""
    bank, tokenizer, rng = _sampled_bank(seed=53)
    sampled = sample_fol_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    src_layer = sampled.start_layer
    rule = sampled.step_rules[0]

    # Rename predicates to produce a schema-matched but unrecognised rule
    src_preds = list(bank.predicates_for_layer(src_layer))
    dst_preds = list(bank.predicates_for_layer(src_layer + 1))
    pred_map = {}
    for idx, pred in enumerate(src_preds):
        pred_map[pred] = f"r70_{idx + 1}"
    for idx, pred in enumerate(dst_preds):
        pred_map[pred] = f"r71_{idx + 1}"

    renamed = _rename_predicates(rule, pred_map)
    subst = {var: list(bank.constants)[i % len(bank.constants)]
             for i, var in enumerate(sorted(renamed.variables()))}
    instantiated = renamed.instantiate(subst)

    rollout_tokenizer = _extended_tokenizer(bank, set(pred_map.values()))

    adapter = _ScriptedAdapter([rollout_tokenizer.encode_completion_texts([instantiated.statement_text])])
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
        tokenizer=rollout_tokenizer,
    )
    assert not result.success
    assert result.failure_reason == FAILURE_WRONG_RULE_ERROR


def test_rollout_succeeds_with_fresh_predicate_bank() -> None:
    """A rollout using a temp_bank that contains fresh layer-0 rules should
    succeed when the model predicts the correct fresh-predicate completions.

    This mirrors the production scenario in experiment 10 where
    ``rule_bank=temp_bank`` is built via ``build_fresh_layer0_bank``.
    """
    rng = np.random.default_rng(60)
    base_bank = build_random_fol_rule_bank(
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

    fresh_preds = generate_fresh_predicate_names(4, rng)
    temp_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=8,
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=2,
        rng=rng,
    )

    # Sample a problem that starts at layer 0 (fresh predicates)
    sampled = sample_fol_problem(
        bank=temp_bank,
        distance=2,
        initial_ant_max=3,
        rng=rng,
    )
    assert sampled.start_layer == 0, "Need a layer-0 start for fresh-predicate test"
    # Verify layer-0 predicates are indeed fresh
    layer0_preds = {atom.predicate for atom in sampled.step_ants[0]}
    assert layer0_preds.issubset(set(fresh_preds)), (
        f"Expected fresh predicates {fresh_preds}, got {layer0_preds}"
    )

    tokenizer = tok.build_tokenizer_from_rule_bank(temp_bank)

    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=sampled.distance,
    )
    adapter = _ScriptedAdapter(
        [_enc1(tokenizer, rule.statement_text) for rule in sampled.step_rules]
    )

    result = run_layer_rollout_fol(
        rule_bank=temp_bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert result.success, f"Expected success, got failure_reason={result.failure_reason}"
    assert result.goal_reached
    assert result.failure_reason is None
    assert result.n_steps == sampled.distance


def test_rollout_uses_adapter_demo_rules() -> None:
    base_bank, temp_bank, fresh_preds, tokenizer, rng = _fresh_demo_setup(seed=75)
    _ = fresh_preds

    sampled = sample_fol_problem(bank=temp_bank, distance=1, initial_ant_max=3, rng=rng)
    while sampled.start_layer != 0:
        sampled = sample_fol_problem(bank=temp_bank, distance=1, initial_ant_max=3, rng=rng)

    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text
    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=sampled.distance,
    )

    demo_rule = None
    for rule in temp_bank.transition_rules(src_layer):
        subst = _find_instantiation_for_rule(
            template=rule,
            lhs_ground=sampled.step_rules[0].lhs,
            rhs_ground=sampled.step_rules[0].rhs,
        )
        if subst is not None:
            demo_rule = rule
            break
    assert demo_rule is not None

    no_demo_adapter = _ScriptedAdapter([_enc1(tokenizer, expected)])
    no_demo_result = run_layer_rollout_fol(
        rule_bank=base_bank,
        example=example,
        model=lambda x: x,
        adapter=no_demo_adapter,
        tokenizer=tokenizer,
    )
    assert not no_demo_result.success
    assert no_demo_result.failure_reason == FAILURE_WRONG_RULE_ERROR

    with_demo_adapter = _ScriptedAdapterWithDemoRules(
        [_enc1(tokenizer, expected)],
        [demo_rule],
    )
    with_demo_result = run_layer_rollout_fol(
        rule_bank=base_bank,
        example=example,
        model=lambda x: x,
        adapter=with_demo_adapter,
        tokenizer=tokenizer,
    )
    assert with_demo_result.success
    assert with_demo_result.failure_reason is None


def test_demo_augmented_adapter_include_oracle_records_oracle_rule() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=76)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    adapter = FOLDemoAugmentedAdapter(
        base_adapter=_ScriptedAdapter([_enc1(tokenizer, sampled.step_rules[0].statement_text)]),
        rule_bank=bank,
        tokenizer=tokenizer,
        min_n_demos=2,
        max_n_demos=2,
        max_unify_solutions=64,
        include_oracle=True,
    )
    adapter.set_oracle_rule(sampled.step_rules[0])
    _ = adapter.predict_completion(
        model=lambda x: x,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
        rng=np.random.default_rng(0),
    )
    demo_rules = adapter.get_last_demo_rules()
    assert len(demo_rules) == 2
    assert any(
        _find_instantiation_for_rule(
            template=demo_rule,
            lhs_ground=sampled.step_rules[0].lhs,
            rhs_ground=sampled.step_rules[0].rhs,
        )
        is not None
        for demo_rule in demo_rules
    )


def test_demo_augmented_adapter_include_oracle_requires_oracle_rule() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=77)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    prompt = tokenizer.tokenize_prompt(
        FOLSequent(ants=sampled.step_ants[0], cons=sampled.goal_atom)
    )
    adapter = FOLDemoAugmentedAdapter(
        base_adapter=_ScriptedAdapter([_enc1(tokenizer, sampled.step_rules[0].statement_text)]),
        rule_bank=bank,
        tokenizer=tokenizer,
        min_n_demos=1,
        max_n_demos=1,
        max_unify_solutions=64,
        include_oracle=True,
    )
    with pytest.raises(ValueError, match="oracle_rule"):
        adapter.predict_completion(
            model=lambda x: x,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            rng=np.random.default_rng(0),
        )


def test_rollout_sets_adapter_oracle_rule_for_demo_augmentation() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=78)
    sampled = sample_fol_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    example = FOLLayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom.text,
        initial_ants=tuple(atom.text for atom in sampled.step_ants[0]),
        max_steps=sampled.distance,
        oracle_rule_statements=(str(sampled.step_rules[0].statement_text),),
    )
    adapter = FOLDemoAugmentedAdapter(
        base_adapter=_ScriptedAdapter([_enc1(tokenizer, sampled.step_rules[0].statement_text)]),
        rule_bank=bank,
        tokenizer=tokenizer,
        min_n_demos=1,
        max_n_demos=1,
        max_unify_solutions=64,
        include_oracle=True,
    )

    result = run_layer_rollout_fol(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )

    assert result.success
    demo_rules = adapter.get_last_demo_rules()
    assert len(demo_rules) == 1
    assert (
        _find_instantiation_for_rule(
            template=demo_rules[0],
            lhs_ground=sampled.step_rules[0].lhs,
            rhs_ground=sampled.step_rules[0].rhs,
        )
        is not None
    )


# ── Demo-rule exact matching tests ─────────────────────────────────────────────


def _fresh_demo_setup(seed: int = 70):
    """Build a base bank + fresh layer-0 bank and return useful objects."""
    rng = np.random.default_rng(seed)
    base_bank = build_random_fol_rule_bank(
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
    fresh_preds = generate_fresh_predicate_names(4, rng)
    temp_bank = build_fresh_layer0_bank(
        base_bank=base_bank,
        fresh_predicates=fresh_preds,
        rules_per_transition=8,
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=2,
        rng=rng,
    )
    tokenizer = tok.build_tokenizer_from_rule_bank(temp_bank)
    return base_bank, temp_bank, fresh_preds, tokenizer, rng


def test_match_rule_demo_rules_exact_instantiation() -> None:
    """A prediction matching a demo rule (with fresh predicates not in the base
    bank) should be classified as correct when demo_rules are provided."""
    base_bank, temp_bank, fresh_preds, tokenizer, rng = _fresh_demo_setup(seed=70)

    sampled = sample_fol_problem(
        bank=temp_bank, distance=1, initial_ant_max=3, rng=rng,
    )
    # Ensure we got a layer-0 rule involving fresh predicates
    while sampled.start_layer != 0:
        sampled = sample_fol_problem(
            bank=temp_bank, distance=1, initial_ant_max=3, rng=rng,
        )

    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text

    # The demo rule is the *template* (uninstantiated) rule from temp_bank
    demo_rule = None
    for rule in temp_bank.transition_rules(src_layer):
        subst = _find_instantiation_for_rule(
            template=rule,
            lhs_ground=sampled.step_rules[0].lhs,
            rhs_ground=sampled.step_rules[0].rhs,
        )
        if subst is not None:
            demo_rule = rule
            break
    assert demo_rule is not None, "Could not find template rule for sampled problem"

    result = match_rule_completion_fol(
        rule_bank=base_bank,
        src_layer=src_layer,
        completion_tokens=_enc1(tokenizer, expected),
        expected_statement_text=expected,
        tokenizer=tokenizer,
        demo_rules=[demo_rule],
    )
    assert result.matched_rule is not None, "Expected matched_rule via demo_rules"
    assert result.is_correct
    assert not result.wrong_rule_error
    assert not result.unknown_rule_error
    assert result.match_source == "demo"


def test_match_rule_demo_rules_wrong_prediction() -> None:
    """A prediction that matches a different demo rule (not the expected one)
    should be classified as wrong_rule_error with matched_rule set."""
    base_bank, temp_bank, fresh_preds, tokenizer, rng = _fresh_demo_setup(seed=71)

    # Collect two distinct layer-0 rules from temp_bank
    layer0_rules = list(temp_bank.transition_rules(0))
    assert len(layer0_rules) >= 2, "Need at least 2 layer-0 rules"

    rule_a = layer0_rules[0]
    rule_b = layer0_rules[1]

    # Instantiate both rules
    consts = list(temp_bank.constants)
    subst_a = {var: consts[i % len(consts)] for i, var in enumerate(sorted(rule_a.variables()))}
    inst_a = rule_a.instantiate(subst_a)
    subst_b = {var: consts[i % len(consts)] for i, var in enumerate(sorted(rule_b.variables()))}
    inst_b = rule_b.instantiate(subst_b)

    # Predict inst_a but expect inst_b
    result = match_rule_completion_fol(
        rule_bank=base_bank,
        src_layer=0,
        completion_tokens=_enc1(tokenizer, inst_a.statement_text),
        expected_statement_text=inst_b.statement_text,
        tokenizer=tokenizer,
        demo_rules=[rule_a, rule_b],
    )
    assert result.matched_rule is not None, "Expected matched_rule via demo_rules"
    assert result.wrong_rule_error
    assert not result.is_correct


def test_match_rule_fresh_bank_correct() -> None:
    """Using temp_bank directly as rule_bank, a correct fresh-predicate
    prediction should be classified as correct."""
    base_bank, temp_bank, fresh_preds, tokenizer, rng = _fresh_demo_setup(seed=72)

    sampled = sample_fol_problem(
        bank=temp_bank, distance=1, initial_ant_max=3, rng=rng,
    )
    while sampled.start_layer != 0:
        sampled = sample_fol_problem(
            bank=temp_bank, distance=1, initial_ant_max=3, rng=rng,
        )

    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text

    result = match_rule_completion_fol(
        rule_bank=temp_bank,
        src_layer=src_layer,
        completion_tokens=_enc1(tokenizer, expected),
        expected_statement_text=expected,
        tokenizer=tokenizer,
    )
    assert result.is_correct
    assert result.matched_rule is not None
    assert not result.wrong_rule_error
    assert not result.unknown_rule_error
    assert result.match_source == "active"


def test_match_rule_completion_uses_active_then_fixed_sources() -> None:
    base_bank, temp_bank, _, tokenizer, rng = _fresh_demo_setup(seed=76)
    sampled = sample_fol_problem(bank=temp_bank, distance=1, initial_ant_max=3, rng=rng)
    while sampled.start_layer != 0:
        sampled = sample_fol_problem(bank=temp_bank, distance=1, initial_ant_max=3, rng=rng)

    src_layer = sampled.step_layers[0]
    expected = sampled.step_rules[0].statement_text
    active_match = match_rule_completion_fol(
        rule_bank=base_bank,
        src_layer=src_layer,
        completion_tokens=_enc1(tokenizer, expected),
        expected_statement_text=expected,
        tokenizer=tokenizer,
        active_rules=list(temp_bank.transition_rules(src_layer)),
    )
    assert active_match.is_correct
    assert active_match.match_source == "active"

    sampled_l1 = sample_fol_problem(bank=base_bank, distance=1, initial_ant_max=3, rng=rng)
    while sampled_l1.start_layer != 1:
        sampled_l1 = sample_fol_problem(bank=base_bank, distance=1, initial_ant_max=3, rng=rng)
    src_layer_l1 = sampled_l1.step_layers[0]
    expected_l1 = sampled_l1.step_rules[0].statement_text
    fixed_match = match_rule_completion_fol(
        rule_bank=base_bank,
        src_layer=src_layer_l1,
        completion_tokens=_enc1(tokenizer, expected_l1),
        expected_statement_text=expected_l1,
        tokenizer=tokenizer,
        active_rules=[],
        fixed_rules=list(base_bank.transition_rules(src_layer_l1)),
    )
    assert fixed_match.is_correct
    assert fixed_match.match_source == "fixed"
