from __future__ import annotations

from flax import nnx
import jax.numpy as jnp
import numpy as np

from model.eval_adapters import make_model_callable
from task.layer import (
    AutoregressiveLogitsAdapter,
    CompletionLogitsAdapter,
    LayerRolloutExample,
    evaluate_layer_rollouts,
    evaluate_rule_matches,
    match_rule_completion,
    run_layer_rollout,
)
from task.layer_gen.util import tokenize_layer as tok
from task.layer_gen.util.rule_bank import build_random_rule_bank, sample_problem
from task.prop_gen.util.elem import Atom, Sequent


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
    bank = build_random_rule_bank(
        n_layers=6,
        props_per_layer=4,
        rules_per_transition=8,
        k_in_max=2,
        k_out_max=2,
        rng=rng,
    )
    tokenizer = tok.build_tokenizer_from_rule_bank(bank)
    return bank, tokenizer, rng


def _make_unknown_statement(bank, src_layer: int) -> str:
    src_atoms = [f"p{src_layer}_{idx}" for idx in range(1, bank.props_per_layer + 1)]
    dst_atoms = [f"p{src_layer + 1}_{idx}" for idx in range(1, bank.props_per_layer + 1)]
    known = bank.statement_set(src_layer)
    for lhs in src_atoms:
        for rhs in dst_atoms:
            statement = f"{lhs} → {rhs}"
            if statement not in known:
                return statement
    raise RuntimeError("Unable to construct unknown statement.")


def test_match_rule_completion_categorizes_errors() -> None:
    bank, tokenizer, _ = _sampled_bank(seed=7)
    src_layer = 1
    rules = bank.transition_rules(src_layer)
    assert len(rules) >= 2

    expected_rule = rules[0]
    wrong_rule = rules[1]

    ok = match_rule_completion(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(expected_rule.statement_text),
        expected_statement_text=expected_rule.statement_text,
        tokenizer=tokenizer,
    )
    assert ok.is_correct
    assert not ok.decode_error
    assert not ok.unknown_rule_error
    assert not ok.wrong_rule_error

    wrong = match_rule_completion(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(wrong_rule.statement_text),
        expected_statement_text=expected_rule.statement_text,
        tokenizer=tokenizer,
    )
    assert wrong.is_valid_rule
    assert not wrong.is_correct
    assert wrong.wrong_rule_error

    unknown_statement = _make_unknown_statement(bank, src_layer)
    unknown = match_rule_completion(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(unknown_statement),
        tokenizer=tokenizer,
    )
    assert unknown.unknown_rule_error
    assert not unknown.decode_error
    assert not unknown.is_valid_rule

    decode_err = match_rule_completion(
        rule_bank=bank,
        src_layer=src_layer,
        completion_tokens=tokenizer.encode_completion(expected_rule.statement_text)[:-1],
        tokenizer=tokenizer,
    )
    assert decode_err.decode_error
    assert not decode_err.is_valid_rule


def test_evaluate_rule_matches_aggregates() -> None:
    bank, tokenizer, _ = _sampled_bank(seed=11)
    src_layer = 2
    rules = bank.transition_rules(src_layer)
    expected = rules[0].statement_text
    wrong = rules[1].statement_text
    unknown = _make_unknown_statement(bank, src_layer)

    metrics = evaluate_rule_matches(
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


def test_run_layer_rollout_success_with_scripted_rules() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=21)
    sampled = sample_problem(bank=bank, distance=3, initial_ant_max=3, rng=rng)

    example = LayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom,
        initial_ants=sampled.step_ants[0],
        max_steps=sampled.distance,
    )
    adapter = _ScriptedAdapter(
        [tokenizer.encode_completion(rule.statement_text) for rule in sampled.step_rules]
    )

    result = run_layer_rollout(
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


def test_run_layer_rollout_unknown_rule_fails() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=22)
    sampled = sample_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    unknown_statement = _make_unknown_statement(bank, sampled.start_layer)
    adapter = _ScriptedAdapter([tokenizer.encode_completion(unknown_statement)])
    example = LayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom,
        initial_ants=sampled.step_ants[0],
        max_steps=sampled.distance,
    )

    result = run_layer_rollout(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert not result.success
    assert result.failure_reason == "unknown_rule_error"


def test_run_layer_rollout_inapplicable_rule_fails() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=23)
    sampled = sample_problem(bank=bank, distance=1, initial_ant_max=1, rng=rng)
    src_layer = sampled.start_layer
    initial_facts = set(sampled.step_ants[0])

    bad_rule = None
    for rule in bank.transition_rules(src_layer):
        if not set(rule.lhs).issubset(initial_facts):
            bad_rule = rule
            break
    assert bad_rule is not None

    adapter = _ScriptedAdapter([tokenizer.encode_completion(bad_rule.statement_text)])
    example = LayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom,
        initial_ants=sampled.step_ants[0],
        max_steps=1,
    )

    result = run_layer_rollout(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert not result.success
    assert result.failure_reason == "inapplicable_rule_error"


def test_run_layer_rollout_goal_not_reached_fails() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=24)
    sampled = sample_problem(bank=bank, distance=2, initial_ant_max=3, rng=rng)
    adapter = _ScriptedAdapter([tokenizer.encode_completion(sampled.step_rules[0].statement_text)])
    example = LayerRolloutExample(
        distance=sampled.distance,
        start_layer=sampled.start_layer,
        goal_atom=sampled.goal_atom,
        initial_ants=sampled.step_ants[0],
        max_steps=1,
    )

    result = run_layer_rollout(
        rule_bank=bank,
        example=example,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert not result.success
    assert result.failure_reason == "goal_not_reached"


def test_evaluate_layer_rollouts_aggregates() -> None:
    bank, tokenizer, rng = _sampled_bank(seed=31)
    sampled_ok = sample_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    sampled_bad = sample_problem(bank=bank, distance=1, initial_ant_max=3, rng=rng)
    unknown_statement = _make_unknown_statement(bank, sampled_bad.start_layer)

    examples = [
        LayerRolloutExample(
            distance=1,
            start_layer=sampled_ok.start_layer,
            goal_atom=sampled_ok.goal_atom,
            initial_ants=sampled_ok.step_ants[0],
            max_steps=1,
        ),
        LayerRolloutExample(
            distance=1,
            start_layer=sampled_bad.start_layer,
            goal_atom=sampled_bad.goal_atom,
            initial_ants=sampled_bad.step_ants[0],
            max_steps=1,
        ),
    ]
    adapter = _ScriptedAdapter(
        [
            tokenizer.encode_completion(sampled_ok.step_rules[0].statement_text),
            tokenizer.encode_completion(unknown_statement),
        ]
    )

    metrics = evaluate_layer_rollouts(
        rule_bank=bank,
        examples=examples,
        model=lambda x: x,
        adapter=adapter,
        tokenizer=tokenizer,
    )
    assert metrics.n_examples == 2
    assert metrics.n_success == 1
    assert metrics.n_failure_unknown_rule_error == 1


def test_completion_logits_adapter_decodes_greedy() -> None:
    bank, tokenizer, _ = _sampled_bank(seed=41)
    src_layer = 0
    rule = bank.transition_rules(src_layer)[0]
    completion = tokenizer.encode_completion(rule.statement_text)
    vocab = tokenizer.vocab_size
    out_len = len(completion)

    def model(_):
        logits = np.zeros((1, out_len, vocab), dtype=np.float32)
        for idx, tok_id in enumerate(completion):
            logits[0, idx, tok_id] = 10.0
        return logits

    adapter = CompletionLogitsAdapter(n_seq=64)
    prompt = tokenizer.tokenize_prompt(Sequent([Atom("p0_1")], Atom("p1_1")))

    decoded = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == completion


def test_autoregressive_logits_adapter_decodes_greedy() -> None:
    _, tokenizer, _ = _sampled_bank(seed=42)
    target = [tokenizer.char_to_id("p0_1"), tokenizer.eot_token_id]
    vocab = tokenizer.vocab_size
    call_idx = {"value": 0}

    def model(xs):
        logits = np.zeros((xs.shape[0], xs.shape[1], vocab), dtype=np.float32)
        tok = target[min(call_idx["value"], len(target) - 1)]
        call_idx["value"] += 1
        last_idx = int(np.max(np.where(xs[0] != 0)[0]))
        logits[0, last_idx, int(tok)] = 8.0
        return logits

    prompt = np.asarray([tokenizer.char_to_id("p0_1"), tokenizer.sep_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(n_seq=16, max_completion_len=4)
    decoded = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == target


def test_autoregressive_logits_adapter_uses_cache_when_supported() -> None:
    _, tokenizer, _ = _sampled_bank(seed=43)
    target = [tokenizer.char_to_id("p0_1"), tokenizer.eot_token_id]
    vocab = tokenizer.vocab_size
    calls: list[tuple[tuple[int, ...], int | None, bool]] = []

    def model(xs, *, cache=None, return_cache=False):
        calls.append((xs.shape, cache, return_cache))
        assert return_cache
        step = 0 if cache is None else int(cache)
        logits = np.zeros((xs.shape[0], xs.shape[1], vocab), dtype=np.float32)
        tok = target[min(step, len(target) - 1)]
        logits[0, -1, int(tok)] = 9.0
        return logits, step + 1

    prompt = np.asarray([tokenizer.char_to_id("p0_1"), tokenizer.sep_token_id], dtype=np.int32)
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


def test_autoregressive_logits_adapter_jit_step_uses_cache_when_supported() -> None:
    _, tokenizer, _ = _sampled_bank(seed=44)
    first_tok = tokenizer.char_to_id("p0_1")
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

    prompt = np.asarray([tokenizer.char_to_id("p0_1"), tokenizer.sep_token_id], dtype=np.int32)
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


def test_autoregressive_logits_adapter_jit_step_sampling_deterministic_with_seeded_rng() -> None:
    _, tokenizer, _ = _sampled_bank(seed=46)
    vocab = int(tokenizer.vocab_size)

    class _SamplingModel(nnx.Module):
        def __init__(self, *, vocab_size: int):
            self.vocab_size = int(vocab_size)

        def __call__(self, xs, *, cache=None, return_cache=False):
            if not return_cache:
                raise RuntimeError("jit_step cache path should call return_cache=True.")
            step = jnp.asarray(0, dtype=jnp.int32) if cache is None else jnp.asarray(cache, dtype=jnp.int32)
            logits = jnp.full((xs.shape[0], xs.shape[1], self.vocab_size), -3.0, dtype=jnp.float32)
            logits = logits.at[0, -1, 3].set(0.0)
            logits = logits.at[0, -1, 4].set(0.2)
            logits = logits.at[0, -1, 5].set(0.4 + 0.1 * step.astype(jnp.float32))
            return logits, step + 1

    prompt = np.asarray([tokenizer.char_to_id("p0_1"), tokenizer.sep_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(
        n_seq=16,
        max_completion_len=3,
        jit_step=True,
    )

    out1 = adapter.predict_completion(
        model=_SamplingModel(vocab_size=vocab),
        prompt_tokens=prompt,
        tokenizer=tokenizer,
        temperature=1.0,
        rng=np.random.default_rng(123),
    )
    out2 = adapter.predict_completion(
        model=_SamplingModel(vocab_size=vocab),
        prompt_tokens=prompt,
        tokenizer=tokenizer,
        temperature=1.0,
        rng=np.random.default_rng(123),
    )
    assert out1.tolist() == out2.tolist()
    assert len(out1) == 3


def test_autoregressive_logits_adapter_jit_step_supports_wrapped_model_callable() -> None:
    _, tokenizer, _ = _sampled_bank(seed=45)
    first_tok = tokenizer.char_to_id("p0_1")
    target = [int(first_tok), int(tokenizer.eot_token_id)]
    vocab = int(tokenizer.vocab_size)

    class _Model(nnx.Module):
        def __init__(self, *, vocab_size: int):
            self.vocab_size = int(vocab_size)

        def __call__(self, xs):
            logits = jnp.zeros((xs.shape[0], xs.shape[1], self.vocab_size), dtype=jnp.float32)
            nonpad = xs[0] != 0
            last_idx = jnp.maximum(jnp.sum(nonpad) - 1, 0)
            last_tok = xs[0, last_idx]
            next_tok = jnp.where(last_tok == first_tok, tokenizer.eot_token_id, first_tok)
            return logits.at[0, last_idx, next_tok].set(8.0)

    class _Opt:
        def __init__(self, model):
            self.model = model

    model_fn = make_model_callable(_Opt(_Model(vocab_size=vocab)), to_numpy=False)
    prompt = np.asarray([tokenizer.char_to_id("p0_1"), tokenizer.sep_token_id], dtype=np.int32)
    adapter = AutoregressiveLogitsAdapter(
        n_seq=16,
        max_completion_len=4,
        jit_step=True,
    )
    decoded = adapter.predict_completion(
        model=model_fn,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    assert decoded.tolist() == target
