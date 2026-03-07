# <codecell>
"""Interactive inspection script for experiment 10 fresh-rule split.

Uses depth3_fresh_icl tasks (every example gets entirely fresh r_XXXX
layer-0 predicates) instead of a fixed disjoint split bundle.
Trains a small Transformer locally and inspects decoded model outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.eval_adapters import make_model_callable
from model.transformer import TransformerConfig
from task.layer_fol import (
    AutoregressiveLogitsAdapter,
    FOLLayerTask,
    _augment_prompt_with_demos,
    _build_tokenizer_for_fresh_icl,
    infer_fol_predicate_layer,
    match_rule_completion_fol,
    run_layer_rollout_fol,
    sample_rollout_examples,
    split_prompt_row_segments,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLLayerRule,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    parse_clause_text,
)
from train import train, warmup_cosine_schedule


# <codecell>
SET_DIR = ROOT / "experiment" / "interactive" / "10_fresh_rule_split" / "set"
SET_DIR.mkdir(parents=True, exist_ok=True)

FRESH_ICL_CFG = {
    "seed": 2043,
    "predicates_per_layer": (1, 32, 16),
    "rules_per_transition": (16, 32),
    "arity_max": 0,
    "vars_per_rule_max": 6,
    "k_in_max": 1,
    "k_out_max": 1,
    "constants": tuple(f"p{i}" for i in range(32)),
    "predicate_name_len": 4,
}
TASK_CFG = {
    "distance_range": (2, 2),
    "initial_ant_max": 2,
    "train_min_n_demos": 4,
    "train_max_n_demos": 8,
    "eval_max_n_demos": 8,
    "train_include_oracle": True,
}


# <codecell>
def _demo_segments_to_rules(
    *,
    demo_segments: list[list[int]],
    src_layer: int,
    tokenizer,
) -> list[FOLLayerRule]:
    out: list[FOLLayerRule] = []
    for demo in demo_segments:
        try:
            demo_text = tokenizer.decode_completion_texts(
                list(demo) + [int(tokenizer.eot_token_id)]
            )[0]
            lhs, rhs = parse_clause_text(demo_text)
            out.append(FOLLayerRule(src_layer=int(src_layer), dst_layer=int(src_layer) + 1, lhs=lhs, rhs=rhs))
        except ValueError:
            continue
    return out


def _rule_texts_to_rules(*, src_layer: int, rule_texts: list[str]) -> list[FOLLayerRule]:
    out: list[FOLLayerRule] = []
    for text in rule_texts:
        try:
            lhs, rhs = parse_clause_text(str(text))
            out.append(
                FOLLayerRule(
                    src_layer=int(src_layer),
                    dst_layer=int(src_layer) + 1,
                    lhs=lhs,
                    rhs=rhs,
                )
            )
        except ValueError:
            continue
    return out


def preview_record(task: FOLLayerTask, record: dict, *, role: str) -> None:
    tokenizer = task.tokenizer
    prompt = np.asarray(record["prompt"], dtype=np.int32)
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    demo_segments, main_segment = split_prompt_row_segments(prompt, tokenizer=tokenizer)

    sequent = tokenizer.decode_prompt(main_segment.tolist())
    completion_text = tokenizer.decode_completion_texts(completion.tolist())[0]

    print(
        f"[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
        f"prompt_len={prompt.size} completion_len={completion.size} n_demos={len(demo_segments)}"
    )
    print("  sequent:", sequent.text)
    print("  completion:", completion_text)
    for idx, demo in enumerate(demo_segments):
        demo_text = tokenizer.decode_completion_texts(
            list(demo) + [int(tokenizer.eot_token_id)]
        )[0]
        print(f"  demo[{idx}]: {demo_text}")


def _decode_single_completion_text(tokenizer, completion_tokens) -> str:
    statements = tokenizer.decode_completion_texts([int(tok) for tok in completion_tokens])
    if len(statements) != 1:
        raise ValueError("Expected a single completion statement.")
    return statements[0]


# <codecell>
# --- Build base bank and preview fresh-ICL samples ---
base_bank = build_random_fol_rule_bank(
    n_layers=3,
    predicates_per_layer=FRESH_ICL_CFG["predicates_per_layer"],
    rules_per_transition=FRESH_ICL_CFG["rules_per_transition"],
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(c) for c in FRESH_ICL_CFG["constants"]),
    k_in_min=1,
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_min=1,
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    rng=np.random.default_rng(int(FRESH_ICL_CFG["seed"])),
    arity_min=0
)
print("base bank: n_layers=3, predicates_per_layer=", FRESH_ICL_CFG["predicates_per_layer"])


# <codecell>
# --- Compute sequence dims from the base bank ---

def _ceil_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_dims_fresh_icl(base_bank, tokenizer, *, max_n_demos_for_shapes: int, initial_ant_max: int):
    all_rules = []
    for src_layer, rules in base_bank.transitions.items():
        all_rules.extend(rules)
    if not all_rules:
        raise ValueError("Base bank has no rules.")

    max_rhs_atoms = max(len(rule.rhs) for rule in all_rules)
    max_prompt_facts = max(int(initial_ant_max), int(max_rhs_atoms))

    # For fresh predicates, estimate max atom length using sentinel predicates
    from task.layer_fol import _fresh_predicate_sentinels
    sentinels = _fresh_predicate_sentinels(
        name_len=int(FRESH_ICL_CFG["predicate_name_len"])
    )
    merged_predicate_arities = dict(base_bank.predicate_arities)
    for s in sentinels:
        if s not in merged_predicate_arities:
            merged_predicate_arities[s] = int(base_bank.arity_max)

    first_const = str(base_bank.constants[0])
    max_atom_len = 1
    for predicate, arity in merged_predicate_arities.items():
        atom_text = f"{predicate}({','.join(first_const for _ in range(int(arity)))})"
        max_atom_len = max(
            max_atom_len,
            len(tokenizer.encode_completion_texts([atom_text])) - 1,
        )

    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1
        + max_atom_len
        + 1
    )
    if int(max_n_demos_for_shapes) > 0:
        max_demo_clause_len = max_atom_len * max(int(base_bank.arity_max), 1) * 3
        max_prompt_len += int(max_n_demos_for_shapes) * (int(max_demo_clause_len) + 1)

    max_completion_len = max(
        len(tokenizer.encode_completion_texts([rule.statement_text]))
        for rule in all_rules
    )
    # Account for fresh layer-0 rules which may be longer
    max_completion_len = max(max_completion_len, max_atom_len * 3 + 10)

    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_vocab = max(512, int(tokenizer.vocab_size))

    return {
        "n_vocab": n_vocab,
        "max_completion_len": max_completion_len,
        "n_seq_ar": n_seq_ar,
    }


tokenizer = _build_tokenizer_for_fresh_icl(
    base_bank=base_bank,
    predicate_name_len=int(FRESH_ICL_CFG["predicate_name_len"]),
)

INITIAL_ANT_MAX = int(TASK_CFG["initial_ant_max"])
dims_train = compute_dims_fresh_icl(
    base_bank, tokenizer,
    max_n_demos_for_shapes=int(TASK_CFG["train_max_n_demos"]),
    initial_ant_max=INITIAL_ANT_MAX,
)
dims_eval = compute_dims_fresh_icl(
    base_bank, tokenizer,
    max_n_demos_for_shapes=int(TASK_CFG["eval_max_n_demos"]),
    initial_ant_max=INITIAL_ANT_MAX,
)

N_VOCAB = max(dims_train["n_vocab"], dims_eval["n_vocab"])
MAX_COMPLETION_LEN = max(dims_train["max_completion_len"], dims_eval["max_completion_len"])
N_SEQ = max(2, _ceil_pow2(max(dims_train["n_seq_ar"], dims_eval["n_seq_ar"])))

print("dims_train:", dims_train)
print("dims_eval:", dims_eval)
print(f"N_VOCAB={N_VOCAB}  N_SEQ={N_SEQ}  MAX_COMPLETION_LEN={MAX_COMPLETION_LEN}")

TRAIN_CFG = {
    "n_layers": 4,
    "n_hidden": 256,
    "n_heads": 4,
    # "lr": 5e-4,
    "lr": warmup_cosine_schedule(1e-4, 1000, warmup_frac=0.05),
    "train_iters": 1000,
    "test_every": 10,
    "test_iters": 1,
    "batch_size": 32,
    "train_max_n_demos": int(TASK_CFG["train_max_n_demos"]),
    "eval_max_n_demos": int(TASK_CFG["eval_max_n_demos"]),
}


# <codecell>
# --- Create tasks and train a small Transformer ---

train_task_ar = FOLLayerTask(
    mode="online",
    task_split="depth3_fresh_icl",
    split_role="train",
    distance_range=TASK_CFG["distance_range"],
    batch_size=TRAIN_CFG["batch_size"],
    seed=111,
    initial_ant_max=INITIAL_ANT_MAX,
    min_n_demos=int(TASK_CFG["train_min_n_demos"]),
    max_n_demos=TRAIN_CFG["train_max_n_demos"],
    prediction_objective="autoregressive",
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=N_SEQ,
    predicates_per_layer=FRESH_ICL_CFG["predicates_per_layer"],
    rules_per_transition=FRESH_ICL_CFG["rules_per_transition"],
    fresh_icl_base_bank_seed=int(FRESH_ICL_CFG["seed"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(c) for c in FRESH_ICL_CFG["constants"]),
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    arity_min=0,
    include_oracle=bool(TASK_CFG["train_include_oracle"]),
    predicate_name_len=int(FRESH_ICL_CFG["predicate_name_len"]),
)

eval_task_ar = FOLLayerTask(
    mode="online",
    task_split="depth3_fresh_icl",
    split_role="eval",
    distance_range=TASK_CFG["distance_range"],
    batch_size=TRAIN_CFG["batch_size"],
    seed=222,
    initial_ant_max=INITIAL_ANT_MAX,
    min_n_demos=TRAIN_CFG["eval_max_n_demos"],
    max_n_demos=TRAIN_CFG["eval_max_n_demos"],
    prediction_objective="autoregressive",
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=N_SEQ,
    predicates_per_layer=FRESH_ICL_CFG["predicates_per_layer"],
    rules_per_transition=FRESH_ICL_CFG["rules_per_transition"],
    fresh_icl_base_bank_seed=int(FRESH_ICL_CFG["seed"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(c) for c in FRESH_ICL_CFG["constants"]),
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    arity_min=0,
    predicate_name_len=int(FRESH_ICL_CFG["predicate_name_len"]),
)

# <codecell>
# --- Preview a few train examples before training ---
N_PREVIEW = 5
print("=" * 60)
print(f"TRAIN TASK PREVIEW ({N_PREVIEW} examples)")
print("=" * 60)
for i in range(N_PREVIEW):
    record = train_task_ar._sample_online_record()
    preview_record(train_task_ar, record, role=f"train_preview #{i}")
print()

# # <codecell>
# [i for i in range(len(train_task_ar._base_bank.transition_rules(1))) if train_task_ar._base_bank.transition_rules(1)[i].rhs[0].predicate == 'r2_5']

# <codecell>
model_config = TransformerConfig(
    n_vocab=N_VOCAB,
    n_seq=N_SEQ,
    n_layers=TRAIN_CFG["n_layers"],
    n_hidden=TRAIN_CFG["n_hidden"],
    n_heads=TRAIN_CFG["n_heads"],
    n_out=N_VOCAB,
    n_pred_tokens=1,
    pos_encoding="rope",
    layer_norm=True,
    use_swiglu=True,
    use_bias=True,
    dropout_rate=0.0,
    output_mode="full_sequence",
    pad_token_id=0,
)

optimizer, hist = train(
    model_config,
    train_iter=train_task_ar,
    test_iter=eval_task_ar,
    loss="ce_mask",
    train_iters=TRAIN_CFG["train_iters"],
    test_iters=TRAIN_CFG["test_iters"],
    test_every=TRAIN_CFG["test_every"],
    lr=TRAIN_CFG["lr"],
)

final_train = hist["train"][-1] if hist["train"] else {}
final_test = hist["test"][-1] if hist["test"] else {}
print("Final train metrics:", final_train)
print("Final test metrics:", final_test)


# <codecell>
# --- Inspect model outputs on train and eval samples ---

model_fn = make_model_callable(optimizer, to_numpy=False)
adapter = AutoregressiveLogitsAdapter(
    n_seq=N_SEQ,
    max_completion_len=MAX_COMPLETION_LEN,
    pad_token_id=0,
    jit_step=True,
)

N_INSPECT_SAMPLES = 10


def inspect_samples(task, *, role: str, n_samples: int):
    rule_bank = task.rule_bank
    for i in range(n_samples):
        record = task._sample_online_record()
        rule_context = record.get("rule_context", {})
        prompt = np.asarray(record["prompt"], dtype=np.int32)
        completion_gt = np.asarray(record["completions"][0], dtype=np.int32)
        src_layer = int(record["src_layer"])

        # Decode the ground-truth completion
        gt_text = _decode_single_completion_text(tokenizer, completion_gt.tolist())

        # Decode the sequent from the prompt
        demo_segments, main_segment = split_prompt_row_segments(prompt, tokenizer=tokenizer)
        sequent = tokenizer.decode_prompt(main_segment.tolist())
        full_prompt_text = tokenizer.decode_batch_ids(
            prompt.reshape(1, -1),
            skip_pad=True,
            include_special_tokens=True,
        )[0]

        # Run autoregressive prediction
        pred_completion = adapter.predict_completion(
            model=model_fn,
            prompt_tokens=prompt.tolist(),
            tokenizer=tokenizer,
            temperature=0.0,
            rng=None,
        )
        pred_text = _decode_single_completion_text(tokenizer, pred_completion.tolist())

        # Match predicted completion against rule bank
        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=pred_completion,
            expected_statement_text=gt_text,
            tokenizer=tokenizer,
            active_rules=(
                _rule_texts_to_rules(
                    src_layer=src_layer,
                    rule_texts=list(rule_context.get("active_rule_texts", [])),
                )
                if "active_rule_texts" in rule_context
                else None
            ),
            fixed_rules=(
                _rule_texts_to_rules(
                    src_layer=src_layer,
                    rule_texts=list(rule_context.get("fixed_rule_texts", [])),
                )
                if "fixed_rule_texts" in rule_context
                else None
            ),
            demo_rules=(
                _rule_texts_to_rules(
                    src_layer=src_layer,
                    rule_texts=list(rule_context.get("demo_schema_texts", [])),
                )
                if "demo_schema_texts" in rule_context
                else None
            ),
        )

        # Classify result
        if matched.decode_error:
            status = "DECODE_ERROR"
        elif matched.unknown_rule_error:
            status = "UNKNOWN_RULE"
        elif matched.wrong_rule_error:
            status = "WRONG_RULE"
        elif matched.is_correct:
            status = "CORRECT"
        else:
            status = "OTHER"

        print(f"[{role} #{i}] status={status} n_demos={len(demo_segments)}")
        print(f"  full_input_prompt: {full_prompt_text}")
        if demo_segments:
            for demo_idx, demo in enumerate(demo_segments):
                demo_text = tokenizer.decode_completion_texts(
                    list(demo) + [int(tokenizer.eot_token_id)]
                )[0]
                print(f"  demo[{demo_idx}]: {demo_text}")
        print(f"  sequent:    {sequent.text}")
        print(f"  expected:   {gt_text}")
        print(f"  predicted:  {pred_text}")
        if matched.matched_rule is not None:
            print(f"  matched_rule: {matched.matched_rule.statement_text}")
        print()


print("=" * 60)
print("TRAIN SAMPLES")
print("=" * 60)
inspect_samples(train_task_ar, role="train", n_samples=N_INSPECT_SAMPLES)

print("=" * 60)
print("EVAL SAMPLES")
print("=" * 60)
inspect_samples(eval_task_ar, role="eval", n_samples=N_INSPECT_SAMPLES)


# <codecell>
# --- Rollout preview with per-example fresh banks ---


def _layer_from_predicate(predicate: str) -> int:
    return infer_fol_predicate_layer(predicate)


class DemoAugmentedAdapter:
    """Prepend sampled demo completions before calling a base adapter."""

    def __init__(
        self,
        *,
        base_adapter,
        rule_bank,
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
        self.augmented_prompts: list[list[int]] = []

    def get_last_demo_rules(self) -> list[FOLLayerRule]:
        return list(self._last_demo_rules)

    def predict_completion(
        self,
        *,
        model,
        prompt_tokens,
        tokenizer,
        temperature: float = 0.0,
        rng=None,
    ):
        self._last_demo_rules = []
        if self.max_n_demos <= 0:
            prompt = np.asarray(prompt_tokens, dtype=np.int32).tolist()
            self.augmented_prompts.append(prompt)
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
            _, main_segment = split_prompt_row_segments(
                np.asarray(prompt, dtype=np.int32),
                tokenizer=self.tokenizer,
            )
            sequent = self.tokenizer.decode_prompt(main_segment.tolist())
            src_layer = int(min(_layer_from_predicate(atom.predicate) for atom in sequent.ants))
            prompt = _augment_prompt_with_demos(
                prompt_tokens=prompt,
                rule_bank=self.rule_bank,
                tokenizer=self.tokenizer,
                rng=rng,
                src_layer=src_layer,
                ants=tuple(sequent.ants),
                min_n_demos=self.min_n_demos,
                max_n_demos=self.max_n_demos,
                max_unify_solutions=self.max_unify_solutions,
            )
            demo_segments, _ = split_prompt_row_segments(
                np.asarray(prompt, dtype=np.int32),
                tokenizer=self.tokenizer,
            )
            self._last_demo_rules = _demo_segments_to_rules(
                demo_segments=[segment.tolist() for segment in demo_segments],
                src_layer=src_layer,
                tokenizer=self.tokenizer,
            )
        except ValueError:
            self._last_demo_rules = []

        self.augmented_prompts.append(prompt)
        return self.base_adapter.predict_completion(
            model=model,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )


N_ROLLOUT_PREVIEW = 10
MAX_UNIFY_SOLUTIONS = 128
rollout_rng = np.random.default_rng(45)


def preview_rollout(
    *,
    model_fn,
    adapter,
    tokenizer,
    base_bank,
    n_examples: int,
    rng,
):
    for i in range(n_examples):
        fresh_preds = generate_fresh_predicate_names(
            len(base_bank.predicates_for_layer(0)),
            rng,
            name_len=int(FRESH_ICL_CFG["predicate_name_len"]),
        )
        temp_bank = build_fresh_layer0_bank(
            base_bank=base_bank,
            fresh_predicates=fresh_preds,
            rules_per_transition=len(base_bank.transition_rules(0)),
            k_in_min=1,
            k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
            k_out_min=1,
            k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
            rng=rng,
        )

        examples = sample_rollout_examples(
            rule_bank=temp_bank,
            distance=2,
            n_examples=1,
            initial_ant_max=INITIAL_ANT_MAX,
            max_steps=2,
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
            rng=rng,
        )
        example = examples[0]

        demo_adapter = DemoAugmentedAdapter(
            base_adapter=adapter,
            rule_bank=temp_bank,
            tokenizer=tokenizer,
            min_n_demos=int(TASK_CFG["eval_max_n_demos"]),
            max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
            max_unify_solutions=MAX_UNIFY_SOLUTIONS,
        )
        demo_adapter.augmented_prompts.clear()

        print(f"{'=' * 60}")
        print(f"ROLLOUT EXAMPLE {i}")
        print(f"  initial_facts: {example.initial_ants}")
        print(f"  goal:          {example.goal_atom}")
        print(f"  oracle_rules:  {example.oracle_rule_statements}")

        result = run_layer_rollout_fol(
            rule_bank=temp_bank,
            example=example,
            model=model_fn,
            adapter=demo_adapter,
            tokenizer=tokenizer,
            temperature=0.0,
            rng=rng,
        )

        status = "SUCCESS" if result.success else f"FAIL ({result.failure_reason})"
        print(f"  result: {status}  n_steps={result.n_steps}")

        for step in result.steps:
            prompt_text = tokenizer.decode_batch_ids(
                np.asarray(step.prompt_tokens, dtype=np.int32).reshape(1, -1),
                skip_pad=True,
                include_special_tokens=True,
            )[0]
            comp_text = _decode_single_completion_text(tokenizer, list(step.completion_tokens))
            print(f"  step[{step.step_idx}] layer={step.src_layer}")
            print(f"    prompt:     {prompt_text}")

            # Show demos from the augmented prompt
            if step.step_idx < len(demo_adapter.augmented_prompts):
                aug_prompt = demo_adapter.augmented_prompts[step.step_idx]
                demo_segments, _ = split_prompt_row_segments(
                    np.asarray(aug_prompt, dtype=np.int32),
                    tokenizer=tokenizer,
                )
                if demo_segments:
                    for demo_idx, demo in enumerate(demo_segments):
                        demo_text = tokenizer.decode_completion_texts(
                            list(demo) + [int(tokenizer.eot_token_id)]
                        )[0]
                        print(f"    demo[{demo_idx}]: {demo_text}")
                else:
                    print(f"    (no demos prepended)")
            else:
                print(f"    (augmented prompt not captured for step {step.step_idx})")

            print(f"    completion: {comp_text}")
            print(f"    decoded:    {step.decoded_statement}")
            print(f"    matched:    {step.matched_rule_statement}")
            if step.decode_error:
                print(f"    >> decode_error")
            if step.unknown_rule_error:
                print(f"    >> unknown_rule_error")
            if step.inapplicable_rule_error:
                print(f"    >> inapplicable_rule_error")
            if step.goal_reached:
                print(f"    >> goal_reached!")

        print(f"  final_facts: {result.final_facts}")
        print()


print("=" * 60)
print("ROLLOUT PREVIEWS")
print("=" * 60)
preview_rollout(
    model_fn=model_fn,
    adapter=adapter,
    tokenizer=tokenizer,
    base_bank=base_bank,
    n_examples=N_ROLLOUT_PREVIEW,
    rng=rollout_rng,
)


# <codecell>
# --- Cleanup ---
train_task_ar.close()
eval_task_ar.close()
