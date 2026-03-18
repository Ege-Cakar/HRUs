# <codecell>
"""Interactive inspection script for experiment 17 full-rank demo ordering.

Uses depth3_fresh_icl tasks with demo_distribution="full_rank" and
configurable demo_ranking_beta.  Trains a small Transformer locally and
inspects decoded model outputs, including per-step reachability checks.
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
    _build_tokenizer_for_fresh_icl,
    _fresh_predicate_sentinels,
    compute_fol_dims,
    match_rule_completion_fol,
    predicted_rule_reaches_goal,
    print_task_preview,
    run_layer_rollout_fol,
    split_prompt_row_segments,
    infer_fol_predicate_layer,
)
from task.layer_gen.util.fol_rule_bank import (
    FOLLayerRule,
    build_random_fol_rule_bank,
    parse_atom_text,
    parse_clause_text,
)
from train import train, warmup_cosine_schedule


# <codecell>
# --- Configuration ---
SET_DIR = ROOT / "experiment" / "interactive" / "17_rank_demo" / "set"
SET_DIR.mkdir(parents=True, exist_ok=True)

# Demo distribution settings (top-level for easy tweaking)
DEMO_DISTRIBUTION = "full_rank"
TRAIN_BETA = 1.0

# Fresh-ICL bank parameters (matching experiment 17)
FRESH_ICL_CFG = {
    "seed": 2047,
    "predicates_per_layer": (16, 256, 16),
    "rules_per_transition": (256, 256),
    "arity_min": 0,
    "arity_max": 0,
    "vars_per_rule_max": 6,
    "k_in_max": 1,
    "k_out_max": 1,
    "constants": ("p0",),
    "predicate_name_len": 4,
}

# Task parameters (matching experiment 17)
TASK_CFG = {
    "distance_range": (2, 2),
    "initial_ant_max": 1,
    "train_min_n_demos": 1,
    "train_max_n_demos": 64,
    "eval_max_n_demos": 64,
    "train_include_oracle": False,
    "sample_max_attempts": 4096,
    "max_unify_solutions": 128,
}

# Interactive model config (smaller than remote for quick iteration)
TRAIN_CFG = {
    "n_layers": 4,
    "n_hidden": 256,
    "n_heads": 4,
    "lr": warmup_cosine_schedule(1e-4, 1000, warmup_frac=0.05),
    "train_iters": 1000,
    "test_every": 10,
    "test_iters": 1,
    "batch_size": 32,
}


# <codecell>
# --- Build base bank, tokenizer, compute dims ---
base_bank = build_random_fol_rule_bank(
    n_layers=3,
    predicates_per_layer=FRESH_ICL_CFG["predicates_per_layer"],
    rules_per_transition=FRESH_ICL_CFG["rules_per_transition"],
    arity_min=int(FRESH_ICL_CFG["arity_min"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    constants=tuple(str(c) for c in FRESH_ICL_CFG["constants"]),
    rng=np.random.default_rng(int(FRESH_ICL_CFG["seed"])),
)
print("base bank: n_layers=3, predicates_per_layer=", FRESH_ICL_CFG["predicates_per_layer"])

tokenizer = _build_tokenizer_for_fresh_icl(
    base_bank=base_bank,
    predicate_name_len=int(FRESH_ICL_CFG["predicate_name_len"]),
)

sentinels = _fresh_predicate_sentinels(name_len=int(FRESH_ICL_CFG["predicate_name_len"]))
extra_arities = {s: int(base_bank.arity_max) for s in sentinels}

dims_train = compute_fol_dims(
    rule_banks=[base_bank],
    tokenizer=tokenizer,
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["train_max_n_demos"]),
    extra_predicate_arities=extra_arities,
    fresh_k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    fresh_k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
)
dims_eval = compute_fol_dims(
    rule_banks=[base_bank],
    tokenizer=tokenizer,
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
    extra_predicate_arities=extra_arities,
    fresh_k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    fresh_k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
)

def _ceil_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

N_VOCAB = max(int(dims_train["n_vocab"]), int(dims_eval["n_vocab"]))
MAX_COMPLETION_LEN = max(int(dims_train["max_completion_len"]), int(dims_eval["max_completion_len"]))
N_SEQ = max(2, _ceil_pow2(max(int(dims_train["n_seq_ar"]), int(dims_eval["n_seq_ar"]))))

print("dims_train:", dims_train)
print("dims_eval:", dims_eval)
print(f"N_VOCAB={N_VOCAB}  N_SEQ={N_SEQ}  MAX_COMPLETION_LEN={MAX_COMPLETION_LEN}")


# <codecell>
# --- Create FOLLayerTask instances ---

_common_task_kwargs = dict(
    mode="online",
    task_split="depth3_fresh_icl",
    distance_range=TASK_CFG["distance_range"],
    batch_size=TRAIN_CFG["batch_size"],
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    prediction_objective="autoregressive",
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=N_SEQ,
    predicates_per_layer=FRESH_ICL_CFG["predicates_per_layer"],
    rules_per_transition=FRESH_ICL_CFG["rules_per_transition"],
    fresh_icl_base_bank_seed=int(FRESH_ICL_CFG["seed"]),
    arity_min=int(FRESH_ICL_CFG["arity_min"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(c) for c in FRESH_ICL_CFG["constants"]),
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    predicate_name_len=int(FRESH_ICL_CFG["predicate_name_len"]),
    sample_max_attempts=int(TASK_CFG["sample_max_attempts"]),
    max_unify_solutions=int(TASK_CFG["max_unify_solutions"]),
    demo_distribution=DEMO_DISTRIBUTION,
    demo_ranking_beta=TRAIN_BETA,
)

train_task_ar = FOLLayerTask(
    **_common_task_kwargs,
    split_role="train",
    seed=111,
    min_n_demos=int(TASK_CFG["train_min_n_demos"]),
    max_n_demos=int(TASK_CFG["train_max_n_demos"]),
    include_oracle=bool(TASK_CFG["train_include_oracle"]),
)

eval_task_ar = FOLLayerTask(
    **_common_task_kwargs,
    split_role="eval",
    seed=222,
    min_n_demos=int(TASK_CFG["eval_max_n_demos"]),
    max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
)


# <codecell>
# --- Preview train/eval examples ---
print("=" * 60)
print("TRAIN TASK PREVIEW")
print("=" * 60)
print_task_preview(train_task_ar, role="train", n_examples=5)
print()

print("=" * 60)
print("EVAL TASK PREVIEW")
print("=" * 60)
print_task_preview(eval_task_ar, role="eval", n_examples=5)
print()


# <codecell>
# --- Train a small Transformer ---
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
# --- Inspect model outputs on samples ---

model_fn = make_model_callable(optimizer, to_numpy=False)
adapter = AutoregressiveLogitsAdapter(
    n_seq=N_SEQ,
    max_completion_len=MAX_COMPLETION_LEN,
    pad_token_id=0,
    jit_step=True,
)

N_INSPECT_SAMPLES = 10


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


def _decode_single_completion_text(tokenizer, completion_tokens) -> str:
    statements = tokenizer.decode_completion_texts([int(tok) for tok in completion_tokens])
    if len(statements) != 1:
        raise ValueError("Expected a single completion statement.")
    return statements[0]


def inspect_samples(task, *, role: str, n_samples: int):
    for i in range(n_samples):
        record = task._sample_online_record()
        rule_context = record.get("rule_context", {})
        prompt = np.asarray(record["prompt"], dtype=np.int32)
        completion_gt = np.asarray(record["completions"][0], dtype=np.int32)
        src_layer = int(record["src_layer"])

        gt_text = _decode_single_completion_text(tokenizer, completion_gt.tolist())

        demo_segments, main_segment = split_prompt_row_segments(prompt, tokenizer=tokenizer)
        sequent = tokenizer.decode_prompt(main_segment.tolist())
        full_prompt_text = tokenizer.decode_batch_ids(
            prompt.reshape(1, -1),
            skip_pad=True,
            include_special_tokens=True,
        )[0]

        # Autoregressive prediction
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
            rule_bank=task.rule_bank,
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

        # Per-step reachability (single-step: is this rule on a path to goal?)
        reachable_tag = ""
        if matched.matched_rule is not None and "goal_text" in rule_context:
            try:
                goal = parse_atom_text(str(rule_context["goal_text"]))
                goal_layer = src_layer + int(record["distance"])
                reachable = predicted_rule_reaches_goal(
                    rule_bank=task.rule_bank,
                    matched_rule=matched.matched_rule,
                    goal=goal,
                    goal_layer=goal_layer,
                    max_unify_solutions=int(TASK_CFG["max_unify_solutions"]),
                )
                reachable_tag = f" reachable={reachable}"
            except (ValueError, RuntimeError):
                reachable_tag = " reachable=?"

        print(f"[{role} #{i}] status={status}{reachable_tag} n_demos={len(demo_segments)}")
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
# --- Rollout preview with fresh temp banks ---

N_ROLLOUT_PREVIEW = 10
MAX_UNIFY_SOLUTIONS = int(TASK_CFG["max_unify_solutions"])
rollout_rng = np.random.default_rng(45)


def preview_rollout(
    *,
    model_fn,
    adapter,
    tokenizer,
    task: FOLLayerTask,
    n_examples: int,
    rng,
):
    rollout_demo_adapter = None

    for i in range(n_examples):
        temp_bank = task.build_fresh_temp_bank(rng)

        if rollout_demo_adapter is None:
            rollout_demo_adapter = task.make_demo_adapter(
                adapter,
                temp_bank,
                min_n_demos=int(TASK_CFG["eval_max_n_demos"]),
                max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
                demo_ranking_beta=TRAIN_BETA,
            )
        else:
            rollout_demo_adapter.rule_bank = temp_bank

        example = task.sample_rollout_example(rng, rule_bank=temp_bank)

        result = run_layer_rollout_fol(
            rule_bank=temp_bank,
            example=example,
            model=model_fn,
            adapter=rollout_demo_adapter,
            tokenizer=tokenizer,
            temperature=0.0,
            rng=rng,
        )

        status = "SUCCESS" if result.success else f"FAIL ({result.failure_reason})"
        print(f"{'=' * 60}")
        print(f"ROLLOUT EXAMPLE {i}")
        print(f"  initial_facts: {example.initial_ants}")
        print(f"  goal:          {example.goal_atom}")
        print(f"  oracle_rules:  {example.oracle_rule_statements}")
        print(f"  result: {status}  n_steps={result.n_steps}")

        # Per-step details with reachability tracking
        goal_atom = parse_atom_text(example.goal_atom)
        goal_layer = int(example.start_layer) + int(example.distance)

        for step in result.steps:
            comp_text = _decode_single_completion_text(tokenizer, list(step.completion_tokens))

            # Check reachability for this step
            reachable_tag = ""
            if step.matched_rule_statement is not None and not step.inapplicable_rule_error:
                try:
                    lhs, rhs = parse_clause_text(str(step.matched_rule_statement))
                    matched_rule = FOLLayerRule(
                        src_layer=int(step.src_layer),
                        dst_layer=int(step.src_layer) + 1,
                        lhs=lhs,
                        rhs=rhs,
                    )
                    reachable = predicted_rule_reaches_goal(
                        rule_bank=temp_bank,
                        matched_rule=matched_rule,
                        goal=goal_atom,
                        goal_layer=goal_layer,
                        max_unify_solutions=MAX_UNIFY_SOLUTIONS,
                    )
                    reachable_tag = f" reachable={reachable}"
                except (ValueError, RuntimeError):
                    reachable_tag = " reachable=?"

            prompt_text = tokenizer.decode_batch_ids(
                np.asarray(step.prompt_tokens, dtype=np.int32).reshape(1, -1),
                skip_pad=True,
                include_special_tokens=True,
            )[0]

            print(f"  step[{step.step_idx}] layer={step.src_layer}{reachable_tag}")
            print(f"    prompt:     {prompt_text}")
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
    task=eval_task_ar,
    n_examples=N_ROLLOUT_PREVIEW,
    rng=rollout_rng,
)


# <codecell>
# --- Cleanup ---
train_task_ar.close()
eval_task_ar.close()
