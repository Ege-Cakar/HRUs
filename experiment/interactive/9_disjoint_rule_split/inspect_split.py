# <codecell>
"""Interactive inspection script for experiment 9 disjoint-rule split.

Loads a disjoint-rule split bundle, trains a small Transformer locally,
and inspects decoded model outputs on train vs eval samples.
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
    match_rule_completion_fol,
)
from task.layer_gen.util.fol_rule_bank import (
    build_depth3_icl_split_bundle,
    load_fol_depth3_icl_split_bundle,
    save_fol_depth3_icl_split_bundle,
)
from train import train


# <codecell>
SET_DIR = ROOT / "experiment" / "interactive" / "9_disjoint_rule_split" / "set"
SET_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_PATH = SET_DIR / "depth3_icl_split_bundle.json"

SPLIT_CFG = {
    "seed": 2031,
    "predicates_per_layer": 16,
    "rules_01_train": 32,
    "rules_01_eval": 32,
    "rules_12_shared": 32,
    "arity_max": 3,
    "vars_per_rule_max": 6,
    "k_in_max": 3,
    "k_out_max": 5,
    "constants": tuple(f"p{i}" for i in range(256)),
}
TASK_CFG = {
    "distance_range": (2, 2),
    "batch_size": 1,
    "initial_ant_max": 5,
    "train_max_n_demos": 2,
    "eval_max_n_demos": 4,
}
REBUILD_SPLIT = False


def ensure_split_bundle(path: Path, *, rebuild: bool) -> Path:
    if path.exists() and not bool(rebuild):
        return path

    bundle = build_depth3_icl_split_bundle(
        predicates_per_layer=int(SPLIT_CFG["predicates_per_layer"]),
        rules_01_train=int(SPLIT_CFG["rules_01_train"]),
        rules_01_eval=int(SPLIT_CFG["rules_01_eval"]),
        rules_12_shared=int(SPLIT_CFG["rules_12_shared"]),
        arity_max=int(SPLIT_CFG["arity_max"]),
        vars_per_rule_max=int(SPLIT_CFG["vars_per_rule_max"]),
        k_in_max=int(SPLIT_CFG["k_in_max"]),
        k_out_max=int(SPLIT_CFG["k_out_max"]),
        constants=tuple(str(c) for c in SPLIT_CFG["constants"]),
        rng=np.random.default_rng(int(SPLIT_CFG["seed"])),
    )
    save_fol_depth3_icl_split_bundle(path, bundle)
    return path


def _split_prompt_segments(prompt_tokens: np.ndarray, sep_token_id: int) -> list[list[int]]:
    out: list[list[int]] = []
    current: list[int] = []
    for tok in prompt_tokens.tolist():
        if int(tok) == int(sep_token_id):
            out.append(current)
            current = []
            continue
        current.append(int(tok))
    return out


def preview_record(task: FOLLayerTask, record: dict, *, role: str) -> None:
    tokenizer = task.tokenizer
    prompt = np.asarray(record["prompt"], dtype=np.int32)
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    segments = _split_prompt_segments(prompt, tokenizer.sep_token_id)
    demo_segments = segments[:-1]
    main_segment = segments[-1] + [int(tokenizer.sep_token_id)]

    sequent = tokenizer.decode_prompt(main_segment)
    completion_text = tokenizer.decode_completion_text(completion.tolist())

    print(
        f"[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
        f"prompt_len={prompt.size} completion_len={completion.size} n_demos={len(demo_segments)}"
    )
    print("  sequent:", sequent.text)
    print("  completion:", completion_text)
    for idx, demo in enumerate(demo_segments):
        demo_text = tokenizer.decode_completion_text(list(demo) + [int(tokenizer.eot_token_id)])
        print(f"  demo[{idx}]: {demo_text}")


SPLIT_PATH = ensure_split_bundle(SPLIT_PATH, rebuild=REBUILD_SPLIT)
split_bundle = load_fol_depth3_icl_split_bundle(SPLIT_PATH)
print("split bundle:", SPLIT_PATH)
print("train layer0 predicates:", len(split_bundle.train_layer0_predicates))
print("eval layer0 predicates:", len(split_bundle.eval_layer0_predicates))
print("shared layer1 predicates:", len(split_bundle.shared_layer1_predicates))
print("shared layer2 predicates:", len(split_bundle.shared_layer2_predicates))


# <codecell>
train_task = FOLLayerTask(
    mode="online",
    task_split="depth3_icl_transfer",
    split_role="train",
    split_rule_bundle_path=SPLIT_PATH,
    distance_range=TASK_CFG["distance_range"],
    batch_size=int(TASK_CFG["batch_size"]),
    seed=111,
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["train_max_n_demos"]),
    online_prefetch_backend="sync",
)

eval_task = FOLLayerTask(
    mode="online",
    task_split="depth3_icl_transfer",
    split_role="eval",
    split_rule_bundle_path=SPLIT_PATH,
    distance_range=TASK_CFG["distance_range"],
    batch_size=int(TASK_CFG["batch_size"]),
    seed=222,
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
    online_prefetch_backend="sync",
)

for _ in range(3):
    preview_record(train_task, train_task._sample_online_record(), role="train")
for _ in range(3):
    preview_record(eval_task, eval_task._sample_online_record(), role="eval")


# <codecell>
train_task.close()
eval_task.close()


# <codecell>
# --- Compute sequence dims from the split bundle ---

def _iter_bundle_rules(bundle):
    seen = set()
    for bank in (bundle.train_bank, bundle.eval_bank):
        for _src_layer, rules in bank.transitions.items():
            for rule in rules:
                key = str(rule.statement_text)
                if key not in seen:
                    seen.add(key)
                    yield rule


def _ceil_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_dims(bundle, tokenizer, *, max_n_demos_for_shapes: int, initial_ant_max: int):
    all_rules = list(_iter_bundle_rules(bundle))
    if not all_rules:
        raise ValueError("Split bundle has no rules.")

    max_rhs_atoms = max(len(rule.rhs) for rule in all_rules)
    max_prompt_facts = max(int(initial_ant_max), int(max_rhs_atoms))

    merged_predicate_arities = dict(bundle.train_bank.predicate_arities)
    merged_predicate_arities.update(bundle.eval_bank.predicate_arities)

    first_const = str(bundle.train_bank.constants[0])
    max_atom_len = 1
    for predicate, arity in merged_predicate_arities.items():
        atom_text = f"{predicate}({','.join(first_const for _ in range(int(arity)))})"
        max_atom_len = max(max_atom_len, len(tokenizer.encode_completion(atom_text)) - 1)

    max_prompt_len = (
        max_prompt_facts * max_atom_len
        + max(0, max_prompt_facts - 1)
        + 1
        + max_atom_len
        + 1
    )
    if int(max_n_demos_for_shapes) > 0:
        max_demo_clause_len = max(
            len(tokenizer.encode_completion(rule.statement_text)) - 1
            for rule in all_rules
        )
        max_prompt_len += int(max_n_demos_for_shapes) * (int(max_demo_clause_len) + 1)

    max_completion_len = max(
        len(tokenizer.encode_completion(rule.statement_text))
        for rule in all_rules
    )
    n_seq_ar = int(max_prompt_len + max_completion_len - 1)
    n_vocab = max(512, int(tokenizer.vocab_size))

    return {
        "n_vocab": n_vocab,
        "max_completion_len": max_completion_len,
        "n_seq_ar": n_seq_ar,
    }


# Rebuild a tokenizer from the split bundle (matches run.py pattern)
from task.layer_fol import _build_tokenizer_for_split_bundle
tokenizer = _build_tokenizer_for_split_bundle(split_bundle)

INITIAL_ANT_MAX = int(TASK_CFG["initial_ant_max"])
dims_train = compute_dims(
    split_bundle, tokenizer,
    max_n_demos_for_shapes=int(TASK_CFG["train_max_n_demos"]),
    initial_ant_max=INITIAL_ANT_MAX,
)
dims_eval = compute_dims(
    split_bundle, tokenizer,
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
    "n_hidden": 128,
    "n_heads": 4,
    "lr": 5e-4,
    "train_iters": 2000,
    "test_every": 500,
    "test_iters": 1,
    "batch_size": 16,
    "train_max_n_demos": int(TASK_CFG["train_max_n_demos"]),
    "eval_max_n_demos": int(TASK_CFG["eval_max_n_demos"]),
}


# <codecell>
# --- Create tasks and train a small Transformer ---

train_task_ar = FOLLayerTask(
    mode="online",
    task_split="depth3_icl_transfer",
    split_role="train",
    split_rule_bundle_path=SPLIT_PATH,
    distance_range=TASK_CFG["distance_range"],
    batch_size=TRAIN_CFG["batch_size"],
    seed=111,
    initial_ant_max=INITIAL_ANT_MAX,
    max_n_demos=TRAIN_CFG["train_max_n_demos"],
    prediction_objective="autoregressive",
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=N_SEQ,
    online_prefetch_backend="sync",
)

eval_task_ar = FOLLayerTask(
    mode="online",
    task_split="depth3_icl_transfer",
    split_role="eval",
    split_rule_bundle_path=SPLIT_PATH,
    distance_range=TASK_CFG["distance_range"],
    batch_size=TRAIN_CFG["batch_size"],
    seed=222,
    initial_ant_max=INITIAL_ANT_MAX,
    max_n_demos=TRAIN_CFG["eval_max_n_demos"],
    prediction_objective="autoregressive",
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=N_SEQ,
    online_prefetch_backend="sync",
)

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
        prompt = np.asarray(record["prompt"], dtype=np.int32)
        completion_gt = np.asarray(record["completions"][0], dtype=np.int32)
        src_layer = int(record["src_layer"])

        # Decode the ground-truth completion
        gt_text = tokenizer.decode_completion_text(completion_gt.tolist())

        # Decode the sequent from the prompt
        segments = _split_prompt_segments(prompt, tokenizer.sep_token_id)
        main_segment = segments[-1] + [int(tokenizer.sep_token_id)]
        sequent = tokenizer.decode_prompt(main_segment)

        # Run autoregressive prediction
        pred_completion = adapter.predict_completion(
            model=model_fn,
            prompt_tokens=prompt.tolist(),
            tokenizer=tokenizer,
            temperature=0.0,
            rng=None,
        )
        pred_text = tokenizer.decode_completion_text(pred_completion.tolist())

        # Match predicted completion against rule bank
        matched = match_rule_completion_fol(
            rule_bank=rule_bank,
            src_layer=src_layer,
            completion_tokens=pred_completion,
            expected_statement_text=gt_text,
            tokenizer=tokenizer,
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

        print(f"[{role} #{i}] status={status}")
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
# --- Cleanup ---
train_task_ar.close()
eval_task_ar.close()
