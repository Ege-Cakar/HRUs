# <codecell>
"""Interactive inspection script for experiment 11 full-completion fresh-rule split."""

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
    extract_prompt_info_from_row_tokens,
)
from task.layer_gen.util.fol_rule_bank import build_random_fol_rule_bank
from train import train, warmup_cosine_schedule


# <codecell>
SET_DIR = ROOT / "experiment" / "interactive" / "11_fresh_rule_split_full_completion" / "set"
SET_DIR.mkdir(parents=True, exist_ok=True)

FRESH_ICL_CFG = {
    "seed": 2042,
    "predicates_per_layer": 16,
    "rules_per_transition": 32,
    "fresh_icl_n_predicates": 16,
    "arity_max": 1,
    "vars_per_rule_max": 6,
    "k_in_max": 1,
    "k_out_max": 3,
    "constants": tuple(f"p{i}" for i in range(8)),
}
TASK_CFG = {
    "distance_range": (2, 2),
    "initial_ant_max": 3,
    "train_min_n_demos": 4,
    "train_max_n_demos": 8,
    "eval_max_n_demos": 8,
}
COMPLETION_STEPS_MAX = 2


# <codecell>
def _ceil_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def preview_record(task: FOLLayerTask, record: dict, *, role: str) -> None:
    tokenizer = task.tokenizer
    prompt = np.asarray(record["prompt"], dtype=np.int32)
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    completion_texts = tokenizer.decode_completion_texts(completion.tolist())
    _, sequent, _, _ = extract_prompt_info_from_row_tokens(prompt, tokenizer=tokenizer)

    print(
        f"[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
        f"prompt_len={prompt.size} completion_len={completion.size} "
        f"n_steps={len(completion_texts)} n_demos={len(record.get('rule_context', {}).get('demo_schema_texts', []))}"
    )
    print("  prompt:", sequent.text)
    for idx, text in enumerate(completion_texts):
        print(f"  completion[{idx}]: {text}")


# <codecell>
base_bank = build_random_fol_rule_bank(
    n_layers=3,
    predicates_per_layer=int(FRESH_ICL_CFG["predicates_per_layer"]),
    rules_per_transition=int(FRESH_ICL_CFG["rules_per_transition"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(c) for c in FRESH_ICL_CFG["constants"]),
    k_in_min=1,
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_min=1,
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    rng=np.random.default_rng(int(FRESH_ICL_CFG["seed"])),
)
tokenizer = _build_tokenizer_for_fresh_icl(base_bank=base_bank)

sentinels = _fresh_predicate_sentinels()
extra_arities = {s: int(base_bank.arity_max) for s in sentinels}
dims_train = compute_fol_dims(
    rule_banks=[base_bank],
    tokenizer=tokenizer,
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["train_max_n_demos"]),
    completion_format="full",
    completion_steps_max=int(COMPLETION_STEPS_MAX),
    extra_predicate_arities=extra_arities,
    fresh_k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    fresh_k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
)
dims_eval = compute_fol_dims(
    rule_banks=[base_bank],
    tokenizer=tokenizer,
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
    completion_format="full",
    completion_steps_max=int(COMPLETION_STEPS_MAX),
    extra_predicate_arities=extra_arities,
    fresh_k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    fresh_k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
)

N_VOCAB = max(int(dims_train["n_vocab"]), int(dims_eval["n_vocab"]))
MAX_COMPLETION_LEN = max(int(dims_train["max_completion_len"]), int(dims_eval["max_completion_len"]))
N_SEQ = max(2, _ceil_pow2(max(int(dims_train["n_seq_ar"]), int(dims_eval["n_seq_ar"]))))

print("dims_train:", dims_train)
print("dims_eval:", dims_eval)
print(f"N_VOCAB={N_VOCAB} N_SEQ={N_SEQ} MAX_COMPLETION_LEN={MAX_COMPLETION_LEN}")

TRAIN_CFG = {
    "n_layers": 4,
    "n_hidden": 256,
    "n_heads": 4,
    "lr": warmup_cosine_schedule(3e-4, 200),
    "train_iters": 200,
    "test_every": 50,
    "test_iters": 1,
    "batch_size": 4,
    "grad_accum_steps": 4,
}


# <codecell>
train_task = FOLLayerTask(
    mode="online",
    task_split="depth3_fresh_icl",
    split_role="train",
    distance_range=TASK_CFG["distance_range"],
    batch_size=int(TRAIN_CFG["batch_size"]),
    shuffle=True,
    seed=101,
    worker_count=0,
    drop_remainder=True,
    prediction_objective="autoregressive",
    completion_format="full",
    predicates_per_layer=int(FRESH_ICL_CFG["predicates_per_layer"]),
    rules_per_transition=int(FRESH_ICL_CFG["rules_per_transition"]),
    fresh_icl_n_predicates=int(FRESH_ICL_CFG["fresh_icl_n_predicates"]),
    fresh_icl_base_bank_seed=int(FRESH_ICL_CFG["seed"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(tok) for tok in FRESH_ICL_CFG["constants"]),
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    min_n_demos=int(TASK_CFG["train_min_n_demos"]),
    max_n_demos=int(TASK_CFG["train_max_n_demos"]),
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=int(N_SEQ),
)
eval_task = FOLLayerTask(
    mode="online",
    task_split="depth3_fresh_icl",
    split_role="eval",
    distance_range=TASK_CFG["distance_range"],
    batch_size=int(TRAIN_CFG["batch_size"]),
    shuffle=True,
    seed=202,
    worker_count=0,
    drop_remainder=False,
    prediction_objective="autoregressive",
    completion_format="full",
    predicates_per_layer=int(FRESH_ICL_CFG["predicates_per_layer"]),
    rules_per_transition=int(FRESH_ICL_CFG["rules_per_transition"]),
    fresh_icl_n_predicates=int(FRESH_ICL_CFG["fresh_icl_n_predicates"]),
    fresh_icl_base_bank_seed=int(FRESH_ICL_CFG["seed"]),
    arity_max=int(FRESH_ICL_CFG["arity_max"]),
    vars_per_rule_max=int(FRESH_ICL_CFG["vars_per_rule_max"]),
    constants=tuple(str(tok) for tok in FRESH_ICL_CFG["constants"]),
    k_in_max=int(FRESH_ICL_CFG["k_in_max"]),
    k_out_max=int(FRESH_ICL_CFG["k_out_max"]),
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    min_n_demos=int(TASK_CFG["eval_max_n_demos"]),
    max_n_demos=int(TASK_CFG["eval_max_n_demos"]),
    fixed_length_mode="next_pow2",
    fixed_length_n_seq=int(N_SEQ),
)

preview_record(train_task, train_task._sample_online_record(), role="train")
preview_record(eval_task, eval_task._sample_online_record(), role="eval")


# <codecell>
config = TransformerConfig(
    n_vocab=N_VOCAB,
    n_seq=N_SEQ,
    n_layers=int(TRAIN_CFG["n_layers"]),
    n_hidden=int(TRAIN_CFG["n_hidden"]),
    n_heads=int(TRAIN_CFG["n_heads"]),
    n_out=N_VOCAB,
    n_pred_tokens=1,
    pos_encoding="rope",
    layer_norm=True,
    use_swiglu=False,
    use_bias=True,
    dropout_rate=0.0,
    output_mode="full_sequence",
    pad_token_id=0,
)

optimizer, hist = train(
    config,
    train_iter=train_task,
    test_iter=eval_task,
    loss="ce_mask",
    train_iters=int(TRAIN_CFG["train_iters"]),
    test_iters=int(TRAIN_CFG["test_iters"]),
    test_every=int(TRAIN_CFG["test_every"]),
    grad_accum_steps=int(TRAIN_CFG["grad_accum_steps"]),
    lr=TRAIN_CFG["lr"],
    seed=42,
)
print("final train metrics:", hist["train"][-1])
print("final eval metrics:", hist["test"][-1])


# <codecell>
adapter = AutoregressiveLogitsAdapter(
    n_seq=int(N_SEQ),
    max_completion_len=int(MAX_COMPLETION_LEN),
    pad_token_id=0,
    jit_step=True,
)
model_fn = make_model_callable(optimizer, to_numpy=False)
record = eval_task._sample_online_record()
pred_completion = adapter.predict_completion(
    model=model_fn,
    prompt_tokens=np.asarray(record["prompt"], dtype=np.int32),
    tokenizer=eval_task.tokenizer,
    temperature=0.0,
    rng=np.random.default_rng(0),
)
print("target:", eval_task.tokenizer.decode_completion_texts(record["completions"][0].tolist()))
print("pred:", eval_task.tokenizer.decode_completion_texts(pred_completion.tolist()))


# <codecell>
for task in (train_task, eval_task):
    close = getattr(task, "close", None)
    if callable(close):
        close()
