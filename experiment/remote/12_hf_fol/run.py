"""Multi-model finetuning sweep on FOL fresh-ICL (distributed via Accelerate).

Compares pretrained Pythia 1B (Transformer) vs Mamba 2 1.3B (SSM) finetuned
on FOL data. The run split selects a combined ``(model, task_shape)``
configuration via ``sys.argv[1]``.

Each GPU rank creates its own FOLLayerTask with a unique seed, so no
DataLoader/sampler wrapping is needed.  The Accelerate library handles
gradient sync across ranks.

Usage:
    # Single GPU test:
    python run.py
    # Multi-GPU via accelerate:
    accelerate launch --num_processes=N run.py 1
    # Via Bolt:
    python bolt/submit.py 12_hf_fol --auto --distributed
"""

# <codecell>
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from accelerate import Accelerator

from common import new_seed
from task.layer_fol import (
    FOLLayerTask,
    _build_tokenizer_for_fresh_icl,
    compute_fol_dims,
    print_task_preview,
)
from task.layer_gen.util.fol_rule_bank import build_random_fol_rule_bank
from train_hf import HFTrainConfig, train_hf
from wandb_utils import make_experiment_wandb_config


# ── Model configs ────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "name": "pythia-1b",
        "model_name_or_path": "EleutherAI/pythia-1b",
        "lr": 5e-5,
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    },
    {
        "name": "mamba2-1.3b",
        # "model_name_or_path": "state-spaces/mamba2-1.3b",
        "model_name_or_path": "AntonV/mamba2-1.3b-hf",
        "lr": 5e-5,
        "torch_dtype": "bfloat16",
    },
]

BASE_NUM_PRED = 16
SWEEP_MID_PRED = [64, 128, 256]
SWEEP_TASK_SHAPES = [
    {
        "predicates_per_layer": (1, p1, BASE_NUM_PRED),
        "rules_per_transition": (BASE_NUM_PRED, BASE_NUM_PRED**2),
    }
    for p1 in SWEEP_MID_PRED
]


# ── Experiment parameters ───────────────────────────────────────────────

RUN_ID = new_seed()
print("RUN ID", RUN_ID)

WANDB_PROJECT = Path(__file__).resolve().parent.name
WANDB_API_KEY_PATH = ROOT / "key" / "wandb.txt"
USE_WANDB = True

BATCH_SIZE = 4  # per device; effective = BATCH_SIZE × GRAD_ACCUM × num_gpus = 32
TRAIN_ITERS = 10_000
TEST_EVERY = 100
TEST_ITERS = 2
GRAD_ACCUM_STEPS = 1

FROM_SCRATCH = False
TOKENIZE_MODE = "native"
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
MAX_GRAD_NORM = 1.0
MIXED_PRECISION = "no"   # models loaded in bf16 via torch_dtype; Accelerate autocast is redundant

N_LAYERS = 3
ARITY_MAX = 1
VARS_PER_RULE_MAX = 6
K_IN_MAX = 1
K_OUT_MAX = 1
INITIAL_ANT_MAX = 1
CONSTANTS = [f"p{i}" for i in range(1)]
SAMPLE_MAX_ATTEMPTS = 4096
MAX_UNIFY_SOLUTIONS = 128
BASE_BANK_SEED = 2042
PREDICATE_NAME_LEN = 4
TRAIN_INCLUDE_ORACLE = True

TRAIN_MIN_N_DEMOS = 4
TRAIN_MAX_N_DEMOS = 8
EVAL_MAX_N_DEMOS = 8
FIXED_LENGTH_MODE = "next_pow2"
PREVIEW_EXAMPLES_PER_SPLIT = 3

BASE_SEED = 1000

### START TEST CONFIGS
# BATCH_SIZE = 4
# TRAIN_ITERS = 20
# TEST_EVERY = 10
# TEST_ITERS = 1
# GRAD_ACCUM_STEPS = 1
# PREDICATES_PER_LAYER = 10
# RULES_PER_TRANSITION = 18
# TRAIN_MAX_N_DEMOS = 4
# EVAL_MAX_N_DEMOS = 4
# MIXED_PRECISION = "no"
# MODEL_NAME = "gpt2"
# FROM_SCRATCH = True
# TOKENIZE_MODE = "direct"
# LR = 3e-4
# USE_WANDB = False
### END TEST CONFIGS

RUN_SPLIT = len(MODEL_CONFIGS) * len(SWEEP_TASK_SHAPES)

run_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
combo_idx = int(run_idx) % int(RUN_SPLIT)
model_idx = int(combo_idx % len(MODEL_CONFIGS))
task_shape_idx = int(combo_idx // len(MODEL_CONFIGS))
model_cfg = MODEL_CONFIGS[model_idx]
TASK_SHAPE = {
    "predicates_per_layer": tuple(int(v) for v in SWEEP_TASK_SHAPES[task_shape_idx]["predicates_per_layer"]),
    "rules_per_transition": tuple(int(v) for v in SWEEP_TASK_SHAPES[task_shape_idx]["rules_per_transition"]),
}

print(
    "SELECTED CONFIG",
    {
        "run_idx": int(run_idx),
        "combo_idx": int(combo_idx),
        "model_idx": int(model_idx),
        "task_shape_idx": int(task_shape_idx),
        "model_name": model_cfg["name"],
        "task_shape": TASK_SHAPE,
    },
)

MODEL_NAME = globals().get("MODEL_NAME", model_cfg["model_name_or_path"])
LR = globals().get("LR", model_cfg["lr"])


# ── Build rule bank & tokenizer ─────────────────────────────────────────
# <codecell>

def _task_shape_tag(task_shape: dict) -> str:
    return f"mid{int(task_shape['predicates_per_layer'][1])}"


def _fresh_icl_config(task_shape: dict) -> dict:
    return {
        "base_bank_seed": BASE_BANK_SEED,
        "predicates_per_layer": tuple(int(v) for v in task_shape["predicates_per_layer"]),
        "rules_per_transition": tuple(int(v) for v in task_shape["rules_per_transition"]),
        "mid_pred": int(task_shape["predicates_per_layer"][1]),
        "n_layers": N_LAYERS,
        "arity_max": ARITY_MAX,
        "vars_per_rule_max": VARS_PER_RULE_MAX,
        "k_in_max": K_IN_MAX,
        "k_out_max": K_OUT_MAX,
        "initial_ant_max": INITIAL_ANT_MAX,
        "constants": list(CONSTANTS),
        "train_max_n_demos": TRAIN_MAX_N_DEMOS,
        "eval_max_n_demos": EVAL_MAX_N_DEMOS,
        "predicate_name_len": int(PREDICATE_NAME_LEN),
        "train_include_oracle": bool(TRAIN_INCLUDE_ORACLE),
        "sample_max_attempts": SAMPLE_MAX_ATTEMPTS,
        "max_unify_solutions": MAX_UNIFY_SOLUTIONS,
    }


def _build_base_bank_and_tokenizer(*, task_shape: dict):
    base_bank = build_random_fol_rule_bank(
        n_layers=int(N_LAYERS),
        predicates_per_layer=task_shape["predicates_per_layer"],
        rules_per_transition=task_shape["rules_per_transition"],
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        constants=tuple(str(c) for c in CONSTANTS),
        rng=np.random.default_rng(int(BASE_BANK_SEED)),
    )
    tokenizer = _build_tokenizer_for_fresh_icl(
        base_bank=base_bank,
        predicate_name_len=int(PREDICATE_NAME_LEN),
    )
    return base_bank, tokenizer


def _ceil_pow2_int(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _make_layer_task(
    *,
    task_shape: dict,
    split_role: str,
    seed: int,
    drop_remainder: bool,
    min_n_demos: int,
    max_n_demos: int,
    include_oracle: bool = False,
):
    return FOLLayerTask(
        distance_range=(2, 2),
        batch_size=int(BATCH_SIZE),
        mode="online",
        task_split="depth3_fresh_icl",
        split_role=str(split_role),
        shuffle=True,
        seed=seed,
        worker_count=0,
        drop_remainder=drop_remainder,
        prediction_objective="autoregressive",
        predicates_per_layer=task_shape["predicates_per_layer"],
        rules_per_transition=task_shape["rules_per_transition"],
        fresh_icl_base_bank_seed=int(BASE_BANK_SEED),
        arity_max=int(ARITY_MAX),
        vars_per_rule_max=int(VARS_PER_RULE_MAX),
        constants=tuple(str(tok) for tok in CONSTANTS),
        k_in_max=int(K_IN_MAX),
        k_out_max=int(K_OUT_MAX),
        initial_ant_max=int(INITIAL_ANT_MAX),
        min_n_demos=int(min_n_demos),
        max_n_demos=int(max_n_demos),
        include_oracle=bool(include_oracle),
        sample_max_attempts=int(SAMPLE_MAX_ATTEMPTS),
        max_unify_solutions=int(MAX_UNIFY_SOLUTIONS),
        fixed_length_mode=str(FIXED_LENGTH_MODE),
        fixed_length_n_seq=int(N_SEQ_AR),
        predicate_name_len=int(PREDICATE_NAME_LEN),
    )


# <codecell>
base_bank, fol_tokenizer = _build_base_bank_and_tokenizer(task_shape=TASK_SHAPE)

from task.layer_fol.common import _fresh_predicate_sentinels

sentinels = _fresh_predicate_sentinels(name_len=int(PREDICATE_NAME_LEN))
extra_arities = {s: int(base_bank.arity_max) for s in sentinels}

dims = compute_fol_dims(
    rule_banks=[base_bank],
    tokenizer=fol_tokenizer,
    initial_ant_max=int(INITIAL_ANT_MAX),
    max_n_demos=int(max(TRAIN_MAX_N_DEMOS, EVAL_MAX_N_DEMOS)),
    extra_predicate_arities=extra_arities,
    fresh_k_in_max=int(K_IN_MAX),
    fresh_k_out_max=int(K_OUT_MAX),
)

N_VOCAB = int(dims["n_vocab"])
N_SEQ_AR_RAW = int(dims["n_seq_ar"])
N_SEQ_AR = int(max(2, _ceil_pow2_int(N_SEQ_AR_RAW)))

print("DIMS", dims)
print(f"N_VOCAB={N_VOCAB}  N_SEQ_AR={N_SEQ_AR}")


# ── Accelerator & per-rank data seeding ─────────────────────────────────
# <codecell>

accelerator = Accelerator(
    mixed_precision=MIXED_PRECISION,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
)

rank_seed = BASE_SEED + accelerator.process_index
print(f"[Rank {accelerator.process_index}] data seed = {rank_seed}")

if accelerator.is_main_process:
    preview_train_task = _make_layer_task(
        task_shape=TASK_SHAPE,
        split_role="train",
        seed=rank_seed,
        drop_remainder=True,
        min_n_demos=int(TRAIN_MIN_N_DEMOS),
        max_n_demos=int(TRAIN_MAX_N_DEMOS),
        include_oracle=bool(TRAIN_INCLUDE_ORACLE),
    )
    preview_eval_task = _make_layer_task(
        task_shape=TASK_SHAPE,
        split_role="eval",
        seed=rank_seed + 500_000,
        drop_remainder=False,
        min_n_demos=int(EVAL_MAX_N_DEMOS),
        max_n_demos=int(EVAL_MAX_N_DEMOS),
    )
    try:
        print_task_preview(
            preview_train_task,
            role="train",
            n_examples=PREVIEW_EXAMPLES_PER_SPLIT,
        )
        print_task_preview(
            preview_eval_task,
            role="eval",
            n_examples=PREVIEW_EXAMPLES_PER_SPLIT,
        )
    finally:
        for task in (preview_train_task, preview_eval_task):
            close = getattr(task, "close", None)
            if callable(close):
                close()

train_task = _make_layer_task(
    task_shape=TASK_SHAPE,
    split_role="train",
    seed=rank_seed,
    drop_remainder=True,
    min_n_demos=int(TRAIN_MIN_N_DEMOS),
    max_n_demos=int(TRAIN_MAX_N_DEMOS),
    include_oracle=bool(TRAIN_INCLUDE_ORACLE),
)

eval_task = _make_layer_task(
    task_shape=TASK_SHAPE,
    split_role="eval",
    seed=rank_seed + 500_000,
    drop_remainder=False,
    min_n_demos=int(EVAL_MAX_N_DEMOS),
    max_n_demos=int(EVAL_MAX_N_DEMOS),
)


# ── Train ───────────────────────────────────────────────────────────────
# <codecell>

hf_config = HFTrainConfig(
    model_name_or_path=MODEL_NAME,
    from_scratch=FROM_SCRATCH,
    tokenize_mode=TOKENIZE_MODE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_frac=WARMUP_FRAC,
    max_grad_norm=MAX_GRAD_NORM,
    train_iters=TRAIN_ITERS,
    test_every=TEST_EVERY,
    test_iters=TEST_ITERS,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    mixed_precision=MIXED_PRECISION,
    use_tqdm=True,
    torch_dtype=model_cfg.get("torch_dtype"),
    attn_implementation=model_cfg.get("attn_implementation"),
)
wandb_cfg = make_experiment_wandb_config(
    enabled=USE_WANDB,
    project=WANDB_PROJECT,
    run_id=RUN_ID,
    run_name=f"{WANDB_PROJECT}-{model_cfg['name']}-{_task_shape_tag(TASK_SHAPE)}-{RUN_ID}",
    api_key_path=WANDB_API_KEY_PATH,
    model_config=hf_config,
    extra_config={
        "model_variant": model_cfg["name"],
        "task_shape_idx": int(task_shape_idx),
        "task_shape_tag": _task_shape_tag(TASK_SHAPE),
        "dims": dims,
        "batch_size": int(BATCH_SIZE),
        "n_vocab": int(N_VOCAB),
        "n_seq_ar": int(N_SEQ_AR),
        "n_gpus": int(accelerator.num_processes),
        "fresh_icl_config": _fresh_icl_config(TASK_SHAPE),
    },
)

train_start = time.perf_counter()

model, hist = train_hf(
    hf_config,
    train_task,
    test_iter=eval_task,
    fol_tokenizer=fol_tokenizer,
    vocab_size=N_VOCAB,
    accelerator=accelerator,
    wandb_cfg=wandb_cfg,
    seed=int(RUN_ID),
)

train_wall_s = time.perf_counter() - train_start


# ── Save results (main process only) ────────────────────────────────────
# <codecell>

if accelerator.is_main_process:
    save_dir = Path("set")
    save_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": RUN_ID,
        "run_split": int(RUN_SPLIT),
        "combo_idx": int(combo_idx),
        "task_shape_idx": int(task_shape_idx),
        "task_shape_tag": _task_shape_tag(TASK_SHAPE),
        "mid_pred": int(TASK_SHAPE["predicates_per_layer"][1]),
        "model_config_name": model_cfg["name"],
        "model_name": MODEL_NAME,
        "from_scratch": FROM_SCRATCH,
        "tokenize_mode": TOKENIZE_MODE,
        "n_vocab": N_VOCAB,
        "n_seq_ar": N_SEQ_AR,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_frac": WARMUP_FRAC,
        "train_iters": TRAIN_ITERS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "mixed_precision": MIXED_PRECISION,
        "batch_size": BATCH_SIZE,
        "n_gpus": accelerator.num_processes,
        "dims": dims,
        "hist": hist,
        "train_wall_s": float(train_wall_s),
        "fresh_icl_config": _fresh_icl_config(TASK_SHAPE),
    }

    out_path = save_dir / f"res.{model_cfg['name']}.{_task_shape_tag(TASK_SHAPE)}.{RUN_ID}.pkl"
    pd.DataFrame([row]).to_pickle(out_path)
    print(f"Results saved to {out_path}")

# Clean up data iterators
for task in (train_task, eval_task):
    close = getattr(task, "close", None)
    if callable(close):
        close()

print("done!")
