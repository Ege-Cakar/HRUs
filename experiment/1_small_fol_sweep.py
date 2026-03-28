"""
Experiment 1: Small FOL sweep — all 4 architectures on internalized rules.

Trains small models (~1-5M params) locally with online data to verify:
1. All architectures learn (loss decreases)
2. Recursive models match non-recursive at same effective depth
3. Recursive models extrapolate beyond training depth D
4. wandb logging works end-to-end

Run:
    python experiment/1_small_fol_sweep.py

This is a LOCAL smoke test — not a cluster experiment. Runs in ~30-60 min
on a laptop CPU / single GPU depending on hardware.
"""

# <codecell> Config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, asdict
import jax
from flax import nnx

from model.architecture import ArchConfig
from model.recursive import RecursiveArchConfig
from task.fol_task_factory import FOLTaskFactory, RuleBankConfig
from train import train, Case, warmup_cosine_schedule
from wandb_utils import WandbConfig

# ── Task ─────────────────────────────────────────────────────────────
SEED = 42
D_TRAIN = 4          # train on depths 1..4
D_EVAL_MAX = 8       # eval up to depth 8 (2x extrapolation)
BATCH_SIZE = 32
TRAIN_ITERS = 2000
TEST_EVERY = 200
LR = 3e-4
WANDB_ENABLED = True
WANDB_PROJECT = "hru-fol-small"

# ── Model dims (small, ~1-5M params) ────────────────────────────────
N_HIDDEN = 64
N_HEADS = 4
N_CORE_REPEATS = 1   # k=1 for non-recursive (6 blocks)
N_RECURRENCES = 1    # T=1 for recursive (6 blocks, weight-tied)

# <codecell> Build task factory
factory = FOLTaskFactory(
    rule_bank_seed=SEED,
    d_train_max=D_TRAIN,
    d_eval_max=D_EVAL_MAX,
    bank_config=RuleBankConfig(
        n_layers=8,
        predicates_per_layer=6,
        arity_min=1,
        arity_max=2,
        constants=("a", "b", "c", "d"),
        rules_per_transition=8,
        k_in_min=1,
        k_in_max=2,
        k_out_min=1,
        k_out_max=1,
        vars_per_rule_max=3,
    ),
)

n_vocab = factory.n_vocab
n_seq = factory.dims_internalized.n_seq_ar

print(f"Vocab: {n_vocab}, Seq len: {n_seq}")
print(f"Training depths: 1..{D_TRAIN}, Eval depths: 1..{D_EVAL_MAX}")

# <codecell> Define architecture cases
shared_model_kw = dict(
    n_vocab=n_vocab,
    n_seq=n_seq,
    n_hidden=N_HIDDEN,
    n_heads=N_HEADS,
    n_out=n_vocab,
    pos_encoding="rope",
    output_mode="full_sequence",
    use_swiglu=True,
    use_bf16=False,  # CPU-friendly; set True for GPU
)

configs = {
    "standard_transformer": ArchConfig(
        core_block="attn",
        n_core_repeats=N_CORE_REPEATS,
        **shared_model_kw,
    ),
    "hybrid": ArchConfig(
        core_block="hybrid",
        n_core_repeats=N_CORE_REPEATS,
        **shared_model_kw,
    ),
    "recursive_transformer": RecursiveArchConfig(
        core_block="attn",
        n_recurrences=N_RECURRENCES,
        **shared_model_kw,
    ),
    "recursive_hybrid_hru": RecursiveArchConfig(
        core_block="hybrid",
        n_recurrences=N_RECURRENCES,
        **shared_model_kw,
    ),
}

# <codecell> Build cases
lr_schedule = warmup_cosine_schedule(LR, train_iters=TRAIN_ITERS, warmup_frac=0.05)

cases = []
for name, config in configs.items():
    train_task = factory.make_internalized_task(
        batch_size=BATCH_SIZE,
        distance_range=(1, D_TRAIN),
    )
    test_task = factory.make_internalized_task(
        batch_size=BATCH_SIZE,
        distance_range=(1, D_TRAIN),
    )

    # Count params
    model_tmp = config.to_model(rngs=nnx.Rngs(0))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model_tmp)))
    del model_tmp
    print(f"{name}: {n_params:,} params, config={type(config).__name__}")

    wandb_cfg = WandbConfig(
        enabled=WANDB_ENABLED,
        project=WANDB_PROJECT,
        name=name,
        config={
            "architecture": name,
            "model_config": asdict(config),
            "seed": SEED,
            "d_train": D_TRAIN,
            "d_eval_max": D_EVAL_MAX,
            "batch_size": BATCH_SIZE,
            "train_iters": TRAIN_ITERS,
            "lr": LR,
            "n_params": n_params,
        },
        api_key_path="key/wandb.txt",
    )

    cases.append(Case(
        name=name,
        config=config,
        train_task=train_task,
        test_task=test_task,
        train_args=dict(
            loss='ce_mask',
            train_iters=TRAIN_ITERS,
            test_every=TEST_EVERY,
            test_iters=2,
            lr=lr_schedule,
            clip=1.0,
            weight_decay=0.1,
            seed=SEED,
            wandb_cfg=wandb_cfg,
        ),
        info={"n_params": n_params},
    ))

# <codecell> Train all cases
for case in cases:
    print(f"\n{'='*60}")
    print(f"  Training: {case.name}")
    print(f"{'='*60}\n")
    case.run()

    final_train = case.hist['train'][-1] if case.hist['train'] else {}
    final_test = case.hist['test'][-1] if case.hist['test'] else {}
    print(f"\n  Final — train_loss={final_train.get('loss', '?'):.4f}  "
          f"test_loss={final_test.get('loss', '?'):.4f}")

# <codecell> Depth extrapolation eval
print(f"\n{'='*60}")
print(f"  Depth Extrapolation Evaluation")
print(f"{'='*60}\n")

from train import loss_and_acc

for case in cases:
    print(f"\n--- {case.name} ---")
    for d in range(1, D_EVAL_MAX + 1):
        eval_task = factory.make_internalized_task(
            batch_size=BATCH_SIZE,
            distance_range=(d, d),
        )
        marker = " *" if d > D_TRAIN else ""
        results = case.eval(eval_task, [loss_and_acc], n_iters=5, prefix=f"depth_{d}")
        depth_results = case.info.get(f"depth_{d}", {})
        loss_val = depth_results.get("loss", "?")
        acc_val = depth_results.get("acc", "?")
        print(f"  D={d}: loss={loss_val:.4f}  acc={acc_val:.4f}{marker}")

print("\n* = extrapolation beyond training depth")
print("\nDone!")
