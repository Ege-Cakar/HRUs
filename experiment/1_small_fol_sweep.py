"""
Experiment 1: Small FOL sweep — all 4 architectures × k=1,2,3,4 on internalized rules.

Sweeps core depth k across all architecture conditions:
  - Standard Transformer:   Attn + [Attn×4]×k + Attn     (4k+2 blocks, independent weights)
  - Hybrid (GDN+Attn):      Attn + [GDN,GDN,GDN,Attn]×k + Attn
  - Recursive Transformer:  Attn + [Attn×4]×k + Attn     (weight-tied, k recursions)
  - HRU (Recursive Hybrid): Attn + [GDN,GDN,GDN,Attn]×k + Attn (weight-tied)

At each k, non-recursive models have k×4 independent core blocks while
recursive models reuse 4 core blocks looped k times. This makes the fair
comparison: same effective depth, different parameter counts.

Includes depth extrapolation eval: train D=1..4, eval D=1..8.

Run:
    python experiment/1_small_fol_sweep.py

Requires: key/wandb.txt with W&B API key (or set WANDB_ENABLED=False).
"""

# <codecell> Config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import asdict
import jax
from flax import nnx

from model.architecture import ArchConfig
from model.recursive import RecursiveArchConfig
from task.fol_task_factory import FOLTaskFactory, RuleBankConfig
from train import train, Case, warmup_cosine_schedule, loss_and_acc
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

# ── Model dims (small) ──────────────────────────────────────────────
N_HIDDEN = 64
N_HEADS = 4
K_VALUES = [1, 2, 3, 4]  # core depth sweep

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
print(f"Core depth sweep: k={K_VALUES}")

# <codecell> Define architecture configs for all (arch, k) pairs
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

ARCH_DEFS = [
    # (name_prefix, config_class, config_key, block_type)
    ("std",     ArchConfig,          "n_core_repeats", "attn"),
    ("hyb",     ArchConfig,          "n_core_repeats", "hybrid"),
    ("rec",     RecursiveArchConfig, "n_recurrences",  "attn"),
    ("hru",     RecursiveArchConfig, "n_recurrences",  "hybrid"),
]

configs = {}
for prefix, ConfigCls, depth_key, block_type in ARCH_DEFS:
    for k in K_VALUES:
        name = f"{prefix}_k{k}"
        n_blocks = 4 * k + 2
        cfg = ConfigCls(
            core_block=block_type,
            **{depth_key: k},
            **shared_model_kw,
        )
        configs[name] = cfg

# <codecell> Build cases
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

    # Extract k from name
    k = int(name.split("_k")[1])
    n_blocks = 4 * k + 2
    arch_type = name.split("_k")[0]

    print(f"{name}: {n_params:,} params, {n_blocks} blocks (k={k})")

    lr_schedule = warmup_cosine_schedule(LR, train_iters=TRAIN_ITERS, warmup_frac=0.05)

    wandb_cfg = WandbConfig(
        enabled=WANDB_ENABLED,
        project=WANDB_PROJECT,
        name=name,
        config={
            "architecture": arch_type,
            "k": k,
            "n_blocks": n_blocks,
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
        info={"n_params": n_params, "k": k, "arch": arch_type},
    ))

print(f"\nTotal cases: {len(cases)} (4 architectures × {len(K_VALUES)} depths)")

# <codecell> Train all cases
for i, case in enumerate(cases):
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(cases)}] Training: {case.name}")
    print(f"  {case.info['n_params']:,} params, k={case.info['k']}")
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

# Header
header = f"{'case':<20}"
for d in range(1, D_EVAL_MAX + 1):
    marker = "*" if d > D_TRAIN else " "
    header += f"  D={d}{marker}"
print(header)
print("-" * len(header))

for case in cases:
    row = f"{case.name:<20}"
    for d in range(1, D_EVAL_MAX + 1):
        eval_task = factory.make_internalized_task(
            batch_size=BATCH_SIZE,
            distance_range=(d, d),
        )
        results = case.eval(eval_task, [loss_and_acc], n_iters=5, prefix=f"depth_{d}")
        depth_results = case.info.get(f"depth_{d}", {})
        acc_val = depth_results.get("acc", 0.0)
        row += f"  {acc_val:.3f}"
    print(row)

print(f"\n* = extrapolation beyond training depth (D > {D_TRAIN})")

# <codecell> Summary table: params vs accuracy
print(f"\n{'='*60}")
print(f"  Parameter Efficiency Summary")
print(f"{'='*60}\n")
print(f"{'case':<20} {'params':>10} {'k':>3} {'train_acc':>10} {'D=4 acc':>10} {'D=8 acc':>10}")
print("-" * 73)
for case in cases:
    final_test = case.hist['test'][-1] if case.hist['test'] else {}
    d4 = case.info.get("depth_4", {}).get("acc", 0.0)
    d8 = case.info.get("depth_8", {}).get("acc", 0.0)
    print(f"{case.name:<20} {case.info['n_params']:>10,} {case.info['k']:>3} "
          f"{final_test.get('acc', 0.0):>10.4f} {d4:>10.4f} {d8:>10.4f}")

print("\nDone!")
