# <codecell>
"""Interactive playground for layered first-order task sampling."""

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from layer_fol import FOLLayerTask
from layer_gen.util.fol_rule_bank import (
    FOLSequent,
    build_random_fol_rule_bank,
    sample_fol_problem,
)
from layer_gen.util import tokenize_layer_fol as tok


def preview_batch(task: FOLLayerTask, n_rows: int = 3) -> None:
    xs, ys = next(task)
    print(f"xs shape={xs.shape}, ys shape={ys.shape}")

    decoded_x = task.tokenizer.decode_batch_ids(xs)
    decoded_y = task.tokenizer.decode_batch_ids(ys)

    n_show = min(n_rows, xs.shape[0])
    for idx in range(n_show):
        print(f"\n[{idx}] prompt+prefix")
        print(decoded_x[idx])
        print(f"[{idx}] targets")
        print(decoded_y[idx])


# <codecell>
# Sample a random FOL rule bank + one multi-step problem.
rng = np.random.default_rng()
rule_bank = build_random_fol_rule_bank(
    n_layers=50,
    predicates_per_layer=8,
    rules_per_transition=32,
    arity_max=3,
    vars_per_rule_max=4,
    constants=("alice", "bob", "carol", "dave"),
    k_in_max=3,
    k_out_max=3,
    rng=rng,
)

problem = sample_fol_problem(
    bank=rule_bank,
    distance=4,
    initial_ant_max=3,
    rng=rng,
)

print("goal:", problem.goal_atom.text)
for idx, (src_layer, ants, templ, inst, subst) in enumerate(
    zip(
        problem.step_layers,
        problem.step_ants,
        problem.step_rule_templates,
        problem.step_rules,
        problem.step_substitutions,
    )
):
    print(f"\nstep={idx} src_layer={src_layer}")
    print("facts:", ", ".join(atom.text for atom in ants))
    print("template:", templ.statement_text)
    print("subst:", subst)
    print("instantiated:", inst.statement_text)


# <codecell>
# Tokenizer roundtrip demo.
tokenizer = tok.build_tokenizer_from_rule_bank(rule_bank)
sequent = FOLSequent(ants=problem.step_ants[0], cons=problem.goal_atom)
statement_text = problem.step_rules[0].statement_text

prompt, completion = tokenizer.tokenize_example(sequent, statement_text)
print("prompt text:", sequent.text)
print("completion text:", statement_text)
print("decoded prompt:", tokenizer.decode_prompt(prompt).text)
print("decoded completion:", tokenizer.decode_completion_text(completion))

# <codecell>
for i in prompt:
    try:
        print(tokenizer.id_to_char(i))
    except ValueError:
        print('skip', i)


# <codecell>
# Online task sampling (autoregressive objective).
task = FOLLayerTask(
    batch_size=8,
    mode="online",
    distance_range=(1, 3),
    n_layers=12,
    predicates_per_layer=6,
    rules_per_transition=16,
    arity_max=3,
    vars_per_rule_max=20,
    constants=(f'p{i}' for i in range(20)),
    k_in_max=3,
    k_out_max=3,
    initial_ant_max=3,
    prediction_objective="autoregressive",
    max_n_demos=0
)

preview_batch(task, n_rows=2)

# <codecell>
task.rule_bank.transition_rules(0)


# <codecell>
# Online task sampling (all-at-once objective).
task_all = FOLLayerTask(
    batch_size=4,
    mode="online",
    seed=321,
    distance_range=(1, 3),
    n_layers=10,
    predicates_per_layer=5,
    rules_per_transition=12,
    arity_max=3,
    vars_per_rule_max=4,
    constants=("alice", "bob", "carol"),
    k_in_max=2,
    k_out_max=2,
    initial_ant_max=3,
    prediction_objective="all_at_once",
)

preview_batch(task_all, n_rows=2)


# <codecell>
# Optional offline mode preview (only runs if dataset exists).
DS_PATH = ROOT / "task" / "layer_gen" / "data" / "toy_layer_fol"
if DS_PATH.exists():
    offline_task = FOLLayerTask(
        ds_path=DS_PATH,
        distance_range=(1, 2),
        batch_size=4,
        mode="offline",
        shuffle=False,
        worker_count=0,
    )
    preview_batch(offline_task, n_rows=2)
else:
    print(f"Offline dataset not found: {DS_PATH}")


# <codecell>
# Quick throughput comparison (optional): online sync/thread/process.
N_WARMUP = 50
N_BATCHES = 500

base_kwargs = {
    "batch_size": 64,
    "prediction_objective": "autoregressive",
    "distance_range": (1, 3),
    "seed": 7,
    "n_layers": 12,
    "predicates_per_layer": 6,
    "rules_per_transition": 16,
    "arity_max": 3,
    "vars_per_rule_max": 4,
    "constants": ("alice", "bob", "carol", "dave"),
    "k_in_max": 3,
    "k_out_max": 3,
    "initial_ant_max": 3,
}

rows = []
for backend in ("sync", "thread", "process"):
    task_bench = FOLLayerTask(
        mode="online",
        online_prefetch_backend=backend,
        online_prefetch_workers=8,
        online_prefetch_buffer_size=256,
        **base_kwargs,
    )
    try:
        for _ in range(N_WARMUP):
            next(task_bench)
        t0 = time.perf_counter()
        for _ in range(N_BATCHES):
            next(task_bench)
        elapsed = time.perf_counter() - t0
    finally:
        task_bench.close()

    bps = N_BATCHES / elapsed
    eps = N_BATCHES * base_kwargs["batch_size"] / elapsed
    rows.append(
        {
            "requested": backend,
            "resolved": task_bench.online_prefetch_backend_resolved,
            "workers": task_bench.online_prefetch_workers_resolved,
            "buffer": task_bench.online_prefetch_buffer_size_resolved,
            "sec": elapsed,
            "bps": bps,
            "eps": eps,
        }
    )

print(
    f"{'requested':>10s} {'resolved':>10s} {'workers':>7s} "
    f"{'buffer':>7s} {'sec':>8s} {'batches/s':>10s} {'examples/s':>11s}"
)
for row in rows:
    print(
        f"{row['requested']:>10s} {row['resolved']:>10s} {row['workers']:>7d} "
        f"{row['buffer']:>7d} {row['sec']:>8.2f} {row['bps']:>10.1f} {row['eps']:>11.1f}"
    )

sync_row = next((row for row in rows if row["requested"] == "sync"), None)
if sync_row is not None:
    for row in rows:
        if row["requested"] == "sync":
            continue
        print(
            f"{row['requested']:>10s} speedup vs sync: "
            f"{row['eps'] / sync_row['eps']:.2f}x"
        )

# Optional offline comparison against sync online.
if DS_PATH.exists():
    offline_task = FOLLayerTask(
        mode="offline",
        ds_path=DS_PATH,
        shuffle=False,
        worker_count=0,
        batch_size=base_kwargs["batch_size"],
        prediction_objective=base_kwargs["prediction_objective"],
        distance_range=base_kwargs["distance_range"],
    )
    try:
        for _ in range(N_WARMUP):
            next(offline_task)
        t0 = time.perf_counter()
        for _ in range(N_BATCHES):
            next(offline_task)
        offline_elapsed = time.perf_counter() - t0
    finally:
        offline_task.close()

    offline_eps = N_BATCHES * base_kwargs["batch_size"] / offline_elapsed
    print(
        f"{'offline':>10s} {'-':>10s} {'-':>7s} {'-':>7s} "
        f"{offline_elapsed:>8.2f} {N_BATCHES / offline_elapsed:>10.1f} {offline_eps:>11.1f}"
    )
else:
    print(f"skip offline benchmark (missing dataset: {DS_PATH})")

# %%
