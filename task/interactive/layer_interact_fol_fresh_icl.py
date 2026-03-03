# <codecell>
"""Interactive visualization for the depth-3 fresh-ICL FOL split.

In this mode, every example gets completely fresh layer-0 predicates (r_abcd names)
and fresh layer-0->1 rules. Layer-1->2 transitions remain fixed. This forces the
model to rely entirely on in-context demonstrations, with no possibility of
memorizing layer-0->1 transitions.
"""

from __future__ import annotations

import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer_fol import FOLLayerTask


# <codecell>
# ---- CONFIG (edit in-place for local iteration) ----
TASK_CFG = {
    "seed": 42,
    "predicates_per_layer": 8,
    "fresh_icl_n_predicates": 8,
    "rules_per_transition": 32,
    "arity_max": 3,
    "vars_per_rule_max": 4,
    "k_in_max": 3,
    "k_out_max": 3,
    "constants": ("a", "b", "c", "d"),
    "initial_ant_max": 10,
    "max_n_demos": 10,
}
ENABLE_PLOT = True


# <codecell>
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


def _symbol_text(tokenizer, token_ids: list[int] | np.ndarray) -> str:
    return " ".join(tokenizer.id_to_char(int(tok)) for tok in list(token_ids))


def _collect_fresh_predicates(text: str) -> list[str]:
    """Extract all r_XXXX fresh predicate names from decoded text."""
    return re.findall(r"r_[a-z0-9]{4}", text)


def preview_record(task: FOLLayerTask, record: dict, *, role: str) -> None:
    tokenizer = task.tokenizer
    prompt = np.asarray(record["prompt"], dtype=np.int32)
    completion = np.asarray(record["completions"][0], dtype=np.int32)
    segments = _split_prompt_segments(prompt, tokenizer.sep_token_id)
    demo_segments = segments[:-1]
    main_segment = segments[-1] + [int(tokenizer.sep_token_id)]

    sequent = tokenizer.decode_prompt(main_segment)
    completion_text = tokenizer.decode_completion_text(completion.tolist())

    fresh_preds = set(_collect_fresh_predicates(sequent.text))
    fresh_preds.update(_collect_fresh_predicates(completion_text))

    print(
        f"\n[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
        f"prompt_len={prompt.size} completion_len={completion.size} n_demos={len(demo_segments)} "
        f"fresh_predicates={sorted(fresh_preds)}"
    )
    print("main sequent:", sequent.text)
    print("target completion:", completion_text)
    if demo_segments:
        for idx, demo in enumerate(demo_segments):
            demo_text = tokenizer.decode_completion_text(
                list(demo) + [int(tokenizer.eot_token_id)]
            )
            print(f"demo[{idx}]: {demo_text}")

    print("prompt tokens:")
    print(_symbol_text(tokenizer, prompt.tolist()))
    print("completion tokens:")
    print(_symbol_text(tokenizer, completion.tolist()))


# <codecell>
train_task = FOLLayerTask(
    mode="online",
    task_split="depth3_fresh_icl",
    split_role="train",
    distance_range=(2, 2),
    batch_size=1,
    seed=int(TASK_CFG["seed"]) + 1,
    predicates_per_layer=int(TASK_CFG["predicates_per_layer"]),
    fresh_icl_n_predicates=int(TASK_CFG["fresh_icl_n_predicates"]),
    rules_per_transition=int(TASK_CFG["rules_per_transition"]),
    arity_max=int(TASK_CFG["arity_max"]),
    vars_per_rule_max=int(TASK_CFG["vars_per_rule_max"]),
    k_in_max=int(TASK_CFG["k_in_max"]),
    k_out_max=int(TASK_CFG["k_out_max"]),
    constants=tuple(str(c) for c in TASK_CFG["constants"]),
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["max_n_demos"]),
    online_prefetch_backend="sync",
    min_n_demos=4,
)

eval_task = FOLLayerTask(
    mode="online",
    task_split="depth3_fresh_icl",
    split_role="eval",
    distance_range=(2, 2),
    batch_size=1,
    seed=int(TASK_CFG["seed"]) + 1000,
    predicates_per_layer=int(TASK_CFG["predicates_per_layer"]),
    fresh_icl_n_predicates=int(TASK_CFG["fresh_icl_n_predicates"]),
    rules_per_transition=int(TASK_CFG["rules_per_transition"]),
    arity_max=int(TASK_CFG["arity_max"]),
    vars_per_rule_max=int(TASK_CFG["vars_per_rule_max"]),
    k_in_max=int(TASK_CFG["k_in_max"]),
    k_out_max=int(TASK_CFG["k_out_max"]),
    constants=tuple(str(c) for c in TASK_CFG["constants"]),
    initial_ant_max=int(TASK_CFG["initial_ant_max"]),
    max_n_demos=int(TASK_CFG["max_n_demos"]),
    online_prefetch_backend="sync",
)

print(f"train task: prefetch={train_task.online_prefetch_backend_resolved}")
print(f"eval task:  prefetch={eval_task.online_prefetch_backend_resolved}")
print(f"base bank layers 1-2 rules: {len(train_task._base_bank.transition_rules(1))}")
print(f"tokenizer vocab size: {train_task.tokenizer.vocab_size}")


# <codecell>
# Preview train samples (step_idx sampled randomly -> can see both src_layer=0 and 1).
for _ in range(3):
    preview_record(train_task, train_task._sample_online_record(), role="train")


# <codecell>
# Preview eval samples (forced step_idx=0 -> always src_layer=0).
for _ in range(3):
    preview_record(eval_task, eval_task._sample_online_record(), role="eval")


# <codecell>
# Verify that fresh predicates change per example.
all_fresh: list[set[str]] = []
for _ in range(10):
    rec = eval_task._sample_online_record()
    prompt = np.asarray(rec["prompt"], dtype=np.int32)
    decoded = eval_task.tokenizer.decode_batch_ids(
        prompt.reshape(1, -1), include_special_tokens=False
    )[0]
    preds = set(_collect_fresh_predicates(decoded))
    all_fresh.append(preds)
    print(f"  example fresh preds: {sorted(preds)}")

n_distinct_sets = len(set(frozenset(s) for s in all_fresh))
print(f"\ndistinct fresh predicate sets across 10 examples: {n_distinct_sets}")
assert n_distinct_sets > 1, "Fresh predicates should differ across examples!"


# <codecell>
# Tokenization diagnostic for a fresh predicate.
rec = eval_task._sample_online_record()
prompt = np.asarray(rec["prompt"], dtype=np.int32)
decoded = eval_task.tokenizer.decode_batch_ids(
    prompt.reshape(1, -1), include_special_tokens=False
)[0]
sample_fresh = _collect_fresh_predicates(decoded)
if sample_fresh:
    fresh_pred = sample_fresh[0]
    print(f"sample fresh predicate: {fresh_pred}")
    print(
        "full predicate token in vocab:",
        fresh_pred in eval_task.tokenizer.token_to_id,
    )
    print("predicate chars + token ids:")
    for ch in fresh_pred:
        print(f"  {ch} -> {eval_task.tokenizer.char_to_id(ch)}")
else:
    print("(no fresh predicates in this sample's prompt)")


# <codecell>
# Batch shape check.
xs, ys = next(train_task)
print(f"train batch: xs={xs.shape} ys={ys.shape} dtype={xs.dtype}")

xs_eval, ys_eval = next(eval_task)
print(f"eval batch:  xs={xs_eval.shape} ys={ys_eval.shape} dtype={xs_eval.dtype}")


# <codecell>
# Optional length distribution plot.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if not ENABLE_PLOT:
    print("plot disabled (set ENABLE_PLOT=True to render histogram).")
elif plt is None:
    print("matplotlib unavailable; skipping plots.")
else:
    n_samples = 128
    train_lengths = []
    eval_lengths = []
    train_src_layers = []
    eval_src_layers = []
    for _ in range(n_samples):
        rec_train = train_task._sample_online_record()
        rec_eval = eval_task._sample_online_record()
        train_lengths.append(
            int(len(rec_train["prompt"])) + int(len(rec_train["completions"][0])) - 1
        )
        eval_lengths.append(
            int(len(rec_eval["prompt"])) + int(len(rec_eval["completions"][0])) - 1
        )
        train_src_layers.append(int(rec_train["src_layer"]))
        eval_src_layers.append(int(rec_eval["src_layer"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.arange(
        min(train_lengths + eval_lengths),
        max(train_lengths + eval_lengths) + 2,
    )
    axes[0].hist(train_lengths, bins=bins, alpha=0.6, label="train", edgecolor="black")
    axes[0].hist(eval_lengths, bins=bins, alpha=0.6, label="eval", edgecolor="black")
    axes[0].set_xlabel("autoregressive sequence length")
    axes[0].set_ylabel("count")
    axes[0].set_title("Length Distribution (fresh ICL)")
    axes[0].legend()

    layer_labels = ["src_layer=0", "src_layer=1"]
    train_layer_counts = [train_src_layers.count(0), train_src_layers.count(1)]
    eval_layer_counts = [eval_src_layers.count(0), eval_src_layers.count(1)]
    x_bar = np.arange(len(layer_labels))
    width = 0.35
    axes[1].bar(x_bar - width / 2, train_layer_counts, width, label="train", alpha=0.8)
    axes[1].bar(x_bar + width / 2, eval_layer_counts, width, label="eval", alpha=0.8)
    axes[1].set_xticks(x_bar)
    axes[1].set_xticklabels(layer_labels)
    axes[1].set_ylabel("count")
    axes[1].set_title("src_layer Distribution")
    axes[1].legend()

    fig.suptitle("Depth-3 Fresh-ICL Visualizations")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


# <codecell>
train_task.close()
eval_task.close()
