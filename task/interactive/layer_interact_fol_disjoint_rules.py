# <codecell>
"""Interactive visualization for the depth-3 disjoint-rule FOL split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer_fol import FOLLayerTask
from task.layer_gen.util.fol_rule_bank import (
    build_depth3_icl_split_bundle,
    load_fol_depth3_icl_split_bundle,
    save_fol_depth3_icl_split_bundle,
)


# <codecell>
SET_DIR = ROOT / "task" / "interactive" / "set"
SET_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_PATH = SET_DIR / "fol_depth3_disjoint_rules.json"

global_rules_per_layer = 512

SPLIT_CFG = {
    "seed": np.random.randint(0, 2**32 - 1),
    "predicates_per_layer": 128,
    "rules_01_train": global_rules_per_layer,
    "rules_01_eval": global_rules_per_layer,
    "rules_12_shared": global_rules_per_layer,
    "arity_max": 3,
    "vars_per_rule_max": 6,
    "k_in_max": 3,
    "k_out_max": 5,
    "constants": [f'p{i}' for i in range(128)],
}
REBUILD_SPLIT = True
ENABLE_PLOT = True


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


SPLIT_PATH = ensure_split_bundle(SPLIT_PATH, rebuild=REBUILD_SPLIT)
split_bundle = load_fol_depth3_icl_split_bundle(SPLIT_PATH)
print(f"split bundle: {SPLIT_PATH}")
print(f"train layer0 predicates: {len(split_bundle.train_layer0_predicates)}")
print(f"eval layer0 predicates: {len(split_bundle.eval_layer0_predicates)}")
print(f"shared layer1 predicates: {len(split_bundle.shared_layer1_predicates)}")
print(f"shared layer2 predicates: {len(split_bundle.shared_layer2_predicates)}")


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


def _atom_text(predicate: str, arity: int, constant: str) -> str:
    args = ",".join(str(constant) for _ in range(int(arity)))
    return f"{str(predicate)}({args})"


def _symbol_text(tokenizer, token_ids: list[int] | np.ndarray) -> str:
    return " ".join(tokenizer.id_to_char(int(tok)) for tok in list(token_ids))


def _predicate_order_key(predicate: str) -> tuple[int, int]:
    layer_text, idx_text = str(predicate).split("_")
    return int(layer_text[1:]), int(idx_text)


def _edge_counts(rules) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for rule in rules:
        for lhs in rule.lhs:
            for rhs in rule.rhs:
                key = (str(lhs.predicate), str(rhs.predicate))
                counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _plot_layer_connections(ax, bundle) -> None:
    train_l0 = tuple(sorted(bundle.train_layer0_predicates, key=_predicate_order_key))
    eval_l0 = tuple(sorted(bundle.eval_layer0_predicates, key=_predicate_order_key))
    shared_l1 = tuple(sorted(bundle.shared_layer1_predicates, key=_predicate_order_key))
    shared_l2 = tuple(sorted(bundle.shared_layer2_predicates, key=_predicate_order_key))

    y_train_l0 = np.linspace(0.55, 0.98, len(train_l0))
    y_eval_l0 = np.linspace(0.02, 0.45, len(eval_l0))
    y_l1 = np.linspace(0.02, 0.98, len(shared_l1))
    y_l2 = np.linspace(0.02, 0.98, len(shared_l2))

    positions: dict[str, tuple[float, float]] = {}
    positions.update({pred: (0.0, float(y)) for pred, y in zip(train_l0, y_train_l0)})
    positions.update({pred: (0.0, float(y)) for pred, y in zip(eval_l0, y_eval_l0)})
    positions.update({pred: (1.0, float(y)) for pred, y in zip(shared_l1, y_l1)})
    positions.update({pred: (2.0, float(y)) for pred, y in zip(shared_l2, y_l2)})

    train_counts = _edge_counts(bundle.train_bank.transition_rules(0))
    eval_counts = _edge_counts(bundle.eval_bank.transition_rules(0))
    shared_counts = _edge_counts(bundle.train_bank.transition_rules(1))

    def _draw_edges(counts: dict[tuple[str, str], int], *, color: str, alpha: float, label: str) -> None:
        if not counts:
            return
        max_count = max(int(v) for v in counts.values())
        for idx, ((src, dst), count) in enumerate(sorted(counts.items())):
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            lw = 0.6 + 2.8 * (float(count) / float(max_count))
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=color,
                alpha=alpha,
                linewidth=lw,
                label=label if idx == 0 else None,
            )

    _draw_edges(shared_counts, color="#7f7f7f", alpha=0.25, label="shared 1→2 rules")
    _draw_edges(train_counts, color="#1f77b4", alpha=0.35, label="train 0→1 rules")
    _draw_edges(eval_counts, color="#ff7f0e", alpha=0.35, label="eval 0→1 rules")

    ax.scatter([0.0] * len(train_l0), y_train_l0, s=45, color="#1f77b4", label="L0 train predicates")
    ax.scatter([0.0] * len(eval_l0), y_eval_l0, s=45, color="#ff7f0e", label="L0 eval predicates")
    ax.scatter([1.0] * len(shared_l1), y_l1, s=35, color="#2ca02c", label="L1 shared predicates")
    ax.scatter([2.0] * len(shared_l2), y_l2, s=35, color="#d62728", label="L2 shared predicates")

    for pred in train_l0:
        x, y = positions[pred]
        ax.text(x - 0.06, y, pred, ha="right", va="center", fontsize=7, color="#1f77b4")
    for pred in eval_l0:
        x, y = positions[pred]
        ax.text(x - 0.06, y, pred, ha="right", va="center", fontsize=7, color="#ff7f0e")
    for pred in shared_l1:
        x, y = positions[pred]
        ax.text(x, y + 0.012, pred, ha="center", va="bottom", fontsize=7, color="#2ca02c")
    for pred in shared_l2:
        x, y = positions[pred]
        ax.text(x + 0.06, y, pred, ha="left", va="center", fontsize=7, color="#d62728")

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels(["Layer 0", "Layer 1", "Layer 2"])
    ax.set_yticks([])
    ax.set_title("Predicate Connectivity")
    ax.legend(loc="lower center", fontsize=7, ncol=2, frameon=False)


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
        f"\n[{role}] distance={int(record['distance'])} src_layer={int(record['src_layer'])} "
        f"prompt_len={prompt.size} completion_len={completion.size} n_demos={len(demo_segments)}"
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
    task_split="depth3_icl_transfer",
    split_role="train",
    split_rule_bundle_path=SPLIT_PATH,
    distance_range=(2, 2),
    batch_size=1,
    seed=111,
    initial_ant_max=5,
    max_n_demos=8,
    online_prefetch_backend="sync",
)

eval_task = FOLLayerTask(
    mode="online",
    task_split="depth3_icl_transfer",
    split_role="eval",
    split_rule_bundle_path=SPLIT_PATH,
    distance_range=(2, 2),
    batch_size=1,
    seed=222,
    initial_ant_max=5,
    max_n_demos=8,
    online_prefetch_backend="sync",
)

for _ in range(2):
    preview_record(train_task, train_task._sample_online_record(), role="train")

for _ in range(2):
    preview_record(eval_task, eval_task._sample_online_record(), role="eval")

# <codecell>
xs, ys = next(train_task)

# <codecell>
type(xs)

# <codecell>
# Tokenization diagnostic for an eval-only predicate.
eval_only_predicate = split_bundle.eval_layer0_predicates[0]
shared_l1_predicate = split_bundle.shared_layer1_predicates[0]
eval_only_arity = int(split_bundle.eval_bank.predicate_arities[eval_only_predicate])
shared_l1_arity = int(split_bundle.eval_bank.predicate_arities[shared_l1_predicate])
const0 = str(split_bundle.eval_bank.constants[0])

print("eval-only predicate:", eval_only_predicate)
print(
    "full predicate token in vocab:",
    eval_only_predicate in train_task.tokenizer.token_to_id,
)
print("predicate chars + token ids:")
for ch in eval_only_predicate:
    print(ch, "->", train_task.tokenizer.char_to_id(ch))

statement_text = (
    f"{_atom_text(eval_only_predicate, eval_only_arity, const0)} "
    f"→ {_atom_text(shared_l1_predicate, shared_l1_arity, const0)}"
)
encoded = train_task.tokenizer.encode_completion(statement_text)
decoded = train_task.tokenizer.decode_completion_text(encoded)
print("statement:", statement_text)
print("encoded ids:", encoded)
print("encoded symbols:", _symbol_text(train_task.tokenizer, encoded))
print("decoded canonical:", decoded)


# <codecell>
# Optional shape visualization.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if not ENABLE_PLOT:
    print("plot disabled (set ENABLE_PLOT=True to render histogram + layer graph).")
elif plt is None:
    print("matplotlib unavailable; skipping plots.")
else:
    n_samples = 64
    train_lengths = []
    eval_lengths = []
    for _ in range(n_samples):
        rec_train = train_task._sample_online_record()
        rec_eval = eval_task._sample_online_record()
        train_lengths.append(
            int(len(rec_train["prompt"])) + int(len(rec_train["completions"][0])) - 1
        )
        eval_lengths.append(
            int(len(rec_eval["prompt"])) + int(len(rec_eval["completions"][0])) - 1
        )

    bins = np.arange(min(train_lengths + eval_lengths), max(train_lengths + eval_lengths) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), width_ratios=(1.0, 1.3))

    axes[0].hist(train_lengths, bins=bins, alpha=0.6, label="train split", edgecolor="black")
    axes[0].hist(eval_lengths, bins=bins, alpha=0.6, label="eval split", edgecolor="black")
    axes[0].set_xlabel("autoregressive sequence length")
    axes[0].set_ylabel("count")
    axes[0].set_title("Length Distribution")
    axes[0].legend()

    _plot_layer_connections(axes[1], split_bundle)
    fig.suptitle("Depth-3 Disjoint-Rule Split Visualizations")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


# <codecell>
train_task.close()
eval_task.close()
