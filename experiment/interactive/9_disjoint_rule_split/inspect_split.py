# <codecell>
"""Interactive inspection script for experiment 9 disjoint-rule split."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer_fol import FOLLayerTask
from task.layer_gen.util.fol_rule_bank import (
    build_depth3_icl_split_bundle,
    load_fol_depth3_icl_split_bundle,
    save_fol_depth3_icl_split_bundle,
)


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
