"""Feature extraction utilities for normative rule-selection baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from task.prop_gen.util.tokenize import logic_char_to_id


TURNSTILE_ID = logic_char_to_id["\u22a2"]
IMPLIES_ID = logic_char_to_id["\u2192"]
OPEN_ID = logic_char_to_id["("]
CLOSE_ID = logic_char_to_id[")"]
FALSE_ID = logic_char_to_id["\u22a5"]
COMMA_ID = logic_char_to_id[","]
MAX_LOGIC_ID = max(logic_char_to_id.values())
RULE_TYPE_COUNT = 11
RULES_WITH_POSITION = {3, 5, 8}
LEFT_RULE_IDS = {3, 5, 8, 10}

SEQUENT_FEATURE_NAMES = [
    "seq_len",
    "n_ants",
    "n_atoms",
    "n_unique_atoms",
    "n_implies",
    "max_depth",
    "depth_after_turnstile",
    "turnstile_pos_norm",
    "goal_atom_in_ants",
    "has_false_antecedent",
    "consequent_contains_implies",
]
ACTION_SCALAR_FEATURE_NAMES = [
    "rule_has_position",
    "rule_is_left_rule",
    "position",
    "position_norm",
    "position_valid",
    "n_candidates",
    "same_rule_count",
]


@dataclass(frozen=True)
class ChoiceExample:
    """Single rule-selection decision with candidate action set."""

    sequent_tokens: np.ndarray
    action_features: np.ndarray
    target_index: int
    target_action: np.ndarray
    candidate_actions: np.ndarray


def trim_nonpad(tokens: np.ndarray) -> np.ndarray:
    """Drop trailing padding tokens (zeros)."""
    arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
    nz = np.nonzero(arr)[0]
    if nz.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return arr[: nz[-1] + 1]


def _turnstile_idx(tokens: np.ndarray) -> int:
    matches = np.where(tokens == TURNSTILE_ID)[0]
    if matches.size == 0:
        return len(tokens)
    return int(matches[0])


def _token_depth_features(tokens: np.ndarray) -> tuple[float, float]:
    depth = 0
    max_depth = 0
    after_turnstile_depth = 0
    turn_idx = _turnstile_idx(tokens)
    for idx, tok in enumerate(tokens):
        if tok == OPEN_ID:
            depth += 1
            max_depth = max(max_depth, depth)
        elif tok == CLOSE_ID and depth > 0:
            depth -= 1
        if idx >= turn_idx:
            after_turnstile_depth = max(after_turnstile_depth, depth)
    return float(max_depth), float(after_turnstile_depth)


def sequent_feature_dict(sequent_tokens: np.ndarray) -> dict[str, float]:
    """Compute deterministic summary features from a tokenized sequent."""
    seq = trim_nonpad(sequent_tokens)
    if seq.size == 0:
        return {name: 0.0 for name in SEQUENT_FEATURE_NAMES}

    turn_idx = _turnstile_idx(seq)
    ants = seq[:turn_idx]
    cons = seq[turn_idx + 1 :] if turn_idx < seq.size else np.zeros((0,), dtype=np.int32)

    atom_mask = seq > MAX_LOGIC_ID
    ant_atom_mask = ants > MAX_LOGIC_ID
    cons_atom_mask = cons > MAX_LOGIC_ID
    ant_atoms = ants[ant_atom_mask]
    cons_atoms = cons[cons_atom_mask]
    n_ants = 0
    if ants.size > 0:
        n_ants = int(np.sum(ants == COMMA_ID)) + 1
    max_depth, after_turnstile_depth = _token_depth_features(seq)
    turnstile_pos = float(turn_idx) / float(max(seq.size - 1, 1))
    goal_atom_in_ants = 0.0
    if cons_atoms.size == 1 and ant_atoms.size > 0:
        goal_atom_in_ants = float(cons_atoms[0] in set(ant_atoms.tolist()))

    return {
        "seq_len": float(seq.size),
        "n_ants": float(n_ants),
        "n_atoms": float(np.sum(atom_mask)),
        "n_unique_atoms": float(np.unique(seq[atom_mask]).size),
        "n_implies": float(np.sum(seq == IMPLIES_ID)),
        "max_depth": max_depth,
        "depth_after_turnstile": after_turnstile_depth,
        "turnstile_pos_norm": turnstile_pos,
        "goal_atom_in_ants": goal_atom_in_ants,
        "has_false_antecedent": float(np.any(ants == FALSE_ID)),
        "consequent_contains_implies": float(np.any(cons == IMPLIES_ID)),
    }


def sequent_feature_vector(sequent_tokens: np.ndarray) -> np.ndarray:
    """Vectorized sequent summary features in fixed order."""
    stats = sequent_feature_dict(sequent_tokens)
    return np.array([stats[name] for name in SEQUENT_FEATURE_NAMES], dtype=np.float32)


def _action_scalar_features(
    sequent_stats: dict[str, float],
    action: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    rule_id = int(action[0])
    pos = int(action[1])
    n_ants = max(int(sequent_stats["n_ants"]), 1)
    requires_pos = rule_id in RULES_WITH_POSITION
    pos_valid = (pos > 0 and pos <= n_ants) if requires_pos else (pos == 0)
    return np.array(
        [
            float(requires_pos),
            float(rule_id in LEFT_RULE_IDS),
            float(pos),
            float(pos / n_ants),
            float(pos_valid),
            float(candidates.shape[0]),
            float(np.sum(candidates[:, 0] == rule_id)),
        ],
        dtype=np.float32,
    )


def action_feature_vector(
    sequent_tokens: np.ndarray,
    action: np.ndarray,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Featurize a candidate action for a tokenized sequent."""
    seq_stats = sequent_feature_dict(sequent_tokens)
    seq_vec = np.array([seq_stats[name] for name in SEQUENT_FEATURE_NAMES], dtype=np.float32)
    action = np.asarray(action, dtype=np.int32).reshape(2)
    if candidates is None:
        candidates = action[None, :]
    candidates = np.asarray(candidates, dtype=np.int32).reshape(-1, 2)
    one_hot = np.zeros((RULE_TYPE_COUNT,), dtype=np.float32)
    rule_id = int(action[0])
    if 1 <= rule_id <= RULE_TYPE_COUNT:
        one_hot[rule_id - 1] = 1.0
    scalars = _action_scalar_features(seq_stats, action, candidates)
    return np.concatenate([seq_vec, one_hot, scalars], axis=0)


def feature_dim() -> int:
    return len(SEQUENT_FEATURE_NAMES) + RULE_TYPE_COUNT + len(ACTION_SCALAR_FEATURE_NAMES)


def extract_candidate_rule_set(
    rule_set_batch: np.ndarray,
    rule_set_mask: np.ndarray,
    idx: int,
) -> np.ndarray:
    """Extract variable-length candidate rule set for one batch element."""
    rules = np.asarray(rule_set_batch[idx], dtype=np.int32)
    mask = np.asarray(rule_set_mask[idx]).astype(bool)
    return rules[mask]


def featurize_rule_set(sequent_tokens: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Featurize all candidates for one example."""
    cand = np.asarray(candidates, dtype=np.int32).reshape(-1, 2)
    if cand.shape[0] == 0:
        raise ValueError("Candidate set is empty.")
    return np.stack(
        [action_feature_vector(sequent_tokens, action, cand) for action in cand],
        axis=0,
    )


def build_choice_example(
    sequent_tokens: np.ndarray,
    target_action: np.ndarray,
    candidates: np.ndarray,
) -> ChoiceExample:
    """Build a training/evaluation example from one sampled action decision."""
    target = np.asarray(target_action, dtype=np.int32).reshape(2)
    cand = np.asarray(candidates, dtype=np.int32).reshape(-1, 2)
    if cand.shape[0] == 0:
        raise ValueError("Candidate set is empty.")
    matches = np.where(np.all(cand == target[None, :], axis=1))[0]
    if matches.size == 0:
        raise ValueError(f"Target action {target.tolist()} not found in candidate set.")
    target_idx = int(matches[0])
    return ChoiceExample(
        sequent_tokens=trim_nonpad(sequent_tokens),
        action_features=featurize_rule_set(sequent_tokens, cand),
        target_index=target_idx,
        target_action=target.copy(),
        candidate_actions=cand.copy(),
    )


def build_choice_dataset(
    seq_batch: np.ndarray,
    target_actions: np.ndarray,
    rule_set_batch: np.ndarray,
    rule_set_mask: np.ndarray,
    *,
    skip_missing: bool = True,
) -> list[ChoiceExample]:
    """Build normative examples from one task batch."""
    xs = np.asarray(seq_batch, dtype=np.int32)
    ys = np.asarray(target_actions, dtype=np.int32)
    all_rules = np.asarray(rule_set_batch, dtype=np.int32)
    all_mask = np.asarray(rule_set_mask).astype(bool)

    examples: list[ChoiceExample] = []
    for idx in range(xs.shape[0]):
        candidates = extract_candidate_rule_set(all_rules, all_mask, idx)
        if candidates.shape[0] == 0:
            continue
        try:
            ex = build_choice_example(xs[idx], ys[idx], candidates)
        except ValueError:
            if not skip_missing:
                raise
            continue
        examples.append(ex)
    return examples


def collate_choice_examples(examples: Iterable[ChoiceExample]) -> tuple[np.ndarray, np.ndarray]:
    """Collate selected action features and labels for debugging utilities."""
    exs = list(examples)
    if not exs:
        return np.zeros((0, feature_dim()), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    selected = np.stack(
        [ex.action_features[ex.target_index] for ex in exs],
        axis=0,
    )
    labels = np.array([ex.target_action[0] for ex in exs], dtype=np.int32)
    return selected, labels
