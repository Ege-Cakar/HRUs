"""Hybrid ICL bank: internalized base rules + fresh in-context rules.

The HybridICLBank wraps a base rule bank (internalized during training)
with pools of fresh rules that are only provided in-context.  Fresh rules
use novel predicates on their LHS but standard predicates on their RHS,
enabling composability: fresh-rule outputs feed into internalized-rule
inputs at the next derivation step.

Three eval conditions of increasing difficulty:

1. **train**: Fresh rules from the training pool (sanity check).
2. **rule_gen**: Same fresh predicates as training, but new rule structures.
3. **pred_gen**: Entirely new predicates + new rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ._build import (
    _sample_transition_rules,
    _validate_bank_params,
    _validate_constants,
    generate_fresh_predicate_names,
)
from ._types import (
    FOLLayerRule,
    FOLRuleBank,
    _rule_sort_key,
)


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HybridICLBank:
    """A base bank + fresh rule pools for hybrid internalized/ICL tasks.

    Attributes
    ----------
    base_bank : FOLRuleBank
        The internalized rule bank (learned during training).
    fresh_predicate_name_len : int
        Character length of fresh predicate suffixes.
    train_predicates_by_layer : dict[int, tuple[str, ...]]
        Fresh predicates used during training, keyed by source layer.
    eval_pred_predicates_by_layer : dict[int, tuple[str, ...]]
        Held-out fresh predicates for predicate-level eval.
    train_rules : dict[int, tuple[FOLLayerRule, ...]]
        Fresh rules for training demos, keyed by source layer.
    eval_rule_rules : dict[int, tuple[FOLLayerRule, ...]]
        Fresh rules with same predicates as train but different structure.
    eval_pred_rules : dict[int, tuple[FOLLayerRule, ...]]
        Fresh rules using held-out predicates.
    p_fresh : float
        Probability that a transition uses a fresh rule (vs internalized).
    """

    base_bank: FOLRuleBank
    fresh_predicate_name_len: int
    train_predicates_by_layer: dict[int, tuple[str, ...]]
    eval_pred_predicates_by_layer: dict[int, tuple[str, ...]]
    train_rules: dict[int, tuple[FOLLayerRule, ...]]
    eval_rule_rules: dict[int, tuple[FOLLayerRule, ...]]
    eval_pred_rules: dict[int, tuple[FOLLayerRule, ...]]
    p_fresh: float

    @property
    def n_layers(self) -> int:
        return self.base_bank.n_layers

    @property
    def n_transitions(self) -> int:
        return self.base_bank.n_layers - 1

    def all_fresh_predicates(self) -> set[str]:
        """All fresh predicates across train and eval-pred pools."""
        preds: set[str] = set()
        for layer_preds in self.train_predicates_by_layer.values():
            preds.update(layer_preds)
        for layer_preds in self.eval_pred_predicates_by_layer.values():
            preds.update(layer_preds)
        return preds

    def all_fresh_rules(self) -> list[FOLLayerRule]:
        """All fresh rules across all pools (for tokenizer construction)."""
        rules: list[FOLLayerRule] = []
        for pool in (self.train_rules, self.eval_rule_rules, self.eval_pred_rules):
            for layer_rules in pool.values():
                rules.extend(layer_rules)
        return rules

    def fresh_rules_for_mode(
        self, eval_mode: str,
    ) -> dict[int, tuple[FOLLayerRule, ...]]:
        """Return the fresh rule pool for a given eval mode."""
        if eval_mode == "train":
            return self.train_rules
        if eval_mode == "rule_gen":
            return self.eval_rule_rules
        if eval_mode == "pred_gen":
            return self.eval_pred_rules
        raise ValueError(
            f"eval_mode must be 'train', 'rule_gen', or 'pred_gen', got {eval_mode!r}"
        )

    def fresh_predicates_for_mode(
        self, eval_mode: str,
    ) -> dict[int, tuple[str, ...]]:
        """Return the fresh predicate pool for a given eval mode."""
        if eval_mode in ("train", "rule_gen"):
            return self.train_predicates_by_layer
        if eval_mode == "pred_gen":
            return self.eval_pred_predicates_by_layer
        raise ValueError(
            f"eval_mode must be 'train', 'rule_gen', or 'pred_gen', got {eval_mode!r}"
        )

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        def _rules_to_payload(pool: dict[int, tuple[FOLLayerRule, ...]]) -> dict:
            return {
                str(layer): [rule.to_dict() for rule in rules]
                for layer, rules in sorted(pool.items())
            }

        def _preds_to_payload(pool: dict[int, tuple[str, ...]]) -> dict:
            return {
                str(layer): list(preds)
                for layer, preds in sorted(pool.items())
            }

        return {
            "version": "hybrid_icl_v1",
            "base_bank": self.base_bank.to_dict(),
            "fresh_predicate_name_len": int(self.fresh_predicate_name_len),
            "train_predicates_by_layer": _preds_to_payload(self.train_predicates_by_layer),
            "eval_pred_predicates_by_layer": _preds_to_payload(self.eval_pred_predicates_by_layer),
            "train_rules": _rules_to_payload(self.train_rules),
            "eval_rule_rules": _rules_to_payload(self.eval_rule_rules),
            "eval_pred_rules": _rules_to_payload(self.eval_pred_rules),
            "p_fresh": float(self.p_fresh),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "HybridICLBank":
        version = str(payload.get("version", ""))
        if version != "hybrid_icl_v1":
            raise ValueError(f"Unsupported HybridICLBank version: {version!r}")

        def _rules_from_payload(raw: dict) -> dict[int, tuple[FOLLayerRule, ...]]:
            return {
                int(layer): tuple(
                    sorted(
                        (FOLLayerRule.from_dict(r) for r in rules),
                        key=_rule_sort_key,
                    )
                )
                for layer, rules in raw.items()
            }

        def _preds_from_payload(raw: dict) -> dict[int, tuple[str, ...]]:
            return {
                int(layer): tuple(str(p) for p in preds)
                for layer, preds in raw.items()
            }

        return cls(
            base_bank=FOLRuleBank.from_dict(payload["base_bank"]),
            fresh_predicate_name_len=int(payload["fresh_predicate_name_len"]),
            train_predicates_by_layer=_preds_from_payload(payload["train_predicates_by_layer"]),
            eval_pred_predicates_by_layer=_preds_from_payload(payload["eval_pred_predicates_by_layer"]),
            train_rules=_rules_from_payload(payload["train_rules"]),
            eval_rule_rules=_rules_from_payload(payload["eval_rule_rules"]),
            eval_pred_rules=_rules_from_payload(payload["eval_pred_rules"]),
            p_fresh=float(payload["p_fresh"]),
        )


# ── Construction ─────────────────────────────────────────────────────────────

def build_hybrid_icl_bank(
    *,
    base_bank: FOLRuleBank,
    fresh_predicates_per_layer: int,
    fresh_rules_per_transition: int,
    pred_train_frac: float = 0.5,
    p_fresh: float = 0.5,
    predicate_name_len: int = 4,
    rng: np.random.Generator,
) -> HybridICLBank:
    """Build a HybridICLBank from an existing base bank.

    Parameters
    ----------
    base_bank : FOLRuleBank
        The internalized rule bank.
    fresh_predicates_per_layer : int
        Number of fresh predicates to generate per source layer.
    fresh_rules_per_transition : int
        Number of fresh rules per transition (for each pool).
    pred_train_frac : float
        Fraction of fresh predicates allocated to training.
        The rest are held out for predicate-level eval.
    p_fresh : float
        Probability that a transition uses a fresh rule during sampling.
    predicate_name_len : int
        Character length of fresh predicate suffixes (e.g. 4 → `r_abcd`).
    rng : np.random.Generator
        Random number generator.
    """
    if fresh_predicates_per_layer < 2:
        raise ValueError(
            f"fresh_predicates_per_layer must be >= 2, got {fresh_predicates_per_layer}"
        )
    if fresh_rules_per_transition < 1:
        raise ValueError(
            f"fresh_rules_per_transition must be >= 1, got {fresh_rules_per_transition}"
        )
    if not (0.0 < pred_train_frac < 1.0):
        raise ValueError(
            f"pred_train_frac must be in (0, 1), got {pred_train_frac}"
        )
    if not (0.0 < p_fresh <= 1.0):
        raise ValueError(f"p_fresh must be in (0, 1], got {p_fresh}")

    n_train = max(1, int(round(fresh_predicates_per_layer * pred_train_frac)))
    n_eval_pred = fresh_predicates_per_layer - n_train
    if n_eval_pred < 1:
        raise ValueError(
            "pred_train_frac too high: need at least 1 predicate for eval-pred pool. "
            f"Got {n_train} train, {n_eval_pred} eval from {fresh_predicates_per_layer} total."
        )

    var_pool = tuple(
        f"x{idx}" for idx in range(1, base_bank.vars_per_rule_max + 1)
    )

    train_predicates_by_layer: dict[int, tuple[str, ...]] = {}
    eval_pred_predicates_by_layer: dict[int, tuple[str, ...]] = {}
    train_rules: dict[int, tuple[FOLLayerRule, ...]] = {}
    eval_rule_rules: dict[int, tuple[FOLLayerRule, ...]] = {}
    eval_pred_rules: dict[int, tuple[FOLLayerRule, ...]] = {}

    # Collect all predicate arities (base + all fresh) for rule construction.
    all_predicate_arities: dict[str, int] = dict(base_bank.predicate_arities)

    for src_layer in range(base_bank.n_layers - 1):
        dst_layer = src_layer + 1
        rhs_predicates = tuple(base_bank.predicates_for_layer(dst_layer))
        if not rhs_predicates:
            continue

        # Generate all fresh predicates for this layer.
        all_fresh = generate_fresh_predicate_names(
            fresh_predicates_per_layer,
            rng,
            name_len=predicate_name_len,
        )

        # Assign arities to all fresh predicates.
        for pred in all_fresh:
            all_predicate_arities[pred] = int(
                rng.integers(base_bank.arity_min, base_bank.arity_max + 1)
            )

        # Split into train and eval-pred pools.
        shuffled = list(all_fresh)
        rng.shuffle(shuffled)
        train_preds = tuple(sorted(shuffled[:n_train]))
        eval_preds = tuple(sorted(shuffled[n_train:]))

        train_predicates_by_layer[src_layer] = train_preds
        eval_pred_predicates_by_layer[src_layer] = eval_preds

        # Build train rules: train predicates → standard RHS predicates.
        train_rules[src_layer] = _sample_transition_rules(
            src_layer=src_layer,
            lhs_predicates=train_preds,
            rhs_predicates=rhs_predicates,
            rules_per_transition=fresh_rules_per_transition,
            k_in_min=base_bank.arity_min if base_bank.arity_min >= 1 else 1,
            k_in_max=min(len(train_preds), base_bank.n_layers),  # reasonable cap
            k_out_min=1,
            k_out_max=min(len(rhs_predicates), 3),
            predicate_arities=all_predicate_arities,
            var_pool=var_pool,
            rng=rng,
        )

        # Build eval-rule rules: SAME train predicates, DIFFERENT rules.
        eval_rule_rules[src_layer] = _sample_transition_rules(
            src_layer=src_layer,
            lhs_predicates=train_preds,
            rhs_predicates=rhs_predicates,
            rules_per_transition=fresh_rules_per_transition,
            k_in_min=base_bank.arity_min if base_bank.arity_min >= 1 else 1,
            k_in_max=min(len(train_preds), base_bank.n_layers),
            k_out_min=1,
            k_out_max=min(len(rhs_predicates), 3),
            predicate_arities=all_predicate_arities,
            var_pool=var_pool,
            rng=rng,
        )

        # Build eval-pred rules: eval predicates → standard RHS predicates.
        eval_pred_rules[src_layer] = _sample_transition_rules(
            src_layer=src_layer,
            lhs_predicates=eval_preds,
            rhs_predicates=rhs_predicates,
            rules_per_transition=fresh_rules_per_transition,
            k_in_min=base_bank.arity_min if base_bank.arity_min >= 1 else 1,
            k_in_max=min(len(eval_preds), base_bank.n_layers),
            k_out_min=1,
            k_out_max=min(len(rhs_predicates), 3),
            predicate_arities=all_predicate_arities,
            var_pool=var_pool,
            rng=rng,
        )

    return HybridICLBank(
        base_bank=base_bank,
        fresh_predicate_name_len=predicate_name_len,
        train_predicates_by_layer=train_predicates_by_layer,
        eval_pred_predicates_by_layer=eval_pred_predicates_by_layer,
        train_rules=train_rules,
        eval_rule_rules=eval_rule_rules,
        eval_pred_rules=eval_pred_rules,
        p_fresh=float(p_fresh),
    )


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_hybrid_icl_bank(path: Path, bank: HybridICLBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank.to_dict(), indent=2))


def load_hybrid_icl_bank(path: Path) -> HybridICLBank:
    payload = json.loads(path.read_text())
    return HybridICLBank.from_dict(payload)
