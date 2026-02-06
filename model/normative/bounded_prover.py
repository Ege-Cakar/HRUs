"""Hand-designed bounded-rational heuristic policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.normative.proof_features import (
    ChoiceExample,
    LEFT_RULE_IDS,
    RULES_WITH_POSITION,
    sequent_feature_dict,
    trim_nonpad,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


@dataclass
class BoundedProverConfig:
    temperature: float = 1.0
    fit_temperature: bool = True
    temperature_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


class BoundedProverPolicy:
    """Heuristic action scorer based on lightweight proof-state signals."""

    _RULE_PRIOR = {
        1: 2.0,   # Axiom
        2: 1.0,   # ImpliesRight
        3: 0.7,   # ImpliesLeft
        4: 0.3,   # AndRight
        5: 0.3,   # AndLeft
        6: 0.1,   # OrRight1
        7: 0.1,   # OrRight2
        8: 0.2,   # OrLeft
        9: 1.0,   # TrueRight
        10: 1.3,  # FalseLeft
        11: 0.4,  # NegationRight
    }

    def __init__(self, config: BoundedProverConfig | None = None) -> None:
        self.config = config or BoundedProverConfig()
        self.temperature = self.config.temperature

    def _score_action(
        self,
        stats: dict[str, float],
        action: np.ndarray,
        candidates: np.ndarray,
    ) -> float:
        rule_id = int(action[0])
        pos = int(action[1])
        n_ants = max(int(stats["n_ants"]), 1)
        score = self._RULE_PRIOR.get(rule_id, -0.5)

        if rule_id == 1 and stats["goal_atom_in_ants"] > 0.0:
            score += 1.0
        if rule_id == 10 and stats["has_false_antecedent"] > 0.0:
            score += 0.8
        if rule_id == 2 and stats["consequent_contains_implies"] > 0.0:
            score += 0.7

        if rule_id in RULES_WITH_POSITION:
            valid = pos > 0 and pos <= n_ants
            score += 0.4 if valid else -0.7
            score -= 0.2 * (pos / n_ants)
        elif pos != 0:
            score -= 0.5

        if rule_id in LEFT_RULE_IDS:
            score += 0.1 * stats["n_ants"]
        score -= 0.05 * stats["max_depth"]
        score -= 0.03 * stats["n_candidates"] if "n_candidates" in stats else 0.0
        score -= 0.15 * float(np.sum(candidates[:, 0] == rule_id) > 1)
        return float(score)

    def _candidate_scores(self, sequent_tokens: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        seq = trim_nonpad(sequent_tokens)
        stats = sequent_feature_dict(seq)
        stats["n_candidates"] = float(candidates.shape[0])
        scores = [self._score_action(stats, action, candidates) for action in candidates]
        return np.asarray(scores, dtype=np.float64)

    def fit(self, examples: list[ChoiceExample]) -> "BoundedProverPolicy":
        if not examples or not self.config.fit_temperature:
            return self

        best_t = self.temperature
        best_nll = float("inf")
        for temp in self.config.temperature_grid:
            nll = 0.0
            for ex in examples:
                scores = self._candidate_scores(ex.sequent_tokens, ex.candidate_actions)
                probs = _softmax(scores / max(float(temp), 1e-6))
                nll += -np.log(max(float(probs[ex.target_index]), 1e-12))
            nll /= len(examples)
            if nll < best_nll:
                best_nll = nll
                best_t = float(temp)
        self.temperature = best_t
        return self

    def predict_proba(
        self,
        sequent_tokens: np.ndarray,
        candidate_actions: np.ndarray,
    ) -> np.ndarray:
        cand = np.asarray(candidate_actions, dtype=np.int32).reshape(-1, 2)
        if cand.shape[0] == 0:
            raise ValueError("Candidate set is empty.")
        scores = self._candidate_scores(sequent_tokens, cand)
        return _softmax(scores / max(self.temperature, 1e-6))

    def predict_index(
        self,
        sequent_tokens: np.ndarray,
        candidate_actions: np.ndarray,
    ) -> int:
        probs = self.predict_proba(sequent_tokens, candidate_actions)
        return int(np.argmax(probs))
