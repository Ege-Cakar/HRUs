"""Kernel retrieval baseline over selected action features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.normative.proof_features import ChoiceExample, feature_dim


@dataclass
class KernelConfig:
    gamma: float = 0.01
    eps: float = 1e-9


class KernelPolicy:
    """Nonparametric policy using RBF similarity to memorized actions."""

    def __init__(self, config: KernelConfig | None = None) -> None:
        self.config = config or KernelConfig()
        self.memory: np.ndarray = np.zeros((0, feature_dim()), dtype=np.float64)

    def fit(self, examples: list[ChoiceExample]) -> "KernelPolicy":
        if not examples:
            raise ValueError("Need at least one choice example to fit kernel policy.")
        self.memory = np.stack(
            [ex.action_features[ex.target_index] for ex in examples],
            axis=0,
        ).astype(np.float64)
        return self

    def _scores(self, action_features: np.ndarray) -> np.ndarray:
        cand = np.asarray(action_features, dtype=np.float64)
        if self.memory.shape[0] == 0:
            return np.ones((cand.shape[0],), dtype=np.float64)
        diffs = cand[:, None, :] - self.memory[None, :, :]
        sq_dist = np.sum(diffs * diffs, axis=-1)
        sims = np.exp(-self.config.gamma * sq_dist)
        return np.sum(sims, axis=1)

    def predict_proba(self, action_features: np.ndarray) -> np.ndarray:
        scores = self._scores(action_features)
        scores = np.maximum(scores, 0.0)
        z = float(np.sum(scores))
        if z <= self.config.eps:
            return np.full((scores.shape[0],), 1.0 / scores.shape[0], dtype=np.float64)
        return scores / z

    def predict_index(self, action_features: np.ndarray) -> int:
        return int(np.argmax(self._scores(action_features)))
