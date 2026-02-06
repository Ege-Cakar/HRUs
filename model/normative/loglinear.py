"""Log-linear policy over variable-size rule candidate sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.normative.proof_features import ChoiceExample, feature_dim


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.sum(exp)


@dataclass
class LogLinearConfig:
    lr: float = 0.05
    weight_decay: float = 1e-4
    max_steps: int = 500
    batch_size: int = 64
    seed: int = 0
    grad_clip: float = 5.0


class LogLinearPolicy:
    """Simple softmax policy trained with SGD on choice sets."""

    def __init__(self, config: LogLinearConfig | None = None) -> None:
        self.config = config or LogLinearConfig()
        self.weights: np.ndarray | None = None
        self.loss_history: list[float] = []

    def _ensure_weights(self) -> None:
        if self.weights is None:
            self.weights = np.zeros((feature_dim(),), dtype=np.float64)

    def fit(self, examples: list[ChoiceExample]) -> "LogLinearPolicy":
        if not examples:
            raise ValueError("Need at least one choice example to fit log-linear policy.")
        self._ensure_weights()
        rng = np.random.default_rng(self.config.seed)
        n = len(examples)
        batch_size = max(1, min(self.config.batch_size, n))
        w = self.weights
        assert w is not None

        for _ in range(self.config.max_steps):
            batch_idx = rng.choice(n, size=batch_size, replace=False)
            grad = np.zeros_like(w)
            batch_loss = 0.0
            for idx in batch_idx:
                ex = examples[int(idx)]
                logits = ex.action_features @ w
                probs = _softmax(logits)
                batch_loss += -np.log(max(probs[ex.target_index], 1e-12))
                probs = probs.copy()
                probs[ex.target_index] -= 1.0
                grad += ex.action_features.T @ probs
            grad /= batch_size
            batch_loss /= batch_size
            if self.config.weight_decay > 0:
                grad += self.config.weight_decay * w
                batch_loss += 0.5 * self.config.weight_decay * float(np.dot(w, w))
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > self.config.grad_clip:
                grad *= self.config.grad_clip / max(grad_norm, 1e-12)
            w -= self.config.lr * grad
            self.loss_history.append(batch_loss)

        self.weights = w
        return self

    def score(self, action_features: np.ndarray) -> np.ndarray:
        self._ensure_weights()
        assert self.weights is not None
        feats = np.asarray(action_features, dtype=np.float64)
        return feats @ self.weights

    def predict_proba(self, action_features: np.ndarray) -> np.ndarray:
        return _softmax(self.score(action_features))

    def predict_index(self, action_features: np.ndarray) -> int:
        return int(np.argmax(self.score(action_features)))
