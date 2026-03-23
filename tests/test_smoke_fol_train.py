"""Smoke test: train standard and hybrid architectures on FOL online data.

Verifies end-to-end that:
1. FOLTaskFactory produces batches compatible with Architecture models
2. Loss decreases over a few training steps
3. Both core_block="attn" and core_block="hybrid" work
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from model.architecture import ArchConfig
from task.fol_task_factory import FOLTaskFactory, RuleBankConfig
from train import create_optimizer, train_step, parse_loss_name


def _make_small_factory():
    """Create a minimal FOL factory for testing."""
    return FOLTaskFactory(
        rule_bank_seed=42,
        d_train_max=3,
        d_eval_max=6,
        bank_config=RuleBankConfig(
            n_layers=4,
            predicates_per_layer=4,
            arity_min=1,
            arity_max=2,
            constants=("a", "b", "c", "d"),
            rules_per_transition=4,
            k_in_min=1,
            k_in_max=2,
            k_out_min=1,
            k_out_max=1,
            vars_per_rule_max=3,
        ),
    )


def _make_config(factory, core_block="attn", k=1):
    n_vocab = factory.n_vocab
    n_seq = factory.dims_internalized.n_seq_ar
    return ArchConfig(
        n_vocab=n_vocab,
        n_seq=n_seq,
        n_hidden=32,
        n_heads=4,
        n_out=n_vocab,
        core_block=core_block,
        n_core_repeats=k,
        output_mode="full_sequence",
        pos_encoding="rope",
        use_bf16=False,
    )


@pytest.mark.slow
class TestSmokeFOLTrain:
    """Smoke tests for training architectures on FOL."""

    def test_standard_transformer_loss_decreases(self):
        factory = _make_small_factory()
        config = _make_config(factory, "attn", k=1)
        task = factory.make_internalized_task(batch_size=8, distance_range=(1, 2))

        model = config.to_model(rngs=nnx.Rngs(42))
        optimizer = create_optimizer(model, lr=1e-3)
        loss_fn = parse_loss_name('ce_mask')

        losses = []
        for _ in range(20):
            xs, ys = next(task)
            loss = train_step(optimizer, (xs, ys), loss_fn)
            losses.append(float(loss))

        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"

    def test_hybrid_loss_decreases(self):
        factory = _make_small_factory()
        config = _make_config(factory, "hybrid", k=1)
        task = factory.make_internalized_task(batch_size=8, distance_range=(1, 2))

        model = config.to_model(rngs=nnx.Rngs(42))
        optimizer = create_optimizer(model, lr=1e-3)
        loss_fn = parse_loss_name('ce_mask')

        losses = []
        for _ in range(20):
            xs, ys = next(task)
            loss = train_step(optimizer, (xs, ys), loss_fn)
            losses.append(float(loss))

        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, f"Loss did not decrease: early={early:.4f}, late={late:.4f}"

    def test_batch_shapes_match(self):
        """Verify FOL task output shapes are compatible with Architecture input."""
        factory = _make_small_factory()
        n_vocab = factory.n_vocab
        task = factory.make_internalized_task(batch_size=4, distance_range=(1, 2))

        xs, ys = next(task)

        assert xs.dtype == jnp.int32 or str(xs.dtype).startswith('int')
        assert xs.ndim == 2  # (batch, seq)
        assert ys.ndim == 2  # (batch, seq) for full_sequence CE
        assert int(xs.max()) < n_vocab
        assert int(ys.max()) < n_vocab

    def test_standard_k2_trains(self):
        """Standard with k=2 (10 blocks) should also train."""
        factory = _make_small_factory()
        config = _make_config(factory, "attn", k=2)
        task = factory.make_internalized_task(batch_size=8, distance_range=(1, 2))

        model = config.to_model(rngs=nnx.Rngs(42))
        optimizer = create_optimizer(model, lr=1e-3)
        loss_fn = parse_loss_name('ce_mask')

        for _ in range(5):
            xs, ys = next(task)
            loss = train_step(optimizer, (xs, ys), loss_fn)
            assert jnp.isfinite(loss)
