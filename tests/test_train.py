"""Tests for training utilities."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from common import split_cases
from model.transformer import Transformer, TransformerConfig
from train import (
    Case,
    _train_with_accumulation,
    ce_mask,
    create_optimizer,
    linear_decay_schedule,
    loss_and_acc,
    parse_loss_name,
    train,
    train_step,
    warmup_constant_schedule,
    warmup_cosine_schedule,
)
from wandb_utils import WandbConfig
import wandb_utils


class TestParseLossName:
    """Tests for parse_loss_name function."""

    def test_bce(self):
        loss_func = parse_loss_name('bce')
        assert loss_func == optax.sigmoid_binary_cross_entropy

    def test_ce(self):
        loss_func = parse_loss_name('ce')
        assert loss_func == optax.softmax_cross_entropy_with_integer_labels

    def test_ce_mask(self):
        loss_func = parse_loss_name('ce_mask')
        assert loss_func == ce_mask

    def test_mse(self):
        loss_func = parse_loss_name('mse')
        assert loss_func == optax.squared_error

    def test_callable(self):
        """Test that callable loss functions are passed through."""
        custom_fn = lambda x, y: jnp.mean((x - y) ** 2)
        result = parse_loss_name(custom_fn)
        assert result is custom_fn

    def test_invalid(self):
        with pytest.raises(ValueError, match="unrecognized loss name"):
            parse_loss_name('invalid_loss')


class TestCeMask:
    """Tests for ce_mask loss function."""

    def test_basic(self):
        # logits: (batch=2, seq=3, vocab=4)
        logits = jnp.array([
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        ], dtype=jnp.float32)
        # labels: (batch=2, seq=3) - 0 is padding
        labels = jnp.array([[1, 2, 0], [1, 3, 0]])
        
        loss = ce_mask(logits, labels)
        # Should only compute loss for non-zero labels
        assert loss.shape == ()
        assert float(loss) > 0

    def test_all_padding(self):
        logits = jnp.ones((2, 3, 4))
        labels = jnp.zeros((2, 3), dtype=jnp.int32)
        
        # All padding - should be inf or nan (division by zero)
        loss = ce_mask(logits, labels)
        assert jnp.isnan(loss) or jnp.isinf(loss)


class TestScheduleHelpers:
    """Tests for LR schedule helper functions."""

    def test_warmup_cosine_returns_callable(self):
        schedule = warmup_cosine_schedule(1e-3, train_iters=1000)
        assert callable(schedule)

    def test_warmup_cosine_values(self):
        schedule = warmup_cosine_schedule(1e-3, train_iters=1000, warmup_frac=0.1, end_lr=0.0)
        # Start at 0
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-7)
        # Peak at end of warmup
        assert float(schedule(100)) == pytest.approx(1e-3, rel=1e-5)
        # End near 0
        assert float(schedule(1000)) == pytest.approx(0.0, abs=1e-5)

    def test_warmup_constant_returns_callable(self):
        schedule = warmup_constant_schedule(1e-3, train_iters=1000)
        assert callable(schedule)

    def test_warmup_constant_values(self):
        schedule = warmup_constant_schedule(1e-3, train_iters=1000, warmup_frac=0.1)
        # Start at 0
        assert float(schedule(0)) == pytest.approx(0.0, abs=1e-7)
        # Peak at end of warmup
        assert float(schedule(100)) == pytest.approx(1e-3, rel=1e-5)
        # Still at peak near end
        assert float(schedule(999)) == pytest.approx(1e-3, rel=1e-5)

    def test_linear_decay_returns_callable(self):
        schedule = linear_decay_schedule(1e-3, train_iters=1000)
        assert callable(schedule)

    def test_linear_decay_values(self):
        schedule = linear_decay_schedule(1e-3, train_iters=1000, end_lr=0.0)
        # Start at init_lr
        assert float(schedule(0)) == pytest.approx(1e-3, rel=1e-5)
        # Midpoint
        assert float(schedule(500)) == pytest.approx(5e-4, rel=1e-5)
        # End at 0
        assert float(schedule(1000)) == pytest.approx(0.0, abs=1e-7)


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_basic(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        
        optimizer = create_optimizer(model, lr=1e-3)
        
        assert isinstance(optimizer, nnx.Optimizer)
        assert optimizer.model is model

    def test_with_clip(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        
        optimizer = create_optimizer(model, lr=1e-3, clip=1.0)
        
        assert isinstance(optimizer, nnx.Optimizer)

    def test_custom_optimizer(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)

        optimizer = create_optimizer(model, lr=1e-3, optim=optax.sgd)

        assert isinstance(optimizer, nnx.Optimizer)

    def test_with_schedule(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)

        schedule = warmup_cosine_schedule(1e-3, train_iters=1000)
        optimizer = create_optimizer(model, lr=schedule)

        assert isinstance(optimizer, nnx.Optimizer)


class TestTrainStep:
    """Tests for train_step function."""

    def test_basic(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4, n_out=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        optimizer = create_optimizer(model, lr=1e-3)
        
        # Create batch
        x = jnp.ones((4, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0])
        batch = (x, y)
        
        loss_func = parse_loss_name('bce')
        
        # Get initial output
        initial_output = optimizer.model(x)
        
        # Run training step
        train_step(optimizer, batch, loss_func)
        
        # Output should change after training
        new_output = optimizer.model(x)
        assert not jnp.allclose(initial_output, new_output)


class TestLossAndAcc:
    """Tests for loss_and_acc function."""

    def test_binary_classification(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4, n_out=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        optimizer = create_optimizer(model, lr=1e-3)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0])
        batch = (x, y)
        
        result = loss_and_acc(optimizer, batch, loss='bce')
        
        assert 'loss' in result
        assert 'acc' in result
        assert 0 <= float(result['acc']) <= 1

    def test_multiclass_classification(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_heads=4, n_out=5)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        optimizer = create_optimizer(model, lr=1e-3)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        y = jnp.array([0, 1, 2, 3])
        batch = (x, y)
        
        result = loss_and_acc(optimizer, batch, loss='ce')
        
        assert 'loss' in result
        assert 'acc' in result



class SimpleIterator:
    """Simple iterator for testing the train function."""
    
    def __init__(self, x, y, batch_size=4):
        self.x = x
        self.y = y
        self.batch_size = batch_size
    
    def __next__(self):
        return self.x, self.y
    
    def __iter__(self):
        return self


class CountingIterator:
    """Iterator that returns a constant batch and counts fetches."""

    def __init__(self, batch):
        self.batch = batch
        self.count = 0
        self.batch_size = int(np.asarray(batch[0]).shape[0])

    def __next__(self):
        self.count += 1
        return self.batch

    def __iter__(self):
        return self


class FakeWandbModule:
    """Minimal wandb stub for metric logging tests."""

    def __init__(self):
        self.login_calls = []
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0

    def login(self, **kwargs):
        self.login_calls.append(kwargs)

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return object()

    def log(self, payload, step=None):
        self.log_calls.append({"payload": payload, "step": step})

    def finish(self):
        self.finish_calls += 1


class CloseTracker:
    """Closable stub that records how many times close() was invoked."""

    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class TestTrain:
    """Tests for the main train function."""

    def test_basic_training(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_iter = SimpleIterator(x, y)
        
        optimizer, hist = train(
            config,
            train_iter=train_iter,
            loss='bce',
            train_iters=10,
            test_every=5,
            seed=42
        )
        
        assert isinstance(optimizer, nnx.Optimizer)
        assert 'train' in hist
        assert 'test' in hist
        assert len(hist['train']) == 2  # Evaluated at step 5 and 10

    def test_with_existing_optimizer(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        initial_optimizer = create_optimizer(model, lr=1e-3)

        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_iter = SimpleIterator(x, y)

        optimizer, hist = train(
            initial_optimizer,  # Pass optimizer instead of config
            train_iter=train_iter,
            loss='bce',
            train_iters=5,
            test_every=5
        )

        assert optimizer is initial_optimizer

    def test_training_with_schedule(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)

        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_iter = SimpleIterator(x, y)

        schedule = warmup_cosine_schedule(1e-3, train_iters=10)
        optimizer, hist = train(
            config,
            train_iter=train_iter,
            loss='bce',
            lr=schedule,
            train_iters=10,
            test_every=5,
            seed=42
        )

        assert isinstance(optimizer, nnx.Optimizer)
        assert len(hist['train']) == 2

    def test_gradient_accumulation_matches_large_batch_update(self):
        config = TransformerConfig(n_vocab=64, n_hidden=16, n_seq=6, n_layers=1, n_heads=2, n_out=1)

        x1 = jnp.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
        y1 = jnp.array([0.0, 1.0], dtype=jnp.float32)
        x2 = jnp.array([[3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]], dtype=jnp.int32)
        y2 = jnp.array([1.0, 0.0], dtype=jnp.float32)

        full_batch = (jnp.concatenate([x1, x2], axis=0), jnp.concatenate([y1, y2], axis=0))

        eval_iter_single = SimpleIterator(*full_batch)
        eval_iter_accum = SimpleIterator(*full_batch)

        optimizer_single, _ = train(
            config,
            train_iter=SimpleIterator(*full_batch),
            test_iter=eval_iter_single,
            loss='bce',
            train_iters=1,
            test_iters=1,
            test_every=1,
            grad_accum_steps=1,
            optim=optax.sgd,
            lr=1e-2,
            seed=7,
        )

        class _AccumIterator:
            def __init__(self, batches):
                self.batches = list(batches)
                self.idx = 0
                self.batch_size = int(np.asarray(self.batches[0][0]).shape[0])

            def __next__(self):
                batch = self.batches[self.idx % len(self.batches)]
                self.idx += 1
                return batch

            def __iter__(self):
                return self

        optimizer_accum, _ = train(
            config,
            train_iter=_AccumIterator([(x1, y1), (x2, y2)]),
            test_iter=eval_iter_accum,
            loss='bce',
            train_iters=1,
            test_iters=1,
            test_every=1,
            grad_accum_steps=2,
            optim=optax.sgd,
            lr=1e-2,
            seed=7,
        )

        state_single = nnx.state(optimizer_single.model)
        state_accum = nnx.state(optimizer_accum.model)
        leaves_single = jax.tree_util.tree_leaves(state_single)
        leaves_accum = jax.tree_util.tree_leaves(state_accum)

        assert len(leaves_single) == len(leaves_accum)
        for lhs, rhs in zip(leaves_single, leaves_accum):
            np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)

    def test_gradient_accumulation_counts_optimizer_steps(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        batch = (
            jnp.ones((8, 10), dtype=jnp.int32),
            jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
        )
        train_iter = CountingIterator(batch)
        test_iter = CountingIterator(batch)
        print_steps = []

        optimizer, hist = train(
            config,
            train_iter=train_iter,
            test_iter=test_iter,
            loss='bce',
            train_iters=4,
            test_iters=1,
            test_every=2,
            grad_accum_steps=3,
            seed=42,
            print_fn=lambda step, hist: print_steps.append(step),
        )

        assert isinstance(optimizer, nnx.Optimizer)
        assert len(hist['train']) == 2
        assert len(hist['test']) == 2
        assert print_steps == [2, 4]
        assert train_iter.count == 14
        assert test_iter.count == 2

    def test_gradient_accumulation_returns_device_scalar_loss(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        optimizer = create_optimizer(model, lr=1e-3)
        batch = (
            jnp.ones((8, 10), dtype=jnp.int32),
            jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
        )

        loss_val = _train_with_accumulation(
            optimizer,
            CountingIterator(batch),
            parse_loss_name('bce'),
            grad_accum_steps=2,
        )

        assert isinstance(loss_val, jax.Array)
        assert loss_val.shape == ()

    def test_gradient_accumulation_rejects_invalid_steps(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        with pytest.raises(ValueError, match='grad_accum_steps'):
            train(
                config,
                train_iter=SimpleIterator(x, y),
                loss='bce',
                train_iters=1,
                grad_accum_steps=0,
                seed=42,
            )

    def test_wandb_disabled_does_not_import_module(self, monkeypatch, tmp_path):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        monkeypatch.setattr(
            wandb_utils.importlib,
            "import_module",
            lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected import: {name}")),
        )

        optimizer, hist = train(
            config,
            train_iter=SimpleIterator(x, y),
            loss='bce',
            train_iters=2,
            test_iters=1,
            test_every=1,
            seed=42,
            wandb_cfg=WandbConfig(
                enabled=False,
                project="disabled",
                api_key_path=tmp_path / "wandb.txt",
            ),
        )

        assert isinstance(optimizer, nnx.Optimizer)
        assert len(hist['train']) == 2

    def test_wandb_logs_train_test_and_summary_metrics(self, monkeypatch, tmp_path):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2, n_out=1)
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        key_path = tmp_path / "wandb.txt"
        key_path.write_text("test-key\n", encoding="utf-8")
        fake_wandb = FakeWandbModule()

        monkeypatch.setattr(
            wandb_utils.importlib,
            "import_module",
            lambda name: fake_wandb,
        )

        _, hist = train(
            config,
            train_iter=SimpleIterator(x, y),
            test_iter=SimpleIterator(x, y),
            loss='bce',
            train_iters=4,
            test_iters=1,
            test_every=2,
            seed=42,
            print_fn=lambda step, hist: None,
            summary_fn=lambda optimizer: {"param_count": 1},
            wandb_cfg=WandbConfig(
                enabled=True,
                project="unit-test",
                name="train-test",
                config={"alpha": 1},
                api_key_path=key_path,
            ),
        )

        assert len(hist['summary']) == 2
        assert fake_wandb.login_calls == [{"key": "test-key", "relogin": True}]
        assert fake_wandb.init_calls[0]["project"] == "unit-test"
        assert fake_wandb.init_calls[0]["name"] == "train-test"
        assert fake_wandb.log_calls[0]["step"] == 2
        assert fake_wandb.log_calls[1]["step"] == 4
        assert "train/loss" in fake_wandb.log_calls[0]["payload"]
        assert "test/loss" in fake_wandb.log_calls[0]["payload"]
        assert fake_wandb.log_calls[0]["payload"]["summary/param_count"] == 1
        assert fake_wandb.finish_calls == 1


class TestSplitCases:
    """Tests for shared case splitting and dropped-case cleanup."""

    @staticmethod
    def _make_case(name):
        train_task = CloseTracker()
        test_task = CloseTracker()
        return Case(
            name=name,
            config=None,
            train_task=train_task,
            test_task=test_task,
            info={
                "train_tracker": train_task,
                "test_tracker": test_task,
            },
        )

    def test_split_cases_closes_dropped_case_tasks(self, monkeypatch):
        cases = [self._make_case(f"case_{idx}") for idx in range(5)]

        monkeypatch.setattr(sys, "argv", ["pytest", "1"])
        selected = split_cases(cases, run_split=3)

        assert [case.name for case in selected] == ["case_2", "case_3"]

        selected_names = {case.name for case in selected}
        for case in cases:
            if case.name in selected_names:
                assert case.train_task is not None
                assert case.test_task is not None
                assert case.train_task.close_calls == 0
                assert case.test_task.close_calls == 0
            else:
                assert case.info["train_tracker"].close_calls == 1
                assert case.info["test_tracker"].close_calls == 1
                assert case.train_task is None
                assert case.test_task is None

    def test_split_cases_cleanup_respects_shuffle_seed(self, monkeypatch):
        cases = [self._make_case(f"case_{idx}") for idx in range(6)]

        shuffled = np.array(cases, dtype=object)
        rng = np.random.default_rng(7)
        rng.shuffle(shuffled)
        expected_selected = [case.name for case in np.array_split(shuffled, 2)[0]]
        expected_dropped = {case.name for case in np.array_split(shuffled, 2)[1]}

        monkeypatch.setattr(sys, "argv", ["pytest", "0"])
        selected = split_cases(cases, run_split=2, shuffle_seed=7)

        assert [case.name for case in selected] == expected_selected
        for case in cases:
            if case.name in expected_dropped:
                assert case.info["train_tracker"].close_calls == 1
                assert case.info["test_tracker"].close_calls == 1
                assert case.train_task is None
                assert case.test_task is None
            else:
                assert case.info["train_tracker"].close_calls == 0
                assert case.info["test_tracker"].close_calls == 0
                assert case.train_task is not None
                assert case.test_task is not None


class TestCase:
    """Tests for the Case dataclass."""

    def test_case_init(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2)
        
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_iter = SimpleIterator(x, y)
        
        case = Case(
            name='test_case',
            config=config,
            train_task=train_iter,
            train_args={'loss': 'bce', 'train_iters': 5, 'test_every': 5}
        )
        
        assert case.name == 'test_case'
        assert case.config == config
        assert case.optimizer is None

    def test_case_run(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2)
        
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_iter = SimpleIterator(x, y)
        
        case = Case(
            name='test_case',
            config=config,
            train_task=train_iter,
            train_args={'loss': 'bce', 'train_iters': 5, 'test_every': 5, 'seed': 42}
        )
        
        case.run()
        
        assert case.optimizer is not None
        assert case.hist is not None

    def test_case_eval(self):
        config = TransformerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_heads=2)
        
        x = jnp.ones((8, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        train_iter = SimpleIterator(x, y)
        test_iter = SimpleIterator(x, y)
        
        case = Case(
            name='test_case',
            config=config,
            train_task=train_iter,
            train_args={'loss': 'bce', 'train_iters': 5, 'test_every': 5, 'seed': 42}
        )
        
        case.run()
        case.eval(test_iter, [loss_and_acc], prefix='test')
        
        assert 'test' in case.info
