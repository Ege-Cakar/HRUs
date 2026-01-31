"""Tests for training utilities."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from model.mlp import Mixer, MixerConfig
from train import (
    Case,
    ce_mask,
    create_optimizer,
    loss_and_acc,
    mse_mask,
    parse_loss_name,
    train,
    train_step,
)


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

    def test_mse_mask(self):
        loss_func = parse_loss_name('mse_mask')
        assert loss_func == mse_mask

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


class TestMseMask:
    """Tests for mse_mask loss function."""

    def test_basic(self):
        logits = jnp.ones((2, 3, 4))
        labels = jnp.array([[1, 2, 0], [1, 3, 0]])
        
        loss = mse_mask(logits, labels)
        assert loss.shape == ()
        assert float(loss) >= 0


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_basic(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        
        optimizer = create_optimizer(model, lr=1e-3)
        
        assert isinstance(optimizer, nnx.Optimizer)
        assert optimizer.model is model

    def test_with_clip(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        
        optimizer = create_optimizer(model, lr=1e-3, clip=1.0)
        
        assert isinstance(optimizer, nnx.Optimizer)

    def test_custom_optimizer(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        
        optimizer = create_optimizer(model, lr=1e-3, optim=optax.sgd)
        
        assert isinstance(optimizer, nnx.Optimizer)


class TestTrainStep:
    """Tests for train_step function."""

    def test_basic(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_out=1)
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
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_out=1)
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
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=1, n_out=5)
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


class TestTrain:
    """Tests for the main train function."""

    def test_basic_training(self):
        config = MixerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_out=1)
        
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
        config = MixerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1, n_out=1)
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


class TestCase:
    """Tests for the Case dataclass."""

    def test_case_init(self):
        config = MixerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1)
        
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
        config = MixerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1)
        
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
        config = MixerConfig(n_vocab=100, n_hidden=32, n_seq=10, n_layers=1)
        
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

