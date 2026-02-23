"""Tests for the Bonsai-adapted Mamba2 implementation."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model.ssm_bonsai import Mamba2Bonsai, Mamba2BonsaiConfig, create_empty_cache
from train import ce_mask, create_optimizer, parse_loss_name, train_step


class TestMamba2BonsaiConfig:
    def test_default_values(self):
        config = Mamba2BonsaiConfig()
        assert config.n_vocab is None
        assert config.n_seq == 128
        assert config.n_layers == 2
        assert config.n_hidden == 128
        assert config.n_heads == 8
        assert config.n_out == 1
        assert config.n_pred_tokens == 1
        assert config.output_mode == "last_token"
        assert config.pad_token_id == 0
        assert config.layer_norm is True
        assert config.use_bias is True
        assert config.dropout_rate == 0.0
        assert config.use_mup is False
        assert config.d_state == 16
        assert config.expand == 2
        assert config.d_conv == 4
        assert config.dt_min == 1e-3
        assert config.dt_max == 0.1
        assert config.scan_chunk_len == 64
        assert config.dt_floor == 1e-4

    def test_to_model(self):
        config = Mamba2BonsaiConfig(n_vocab=100, n_hidden=32, n_layers=2, n_heads=4)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert isinstance(model, Mamba2Bonsai)


def test_forward_with_embedding_shape():
    config = Mamba2BonsaiConfig(
        n_vocab=100,
        n_hidden=32,
        n_seq=16,
        n_layers=2,
        n_heads=4,
        n_out=5,
    )
    model = Mamba2Bonsai(config, rngs=nnx.Rngs(0))
    x = jnp.ones((4, 10), dtype=jnp.int32)
    out = model(x)
    assert out.shape == (4, 5)


def test_forward_without_embedding_shape():
    config = Mamba2BonsaiConfig(
        n_vocab=None,
        n_hidden=32,
        n_seq=16,
        n_layers=2,
        n_heads=4,
        n_out=3,
    )
    model = Mamba2Bonsai(config, rngs=nnx.Rngs(0))
    x = jnp.ones((4, 10, 32))
    out = model(x)
    assert out.shape == (4, 3)


def test_cache_shapes():
    config = Mamba2BonsaiConfig(
        n_hidden=32,
        n_heads=4,
        n_layers=3,
        d_state=8,
        expand=2,
        d_conv=5,
    )
    cache = create_empty_cache(config, batch_size=2)
    assert len(cache.conv_states) == 3
    assert len(cache.ssm_states) == 3
    assert cache.conv_states[0].shape == (2, 80, 4)
    assert cache.ssm_states[0].shape == (2, 4, 16, 8)


def test_cached_incremental_matches_full_sequence():
    config = Mamba2BonsaiConfig(
        n_vocab=64,
        n_hidden=32,
        n_seq=16,
        n_layers=2,
        n_heads=4,
        n_out=7,
        output_mode="full_sequence",
        d_state=8,
        scan_chunk_len=8,
    )
    model = Mamba2Bonsai(config, rngs=nnx.Rngs(42))

    x = jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 0, 0, 0, 0],
            [7, 8, 9, 1, 2, 3, 4, 5, 0, 0],
        ],
        dtype=jnp.int32,
    )

    y_full = model(x)

    cache = create_empty_cache(config, batch_size=x.shape[0], dtype=jnp.float32)
    y_parts = []
    for t in range(x.shape[1]):
        y_t, cache = model(x[:, t : t + 1], cache=cache, return_cache=True)
        y_parts.append(y_t)
    y_incr = jnp.concatenate(y_parts, axis=1)

    np.testing.assert_allclose(y_full, y_incr, rtol=2e-3, atol=2e-3)


def test_last_nonpad_requires_tokens():
    config = Mamba2BonsaiConfig(
        n_vocab=None,
        n_hidden=32,
        n_heads=4,
        output_mode="last_nonpad",
    )
    model = Mamba2Bonsai(config, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="requires token indices"):
        model(jnp.ones((2, 8, 32)))


def test_train_step_updates_output():
    config = Mamba2BonsaiConfig(
        n_vocab=64,
        n_hidden=32,
        n_heads=4,
        n_layers=1,
        n_out=64,
        output_mode="full_sequence",
    )
    model = config.to_model(rngs=nnx.Rngs(42))
    optimizer = create_optimizer(model, lr=1e-3)

    x = jnp.ones((4, 10), dtype=jnp.int32)
    y = jnp.ones((4, 10), dtype=jnp.int32)
    batch = (x, y)
    loss_func = parse_loss_name("ce_mask")

    initial_output = optimizer.model(x)
    train_step(optimizer, batch, loss_func)
    new_output = optimizer.model(x)
    assert not jnp.allclose(initial_output, new_output)

    loss = ce_mask(new_output, y)
    assert jnp.isfinite(loss)
