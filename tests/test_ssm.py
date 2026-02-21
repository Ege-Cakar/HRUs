"""Tests for Mamba-style state space models."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model.ssm import Mamba, Mamba2, Mamba2Config, MambaConfig
from train import ce_mask, create_optimizer, parse_loss_name, train_step


class TestMambaConfig:
    def test_default_values(self):
        config = MambaConfig()
        assert config.n_vocab is None
        assert config.n_seq == 128
        assert config.n_layers == 2
        assert config.n_hidden == 128
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
        assert config.dt_rank == "auto"
        assert config.d_conv == 4
        assert config.scan_backend == "reference"
        assert config.scan_chunk_len == 64
        assert config.scan_debug_checks is False

    def test_custom_values(self):
        config = MambaConfig(
            n_vocab=512,
            n_seq=256,
            n_layers=4,
            n_hidden=64,
            n_out=5,
            n_pred_tokens=2,
            output_mode="full_sequence",
            pad_token_id=3,
            layer_norm=False,
            use_bias=False,
            dropout_rate=0.1,
            use_mup=True,
            d_state=8,
            expand=3,
            dt_rank=4,
            d_conv=3,
            dt_min=1e-4,
            dt_max=0.2,
            scan_backend="auto",
            scan_chunk_len=32,
            scan_debug_checks=True,
        )
        assert config.n_vocab == 512
        assert config.n_seq == 256
        assert config.n_layers == 4
        assert config.n_hidden == 64
        assert config.n_out == 5
        assert config.n_pred_tokens == 2
        assert config.output_mode == "full_sequence"
        assert config.pad_token_id == 3
        assert config.layer_norm is False
        assert config.use_bias is False
        assert config.dropout_rate == 0.1
        assert config.use_mup is True
        assert config.d_state == 8
        assert config.expand == 3
        assert config.dt_rank == 4
        assert config.d_conv == 3
        assert config.dt_min == 1e-4
        assert config.dt_max == 0.2
        assert config.scan_backend == "auto"
        assert config.scan_chunk_len == 32
        assert config.scan_debug_checks is True

    def test_to_model(self):
        config = MambaConfig(n_vocab=100, n_hidden=32, n_layers=2)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert isinstance(model, Mamba)


class TestMamba2Config:
    def test_default_values(self):
        config = Mamba2Config()
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
        assert config.dt_rank == "auto"
        assert config.d_conv == 4
        assert config.scan_backend == "reference"
        assert config.scan_chunk_len == 64
        assert config.scan_debug_checks is False

    def test_custom_values(self):
        config = Mamba2Config(
            n_vocab=256,
            n_seq=64,
            n_layers=3,
            n_hidden=96,
            n_heads=6,
            n_out=7,
            n_pred_tokens=3,
            output_mode="full_sequence",
            pad_token_id=2,
            layer_norm=False,
            use_bias=False,
            dropout_rate=0.05,
            use_mup=True,
            d_state=12,
            expand=2,
            dt_rank=6,
            d_conv=5,
            dt_min=5e-4,
            dt_max=0.15,
            scan_backend="auto",
            scan_chunk_len=16,
            scan_debug_checks=True,
        )
        assert config.n_vocab == 256
        assert config.n_seq == 64
        assert config.n_layers == 3
        assert config.n_hidden == 96
        assert config.n_heads == 6
        assert config.n_out == 7
        assert config.n_pred_tokens == 3
        assert config.output_mode == "full_sequence"
        assert config.pad_token_id == 2
        assert config.layer_norm is False
        assert config.use_bias is False
        assert config.dropout_rate == 0.05
        assert config.use_mup is True
        assert config.d_state == 12
        assert config.expand == 2
        assert config.dt_rank == 6
        assert config.d_conv == 5
        assert config.dt_min == 5e-4
        assert config.dt_max == 0.15
        assert config.scan_backend == "auto"
        assert config.scan_chunk_len == 16
        assert config.scan_debug_checks is True

    def test_to_model(self):
        config = Mamba2Config(n_vocab=100, n_hidden=32, n_layers=2, n_heads=4)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert isinstance(model, Mamba2)


@pytest.mark.parametrize(
    "config_cls,model_cls,extra",
    [
        (MambaConfig, Mamba, {}),
        (Mamba2Config, Mamba2, {"n_heads": 4}),
    ],
)
class TestSSMForward:
    def test_forward_with_embedding(self, config_cls, model_cls, extra):
        config = config_cls(n_vocab=100, n_hidden=32, n_seq=16, n_layers=2, n_out=5, **extra)
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (4, 5)

    def test_forward_without_embedding(self, config_cls, model_cls, extra):
        config = config_cls(n_vocab=None, n_hidden=32, n_seq=16, n_layers=2, n_out=3, **extra)
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10, 32))
        out = model(x)
        assert out.shape == (4, 3)

    def test_forward_full_sequence(self, config_cls, model_cls, extra):
        config = config_cls(
            n_vocab=100,
            n_hidden=32,
            n_seq=16,
            n_layers=2,
            n_out=5,
            output_mode="full_sequence",
            **extra,
        )
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (4, 10, 5)

    def test_forward_single_output(self, config_cls, model_cls, extra):
        config = config_cls(n_vocab=100, n_hidden=32, n_seq=16, n_layers=2, n_out=1, **extra)
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (4,)

    def test_forward_multi_token_output(self, config_cls, model_cls, extra):
        config = config_cls(
            n_vocab=100,
            n_hidden=32,
            n_seq=16,
            n_layers=2,
            n_out=7,
            n_pred_tokens=2,
            **extra,
        )
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (4, 2, 7)

    def test_forward_multi_token_output_full_sequence(self, config_cls, model_cls, extra):
        config = config_cls(
            n_vocab=100,
            n_hidden=32,
            n_seq=16,
            n_layers=2,
            n_out=7,
            n_pred_tokens=2,
            output_mode="full_sequence",
            **extra,
        )
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (4, 10, 2, 7)

    def test_forward_last_nonpad(self, config_cls, model_cls, extra):
        config_full = config_cls(
            n_vocab=100,
            n_hidden=32,
            n_seq=12,
            n_layers=2,
            n_out=3,
            output_mode="full_sequence",
            **extra,
        )
        config_last = config_cls(
            n_vocab=100,
            n_hidden=32,
            n_seq=12,
            n_layers=2,
            n_out=3,
            output_mode="last_nonpad",
            **extra,
        )
        model_full = model_cls(config_full, rngs=nnx.Rngs(42))
        model_last = model_cls(config_last, rngs=nnx.Rngs(42))

        x = jnp.array(
            [
                [5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 2, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=jnp.int32,
        )

        output_full = model_full(x)
        output_last = model_last(x)

        lengths = jnp.sum(x != 0, axis=1)
        last_index = jnp.maximum(lengths - 1, 0)
        batch_idx = jnp.arange(x.shape[0])
        expected = output_full[batch_idx, last_index, :]
        np.testing.assert_array_almost_equal(output_last, expected)

    def test_last_nonpad_requires_tokens(self, config_cls, model_cls, extra):
        config = config_cls(
            n_vocab=None,
            n_hidden=32,
            n_seq=16,
            n_layers=2,
            output_mode="last_nonpad",
            **extra,
        )
        model = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10, 32))
        with pytest.raises(ValueError, match="requires token indices"):
            model(x)

    def test_forward_deterministic(self, config_cls, model_cls, extra):
        config = config_cls(n_vocab=100, n_hidden=32, n_seq=16, n_layers=2, **extra)
        model1 = model_cls(config, rngs=nnx.Rngs(42))
        model2 = model_cls(config, rngs=nnx.Rngs(42))
        x = jnp.ones((4, 10), dtype=jnp.int32)
        out1 = model1(x)
        out2 = model2(x)
        np.testing.assert_array_almost_equal(out1, out2)


class TestSSMErrors:
    def test_invalid_output_mode(self):
        config = MambaConfig(output_mode="invalid")
        with pytest.raises(ValueError, match="output_mode"):
            Mamba(config, rngs=nnx.Rngs(42))

    def test_invalid_n_pred_tokens(self):
        config = MambaConfig(n_pred_tokens=0)
        with pytest.raises(ValueError, match="n_pred_tokens"):
            Mamba(config, rngs=nnx.Rngs(42))

    def test_invalid_mamba2_head_dim(self):
        config = Mamba2Config(n_hidden=30, n_heads=8)
        with pytest.raises(ValueError, match="must be divisible"):
            Mamba2(config, rngs=nnx.Rngs(42))

    def test_invalid_scan_backend(self):
        config = MambaConfig(scan_backend="invalid")
        with pytest.raises(ValueError, match="scan_backend"):
            Mamba(config, rngs=nnx.Rngs(42))

    def test_invalid_scan_chunk_len(self):
        config = MambaConfig(scan_chunk_len=0)
        with pytest.raises(ValueError, match="scan_chunk_len"):
            Mamba(config, rngs=nnx.Rngs(42))


class TestSSMTrainIntegration:
    @pytest.mark.parametrize(
        "config",
        [
            MambaConfig(n_vocab=100, n_hidden=32, n_seq=16, n_layers=1, n_out=1),
            Mamba2Config(n_vocab=100, n_hidden=32, n_seq=16, n_layers=1, n_heads=4, n_out=1),
        ],
    )
    def test_train_step_updates_output(self, config):
        model = config.to_model(rngs=nnx.Rngs(42))
        optimizer = create_optimizer(model, lr=1e-3)

        x = jnp.ones((4, 10), dtype=jnp.int32)
        y = jnp.array([0.0, 1.0, 0.0, 1.0])
        batch = (x, y)

        loss_func = parse_loss_name("bce")
        initial_output = optimizer.model(x)
        train_step(optimizer, batch, loss_func)
        new_output = optimizer.model(x)

        assert not jnp.allclose(initial_output, new_output)

    @pytest.mark.parametrize(
        "config",
        [
            MambaConfig(
                n_vocab=64,
                n_hidden=32,
                n_seq=16,
                n_layers=1,
                n_out=64,
                output_mode="full_sequence",
            ),
            Mamba2Config(
                n_vocab=64,
                n_hidden=32,
                n_seq=16,
                n_layers=1,
                n_heads=4,
                n_out=64,
                output_mode="full_sequence",
            ),
        ],
    )
    def test_ce_mask_path(self, config):
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.array(
            [
                [1, 2, 3, 4, 0, 0],
                [4, 3, 2, 1, 5, 0],
            ],
            dtype=jnp.int32,
        )
        labels = jnp.array(
            [
                [2, 3, 4, 0, 0, 0],
                [3, 2, 1, 5, 0, 0],
            ],
            dtype=jnp.int32,
        )

        logits = model(x)
        loss = ce_mask(logits, labels)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
