"""Tests for the MLP-Mixer model."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model.mlp import Mlp, MlpConfig, Mixer, MixerConfig, parse_act_fn


class TestParseActFn:
    """Tests for the parse_act_fn utility."""

    def test_relu(self):
        fn = parse_act_fn('relu')
        x = jnp.array([-1.0, 0.0, 1.0])
        result = fn(x)
        expected = jnp.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_linear(self):
        fn = parse_act_fn('linear')
        x = jnp.array([-1.0, 0.0, 1.0])
        result = fn(x)
        np.testing.assert_array_almost_equal(result, x)

    def test_gelu(self):
        fn = parse_act_fn('gelu')
        x = jnp.array([0.0, 1.0])
        result = fn(x)
        # GELU(0) = 0, GELU(1) ≈ 0.841
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(0.841, abs=0.01)

    def test_quadratic(self):
        fn = parse_act_fn('quadratic')
        x = jnp.array([-2.0, 0.0, 3.0])
        result = fn(x)
        expected = jnp.array([4.0, 0.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_fn(self):
        with pytest.raises(ValueError, match="function not recognized"):
            parse_act_fn('invalid')


class TestMixerConfig:
    """Tests for MixerConfig dataclass."""

    def test_default_values(self):
        config = MixerConfig()
        assert config.n_vocab is None
        assert config.n_seq == 16
        assert config.n_layers == 2
        assert config.n_hidden == 128
        assert config.n_channels == 16
        assert config.n_out == 1
        assert config.act_fn == 'relu'
        assert config.last_token_only is True
        assert config.layer_norm is False
        assert config.use_bias is True
        assert config.use_mup is False

    def test_custom_values(self):
        config = MixerConfig(
            n_vocab=512,
            n_seq=32,
            n_layers=4,
            n_hidden=256,
            n_channels=32,
            n_out=10,
            act_fn='gelu',
            last_token_only=False,
            layer_norm=True,
            use_bias=False,
            use_mup=True
        )
        assert config.n_vocab == 512
        assert config.n_seq == 32
        assert config.n_layers == 4
        assert config.n_hidden == 256
        assert config.n_channels == 32
        assert config.n_out == 10
        assert config.act_fn == 'gelu'
        assert config.last_token_only is False
        assert config.layer_norm is True
        assert config.use_bias is False
        assert config.use_mup is True

    def test_to_model(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=2)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        assert isinstance(model, Mixer)


class TestMlpConfig:
    """Tests for MlpConfig dataclass."""

    def test_default_values(self):
        config = MlpConfig()
        assert config.n_vocab is None
        assert config.n_layers == 2
        assert config.n_hidden == 128
        assert config.n_out == 1
        assert config.act_fn == 'relu'
        assert config.use_bias is True
        assert config.use_mup is False

    def test_custom_values(self):
        config = MlpConfig(
            n_vocab=256,
            n_layers=3,
            n_hidden=64,
            n_out=4,
            act_fn='gelu',
            use_bias=False,
            use_mup=True
        )
        assert config.n_vocab == 256
        assert config.n_layers == 3
        assert config.n_hidden == 64
        assert config.n_out == 4
        assert config.act_fn == 'gelu'
        assert config.use_bias is False
        assert config.use_mup is True

    def test_to_model(self):
        config = MlpConfig(n_vocab=100, n_hidden=64, n_layers=2)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        assert isinstance(model, Mlp)


class TestMlp:
    """Tests for the Mlp model."""

    def test_forward_with_embedding(self):
        config = MlpConfig(n_vocab=100, n_hidden=32, n_layers=2, n_out=5)
        rngs = nnx.Rngs(42)
        model = Mlp(config, rngs=rngs)

        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)

        assert output.shape == (4, 5)

    def test_forward_without_embedding(self):
        config = MlpConfig(n_vocab=None, n_hidden=32, n_layers=2, n_out=3)
        rngs = nnx.Rngs(42)
        model = Mlp(config, rngs=rngs)

        x = jnp.ones((4, 32))
        output = model(x)

        assert output.shape == (4, 3)

    def test_forward_single_output(self):
        config = MlpConfig(n_vocab=100, n_hidden=32, n_layers=2, n_out=1)
        rngs = nnx.Rngs(42)
        model = Mlp(config, rngs=rngs)

        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)

        assert output.shape == (4,)

    def test_forward_deterministic(self):
        config = MlpConfig(n_vocab=100, n_hidden=32, n_layers=2, n_out=2)

        rngs1 = nnx.Rngs(42)
        model1 = Mlp(config, rngs=rngs1)

        rngs2 = nnx.Rngs(42)
        model2 = Mlp(config, rngs=rngs2)

        x = jnp.ones((4, 10), dtype=jnp.int32)
        output1 = model1(x)
        output2 = model2(x)

        np.testing.assert_array_almost_equal(output1, output2)

    def test_invalid_input_shape(self):
        config = MlpConfig(n_vocab=None, n_hidden=32, n_layers=2, n_out=1)
        rngs = nnx.Rngs(42)
        model = Mlp(config, rngs=rngs)

        x = jnp.ones((4, 10, 32))
        with pytest.raises(AssertionError, match="Expected 2D input"):
            model(x)


class TestMixer:
    """Tests for the Mixer model."""

    def test_init_with_embedding(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=2, n_channels=8)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        assert model.embed is not None
        assert len(model.token_mixing_layers) == 2
        assert len(model.channel_mixing_layers) == 2

    def test_init_without_embedding(self):
        config = MixerConfig(n_vocab=None, n_hidden=64, n_seq=10, n_layers=2, n_channels=8)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        assert model.embed is None

    def test_init_with_layer_norm(self):
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=3, layer_norm=True)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        assert len(model.layer_norms) == 3

    def test_forward_with_embedding(self):
        """Test forward pass with embedding layer."""
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=2, n_channels=8, n_out=5)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        # Input: batch of token indices (batch_size=4, seq_len=10)
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # With last_token_only=True and n_out=5, output should be (4, 5)
        assert output.shape == (4, 5)

    def test_forward_without_embedding(self):
        """Test forward pass with pre-embedded input."""
        config = MixerConfig(n_vocab=None, n_hidden=64, n_seq=10, n_layers=2, n_channels=8, n_out=3)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        # Input: pre-embedded (batch_size=4, seq_len=10, features=64)
        x = jnp.ones((4, 10, 64))
        output = model(x)
        
        assert output.shape == (4, 3)

    def test_forward_last_token_only_false(self):
        """Test forward pass with last_token_only=False (flatten all)."""
        config = MixerConfig(
            n_vocab=100, n_hidden=64, n_seq=10, n_layers=2, n_channels=8, 
            n_out=5, last_token_only=False
        )
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # Output should be (4, 5)
        assert output.shape == (4, 5)

    def test_forward_single_output(self):
        """Test forward pass with n_out=1 (flattened output)."""
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=2, n_channels=8, n_out=1)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # With n_out=1, output should be flattened to (4,)
        assert output.shape == (4,)

    def test_forward_deterministic(self):
        """Test that same seed produces same results."""
        config = MixerConfig(n_vocab=100, n_hidden=64, n_seq=10, n_layers=2)
        
        rngs1 = nnx.Rngs(42)
        model1 = Mixer(config, rngs=rngs1)
        
        rngs2 = nnx.Rngs(42)
        model2 = Mixer(config, rngs=rngs2)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output1 = model1(x)
        output2 = model2(x)
        
        np.testing.assert_array_almost_equal(output1, output2)

    def test_invalid_input_shape(self):
        """Test that 2D input without embedding raises error."""
        config = MixerConfig(n_vocab=None, n_hidden=64, n_seq=10, n_layers=2)
        rngs = nnx.Rngs(42)
        model = Mixer(config, rngs=rngs)
        
        # 2D input without embedding should fail
        x = jnp.ones((4, 64))
        with pytest.raises(AssertionError, match="Expected 3D input"):
            model(x)
