"""Tests for the Transformer model."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model.transformer import (
    Transformer, 
    TransformerConfig, 
    MultiHeadAttention, 
    TransformerBlock,
    sinusoidal_pos_embedding
)


class TestSinusoidalPosEmbedding:
    """Tests for the sinusoidal positional embedding function."""

    def test_output_shape(self):
        pe = sinusoidal_pos_embedding(seq_len=32, dim=64)
        assert pe.shape == (32, 64)

    def test_different_dims(self):
        pe = sinusoidal_pos_embedding(seq_len=16, dim=128)
        assert pe.shape == (16, 128)

    def test_values_bounded(self):
        pe = sinusoidal_pos_embedding(seq_len=100, dim=64)
        # Sinusoidal values should be between -1 and 1
        assert jnp.all(pe >= -1.0)
        assert jnp.all(pe <= 1.0)

    def test_first_position_pattern(self):
        pe = sinusoidal_pos_embedding(seq_len=10, dim=4)
        # At position 0, sin(0) = 0 for all frequencies
        np.testing.assert_array_almost_equal(pe[0, 0::2], 0.0)


class TestTransformerConfig:
    """Tests for TransformerConfig dataclass."""

    def test_default_values(self):
        config = TransformerConfig()
        assert config.n_vocab is None
        assert config.n_seq == 128
        assert config.n_layers == 2
        assert config.n_hidden == 128
        assert config.n_heads == 4
        assert config.n_mlp_hidden is None
        assert config.n_out == 1
        assert config.pos_emb is True
        assert config.layer_norm is True
        assert config.use_bias is True
        assert config.dropout_rate == 0.0
        assert config.last_token_only is True
        assert config.use_mup is False

    def test_custom_values(self):
        config = TransformerConfig(
            n_vocab=512,
            n_seq=256,
            n_layers=6,
            n_hidden=256,
            n_heads=8,
            n_mlp_hidden=1024,
            n_out=10,
            pos_emb=False,
            layer_norm=False,
            use_bias=False,
            dropout_rate=0.1,
            last_token_only=False,
            use_mup=True
        )
        assert config.n_vocab == 512
        assert config.n_seq == 256
        assert config.n_layers == 6
        assert config.n_hidden == 256
        assert config.n_heads == 8
        assert config.n_mlp_hidden == 1024
        assert config.n_out == 10
        assert config.pos_emb is False
        assert config.layer_norm is False
        assert config.use_bias is False
        assert config.dropout_rate == 0.1
        assert config.last_token_only is False
        assert config.use_mup is True

    def test_to_model(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = config.to_model(rngs=rngs)
        assert isinstance(model, Transformer)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    def test_init(self):
        rngs = nnx.Rngs(42)
        attn = MultiHeadAttention(n_hidden=64, n_heads=4, rngs=rngs)
        assert attn.n_hidden == 64
        assert attn.n_heads == 4
        assert attn.head_dim == 16

    def test_invalid_head_dim(self):
        rngs = nnx.Rngs(42)
        with pytest.raises(AssertionError, match="must be divisible"):
            MultiHeadAttention(n_hidden=63, n_heads=4, rngs=rngs)

    def test_forward_shape(self):
        rngs = nnx.Rngs(42)
        attn = MultiHeadAttention(n_hidden=64, n_heads=4, rngs=rngs)
        x = jnp.ones((2, 10, 64))
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_forward_with_mask(self):
        rngs = nnx.Rngs(42)
        attn = MultiHeadAttention(n_hidden=64, n_heads=4, rngs=rngs)
        x = jnp.ones((2, 10, 64))
        mask = jnp.tril(jnp.ones((10, 10), dtype=bool))
        out = attn(x, mask=mask)
        assert out.shape == (2, 10, 64)

    def test_forward_with_dropout(self):
        rngs = nnx.Rngs(42)
        attn = MultiHeadAttention(n_hidden=64, n_heads=4, dropout_rate=0.1, rngs=rngs)
        x = jnp.ones((2, 10, 64))
        out = attn(x)
        assert out.shape == (2, 10, 64)


class TestTransformerBlock:
    """Tests for TransformerBlock module."""

    def test_init_with_layer_norm(self):
        rngs = nnx.Rngs(42)
        block = TransformerBlock(
            n_hidden=64, n_heads=4, n_mlp_hidden=256, 
            layer_norm=True, rngs=rngs
        )
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'ln2')

    def test_init_without_layer_norm(self):
        rngs = nnx.Rngs(42)
        block = TransformerBlock(
            n_hidden=64, n_heads=4, n_mlp_hidden=256, 
            layer_norm=False, rngs=rngs
        )
        assert block.layer_norm is False

    def test_forward_shape(self):
        rngs = nnx.Rngs(42)
        block = TransformerBlock(
            n_hidden=64, n_heads=4, n_mlp_hidden=256, rngs=rngs
        )
        x = jnp.ones((2, 10, 64))
        out = block(x)
        assert out.shape == (2, 10, 64)

    def test_forward_with_mask(self):
        rngs = nnx.Rngs(42)
        block = TransformerBlock(
            n_hidden=64, n_heads=4, n_mlp_hidden=256, rngs=rngs
        )
        x = jnp.ones((2, 10, 64))
        mask = jnp.tril(jnp.ones((10, 10), dtype=bool))
        out = block(x, mask=mask)
        assert out.shape == (2, 10, 64)


class TestTransformer:
    """Tests for the Transformer model."""

    def test_init_with_embedding(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        assert model.embed is not None
        assert len(model.blocks) == 2
        assert model.pos_embedding is not None

    def test_init_without_embedding(self):
        config = TransformerConfig(n_vocab=None, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        assert model.embed is None

    def test_init_without_pos_emb(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, pos_emb=False)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        assert model.pos_embedding is None

    def test_init_without_layer_norm(self):
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, layer_norm=False)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        assert model.final_ln is None

    def test_forward_with_embedding(self):
        """Test forward pass with embedding layer."""
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, n_out=5)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        # Input: batch of token indices (batch_size=4, seq_len=10)
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # With last_token_only=True and n_out=5, output should be (4, 5)
        assert output.shape == (4, 5)

    def test_forward_without_embedding(self):
        """Test forward pass with pre-embedded input."""
        config = TransformerConfig(n_vocab=None, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, n_out=3)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        # Input: pre-embedded (batch_size=4, seq_len=10, features=64)
        x = jnp.ones((4, 10, 64))
        output = model(x)
        
        assert output.shape == (4, 3)

    def test_forward_last_token_only_false(self):
        """Test forward pass with last_token_only=False (full sequence output)."""
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4,
            n_out=5, last_token_only=False
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # Output should be (4, 10, 5) - full sequence
        assert output.shape == (4, 10, 5)

    def test_forward_single_output(self):
        """Test forward pass with n_out=1 (squeezed output)."""
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, n_out=1)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # With n_out=1 and last_token_only=True, output should be (4,)
        assert output.shape == (4,)

    def test_forward_single_output_full_sequence(self):
        """Test forward pass with n_out=1 and last_token_only=False."""
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4,
            n_out=1, last_token_only=False
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # With n_out=1 and last_token_only=False, output should be (4, 10)
        assert output.shape == (4, 10)

    def test_forward_deterministic(self):
        """Test that same seed produces same results."""
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        
        rngs1 = nnx.Rngs(42)
        model1 = Transformer(config, rngs=rngs1)
        
        rngs2 = nnx.Rngs(42)
        model2 = Transformer(config, rngs=rngs2)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output1 = model1(x)
        output2 = model2(x)
        
        np.testing.assert_array_almost_equal(output1, output2)

    def test_causal_mask(self):
        """Test that causal mask is correctly precomputed."""
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        # Check the precomputed mask (first 5x5 slice)
        mask = model._causal_mask[:5, :5]
        expected = jnp.array([
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])
        np.testing.assert_array_equal(mask, expected)

    def test_invalid_input_shape_with_embedding(self):
        """Test that 3D input with embedding raises error."""
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        # 3D input with embedding should fail
        x = jnp.ones((4, 10, 64))
        with pytest.raises(AssertionError, match="Expected 2D input"):
            model(x)

    def test_invalid_input_shape_without_embedding(self):
        """Test that 2D input without embedding raises error."""
        config = TransformerConfig(n_vocab=None, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        # 2D input without embedding should fail
        x = jnp.ones((4, 10))
        with pytest.raises(AssertionError, match="Expected 3D input"):
            model(x)

    def test_mlp_hidden_default(self):
        """Test that n_mlp_hidden defaults to 4 * n_hidden."""
        config = TransformerConfig(n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        # Check that the MLP layers have 4 * n_hidden dimensions
        block = model.blocks[0]
        # The first MLP linear layer should output 4 * 64 = 256
        x = jnp.ones((1, 10, 64))
        after_fc1 = block.mlp_fc1(x)
        assert after_fc1.shape == (1, 10, 256)
