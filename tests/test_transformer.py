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
    create_empty_kv_cache,
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
        assert config.n_pred_tokens == 1
        assert config.pos_encoding == "none"
        assert config.layer_norm is True
        assert config.use_swiglu is False
        assert config.use_bias is True
        assert config.dropout_rate == 0.0
        assert config.output_mode == "last_token"
        assert config.pad_token_id == 0
        assert config.use_mup is False
        assert config.use_bf16 is True

    def test_custom_values(self):
        config = TransformerConfig(
            n_vocab=512,
            n_seq=256,
            n_layers=6,
            n_hidden=256,
            n_heads=8,
            n_mlp_hidden=1024,
            n_out=10,
            n_pred_tokens=2,
            pos_encoding="none",
            layer_norm=False,
            use_swiglu=True,
            use_bias=False,
            dropout_rate=0.1,
            output_mode="full_sequence",
            pad_token_id=3,
            use_mup=True,
            use_bf16=False,
        )
        assert config.n_vocab == 512
        assert config.n_seq == 256
        assert config.n_layers == 6
        assert config.n_hidden == 256
        assert config.n_heads == 8
        assert config.n_mlp_hidden == 1024
        assert config.n_out == 10
        assert config.n_pred_tokens == 2
        assert config.pos_encoding == "none"
        assert config.layer_norm is False
        assert config.use_swiglu is True
        assert config.use_bias is False
        assert config.dropout_rate == 0.1
        assert config.output_mode == "full_sequence"
        assert config.pad_token_id == 3
        assert config.use_mup is True
        assert config.use_bf16 is False

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
        assert model.pos_embedding is None

    def test_init_without_embedding(self):
        config = TransformerConfig(n_vocab=None, n_hidden=64, n_seq=16, n_layers=2, n_heads=4)
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        assert model.embed is None

    def test_init_without_pos_emb(self):
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, pos_encoding="none"
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        assert model.pos_embedding is None

    def test_init_with_rope(self):
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, pos_encoding="rope"
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)

        assert model.pos_embedding is None
        assert model.rotary_cos is not None
        assert model.rotary_sin is not None

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
        
        # With output_mode='last_token' and n_out=5, output should be (4, 5)
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
        assert output.dtype == jnp.bfloat16

    def test_forward_without_embedding_fp32_opt_out(self):
        config = TransformerConfig(
            n_vocab=None,
            n_hidden=64,
            n_seq=16,
            n_layers=2,
            n_heads=4,
            n_out=3,
            use_bf16=False,
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)

        x = jnp.ones((4, 10, 64), dtype=jnp.float32)
        output = model(x)

        assert output.shape == (4, 3)
        assert output.dtype == jnp.float32

    def test_forward_last_token_only_false(self):
        """Test forward pass with output_mode='full_sequence' (full sequence output)."""
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4,
            n_out=5, output_mode="full_sequence"
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
        
        # With n_out=1 and output_mode='last_token', output should be (4,)
        assert output.shape == (4,)

    def test_forward_single_output_full_sequence(self):
        """Test forward pass with n_out=1 and output_mode='full_sequence'."""
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4,
            n_out=1, output_mode="full_sequence"
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)
        
        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)
        
        # With n_out=1 and output_mode='full_sequence', output should be (4, 10)
        assert output.shape == (4, 10)

    def test_forward_multi_token_output(self):
        """Test forward pass with multiple predicted tokens."""
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4,
            n_out=7, n_pred_tokens=2
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)

        x = jnp.ones((4, 10), dtype=jnp.int32)
        output = model(x)

        assert output.shape == (4, 2, 7)

    def test_forward_last_nonpad(self):
        """Test forward pass with output_mode='last_nonpad'."""
        config_full = TransformerConfig(
            n_vocab=100, n_hidden=32, n_seq=8, n_layers=2, n_heads=4,
            n_out=3, output_mode="full_sequence"
        )
        config_last = TransformerConfig(
            n_vocab=100, n_hidden=32, n_seq=8, n_layers=2, n_heads=4,
            n_out=3, output_mode="last_nonpad"
        )
        rngs1 = nnx.Rngs(42)
        model_full = Transformer(config_full, rngs=rngs1)
        rngs2 = nnx.Rngs(42)
        model_last = Transformer(config_last, rngs=rngs2)

        x = jnp.array([
            [5, 6, 7, 0, 0, 0, 0, 0],
            [9, 3, 0, 0, 0, 0, 0, 0],
            [4, 2, 1, 8, 0, 0, 0, 0],
        ], dtype=jnp.int32)

        output_full = model_full(x)
        output_last = model_last(x)

        lengths = jnp.sum(x != 0, axis=1)
        last_index = jnp.maximum(lengths - 1, 0)
        batch_idx = jnp.arange(x.shape[0])
        expected = output_full[batch_idx, last_index, :]

        np.testing.assert_array_almost_equal(output_last, expected)

    def test_last_nonpad_requires_tokens(self):
        config = TransformerConfig(
            n_vocab=None, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, output_mode="last_nonpad"
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)

        x = jnp.ones((4, 10, 64))
        with pytest.raises(ValueError, match="requires token indices"):
            model(x)

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

    def test_mlp_hidden_default_swiglu(self):
        """Test that n_mlp_hidden defaults to 8/3 * n_hidden for SwiGLU."""
        config = TransformerConfig(
            n_vocab=100, n_hidden=64, n_seq=16, n_layers=2, n_heads=4, use_swiglu=True
        )
        rngs = nnx.Rngs(42)
        model = Transformer(config, rngs=rngs)

        block = model.blocks[0]
        x = jnp.ones((1, 10, 64))
        expected_hidden = int(4 * config.n_hidden * 2 / 3)
        after_gate = block.mlp_gate(x)
        assert after_gate.shape == (1, 10, expected_hidden)

    def test_create_empty_kv_cache_shapes(self):
        config = TransformerConfig(n_hidden=32, n_heads=4, n_layers=3, n_seq=16)
        cache = create_empty_kv_cache(config, batch_size=2)
        assert len(cache.keys) == 3
        assert len(cache.values) == 3
        assert cache.keys[0].shape == (2, 4, 16, 8)
        assert cache.values[0].shape == (2, 4, 16, 8)
        assert int(cache.length) == 0

    def test_mixed_precision_forward_accepts_fp32_cache(self):
        config = TransformerConfig(
            n_vocab=64,
            n_hidden=32,
            n_seq=8,
            n_layers=2,
            n_heads=4,
            n_out=7,
            output_mode="full_sequence",
        )
        model = Transformer(config, rngs=nnx.Rngs(0))
        x = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        cache = create_empty_kv_cache(config, batch_size=1, dtype=jnp.float32)

        y, new_cache = model(x, cache=cache, return_cache=True)
        assert y.shape == (1, 4, 7)
        assert y.dtype == jnp.bfloat16
        assert new_cache.keys[0].dtype == jnp.float32
        assert new_cache.values[0].dtype == jnp.float32

    @pytest.mark.parametrize("pos_encoding", ["absolute", "rope"])
    def test_cached_incremental_matches_full_sequence(self, pos_encoding):
        config = TransformerConfig(
            n_vocab=64,
            n_hidden=32,
            n_seq=16,
            n_layers=2,
            n_heads=4,
            n_out=7,
            output_mode="full_sequence",
            pos_encoding=pos_encoding,
            use_bf16=False,
        )
        model = Transformer(config, rngs=nnx.Rngs(42))
        x = jnp.array(
            [
                [1, 2, 3, 4, 5, 6, 0, 0],
                [7, 8, 9, 1, 2, 0, 0, 0],
            ],
            dtype=jnp.int32,
        )

        y_full = model(x)

        cache = create_empty_kv_cache(config, batch_size=x.shape[0], dtype=jnp.float32)
        y_parts = []
        for t in range(x.shape[1]):
            y_t, cache = model(x[:, t : t + 1], cache=cache, return_cache=True)
            y_parts.append(y_t)
            assert int(cache.length) == (t + 1)
        y_incr = jnp.concatenate(y_parts, axis=1)

        np.testing.assert_allclose(y_full, y_incr, rtol=2e-3, atol=2e-3)

    def test_cache_overflow_raises(self):
        config = TransformerConfig(
            n_vocab=32,
            n_hidden=32,
            n_seq=4,
            n_layers=2,
            n_heads=4,
            output_mode="full_sequence",
        )
        model = Transformer(config, rngs=nnx.Rngs(0))
        cache = create_empty_kv_cache(config, batch_size=1, dtype=jnp.float32)
        x = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        _, cache = model(x, cache=cache, return_cache=True)
        with pytest.raises(ValueError, match="exceeds n_seq"):
            model(jnp.array([[5]], dtype=jnp.int32), cache=cache, return_cache=True)

    def test_last_nonpad_with_cache_raises(self):
        config = TransformerConfig(
            n_vocab=32,
            n_hidden=32,
            n_seq=8,
            n_layers=2,
            n_heads=4,
            n_out=3,
            output_mode="last_nonpad",
        )
        model = Transformer(config, rngs=nnx.Rngs(0))
        cache = create_empty_kv_cache(config, batch_size=1, dtype=jnp.float32)
        with pytest.raises(ValueError, match="does not support cache-based inference"):
            model(jnp.array([[1, 2]], dtype=jnp.int32), cache=cache)
