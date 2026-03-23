"""Tests for the unified Architecture model and GDN block."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from model.architecture import ArchConfig, Architecture, _build_block_sequence
from model.gated_delta_net import GatedDeltaNetAttention, GDNBlock


# ── GDN Attention ─────────────────────────────────────────────────────

class TestGatedDeltaNetAttention:
    def test_output_shape(self):
        attn = GatedDeltaNetAttention(
            n_hidden=32, n_heads=4, rngs=nnx.Rngs(42),
        )
        x = jnp.ones((2, 8, 32))
        out = attn(x)
        assert out.shape == (2, 8, 32)

    def test_single_head(self):
        attn = GatedDeltaNetAttention(
            n_hidden=16, n_heads=1, rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 4, 16))
        out = attn(x)
        assert out.shape == (1, 4, 16)

    def test_causal_by_construction(self):
        """GDN is causal via scan — output at t depends only on inputs 0..t."""
        attn = GatedDeltaNetAttention(
            n_hidden=16, n_heads=2, rngs=nnx.Rngs(42),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 6, 16))
        full_out = attn(x)

        # Output at position 2 should be the same whether we pass 3 or 6 tokens
        x_short = x[:, :3, :]
        short_out = attn(x_short)
        # Positions 0-2 should match
        assert jnp.allclose(full_out[:, :3, :], short_out, atol=1e-5)

    def test_different_seq_lengths(self):
        attn = GatedDeltaNetAttention(
            n_hidden=32, n_heads=4, rngs=nnx.Rngs(42),
        )
        for seq_len in [1, 4, 16, 32]:
            x = jnp.ones((1, seq_len, 32))
            out = attn(x)
            assert out.shape == (1, seq_len, 32)


# ── GDN Block ─────────────────────────────────────────────────────────

class TestGDNBlock:
    def test_output_shape(self):
        block = GDNBlock(
            n_hidden=32, n_heads=4, n_mlp_hidden=64,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((2, 8, 32))
        out = block(x)
        assert out.shape == (2, 8, 32)

    def test_interface_compat_with_transformer_block(self):
        """GDNBlock accepts same kwargs as TransformerBlock."""
        block = GDNBlock(
            n_hidden=32, n_heads=4, n_mlp_hidden=64,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 4, 32))
        # These kwargs should be accepted (and ignored)
        out = block(x, mask=None, is_causal=True, pos_offset=0)
        assert out.shape == (1, 4, 32)

    def test_return_kv_compat(self):
        block = GDNBlock(
            n_hidden=32, n_heads=4, n_mlp_hidden=64,
            rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 4, 32))
        out, k, v = block(x, return_kv=True)
        assert out.shape == (1, 4, 32)

    def test_swiglu(self):
        block = GDNBlock(
            n_hidden=32, n_heads=4, n_mlp_hidden=64,
            use_swiglu=True, rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 4, 32))
        out = block(x)
        assert out.shape == (1, 4, 32)

    def test_no_layer_norm(self):
        block = GDNBlock(
            n_hidden=32, n_heads=4, n_mlp_hidden=64,
            layer_norm=False, rngs=nnx.Rngs(42),
        )
        x = jnp.ones((1, 4, 32))
        out = block(x)
        assert out.shape == (1, 4, 32)


# ── Block Sequence ────────────────────────────────────────────────────

class TestBlockSequence:
    def test_standard_k1(self):
        config = ArchConfig(core_block="attn", n_core_repeats=1)
        seq = _build_block_sequence(config, rngs=nnx.Rngs(0))
        assert seq == ["attn", "attn", "attn", "attn", "attn", "attn"]
        assert len(seq) == 6

    def test_hybrid_k1(self):
        config = ArchConfig(core_block="hybrid", n_core_repeats=1)
        seq = _build_block_sequence(config, rngs=nnx.Rngs(0))
        assert seq == ["attn", "gdn", "gdn", "gdn", "attn", "attn"]
        assert len(seq) == 6

    def test_standard_k2(self):
        config = ArchConfig(core_block="attn", n_core_repeats=2)
        seq = _build_block_sequence(config, rngs=nnx.Rngs(0))
        assert seq == ["attn"] + ["attn"] * 8 + ["attn"]
        assert len(seq) == 10

    def test_hybrid_k3(self):
        config = ArchConfig(core_block="hybrid", n_core_repeats=3)
        seq = _build_block_sequence(config, rngs=nnx.Rngs(0))
        expected = ["attn"] + ["gdn", "gdn", "gdn", "attn"] * 3 + ["attn"]
        assert seq == expected
        assert len(seq) == 14

    def test_n_layers_property(self):
        for k in [1, 2, 3, 5]:
            config = ArchConfig(n_core_repeats=k)
            assert config.n_layers == 4 * k + 2

    def test_invalid_core_block(self):
        config = ArchConfig(core_block="invalid")
        with pytest.raises(ValueError, match="core_block"):
            _build_block_sequence(config, rngs=nnx.Rngs(0))


# ── Architecture Model ────────────────────────────────────────────────

class TestArchitecture:
    def _make_config(self, core_block="attn", k=1, **kwargs):
        defaults = dict(
            n_vocab=50, n_seq=16, n_hidden=32, n_heads=4,
            n_out=50, n_pred_tokens=1,
            core_block=core_block, n_core_repeats=k,
            pos_encoding="rope", output_mode="full_sequence",
            use_bf16=False,
        )
        defaults.update(kwargs)
        return ArchConfig(**defaults)

    def test_standard_k1_forward(self):
        config = self._make_config("attn", k=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_hybrid_k1_forward(self):
        config = self._make_config("hybrid", k=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_standard_k2_forward(self):
        config = self._make_config("attn", k=2)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_hybrid_k2_forward(self):
        config = self._make_config("hybrid", k=2)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_block_count(self):
        config = self._make_config("hybrid", k=3)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert len(model.blocks) == 14  # 4*3 + 2

    def test_block_types_standard(self):
        config = self._make_config("attn", k=2)
        model = config.to_model(rngs=nnx.Rngs(42))
        # All blocks should be TransformerBlock
        from model.transformer import TransformerBlock
        for block in model.blocks:
            assert isinstance(block, TransformerBlock)

    def test_block_types_hybrid(self):
        config = self._make_config("hybrid", k=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        from model.transformer import TransformerBlock
        from model.gated_delta_net import GDNBlock
        types = [type(b).__name__ for b in model.blocks]
        # Attn + [GDN GDN GDN Attn] + Attn
        assert types == [
            "TransformerBlock",
            "GDNBlock", "GDNBlock", "GDNBlock", "TransformerBlock",
            "TransformerBlock",
        ]

    def test_last_token_output_mode(self):
        config = self._make_config("attn", k=1, output_mode="last_token")
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 50)

    def test_no_vocab_pre_embedded(self):
        config = self._make_config("hybrid", k=1, n_vocab=None)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8, 32))
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_swiglu(self):
        config = self._make_config("hybrid", k=1, use_swiglu=True)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_absolute_pos_encoding(self):
        config = self._make_config("attn", k=1, pos_encoding="absolute")
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_no_pos_encoding(self):
        config = self._make_config("attn", k=1, pos_encoding="none")
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_gradient_flows(self):
        """Verify gradients flow through the full architecture."""
        config = self._make_config("hybrid", k=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        labels = jnp.zeros((2, 8), dtype=jnp.int32)

        def loss_fn(model):
            logits = model(x)
            return jnp.mean(
                jax.vmap(jax.vmap(jax.nn.log_softmax))(logits)[:, :, 0]
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        # Check at least one grad is nonzero
        flat_grads = jax.tree.leaves(grads)
        any_nonzero = any(jnp.any(g != 0) for g in flat_grads if hasattr(g, '__len__'))
        assert any_nonzero

    def test_standard_matches_more_blocks(self):
        """Standard k=2 should have more params than k=1."""
        config_k1 = self._make_config("attn", k=1)
        config_k2 = self._make_config("attn", k=2)
        model_k1 = config_k1.to_model(rngs=nnx.Rngs(42))
        model_k2 = config_k2.to_model(rngs=nnx.Rngs(42))
        params_k1 = sum(p.size for p in jax.tree.leaves(nnx.state(model_k1)))
        params_k2 = sum(p.size for p in jax.tree.leaves(nnx.state(model_k2)))
        assert params_k2 > params_k1
