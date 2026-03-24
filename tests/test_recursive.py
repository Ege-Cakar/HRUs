"""Tests for the recursive (depth-recurrent) architecture."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from model.recursive import RecursiveArchConfig, RecursiveArchitecture


class TestRecursiveArchConfig:
    def test_n_layers_unrolled(self):
        for T in [1, 2, 3, 5]:
            config = RecursiveArchConfig(n_recurrences=T)
            assert config.n_layers_unrolled == 4 * T + 2

    def test_n_params_blocks_constant(self):
        """Parameter count is constant regardless of recurrence count."""
        config_t1 = RecursiveArchConfig(n_recurrences=1)
        config_t5 = RecursiveArchConfig(n_recurrences=5)
        assert config_t1.n_params_blocks == 6
        assert config_t5.n_params_blocks == 6

    def test_invalid_core_block(self):
        config = RecursiveArchConfig(core_block="invalid")
        with pytest.raises(ValueError, match="core_block"):
            config.to_model(rngs=nnx.Rngs(42))


class TestRecursiveArchitecture:
    def _make_config(self, core_block="attn", T=1, **kwargs):
        defaults = dict(
            n_vocab=50, n_seq=16, n_hidden=32, n_heads=4,
            n_out=50, n_pred_tokens=1,
            core_block=core_block, n_recurrences=T,
            pos_encoding="rope", output_mode="full_sequence",
            use_bf16=False,
        )
        defaults.update(kwargs)
        return RecursiveArchConfig(**defaults)

    def test_standard_recursive_forward(self):
        config = self._make_config("attn", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_hybrid_recursive_forward(self):
        config = self._make_config("hybrid", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_multiple_recurrences(self):
        config = self._make_config("attn", T=3)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_override_recurrences_at_call_time(self):
        """n_recurrences can differ between train and test."""
        config = self._make_config("attn", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)

        out_t1 = model(x, n_recurrences=1)
        out_t3 = model(x, n_recurrences=3)
        out_t5 = model(x, n_recurrences=5)

        # All should have correct shape
        assert out_t1.shape == (2, 8, 50)
        assert out_t3.shape == (2, 8, 50)
        assert out_t5.shape == (2, 8, 50)

        # Different T should produce different outputs
        assert not jnp.allclose(out_t1, out_t3)
        assert not jnp.allclose(out_t3, out_t5)

    def test_weight_sharing(self):
        """Core blocks are weight-tied — same params regardless of T."""
        config = self._make_config("attn", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))

        # Count parameters — should be same whether T=1 or T=100
        params_leaves = jax.tree.leaves(nnx.state(model))
        n_params = sum(p.size for p in params_leaves)

        # A non-recursive model with k=3 should have MORE params
        from model.architecture import ArchConfig
        nonrec_config = ArchConfig(
            n_vocab=50, n_seq=16, n_hidden=32, n_heads=4,
            n_out=50, core_block="attn", n_core_repeats=3,
            pos_encoding="rope", output_mode="full_sequence",
            use_bf16=False,
        )
        nonrec_model = nonrec_config.to_model(rngs=nnx.Rngs(42))
        n_params_nonrec = sum(
            p.size for p in jax.tree.leaves(nnx.state(nonrec_model))
        )

        # Recursive has fewer params (weight-tied core) but same effective depth
        # when called with n_recurrences=3
        assert n_params < n_params_nonrec

    def test_recursive_fewer_params_than_unrolled(self):
        """Recursive T=5 has same params as T=1, unrolled k=5 has 5x core params."""
        config = self._make_config("hybrid", T=5)
        model = config.to_model(rngs=nnx.Rngs(42))
        rec_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)))

        from model.architecture import ArchConfig
        unrolled = ArchConfig(
            n_vocab=50, n_seq=16, n_hidden=32, n_heads=4,
            n_out=50, core_block="hybrid", n_core_repeats=5,
            pos_encoding="rope", output_mode="full_sequence",
            use_bf16=False,
        ).to_model(rngs=nnx.Rngs(42))
        unrolled_params = sum(p.size for p in jax.tree.leaves(nnx.state(unrolled)))

        assert rec_params < unrolled_params

    def test_gradient_flows(self):
        config = self._make_config("hybrid", T=2)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)

        def loss_fn(model):
            logits = model(x)
            return jnp.mean(
                jax.vmap(jax.vmap(jax.nn.log_softmax))(logits)[:, :, 0]
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        flat_grads = jax.tree.leaves(grads)
        any_nonzero = any(jnp.any(g != 0) for g in flat_grads if hasattr(g, '__len__'))
        assert any_nonzero

    def test_input_injection_exists(self):
        """Model should have input_injection linear layer."""
        config = self._make_config("attn", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert hasattr(model, 'input_injection')
        assert isinstance(model.input_injection, nnx.Linear)
        # Input: 2 * n_hidden, Output: n_hidden
        assert model.input_injection.kernel.value.shape == (64, 32)

    def test_last_token_output_mode(self):
        config = self._make_config("attn", T=1, output_mode="last_token")
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        out = model(x)
        assert out.shape == (2, 50)

    def test_no_vocab_pre_embedded(self):
        config = self._make_config("hybrid", T=2, n_vocab=None)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8, 32))
        out = model(x)
        assert out.shape == (2, 8, 50)

    def test_cache_not_supported(self):
        config = self._make_config("attn", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        x = jnp.ones((2, 8), dtype=jnp.int32)
        with pytest.raises(NotImplementedError, match="KV cache"):
            model(x, return_cache=True)

    def test_core_block_types_standard(self):
        config = self._make_config("attn", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert model._core_types == ["attn", "attn", "attn", "attn"]
        assert len(model.core) == 4
        for block in model.core:
            assert isinstance(block, TransformerBlock)

    def test_core_block_types_hybrid(self):
        config = self._make_config("hybrid", T=1)
        model = config.to_model(rngs=nnx.Rngs(42))
        assert model._core_types == ["gdn", "gdn", "gdn", "attn"]
        from model.gated_delta_net import GDNBlock
        assert isinstance(model.core[0], GDNBlock)
        assert isinstance(model.core[3], TransformerBlock)


# Import here to avoid circular at module level
from model.transformer import TransformerBlock


@pytest.mark.slow
class TestRecursiveFOLSmoke:
    """Smoke test: recursive models learn on FOL."""

    def test_recursive_transformer_loss_decreases(self):
        from task.fol_task_factory import FOLTaskFactory, RuleBankConfig
        from train import create_optimizer, train_step, parse_loss_name

        factory = FOLTaskFactory(
            rule_bank_seed=42, d_train_max=3, d_eval_max=6,
            bank_config=RuleBankConfig(
                n_layers=4, predicates_per_layer=4,
                arity_min=1, arity_max=2,
                constants=("a", "b", "c", "d"),
                rules_per_transition=4,
                k_in_min=1, k_in_max=2, k_out_min=1, k_out_max=1,
                vars_per_rule_max=3,
            ),
        )
        n_vocab = factory.n_vocab
        n_seq = factory.dims_internalized.n_seq_ar
        task = factory.make_internalized_task(batch_size=8, distance_range=(1, 2))

        config = RecursiveArchConfig(
            n_vocab=n_vocab, n_seq=n_seq, n_hidden=32, n_heads=4,
            n_out=n_vocab, core_block="attn", n_recurrences=2,
            output_mode="full_sequence", pos_encoding="rope", use_bf16=False,
        )
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
        assert late < early, f"Loss did not decrease: {early:.4f} -> {late:.4f}"

    def test_recursive_hybrid_loss_decreases(self):
        from task.fol_task_factory import FOLTaskFactory, RuleBankConfig
        from train import create_optimizer, train_step, parse_loss_name

        factory = FOLTaskFactory(
            rule_bank_seed=42, d_train_max=3, d_eval_max=6,
            bank_config=RuleBankConfig(
                n_layers=4, predicates_per_layer=4,
                arity_min=1, arity_max=2,
                constants=("a", "b", "c", "d"),
                rules_per_transition=4,
                k_in_min=1, k_in_max=2, k_out_min=1, k_out_max=1,
                vars_per_rule_max=3,
            ),
        )
        n_vocab = factory.n_vocab
        n_seq = factory.dims_internalized.n_seq_ar
        task = factory.make_internalized_task(batch_size=8, distance_range=(1, 2))

        config = RecursiveArchConfig(
            n_vocab=n_vocab, n_seq=n_seq, n_hidden=32, n_heads=4,
            n_out=n_vocab, core_block="hybrid", n_recurrences=2,
            output_mode="full_sequence", pos_encoding="rope", use_bf16=False,
        )
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
        assert late < early, f"Loss did not decrease: {early:.4f} -> {late:.4f}"
