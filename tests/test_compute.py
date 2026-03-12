"""Tests for model.compute — analytical compute-efficiency utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model.compute import (
    compute_metrics_from_info,
    mamba2_bonsai_flops_forward,
    mamba2_bonsai_param_count,
    memory_bytes_estimate,
    training_flops_total,
    transformer_flops_forward,
    transformer_param_count,
)
from model.ssm_bonsai import Mamba2BonsaiConfig
from model.transformer import TransformerConfig


def _count_params(model):
    """Count total parameters in a Flax NNX model."""
    return sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))


# --- 2a. Parameter count validation ---


class TestTransformerParamCount:
    """Analytical param counts must match instantiated models exactly."""

    def test_swiglu_small(self):
        config = TransformerConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_swiglu=True, use_bias=True,
            layer_norm=True, pos_encoding="rope", use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = transformer_param_count(
            n_vocab=32, n_layers=2, n_hidden=64, n_heads=4,
            use_swiglu=True, use_bias=True, n_out=32, n_pred_tokens=1,
        )
        assert analytical["total"] == actual

    def test_gelu_small(self):
        config = TransformerConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_swiglu=False, use_bias=True,
            layer_norm=True, pos_encoding="rope", use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = transformer_param_count(
            n_vocab=32, n_layers=2, n_hidden=64, n_heads=4,
            use_swiglu=False, use_bias=True, n_out=32, n_pred_tokens=1,
        )
        assert analytical["total"] == actual

    def test_no_bias(self):
        config = TransformerConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_swiglu=True, use_bias=False,
            layer_norm=True, pos_encoding="rope", use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = transformer_param_count(
            n_vocab=32, n_layers=2, n_hidden=64, n_heads=4,
            use_swiglu=True, use_bias=False, n_out=32, n_pred_tokens=1,
        )
        assert analytical["total"] == actual

    def test_larger_config(self):
        config = TransformerConfig(
            n_vocab=128, n_seq=256, n_layers=4, n_hidden=256, n_heads=8,
            n_out=128, n_pred_tokens=1, use_swiglu=True, use_bias=True,
            layer_norm=True, pos_encoding="rope", use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = transformer_param_count(
            n_vocab=128, n_layers=4, n_hidden=256, n_heads=8,
            use_swiglu=True, use_bias=True, n_out=128, n_pred_tokens=1,
        )
        assert analytical["total"] == actual


class TestMamba2BonsaiParamCount:
    """Analytical param counts must match instantiated models exactly."""

    def test_small(self):
        config = Mamba2BonsaiConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_bias=True, layer_norm=True,
            d_state=8, d_conv=4, expand=2, use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = mamba2_bonsai_param_count(
            n_vocab=32, n_layers=2, n_hidden=64, n_heads=4,
            d_state=8, d_conv=4, expand=2,
            use_bias=True, n_out=32, n_pred_tokens=1,
        )
        assert analytical["total"] == actual

    def test_no_bias(self):
        config = Mamba2BonsaiConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_bias=False, layer_norm=True,
            d_state=8, d_conv=4, expand=2, use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = mamba2_bonsai_param_count(
            n_vocab=32, n_layers=2, n_hidden=64, n_heads=4,
            d_state=8, d_conv=4, expand=2,
            use_bias=False, n_out=32, n_pred_tokens=1,
        )
        assert analytical["total"] == actual

    def test_larger_config(self):
        config = Mamba2BonsaiConfig(
            n_vocab=128, n_seq=256, n_layers=4, n_hidden=128, n_heads=8,
            n_out=128, n_pred_tokens=1, use_bias=True, layer_norm=True,
            d_state=16, d_conv=4, expand=2, use_bf16=False,
        )
        model = config.to_model(rngs=nnx.Rngs(42))
        actual = _count_params(model)
        analytical = mamba2_bonsai_param_count(
            n_vocab=128, n_layers=4, n_hidden=128, n_heads=8,
            d_state=16, d_conv=4, expand=2,
            use_bias=True, n_out=128, n_pred_tokens=1,
        )
        assert analytical["total"] == actual


# --- 2b. FLOPs validation against JAX AOT cost analysis ---


class TestFlopsVsJaxCostAnalysis:
    """Analytical FLOPs should be within reasonable tolerance of JAX's estimates."""

    def _get_jax_flops(self, config, seq_len):
        model = config.to_model(rngs=nnx.Rngs(42))
        dummy_x = jnp.ones((1, seq_len), dtype=jnp.int32)
        lowered = jax.jit(model).lower(dummy_x)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        # cost_analysis returns a dict (CPU/single-device) or list of dicts
        if isinstance(cost, (list, tuple)):
            cost = cost[0] if cost else {}
        if not isinstance(cost, dict) or "flops" not in cost:
            pytest.skip("JAX cost_analysis does not report flops on this backend")
        return cost["flops"]

    def test_transformer_flops(self):
        config = TransformerConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_swiglu=True, use_bias=True,
            layer_norm=True, pos_encoding="rope", output_mode="full_sequence",
            use_bf16=False,
        )
        seq_len = 32
        jax_flops = self._get_jax_flops(config, seq_len)
        analytical = transformer_flops_forward(
            n_seq=seq_len, n_layers=2, n_hidden=64, n_heads=4,
            use_swiglu=True, n_vocab=32,
        )
        ratio = analytical["total"] / jax_flops
        assert 0.5 < ratio < 2.0, f"Analytical/JAX ratio {ratio:.2f} outside tolerance"

    def test_mamba2_flops(self):
        config = Mamba2BonsaiConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_bias=True, layer_norm=True,
            d_state=8, d_conv=4, expand=2, scan_chunk_len=16,
            output_mode="full_sequence", use_bf16=False,
        )
        seq_len = 32
        jax_flops = self._get_jax_flops(config, seq_len)
        analytical = mamba2_bonsai_flops_forward(
            n_seq=seq_len, n_layers=2, n_hidden=64, n_heads=4,
            d_state=8, d_conv=4, expand=2, scan_chunk_len=16, n_vocab=32,
        )
        ratio = analytical["total"] / jax_flops
        assert 0.5 < ratio < 2.0, f"Analytical/JAX ratio {ratio:.2f} outside tolerance"

    def test_transformer_scaling_vs_jax(self):
        """Both analytical and JAX FLOPs should show similar scaling."""
        config = TransformerConfig(
            n_vocab=32, n_seq=64, n_layers=2, n_hidden=64, n_heads=4,
            n_out=32, n_pred_tokens=1, use_swiglu=True, use_bias=True,
            layer_norm=True, pos_encoding="rope", output_mode="full_sequence",
            use_bf16=False,
        )
        seq_lens = [16, 32]
        jax_vals = []
        analytical_vals = []
        for s in seq_lens:
            jax_vals.append(self._get_jax_flops(config, s))
            analytical_vals.append(
                transformer_flops_forward(
                    n_seq=s, n_layers=2, n_hidden=64, n_heads=4,
                    use_swiglu=True, n_vocab=32,
                )["total"]
            )
        jax_growth = jax_vals[1] / jax_vals[0]
        analytical_growth = analytical_vals[1] / analytical_vals[0]
        assert abs(jax_growth - analytical_growth) / jax_growth < 0.3


# --- 2c. Scaling tests ---


class TestScaling:
    """Verify quadratic vs linear scaling properties."""

    def test_transformer_attention_quadratic(self):
        """Doubling n_seq should ~4x the attention FLOPs when S >> D."""
        # Use large S relative to D so 4*S²*D dominates over 8*S*D²
        s1 = transformer_flops_forward(
            n_seq=512, n_layers=4, n_hidden=64, n_heads=4,
            use_swiglu=True, n_vocab=128,
        )
        s2 = transformer_flops_forward(
            n_seq=1024, n_layers=4, n_hidden=64, n_heads=4,
            use_swiglu=True, n_vocab=128,
        )
        attn_ratio = s2["attn_per_layer"] / s1["attn_per_layer"]
        # 8*S*D² + 4*S²*D — quadratic dominates at larger S/D ratio
        assert attn_ratio > 3.0, f"Attention scaling ratio {attn_ratio:.2f}, expected > 3"

    def test_mamba2_ssd_approximately_linear(self):
        """Doubling n_seq should ~2x the SSD FLOPs (linear dominant term)."""
        s1 = mamba2_bonsai_flops_forward(
            n_seq=64, n_layers=4, n_hidden=256, n_heads=8,
            d_state=16, d_conv=4, expand=2, scan_chunk_len=64, n_vocab=128,
        )
        s2 = mamba2_bonsai_flops_forward(
            n_seq=128, n_layers=4, n_hidden=256, n_heads=8,
            d_state=16, d_conv=4, expand=2, scan_chunk_len=64, n_vocab=128,
        )
        ssd_ratio = s2["ssd_per_layer"] / s1["ssd_per_layer"]
        assert 1.5 < ssd_ratio < 3.0, f"SSD scaling ratio {ssd_ratio:.2f}, expected ~2"

    def test_transformer_grows_faster_than_mamba2(self):
        """At large seq len, transformer FLOPs ratio over Mamba2 should increase."""
        D, H = 256, 8
        results = {}
        for S in [64, 256, 1024]:
            tf = transformer_flops_forward(
                n_seq=S, n_layers=4, n_hidden=D, n_heads=H,
                use_swiglu=True, n_vocab=128,
            )
            mb = mamba2_bonsai_flops_forward(
                n_seq=S, n_layers=4, n_hidden=D, n_heads=H,
                d_state=16, d_conv=4, expand=2, scan_chunk_len=64, n_vocab=128,
            )
            results[S] = (tf["all_layers"], mb["all_layers"])

        ratio_64 = results[64][0] / results[64][1]
        ratio_1024 = results[1024][0] / results[1024][1]
        assert ratio_1024 > ratio_64


class TestDispatcher:
    """Test compute_metrics_from_info unified dispatcher."""

    def test_transformer_info(self):
        info = {
            "model_family": "transformer",
            "n_vocab": 128, "n_layers": 4, "n_hidden": 256,
            "n_heads": 8, "use_swiglu": True, "n_seq": 512,
        }
        result = compute_metrics_from_info(info)
        assert result["n_params"] > 0
        assert result["forward_flops"] > 0

    def test_mamba2_info(self):
        info = {
            "model_family": "mamba2_bonsai",
            "n_vocab": 128, "n_layers": 4, "n_hidden": 256,
            "n_heads": 8, "d_state": 16, "d_conv": 4,
            "scan_chunk_len": 64, "n_seq": 512,
        }
        result = compute_metrics_from_info(info)
        assert result["n_params"] > 0
        assert result["forward_flops"] > 0

    def test_n_seq_override(self):
        info = {
            "model_family": "transformer",
            "n_vocab": 128, "n_layers": 4, "n_hidden": 256,
            "n_heads": 8, "use_swiglu": True, "n_seq": 512,
        }
        r1 = compute_metrics_from_info(info, n_seq_override=64)
        r2 = compute_metrics_from_info(info, n_seq_override=128)
        assert r2["forward_flops"] > r1["forward_flops"]
        assert r1["n_params"] == r2["n_params"]

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown model_family"):
            compute_metrics_from_info({"model_family": "gpt42"})


class TestTrainingFlops:
    def test_basic(self):
        result = training_flops_total(
            1000, train_iters=100, batch_size=8, grad_accum_steps=2,
        )
        assert result == 3 * 1000 * 8 * 2 * 100


class TestMemory:
    def test_transformer_memory(self):
        result = memory_bytes_estimate(
            1_000_000, batch_size=8, n_seq=512, n_hidden=256,
            n_layers=4, n_heads=8, model_family="transformer",
        )
        assert result["total_bytes"] > 0
        assert result["params_bytes"] == 1_000_000 * 4
        assert result["optimizer_bytes"] == 2 * 1_000_000 * 4

    def test_mamba2_less_activation_memory(self):
        """Mamba2 should use less activation memory (no attention matrices)."""
        tf = memory_bytes_estimate(
            1_000_000, batch_size=8, n_seq=512, n_hidden=256,
            n_layers=4, n_heads=8, model_family="transformer",
        )
        mb = memory_bytes_estimate(
            1_000_000, batch_size=8, n_seq=512, n_hidden=256,
            n_layers=4, model_family="mamba2_bonsai",
        )
        assert mb["activations_bytes"] < tf["activations_bytes"]
