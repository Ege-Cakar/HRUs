"""muP helpers."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def mup_attention_scale(head_dim: int) -> float:
    """Return the muP attention scaling factor for a head dimension."""
    return 1.0 / head_dim


class MuReadout(nnx.Module):
    """Readout layer with muP output scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
        scale: float | None = None,
    ):
        # muP readout uses unit-variance init with an explicit output scale.
        self.linear = nnx.Linear(
            in_features,
            out_features,
            use_bias=use_bias,
            kernel_init=jax.nn.initializers.normal(stddev=1.0),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.scale = (1.0 / in_features) if scale is None else scale

    def __call__(self, x):
        out = self.linear(x)
        bias = getattr(self.linear, "bias", None)
        if bias is None:
            return out * self.scale
        # Scale only the weight contribution; keep bias unscaled.
        return (out - bias) * self.scale + bias
