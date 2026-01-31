"""Coordinate checks for muP scaling."""

import numpy as np
import optax
import jax.numpy as jnp
from flax import nnx

from model.mlp import MlpConfig, MixerConfig
from model.transformer import TransformerConfig


def _train_steps(model, x, y, *, steps: int, lr: float):
    optimizer = nnx.ModelAndOptimizer(model, optax.sgd(lr))

    def loss_fn(m):
        preds = m(x)
        return jnp.mean((preds - y) ** 2)

    for _ in range(steps):
        _, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, optimizer.wrt)
        )(optimizer.model)
        optimizer.update(grads)

    return optimizer.model


def _mlp_features(model, x):
    if model.embed is not None:
        x = model.embed(x)
    for layer in model.layers:
        x = model.act_fn(layer(x))
    return x


def _mixer_features(model, x):
    config = model.config
    if model.embed is not None:
        x = model.embed(x)

    for i in range(config.n_layers):
        x = model.token_mixing_layers[i](x)
        x = jnp.transpose(x, (0, 2, 1))

        x = model.channel_mixing_layers[i](x)
        x = jnp.transpose(x, (0, 2, 1))

        if config.layer_norm:
            x = model.layer_norms[i](x)

        x = model.act_fn(x)

    if config.last_token_only:
        x = x[:, -1, :]
    else:
        x = x.reshape(x.shape[0], -1)

    return x


def _transformer_features(model, x):
    config = model.config
    if model.embed is not None:
        x = model.embed(x)

    if model.pos_embedding is not None:
        x = x + model.pos_embedding[: x.shape[1]]

    mask = model._causal_mask[: x.shape[1], : x.shape[1]]
    for block in model.blocks:
        x = block(x, mask=mask)

    if model.final_ln is not None:
        x = model.final_ln(x)

    if config.last_token_only:
        x = x[:, -1, :]

    return x


def _coord_l1(model, x, y, feature_fn, *, steps: int, lr: float) -> float:
    model = _train_steps(model, x, y, steps=steps, lr=lr)
    features = feature_fn(model, x)
    return float(jnp.mean(jnp.abs(features)))


def _assert_width_stable(l1s, *, max_ratio: float = 2.0):
    assert all(np.isfinite(l1) for l1 in l1s)
    ratio = max(l1s) / min(l1s)
    assert ratio < max_ratio


def test_mup_coord_check_mlp():
    widths = [32, 64]
    vocab = 32
    batch = 8
    l1s = []
    for width in widths:
        config = MlpConfig(
            n_vocab=vocab,
            n_layers=2,
            n_hidden=width,
            n_out=1,
            use_mup=True,
        )
        rngs = nnx.Rngs(0)
        model = config.to_model(rngs=rngs)
        x = (jnp.arange(batch) % vocab).astype(jnp.int32)
        y = jnp.ones((batch,), dtype=jnp.float32)
        l1 = _coord_l1(model, x, y, _mlp_features, steps=2, lr=0.1)
        l1s.append(l1)
    _assert_width_stable(l1s)


def test_mup_coord_check_mixer():
    widths = [32, 64]
    vocab = 32
    batch = 4
    seq = 8
    l1s = []
    x = (jnp.arange(batch * seq).reshape(batch, seq) % vocab).astype(jnp.int32)
    y = jnp.ones((batch,), dtype=jnp.float32)
    for width in widths:
        config = MixerConfig(
            n_vocab=vocab,
            n_seq=seq,
            n_layers=2,
            n_hidden=width,
            n_channels=4,
            n_out=1,
            use_mup=True,
        )
        rngs = nnx.Rngs(0)
        model = config.to_model(rngs=rngs)
        l1 = _coord_l1(model, x, y, _mixer_features, steps=2, lr=0.1)
        l1s.append(l1)
    _assert_width_stable(l1s)


def test_mup_coord_check_transformer():
    widths = [32, 64]
    vocab = 32
    batch = 4
    seq = 8
    l1s = []
    x = (jnp.arange(batch * seq).reshape(batch, seq) % vocab).astype(jnp.int32)
    y = jnp.ones((batch,), dtype=jnp.float32)
    for width in widths:
        config = TransformerConfig(
            n_vocab=vocab,
            n_seq=seq,
            n_layers=2,
            n_hidden=width,
            n_heads=4,
            n_out=1,
            dropout_rate=0.0,
            use_mup=True,
        )
        rngs = nnx.Rngs(0)
        model = config.to_model(rngs=rngs)
        l1 = _coord_l1(model, x, y, _transformer_features, steps=2, lr=0.1)
        l1s.append(l1)
    _assert_width_stable(l1s)
