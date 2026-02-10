"""
MLP-Mixer model using Flax NNX
"""
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from model.mup import MuReadout

def parse_act_fn(fn: str) -> Callable:
    if fn == 'relu':
        return jax.nn.relu
    elif fn == 'linear':
        return lambda x: x
    elif fn == 'gelu':
        return jax.nn.gelu
    elif fn == 'quadratic':
        return lambda x: x**2
    else:
        raise ValueError(f'function not recognized: {fn}')


@dataclass
class MlpConfig:
    """Global hyperparameters for Mixer"""
    n_vocab: int | None = None
    n_layers: int = 2
    n_hidden: int = 128
    n_out: int = 1
    act_fn: str = 'relu'
    use_bias: bool = True
    use_mup: bool = False

    def to_model(self, *, rngs: nnx.Rngs) -> 'Mlp':
        return Mlp(self, rngs=rngs)


class Mlp(nnx.Module):
    """Simple MLP architecture using Flax NNX"""

    def __init__(self, config: MlpConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.act_fn = parse_act_fn(config.act_fn)

        # Embedding layer (optional)
        if config.n_vocab is not None:
            self.embed = nnx.Embed(
                num_embeddings=config.n_vocab,
                features=config.n_hidden,
                rngs=rngs
            )
        else:
            self.embed = None

        # MLP layers
        self.layers = nnx.List([
            nnx.Linear(config.n_hidden, config.n_hidden, use_bias=config.use_bias, rngs=rngs)
            for _ in range(config.n_layers)
        ])

        # Output layer
        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden,
                config.n_out,
                use_bias=config.use_bias,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(config.n_hidden, config.n_out, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, x):
        if self.embed is not None:
            x = self.embed(x)
            if x.ndim == 3:
                x = x.mean(axis=1)

        assert len(x.shape) == 2, f"Expected 2D input (batch, features), got {x.shape}"

        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)

        out = self.output(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out


@dataclass
class MixerConfig:
    """Global hyperparameters for Mixer"""
    n_vocab: int | None = None
    n_seq: int = 16  # Sequence length (required for channel mixing)
    n_layers: int = 2
    n_hidden: int = 128
    n_channels: int = 16
    n_out: int = 1
    act_fn: str = 'relu'
    last_token_only: bool = True
    layer_norm: bool = False
    use_bias: bool = True
    use_mup: bool = False

    def to_model(self, *, rngs: nnx.Rngs) -> 'Mixer':
        return Mixer(self, rngs=rngs)


class Mixer(nnx.Module):
    """MLP-Mixer architecture using Flax NNX"""

    def __init__(self, config: MixerConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.act_fn = parse_act_fn(config.act_fn)

        # Embedding layer (optional)
        if config.n_vocab is not None:
            self.embed = nnx.Embed(
                num_embeddings=config.n_vocab,
                features=config.n_hidden,
                rngs=rngs
            )
        else:
            self.embed = None

        # Mixer layers
        self.token_mixing_layers = nnx.List()
        self.channel_mixing_layers = nnx.List()
        self.layer_norms = nnx.List()

        for i in range(config.n_layers):
            # Token mixing: operates on features dimension
            self.token_mixing_layers.append(
                nnx.Linear(config.n_hidden, config.n_hidden, use_bias=config.use_bias, rngs=rngs)
            )
            # Channel mixing: first layer maps n_seq -> n_channels,
            # subsequent layers map n_channels -> n_channels
            in_channels = config.n_seq if i == 0 else config.n_channels
            self.channel_mixing_layers.append(
                nnx.Linear(in_channels, config.n_channels, use_bias=config.use_bias, rngs=rngs)
            )
            if config.layer_norm:
                self.layer_norms.append(nnx.LayerNorm(config.n_hidden, rngs=rngs))

        # Output layer
        # When last_token_only=True: input is (batch, n_hidden) from last token
        # When last_token_only=False: input is (batch, n_channels * n_hidden) from flatten
        out_in_features = config.n_hidden if config.last_token_only else config.n_channels * config.n_hidden
        if config.use_mup:
            self.output = MuReadout(
                out_in_features,
                config.n_out,
                use_bias=config.use_bias,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(out_in_features, config.n_out, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, x):
        config = self.config

        if self.embed is not None:
            x = self.embed(x)

        assert len(x.shape) == 3, f"Expected 3D input (batch, seq, features), got {x.shape}"

        for i in range(config.n_layers):
            x = self.token_mixing_layers[i](x)
            x = jnp.transpose(x, (0, 2, 1))

            x = self.channel_mixing_layers[i](x)
            x = jnp.transpose(x, (0, 2, 1))

            if config.layer_norm:
                x = self.layer_norms[i](x)

            x = self.act_fn(x)

        if config.last_token_only:
            x = x[:, -1, :]
        else:
            x = x.reshape(x.shape[0], -1)

        out = self.output(x)

        if config.n_out == 1:
            out = out.flatten()

        return out


@dataclass
class CompletionMixerConfig:
    """Hyperparameters for prompt-to-completion Mixer."""
    n_vocab: int | None = None
    n_seq: int = 128
    n_layers: int = 4
    n_hidden: int = 128
    n_channels: int = 128
    n_out_seq: int = 64
    n_out_vocab: int = 128
    act_fn: str = 'gelu'
    layer_norm: bool = True
    use_bias: bool = True
    use_mup: bool = False

    def to_model(self, *, rngs: nnx.Rngs) -> 'CompletionMixer':
        return CompletionMixer(self, rngs=rngs)


class CompletionMixer(nnx.Module):
    """Mixer that maps a full prompt sequence to a full completion sequence."""

    def __init__(self, config: CompletionMixerConfig, *, rngs: nnx.Rngs):
        if config.n_out_seq > config.n_seq:
            raise ValueError(
                f"n_out_seq ({config.n_out_seq}) must be <= n_seq ({config.n_seq})"
            )

        self.config = config
        self.act_fn = parse_act_fn(config.act_fn)

        if config.n_vocab is not None:
            self.embed = nnx.Embed(
                num_embeddings=config.n_vocab,
                features=config.n_hidden,
                rngs=rngs,
            )
        else:
            self.embed = None

        self.token_mixing_layers = nnx.List()
        self.channel_up_layers = nnx.List()
        self.channel_down_layers = nnx.List()
        self.token_norms = nnx.List()
        self.channel_norms = nnx.List()

        for _ in range(config.n_layers):
            self.token_mixing_layers.append(
                nnx.Linear(config.n_hidden, config.n_hidden, use_bias=config.use_bias, rngs=rngs)
            )
            self.channel_up_layers.append(
                nnx.Linear(config.n_seq, config.n_channels, use_bias=config.use_bias, rngs=rngs)
            )
            self.channel_down_layers.append(
                nnx.Linear(config.n_channels, config.n_seq, use_bias=config.use_bias, rngs=rngs)
            )
            if config.layer_norm:
                self.token_norms.append(nnx.LayerNorm(config.n_hidden, rngs=rngs))
                self.channel_norms.append(nnx.LayerNorm(config.n_hidden, rngs=rngs))

        if config.use_mup:
            self.output = MuReadout(
                config.n_hidden,
                config.n_out_vocab,
                use_bias=config.use_bias,
                rngs=rngs,
            )
        else:
            self.output = nnx.Linear(
                config.n_hidden,
                config.n_out_vocab,
                use_bias=config.use_bias,
                rngs=rngs,
            )

    def __call__(self, x):
        config = self.config
        if self.embed is not None:
            assert x.ndim == 2, f"Expected 2D token ids (batch, seq), got {x.shape}"
            x = self.embed(x)
        else:
            assert x.ndim == 3, f"Expected 3D input (batch, seq, features), got {x.shape}"

        if x.shape[1] != config.n_seq:
            raise ValueError(
                f"Expected sequence length {config.n_seq}, got {x.shape[1]}"
            )

        for i in range(config.n_layers):
            residual = x
            if config.layer_norm:
                x = self.token_norms[i](x)
            x = self.token_mixing_layers[i](x)
            x = self.act_fn(x)
            x = x + residual

            residual = x
            if config.layer_norm:
                x = self.channel_norms[i](x)
            x_t = jnp.transpose(x, (0, 2, 1))
            x_t = self.channel_up_layers[i](x_t)
            x_t = self.act_fn(x_t)
            x_t = self.channel_down_layers[i](x_t)
            x = jnp.transpose(x_t, (0, 2, 1))
            x = x + residual

        x = x[:, :config.n_out_seq, :]
        return self.output(x)
