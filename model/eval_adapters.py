"""Shared adapter utilities for autoregressive and completion-style evaluation."""

from __future__ import annotations

import inspect
import uuid
import weakref
from typing import Any, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

_DECODE_CACHE_KEY_ATTR = "_decode_cache_key"
_MODEL_INSTANCE_CACHE_IDS: weakref.WeakKeyDictionary[object, str] = weakref.WeakKeyDictionary()


def _resolve_decode_cache_key(model: Callable[..., Any]) -> object:
    key = getattr(model, _DECODE_CACHE_KEY_ATTR, None)
    if key is not None:
        return key
    return int(id(model))


def _random_model_instance_id(model_obj: object) -> str:
    try:
        cached = _MODEL_INSTANCE_CACHE_IDS.get(model_obj)
        if cached is not None:
            return str(cached)
        out = uuid.uuid4().hex
        _MODEL_INSTANCE_CACHE_IDS[model_obj] = out
        return out
    except TypeError:
        return f"id_{int(id(model_obj))}"


class CompletionLogitsAdapter:
    """Decode fixed-length completion logits into completion tokens."""

    def __init__(self, *, n_seq: int, pad_token_id: int = 0) -> None:
        self.n_seq = int(n_seq)
        self.pad_token_id = int(pad_token_id)

    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens: list[int] | np.ndarray,
        tokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        prompt_only = _build_prompt_only_inputs_single(
            prompt_tokens=prompt_tokens,
            n_seq=self.n_seq,
            sep_token_id=int(tokenizer.sep_token_id),
            pad_token_id=self.pad_token_id,
        )

        logits = np.asarray(model(prompt_only[None, :]))
        logits = _normalize_completion_logits(logits)

        out: list[int] = []
        for idx in range(logits.shape[0]):
            tok = _sample_token_from_logits(
                logits[idx],
                temperature=temperature,
                rng=rng,
            )
            out.append(tok)
            if tok == int(tokenizer.eot_token_id):
                break
        return np.asarray(out, dtype=np.int32)


class AutoregressiveLogitsAdapter:
    """Decode next-token logits autoregressively until EOT or length cap."""

    def __init__(
        self,
        *,
        n_seq: int,
        max_completion_len: int,
        pad_token_id: int = 0,
        jit_step: bool = False,
    ) -> None:
        self.n_seq = int(n_seq)
        self.max_completion_len = int(max_completion_len)
        self.pad_token_id = int(pad_token_id)
        self.jit_step = bool(jit_step)
        self._noncache_decode_fn_cache: dict[object, Callable[..., tuple[jnp.ndarray, jnp.ndarray]]] = {}
        self._cached_decode_fn_cache: dict[object, Callable[..., tuple[jnp.ndarray, jnp.ndarray]]] = {}

    def predict_completion(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt_tokens: list[int] | np.ndarray,
        tokenizer,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        prompt = np.asarray(prompt_tokens, dtype=np.int32)
        if prompt.ndim != 1:
            raise ValueError(f"Prompt must be 1D, got {prompt.shape}")
        if prompt.shape[0] < 1:
            raise ValueError("Prompt cannot be empty.")
        if prompt.shape[0] >= self.n_seq:
            raise ValueError(
                f"Prompt length {prompt.shape[0]} must be < n_seq={self.n_seq}."
            )

        if self.jit_step:
            if _model_supports_cache_kwargs(model):
                cached = self._predict_completion_with_cache_jit(
                    model=model,
                    prompt=prompt,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    rng=rng,
                )
                if cached is not None:
                    return cached
            return self._predict_completion_step_jit(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                temperature=temperature,
                rng=rng,
            )

        if _model_supports_cache_kwargs(model):
            cached = self._predict_completion_with_cache(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                temperature=temperature,
                rng=rng,
            )
            if cached is not None:
                return cached

        return self._predict_completion_without_cache(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            temperature=temperature,
            rng=rng,
        )

    def _predict_completion_without_cache(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt: np.ndarray,
        tokenizer,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        xs = np.full((1, self.n_seq), self.pad_token_id, dtype=np.int32)
        xs[0, : prompt.shape[0]] = prompt

        cursor = prompt.shape[0] - 1
        out: list[int] = []
        for _ in range(self.max_completion_len):
            logits = np.asarray(model(xs))
            tok_logits = _select_ar_token_logits(logits, cursor)
            next_tok = _sample_token_from_logits(
                tok_logits,
                temperature=temperature,
                rng=rng,
            )

            next_idx = cursor + 1
            if next_idx >= self.n_seq:
                break

            xs[0, next_idx] = int(next_tok)
            out.append(int(next_tok))
            cursor = next_idx

            if next_tok == int(tokenizer.eot_token_id):
                break
        return np.asarray(out, dtype=np.int32)

    def _predict_completion_step_jit(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt: np.ndarray,
        tokenizer,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        xs = jnp.full((1, self.n_seq), self.pad_token_id, dtype=jnp.int32)
        xs = xs.at[0, : prompt.shape[0]].set(jnp.asarray(prompt, dtype=jnp.int32))

        prompt_len = int(prompt.shape[0])
        max_steps = min(self.max_completion_len, self.n_seq - prompt_len)
        if max_steps <= 0:
            return np.asarray([], dtype=np.int32)

        decode_fn = self._get_noncache_decode_fn_for_model(model)
        key = _jax_key_from_numpy_rng(rng)
        out_tokens, steps = decode_fn(
            xs,
            jnp.asarray(prompt_len - 1, dtype=jnp.int32),
            jnp.asarray(max_steps, dtype=jnp.int32),
            jnp.asarray(int(tokenizer.eot_token_id), dtype=jnp.int32),
            jnp.asarray(float(temperature), dtype=jnp.float32),
            key,
        )
        steps_int = int(np.asarray(steps))
        return np.asarray(out_tokens[:steps_int], dtype=np.int32)

    def _get_noncache_decode_fn_for_model(
        self,
        model: Callable[[np.ndarray], Any],
    ) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
        key = _resolve_decode_cache_key(model)
        cached = self._noncache_decode_fn_cache.get(key)
        if cached is not None:
            return cached

        n_seq = self.n_seq
        max_completion_len = self.max_completion_len

        @nnx.jit
        def _decode(xs, cursor, max_steps, eot_token_id, temperature, rng_key):
            tokens = jnp.zeros((max_completion_len,), dtype=jnp.int32)
            state = (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                xs,
                cursor,
                tokens,
                rng_key,
            )

            def _cond_fn(loop_state):
                step_idx, done, *_ = loop_state
                return jnp.logical_and(step_idx < max_steps, jnp.logical_not(done))

            def _body_fn(loop_state):
                step_idx, _, xs_state, cursor_state, out_tokens, key_state = loop_state
                key_state, sample_key = jax.random.split(key_state)
                logits = model(xs_state)
                tok_logits = _select_ar_token_logits_jax(logits, cursor_state)
                next_tok = _sample_token_from_logits_jax(
                    tok_logits,
                    temperature=temperature,
                    key=sample_key,
                )
                out_tokens = out_tokens.at[step_idx].set(next_tok)

                next_cursor = cursor_state + 1
                xs_state = xs_state.at[0, next_cursor].set(next_tok)
                done = jnp.logical_or(
                    next_tok == eot_token_id,
                    jnp.logical_or(next_cursor >= (n_seq - 1), (step_idx + 1) >= max_steps),
                )

                return (
                    step_idx + 1,
                    done,
                    xs_state,
                    next_cursor,
                    out_tokens,
                    key_state,
                )

            steps, _, _, _, out_tokens, _ = jax.lax.while_loop(_cond_fn, _body_fn, state)
            return out_tokens, steps

        self._noncache_decode_fn_cache[key] = _decode
        return _decode

    def _predict_completion_with_cache_jit(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt: np.ndarray,
        tokenizer,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray | None:
        try:
            logits, cache = _call_cached_model_jax(
                model=model,
                batch_tokens=jnp.asarray(prompt[None, :], dtype=jnp.int32),
                cache=None,
            )
            tok_logits = _select_last_ar_token_logits_jax(logits)
            decode_fn = self._get_cached_decode_fn_for_model(model)
        except TypeError:
            return None

        prompt_len = int(prompt.shape[0])
        max_steps = min(self.max_completion_len, self.n_seq - prompt_len)
        if max_steps <= 0:
            return np.asarray([], dtype=np.int32)

        try:
            key = _jax_key_from_numpy_rng(rng)
            out_tokens, steps = decode_fn(
                cache,
                tok_logits,
                key,
                jnp.asarray(max_steps, dtype=jnp.int32),
                jnp.asarray(int(tokenizer.eot_token_id), dtype=jnp.int32),
                jnp.asarray(float(temperature), dtype=jnp.float32),
            )
        except Exception:
            return None

        steps_int = int(np.asarray(steps))
        return np.asarray(out_tokens[:steps_int], dtype=np.int32)

    def _get_cached_decode_fn_for_model(
        self,
        model: Callable[[np.ndarray], Any],
    ) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
        key = _resolve_decode_cache_key(model)
        cached = self._cached_decode_fn_cache.get(key)
        if cached is not None:
            return cached

        max_completion_len = self.max_completion_len

        @nnx.jit
        def _decode(cache, tok_logits, rng_key, max_steps, eot_token_id, temperature):
            tokens = jnp.zeros((max_completion_len,), dtype=jnp.int32)
            state = (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                cache,
                tok_logits,
                tokens,
                rng_key,
            )

            def _cond_fn(loop_state):
                step_idx, done, *_ = loop_state
                return jnp.logical_and(step_idx < max_steps, jnp.logical_not(done))

            def _body_fn(loop_state):
                step_idx, _, cache_state, step_logits, out_tokens, key_state = loop_state
                key_state, sample_key = jax.random.split(key_state)
                next_tok = _sample_token_from_logits_jax(
                    step_logits,
                    temperature=temperature,
                    key=sample_key,
                )
                out_tokens = out_tokens.at[step_idx].set(next_tok)

                next_idx = step_idx + 1
                done = jnp.logical_or(next_tok == eot_token_id, next_idx >= max_steps)

                def _advance(arg):
                    cache_in, tok_in = arg
                    logits_out, next_cache = model(
                        jnp.asarray([[tok_in]], dtype=jnp.int32),
                        cache=cache_in,
                        return_cache=True,
                    )
                    return next_cache, _select_last_ar_token_logits_jax(logits_out)

                next_cache_state, next_logits = jax.lax.cond(
                    done,
                    lambda _: (cache_state, step_logits),
                    _advance,
                    operand=(cache_state, next_tok),
                )
                return (
                    next_idx,
                    done,
                    next_cache_state,
                    next_logits,
                    out_tokens,
                    key_state,
                )

            steps, _, _, _, out_tokens, _ = jax.lax.while_loop(_cond_fn, _body_fn, state)
            return out_tokens, steps

        self._cached_decode_fn_cache[key] = _decode
        return _decode

    def _predict_completion_with_cache(
        self,
        *,
        model: Callable[[np.ndarray], Any],
        prompt: np.ndarray,
        tokenizer,
        temperature: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray | None:
        try:
            logits, cache = _call_cached_model(
                model=model,
                batch_tokens=prompt[None, :],
                cache=None,
            )
        except TypeError:
            return None

        current_len = int(prompt.shape[0])
        out: list[int] = []

        for _ in range(self.max_completion_len):
            tok_logits = _select_last_ar_token_logits(logits)
            next_tok = _sample_token_from_logits(
                tok_logits,
                temperature=temperature,
                rng=rng,
            )

            if current_len + 1 > self.n_seq:
                break

            out.append(int(next_tok))
            current_len += 1

            if next_tok == int(tokenizer.eot_token_id):
                break

            logits, cache = _call_cached_model(
                model=model,
                batch_tokens=np.asarray([[next_tok]], dtype=np.int32),
                cache=cache,
            )

        return np.asarray(out, dtype=np.int32)


def make_model_callable(optimizer, *, to_numpy: bool = True):
    """Wrap ``optimizer.model`` with optional cache forwarding and dtype normalization."""
    model_obj = optimizer.model
    try:
        sig = inspect.signature(model_obj)
    except (TypeError, ValueError):
        sig = None

    supports_cache = False
    if sig is not None:
        supports_cache = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        ) or ("cache" in sig.parameters and "return_cache" in sig.parameters)
    model_id = _random_model_instance_id(model_obj)
    decode_cache_key = (
        f"model={model_id}|supports_cache={int(bool(supports_cache))}|to_numpy={int(bool(to_numpy))}"
    )

    if supports_cache:
        def _model(batch_tokens: np.ndarray, *, cache=None, return_cache: bool = False):
            out = model_obj(
                jnp.asarray(batch_tokens, dtype=jnp.int32),
                cache=cache,
                return_cache=return_cache,
            )
            if return_cache:
                logits, new_cache = out
                if to_numpy:
                    return np.asarray(logits), new_cache
                return logits, new_cache
            if to_numpy:
                return np.asarray(out)
            return out
        setattr(_model, _DECODE_CACHE_KEY_ATTR, decode_cache_key)
        return _model

    def _model(batch_tokens: np.ndarray):
        out = model_obj(jnp.asarray(batch_tokens, dtype=jnp.int32))
        if to_numpy:
            return np.asarray(out)
        return out

    setattr(_model, _DECODE_CACHE_KEY_ATTR, decode_cache_key)
    return _model


def _build_prompt_only_inputs_single(
    *,
    prompt_tokens: list[int] | np.ndarray,
    n_seq: int,
    sep_token_id: int,
    pad_token_id: int,
) -> np.ndarray:
    prompt = np.asarray(prompt_tokens, dtype=np.int32)
    if prompt.ndim != 1:
        raise ValueError(f"Prompt must be 1D, got {prompt.shape}")
    if prompt.shape[0] > n_seq:
        raise ValueError(f"Prompt length {prompt.shape[0]} exceeds n_seq={n_seq}")

    out = np.full((n_seq,), int(pad_token_id), dtype=np.int32)
    out[: prompt.shape[0]] = prompt
    sep_hits = np.where(out == int(sep_token_id))[0]
    if sep_hits.size == 0:
        raise ValueError("Prompt must include SEP token.")
    sep_idx = int(sep_hits[0])
    out[sep_idx + 1 :] = int(pad_token_id)
    return out


def _normalize_completion_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(
                f"Expected completion logits with batch size 1, got {logits.shape}"
            )
        return logits[0]
    raise ValueError(f"Expected completion logits with ndim 2 or 3, got {logits.shape}")


def _select_ar_token_logits(logits: np.ndarray, cursor: int) -> np.ndarray:
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError(
                f"Expected AR logits with batch size 1, got {logits.shape}"
            )
        return logits[0]
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(
                f"Expected AR logits with batch size 1, got {logits.shape}"
            )
        if cursor < 0 or cursor >= logits.shape[1]:
            raise ValueError(f"Cursor index {cursor} out of bounds for {logits.shape}")
        return logits[0, cursor]
    raise ValueError(f"Unsupported AR logits shape: {logits.shape}")


def _select_ar_token_logits_jax(logits: jnp.ndarray, cursor: jnp.ndarray) -> jnp.ndarray:
    if logits.ndim == 2:
        return logits[0]
    if logits.ndim == 3:
        return logits[0, cursor]
    raise ValueError(f"Unsupported AR logits shape: {logits.shape}")


def _select_last_ar_token_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 2:
        return _select_ar_token_logits(logits, 0)
    if logits.ndim == 3:
        return _select_ar_token_logits(logits, logits.shape[1] - 1)
    raise ValueError(f"Unsupported AR logits shape: {logits.shape}")


def _select_last_ar_token_logits_jax(logits: jnp.ndarray) -> jnp.ndarray:
    if logits.ndim == 2:
        return logits[0]
    if logits.ndim == 3:
        return logits[0, logits.shape[1] - 1]
    raise ValueError(f"Unsupported AR logits shape: {logits.shape}")


def _model_supports_cache_kwargs(model: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(model)
    except (TypeError, ValueError):
        return False

    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    ):
        return True
    return "cache" in sig.parameters and "return_cache" in sig.parameters


def _call_cached_model(
    *,
    model: Callable[..., Any],
    batch_tokens: np.ndarray,
    cache: Any,
) -> tuple[np.ndarray, Any]:
    out = model(batch_tokens, cache=cache, return_cache=True)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(
            "Cache-enabled model must return (logits, cache) when return_cache=True."
        )
    logits, new_cache = out
    return np.asarray(logits), new_cache


def _call_cached_model_jax(
    *,
    model: Callable[..., Any],
    batch_tokens: jnp.ndarray,
    cache: Any,
) -> tuple[jnp.ndarray, Any]:
    out = model(batch_tokens, cache=cache, return_cache=True)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(
            "Cache-enabled model must return (logits, cache) when return_cache=True."
        )
    logits, new_cache = out
    return jnp.asarray(logits), new_cache


def _jax_key_from_numpy_rng(rng: np.random.Generator | None) -> jax.Array:
    if rng is None:
        rng = np.random.default_rng()
    seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return jax.random.PRNGKey(seed)


def _sample_token_from_logits_jax(
    logits: jnp.ndarray,
    *,
    temperature: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    logits = jnp.asarray(logits)

    def _greedy(_):
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    def _sample(_):
        safe_temp = jnp.maximum(temperature, jnp.asarray(1e-6, dtype=logits.dtype))
        scaled = logits / safe_temp
        return jax.random.categorical(key, scaled, axis=-1).astype(jnp.int32)

    return jax.lax.cond(temperature <= 0, _greedy, _sample, operand=None)


def _sample_token_from_logits(
    logits: np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator | None,
) -> int:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {logits.shape}")
    if logits.shape[0] < 1:
        raise ValueError("Cannot sample from empty logits.")

    if temperature <= 0:
        return int(np.argmax(logits))

    if rng is None:
        rng = np.random.default_rng()

    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        return int(np.argmax(logits))
    probs = probs / probs_sum
    return int(rng.choice(np.arange(logits.shape[0]), p=probs))
