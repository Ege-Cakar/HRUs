"""Benchmark original Mamba2 vs Bonsai-adapted Mamba2 on train and inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from model.ssm import Mamba2Config
from model.ssm_bonsai import Mamba2BonsaiConfig, create_empty_cache
from train import create_optimizer, parse_loss_name, train_step


def _timed_loop(step_fn, *, warmup_steps: int, timed_steps: int):
    for _ in range(warmup_steps):
        out = step_fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        out = step_fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return {
        "mean_step_s": float(np.mean(times)),
        "p95_step_s": float(np.percentile(times, 95)),
        "steps_per_s": float(1.0 / np.mean(times)),
    }


def _make_original_config(
    *,
    n_vocab: int,
    n_seq: int,
    n_layers: int,
    n_hidden: int,
    n_heads: int,
    d_state: int,
    scan_chunk_len: int,
    output_mode: str,
    n_out: int,
    scan_backend: str,
) -> Mamba2Config:
    return Mamba2Config(
        n_vocab=n_vocab,
        n_seq=n_seq,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_heads=n_heads,
        n_out=n_out,
        n_pred_tokens=1,
        output_mode=output_mode,
        d_state=d_state,
        scan_chunk_len=scan_chunk_len,
        scan_backend=scan_backend,
    )


def _make_bonsai_config(
    *,
    n_vocab: int,
    n_seq: int,
    n_layers: int,
    n_hidden: int,
    n_heads: int,
    d_state: int,
    scan_chunk_len: int,
    output_mode: str,
    n_out: int,
) -> Mamba2BonsaiConfig:
    return Mamba2BonsaiConfig(
        n_vocab=n_vocab,
        n_seq=n_seq,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_heads=n_heads,
        n_out=n_out,
        n_pred_tokens=1,
        output_mode=output_mode,
        d_state=d_state,
        scan_chunk_len=scan_chunk_len,
    )


def _bench_training(
    *,
    model_name: str,
    config,
    batch_size: int,
    n_seq: int,
    train_warmup_steps: int,
    train_timed_steps: int,
) -> dict[str, float | str]:
    optimizer = create_optimizer(
        config.to_model(rngs=nnx.Rngs(0)),
        lr=3e-4,
        optim=optax.adamw,
    )
    loss_func = parse_loss_name("ce_mask")
    xs = jnp.ones((batch_size, n_seq), dtype=jnp.int32)
    ys = jnp.ones((batch_size, n_seq), dtype=jnp.int32)

    def _step():
        return train_step(optimizer, (xs, ys), loss_func)

    stats = _timed_loop(
        _step,
        warmup_steps=train_warmup_steps,
        timed_steps=train_timed_steps,
    )
    return {"impl": model_name, "mode": "training"} | stats


def _bench_inference_original(
    *,
    config: Mamba2Config,
    batch_size: int,
    n_seq: int,
    infer_warmup_steps: int,
    infer_timed_steps: int,
) -> dict[str, float | str]:
    model = config.to_model(rngs=nnx.Rngs(0))
    tokens = jnp.ones((batch_size, n_seq), dtype=jnp.int32)

    @nnx.jit
    def _infer(mod, x):
        return mod(x)

    def _step():
        return _infer(model, tokens)

    stats = _timed_loop(
        _step,
        warmup_steps=infer_warmup_steps,
        timed_steps=infer_timed_steps,
    )
    return {"impl": "original", "mode": "inference_no_cache"} | stats


def _bench_inference_bonsai(
    *,
    config: Mamba2BonsaiConfig,
    batch_size: int,
    n_seq: int,
    infer_warmup_steps: int,
    infer_timed_steps: int,
) -> dict[str, float | str]:
    model = config.to_model(rngs=nnx.Rngs(0))
    prefill = jnp.ones((batch_size, n_seq), dtype=jnp.int32)
    one_token = jnp.ones((batch_size, 1), dtype=jnp.int32)
    cache = create_empty_cache(config, batch_size=batch_size, dtype=jnp.float32)
    _, cache = model(prefill, cache=cache, return_cache=True)

    @nnx.jit
    def _infer_cached(mod, x, kv):
        return mod(x, cache=kv, return_cache=True)

    for _ in range(infer_warmup_steps):
        logits, cache = _infer_cached(model, one_token, cache)
        jax.block_until_ready(logits)

    times = []
    for _ in range(infer_timed_steps):
        t0 = time.perf_counter()
        logits, cache = _infer_cached(model, one_token, cache)
        jax.block_until_ready(logits)
        times.append(time.perf_counter() - t0)

    return {
        "impl": "bonsai",
        "mode": "inference_cached",
        "mean_step_s": float(np.mean(times)),
        "p95_step_s": float(np.percentile(times, 95)),
        "steps_per_s": float(1.0 / np.mean(times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-seq", type=int, default=64)
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--scan-chunk-len", type=int, default=64)
    parser.add_argument("--original-backend", choices=("reference", "auto", "pallas"), default="reference")
    parser.add_argument("--train-warmup-steps", type=int, default=4)
    parser.add_argument("--train-timed-steps", type=int, default=16)
    parser.add_argument("--infer-warmup-steps", type=int, default=8)
    parser.add_argument("--infer-timed-steps", type=int, default=64)
    args = parser.parse_args()

    print("JAX backend:", jax.default_backend())
    print("Original Mamba2 backend:", args.original_backend)

    original_train_cfg = _make_original_config(
        n_vocab=args.n_vocab,
        n_seq=args.n_seq,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        d_state=args.d_state,
        scan_chunk_len=args.scan_chunk_len,
        output_mode="full_sequence",
        n_out=args.n_vocab,
        scan_backend=args.original_backend,
    )
    bonsai_train_cfg = _make_bonsai_config(
        n_vocab=args.n_vocab,
        n_seq=args.n_seq,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        d_state=args.d_state,
        scan_chunk_len=args.scan_chunk_len,
        output_mode="full_sequence",
        n_out=args.n_vocab,
    )

    original_infer_cfg = _make_original_config(
        n_vocab=args.n_vocab,
        n_seq=args.n_seq,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        d_state=args.d_state,
        scan_chunk_len=args.scan_chunk_len,
        output_mode="last_token",
        n_out=args.n_vocab,
        scan_backend=args.original_backend,
    )
    bonsai_infer_cfg = _make_bonsai_config(
        n_vocab=args.n_vocab,
        n_seq=args.n_seq,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        d_state=args.d_state,
        scan_chunk_len=args.scan_chunk_len,
        output_mode="last_token",
        n_out=args.n_vocab,
    )

    results = []
    results.append(
        _bench_training(
            model_name="original",
            config=original_train_cfg,
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            train_warmup_steps=args.train_warmup_steps,
            train_timed_steps=args.train_timed_steps,
        )
    )
    results.append(
        _bench_training(
            model_name="bonsai",
            config=bonsai_train_cfg,
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            train_warmup_steps=args.train_warmup_steps,
            train_timed_steps=args.train_timed_steps,
        )
    )
    results.append(
        _bench_inference_original(
            config=original_infer_cfg,
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            infer_warmup_steps=args.infer_warmup_steps,
            infer_timed_steps=args.infer_timed_steps,
        )
    )
    results.append(
        _bench_inference_bonsai(
            config=bonsai_infer_cfg,
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            infer_warmup_steps=args.infer_warmup_steps,
            infer_timed_steps=args.infer_timed_steps,
        )
    )

    for res in results:
        print(
            f"{res['impl']:<8} {res['mode']:<19} mean={res['mean_step_s']:.6f}s "
            f"p95={res['p95_step_s']:.6f}s steps/s={res['steps_per_s']:.2f}"
        )

    train_orig = next(float(x["steps_per_s"]) for x in results if x["impl"] == "original" and x["mode"] == "training")
    train_bonsai = next(float(x["steps_per_s"]) for x in results if x["impl"] == "bonsai" and x["mode"] == "training")
    infer_orig = next(
        float(x["steps_per_s"]) for x in results if x["impl"] == "original" and x["mode"] == "inference_no_cache"
    )
    infer_bonsai = next(
        float(x["steps_per_s"]) for x in results if x["impl"] == "bonsai" and x["mode"] == "inference_cached"
    )

    print(f"training speedup (bonsai/original): {train_bonsai / train_orig:.2f}x")
    print(f"inference speedup (bonsai/original): {infer_bonsai / infer_orig:.2f}x")


if __name__ == "__main__":
    main()
