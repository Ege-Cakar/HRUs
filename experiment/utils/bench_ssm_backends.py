"""Benchmark SSM scan backends under train_step workload."""

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

from model.ssm import Mamba2Config, MambaConfig
from train import create_optimizer, parse_loss_name, train_step


def _bench_backend(
    *,
    model_family: str,
    backend: str,
    batch_size: int,
    n_seq: int,
    n_vocab: int,
    n_layers: int,
    n_hidden: int,
    n_heads: int,
    d_state: int,
    scan_chunk_len: int,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float | str]:
    if model_family == "mamba2":
        config = Mamba2Config(
            n_vocab=n_vocab,
            n_seq=n_seq,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_heads=n_heads,
            n_out=n_vocab,
            n_pred_tokens=1,
            output_mode="full_sequence",
            d_state=d_state,
            scan_backend=backend,
            scan_chunk_len=scan_chunk_len,
        )
    else:
        config = MambaConfig(
            n_vocab=n_vocab,
            n_seq=n_seq,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_out=n_vocab,
            n_pred_tokens=1,
            output_mode="full_sequence",
            d_state=d_state,
            scan_backend=backend,
            scan_chunk_len=scan_chunk_len,
        )
    optimizer = create_optimizer(
        config.to_model(rngs=nnx.Rngs(0)),
        lr=3e-4,
        optim=optax.adamw,
    )
    loss_func = parse_loss_name("ce_mask")

    xs = jnp.ones((batch_size, n_seq), dtype=jnp.int32)
    ys = jnp.ones((batch_size, n_seq), dtype=jnp.int32)

    for _ in range(warmup_steps):
        loss_val = train_step(optimizer, (xs, ys), loss_func)
        jax.block_until_ready(loss_val)

    times = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        loss_val = train_step(optimizer, (xs, ys), loss_func)
        jax.block_until_ready(loss_val)
        times.append(time.perf_counter() - t0)

    return {
        "backend": backend,
        "mean_step_s": float(np.mean(times)),
        "p95_step_s": float(np.percentile(times, 95)),
        "steps_per_s": float(1.0 / np.mean(times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-family", choices=("mamba", "mamba2"), default="mamba2")
    parser.add_argument("--n-seq", type=int, default=64)
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--scan-chunk-len", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--timed-steps", type=int, default=16)
    parser.add_argument("--speed-gate", type=float, default=1.5)
    args = parser.parse_args()

    print("JAX backend:", jax.default_backend())
    print("Model family:", args.model_family)
    print("Speed gate:", args.speed_gate)
    results = []
    for backend in ("reference", "auto", "pallas"):
        res = _bench_backend(
            model_family=args.model_family,
            backend=backend,
            batch_size=args.batch_size,
            n_seq=args.n_seq,
            n_vocab=args.n_vocab,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            n_heads=args.n_heads,
            d_state=args.d_state,
            scan_chunk_len=args.scan_chunk_len,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        results.append(res)
        print(
            f"{res['backend']}: mean={res['mean_step_s']:.4f}s "
            f"p95={res['p95_step_s']:.4f}s steps/s={res['steps_per_s']:.2f}"
        )

    ref_steps = next(float(r["steps_per_s"]) for r in results if r["backend"] == "reference")
    pallas_steps = next(float(r["steps_per_s"]) for r in results if r["backend"] == "pallas")
    speedup = pallas_steps / ref_steps if ref_steps > 0 else float("nan")
    print(f"pallas/reference speedup: {speedup:.2f}x")
    print(
        "speed gate:",
        "PASS" if speedup >= args.speed_gate else "FAIL",
        f"(threshold {args.speed_gate:.2f}x)",
    )


if __name__ == "__main__":
    main()
