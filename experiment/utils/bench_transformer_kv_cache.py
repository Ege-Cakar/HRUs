"""Benchmark autoregressive decode adapter modes across model families.

Transformer variants benchmarked across:
1) mixed precision on/off (`use_bf16`)
2) flash attention on/off (cuDNN path toggle)

For each Transformer variant:
- KV cache + no JIT step
- No KV cache + JIT step
- KV cache + JIT step

Mamba2 Bonsai:
- No cache + JIT step
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from model.eval_adapters import AutoregressiveLogitsAdapter
from model.ssm_bonsai import Mamba2BonsaiConfig
from model.transformer import TransformerConfig


class _TokenizerStub:
    def __init__(self, *, start_token_id: int, eot_token_id: int):
        self.start_token_id = int(start_token_id)
        self.eot_token_id = int(eot_token_id)


def _decode_with_adapter(
    *,
    adapter: AutoregressiveLogitsAdapter,
    model,
    prompt_tokens: np.ndarray,
    tokenizer: _TokenizerStub,
) -> int:
    out = adapter.predict_completion(
        model=model,
        prompt_tokens=prompt_tokens,
        tokenizer=tokenizer,
        temperature=0.0,
        rng=None,
    )
    return int(np.asarray(out).shape[0])


def _time_runs(run_fn, *, warmup_runs: int, timed_runs: int) -> list[float]:
    for _ in range(warmup_runs):
        run_fn()

    times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        run_fn()
        times.append(time.perf_counter() - t0)
    return times


def _set_transformer_flash_attention(model, *, enabled: bool) -> bool:
    """Toggle whether Transformer attention may use cuDNN flash attention."""
    uses_bf16 = getattr(model, "compute_dtype", None) == jnp.bfloat16
    cudnn_allowed = bool(enabled and uses_bf16 and jax.default_backend() == "gpu")
    for block in model.blocks:
        block.attn._prefer_cudnn = cudnn_allowed
    return cudnn_allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vocab", type=int, default=512)
    parser.add_argument("--n-seq", type=int, default=2048)
    parser.add_argument("--prompt-len", type=int, default=1536)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--pos-encoding", choices=("none", "absolute", "rope"), default="rope")
    parser.add_argument("--bonsai-scan-chunk-len", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.n_vocab <= 3:
        raise ValueError(f"n-vocab must be > 3, got {args.n_vocab}")
    if args.prompt_len < 1:
        raise ValueError(f"prompt-len must be >= 1, got {args.prompt_len}")
    if args.prompt_len >= args.n_seq:
        raise ValueError(
            f"prompt-len ({args.prompt_len}) must be < n-seq ({args.n_seq})"
        )
    if args.max_new < 1:
        raise ValueError(f"max-new must be >= 1, got {args.max_new}")

    max_possible_new = int(args.n_seq - args.prompt_len)
    if args.max_new > max_possible_new:
        raise ValueError(
            f"max-new ({args.max_new}) exceeds capacity n-seq-prompt-len ({max_possible_new})"
        )

    print("JAX backend:", jax.default_backend())
    print("Device:", jax.devices()[0])

    mamba2 = Mamba2BonsaiConfig(
        n_vocab=args.n_vocab,
        n_seq=args.n_seq,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_heads=args.n_heads,
        n_out=args.n_vocab,
        n_pred_tokens=1,
        output_mode="full_sequence",
        scan_chunk_len=args.bonsai_scan_chunk_len,
    ).to_model(rngs=nnx.Rngs(args.seed + 1))

    rng = np.random.default_rng(args.seed)
    prompt = rng.integers(2, args.n_vocab, size=(args.prompt_len,), dtype=np.int32)
    prompt[-1] = 1  # Ensure a START token is present for adapter prompt validation.
    tokenizer = _TokenizerStub(start_token_id=1, eot_token_id=-1)

    def model_mamba2_no_cache(batch_tokens: np.ndarray):
        return mamba2(jnp.asarray(batch_tokens, dtype=jnp.int32))

    cache_no_jit = AutoregressiveLogitsAdapter(
        n_seq=args.n_seq,
        max_completion_len=args.max_new,
        pad_token_id=0,
        jit_step=False,
    )
    no_cache_jit = AutoregressiveLogitsAdapter(
        n_seq=args.n_seq,
        max_completion_len=args.max_new,
        pad_token_id=0,
        jit_step=True,
    )
    cache_jit = AutoregressiveLogitsAdapter(
        n_seq=args.n_seq,
        max_completion_len=args.max_new,
        pad_token_id=0,
        jit_step=True,
    )
    mamba2_no_cache_jit = AutoregressiveLogitsAdapter(
        n_seq=args.n_seq,
        max_completion_len=args.max_new,
        pad_token_id=0,
        jit_step=True,
    )

    transformer_variants = [
        {"name": "transformer_bf16_flash", "use_bf16": True, "enable_flash": True},
        {"name": "transformer_bf16_no_flash", "use_bf16": True, "enable_flash": False},
        {"name": "transformer_fp32_flash", "use_bf16": False, "enable_flash": True},
        {"name": "transformer_fp32_no_flash", "use_bf16": False, "enable_flash": False},
    ]

    print("\nConfig:")
    print(
        f"  n_seq={args.n_seq} prompt_len={args.prompt_len} max_new={args.max_new} "
        f"layers={args.n_layers} hidden={args.n_hidden} heads={args.n_heads} pos={args.pos_encoding}"
    )
    print(f"  bonsai_scan_chunk_len={args.bonsai_scan_chunk_len}")
    print("\nDecode timing (raw per-run seconds):")

    for variant_idx, variant in enumerate(transformer_variants):
        config = TransformerConfig(
            n_vocab=args.n_vocab,
            n_seq=args.n_seq,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            n_heads=args.n_heads,
            n_out=args.n_vocab,
            n_pred_tokens=1,
            output_mode="full_sequence",
            pos_encoding=args.pos_encoding,
            dropout_rate=0.0,
            use_bf16=variant["use_bf16"],
        )
        model = config.to_model(rngs=nnx.Rngs(args.seed + variant_idx))
        cudnn_enabled = _set_transformer_flash_attention(
            model, enabled=variant["enable_flash"]
        )

        def model_with_cache(batch_tokens: np.ndarray, *, cache=None, return_cache: bool = False):
            out = model(
                jnp.asarray(batch_tokens, dtype=jnp.int32),
                cache=cache,
                return_cache=return_cache,
            )
            return out

        def model_no_cache(batch_tokens: np.ndarray):
            return model(jnp.asarray(batch_tokens, dtype=jnp.int32))

        cache_steps = _decode_with_adapter(
            adapter=cache_no_jit,
            model=model_with_cache,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
        )
        jit_steps = _decode_with_adapter(
            adapter=no_cache_jit,
            model=model_no_cache,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
        )
        cache_jit_steps = _decode_with_adapter(
            adapter=cache_jit,
            model=model_with_cache,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
        )

        cache_times = _time_runs(
            lambda: _decode_with_adapter(
                adapter=cache_no_jit,
                model=model_with_cache,
                prompt_tokens=prompt,
                tokenizer=tokenizer,
            ),
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        jit_times = _time_runs(
            lambda: _decode_with_adapter(
                adapter=no_cache_jit,
                model=model_no_cache,
                prompt_tokens=prompt,
                tokenizer=tokenizer,
            ),
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        cache_jit_times = _time_runs(
            lambda: _decode_with_adapter(
                adapter=cache_jit,
                model=model_with_cache,
                prompt_tokens=prompt,
                tokenizer=tokenizer,
            ),
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )

        print(
            f"  {variant['name']}: use_bf16={variant['use_bf16']} "
            f"flash_requested={variant['enable_flash']} cudnn_enabled={cudnn_enabled}"
        )
        print(f"    cache_no_jit_steps: {cache_steps}")
        print(f"    no_cache_jit_steps: {jit_steps}")
        print(f"    cache_jit_steps: {cache_jit_steps}")
        print("    cache_no_jit_times_s:", [round(t, 3) for t in cache_times])
        print("    no_cache_jit_times_s:", [round(t, 3) for t in jit_times])
        print("    cache_jit_times_s:", [round(t, 3) for t in cache_jit_times])

    mamba2_jit_steps = _decode_with_adapter(
        adapter=mamba2_no_cache_jit,
        model=model_mamba2_no_cache,
        prompt_tokens=prompt,
        tokenizer=tokenizer,
    )
    mamba2_jit_times = _time_runs(
        lambda: _decode_with_adapter(
            adapter=mamba2_no_cache_jit,
            model=model_mamba2_no_cache,
            prompt_tokens=prompt,
            tokenizer=tokenizer,
        ),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )
    print("  mamba2_no_cache_jit")
    print(f"    no_cache_jit_steps: {mamba2_jit_steps}")
    print("    no_cache_jit_times_s:", [round(t, 3) for t in mamba2_jit_times])


if __name__ == "__main__":
    main()
