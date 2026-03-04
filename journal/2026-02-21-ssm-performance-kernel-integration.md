# 2026-02-21 - SSM Performance Investigation + Kernel Integration Pass

## Task Summary
Investigate why SSM (Mamba/Mamba2 in `model/ssm.py`) trains much slower than Transformer/Mixer on LayerTask, implement immediate speed/debug fixes, and prototype a specialized-kernel backend path with safe fallback behavior.

## Findings
- Two distinct slowdown sources were confirmed:
  1. **Shape-driven JIT recompilation** in autoregressive LayerTask batches (variable sequence length per batch).
  2. **Intrinsic SSM compute cost** of the current reference scan implementation versus Transformer/Mixer at fixed shape.
- Local timing (CPU fallback env): Mamba remained significantly slower than Transformer/Mixer even at fixed shape, so recompilation was not the whole story.
- On GPU Pallas path, two concrete failures occurred during initial integration:
  - `program_id` assertion during AD/JVP through `pallas_call`.
  - Mosaic lowering failure from shared memory overuse:
    `smem_bytes=13126664 > max_smem_bytes=166912`.

## Changes Implemented

### 1) LayerTask fixed-length control (recompile mitigation)
File: `task/layer.py`
- Added:
  - `fixed_length_mode`: `"batch_max" | "global_max"`.
  - `fixed_length_n_seq`: optional explicit fixed length.
- Implemented global-length derivation:
  - Offline: from metadata `max_seq`.
  - Online: from rule bank/tokenizer max prompt + completion length.
- Added autoregressive-only global padding path in `__next__`.

Files: `tests/test_layer_task.py`
- Added coverage for:
  - offline global max behavior,
  - online stable-length behavior,
  - explicit override,
  - invalid mode rejection,
  - all-at-once non-impact.

### 2) SSM backend abstraction + kernel module
Files:
- `model/ssm.py`
- `model/ssm_kernels.py` (new)

Added config options to both Mamba configs:
- `scan_backend`: `"reference" | "pallas" | "auto"` (default `"reference"`)
- `scan_chunk_len`
- `scan_debug_checks`

Refactored `_scan` in both blocks to dispatch through `model/ssm_kernels.py`.

`model/ssm_kernels.py` now contains:
- backend resolver and warning-once fallback behavior,
- chunked reference selective scan implementations,
- initial Pallas implementations for Mamba and Mamba2,
- AD-safe custom VJP wrappers:
  - Pallas used for forward,
  - reference path used for backward (avoids `program_id` AD assert),
- pre-lowering shared-memory guard:
  - estimates naive Pallas memory footprint,
  - auto-falls back to reference if estimate exceeds budget,
  - budget configurable via env var:
    - `MATH_PALLAS_MAX_SMEM_BYTES` (default `166912`).

### 3) Dependency + benchmark tooling
File: `pyproject.toml`
- Added dependency group:
  - `[dependency-groups].ssm-kernels = ["jax-triton>=0.3.0"]`

File: `experiment/utils/bench_ssm_backends.py` (new)
- Added benchmark script comparing `reference`, `auto`, `pallas` backends under real `train_step` workload.

### 4) Sweep wiring for shadow benchmarking
Files:
- `experiment/remote/5_arch_sweep/run.py`
- `experiment/remote/6_layer_sweep/run.py`

Added `MAMBA_SCAN_BACKENDS` list (default `['reference']`) and threaded `scan_backend` into Mamba case generation + metadata, so backend A/B can be done via config list rather than code edits.

## Current Behavior Status
- On environments without usable Pallas GPU runtime: backend falls back to reference with warning.
- On GPU where Pallas path would exceed shared memory with current naive kernel form: now falls back to reference instead of crashing.
- This means robustness is improved, but **large-shape speedup is not yet achieved** because fallback will often trigger with current naive kernel layout.

## Tests and Validation Run
- `tests/test_layer_task.py`: passed after fixed-length changes.
- `tests/test_ssm.py`: passed after backend integration.
- `tests/test_ssm_kernels.py`: added and passing.
- `tests/test_train.py`: passing (regression sanity).
- `experiment/utils/bench_ssm_backends.py`: runs end-to-end; currently reports fallback behavior in local CPU env.

## Why Pallas Is Still Not Fast Yet
Current Pallas kernels are correctness-oriented and naive about memory layout. Without explicit `BlockSpec` tiling and bounded on-chip buffering, Mosaic lowers kernels that can require too much shared memory at realistic training shapes.

## Next Steps (Priority Order)
1. **Implement tiled Pallas kernels with explicit BlockSpecs**
   - Tile over `(batch, inner)` or `(batch, head, head_dim)` and keep state vectors in registers/shared memory with bounded footprint.
   - Avoid materializing full sequence/state tensors in shared memory.
2. **Add hard runtime logging for backend decisions**
   - Emit per-run summary: requested backend, resolved backend, fallback reason.
3. **GPU benchmark matrix on cluster**
   - Use `bench_ssm_backends.py` across representative shapes from arch/layer sweeps.
   - Record compile time and steady-state tokens/sec.
4. **Promote to `scan_backend='auto'` in sweeps only after speed gates pass**
   - Keep default `reference` until measured wins are consistent.
5. **Optional optimization track**
   - Introduce sequence-length bucketing in autoregressive training to reduce padding waste while preserving low compile churn.

## Resume Commands
- Quick backend benchmark:
  - `./.venv/bin/python experiment/utils/bench_ssm_backends.py --batch-size 64 --n-seq 64 --n-layers 4 --n-hidden 128 --d-state 16 --warmup-steps 8 --timed-steps 40`
- Force/adjust shared-memory fallback threshold:
  - `export MATH_PALLAS_MAX_SMEM_BYTES=166912`
- SSM tests:
  - `./.venv/bin/python -m pytest tests/test_ssm.py tests/test_ssm_kernels.py -v --tb=short`

## Notes
- Full test suite still has unrelated collection issues in environments missing some experiment datasets; SSM/layer-specific tests used for regression confidence in this session.

---

## 2026-02-23 - Mamba2 Tiled Kernel Follow-up (Implementation Pass)

## Task Summary
Implement the Mamba2-first kernel plan: replace naive Mamba2 Pallas scan with tiled kernels, keep reference backward for gradients, improve backend diagnostics/fallback behavior, and validate with tests + benchmark gate.

## Changes Implemented

### 1) Mamba2 tiled Pallas forward path
File: `model/ssm_kernels.py`
- Replaced the old naive Mamba2 Pallas implementation with a two-stage tiled path:
  1. Chunk-summary kernel computes per-chunk affine recurrence summaries `(F_chunk, G_chunk)`.
  2. `jax.lax.associative_scan` composes chunk summaries across time to derive per-chunk initial states.
  3. Chunk-output kernel replays each chunk from the composed initial state to write `y`.
- Kept sequence padding/unpadding internal so non-multiple sequence lengths are supported.
- Wired Mamba2 Pallas calls with explicit `BlockSpec`s and `backend="triton"` to avoid the prior large naive-memory behavior.

### 2) Backward path policy (kept reference)
File: `model/ssm_kernels.py`
- Preserved custom VJP strategy:
  - forward: tiled Pallas kernel
  - backward: reference scan via `jax.vjp`
- Added `scan_chunk_len` as a static nondiff argument for the Mamba2 custom VJP wrapper.

### 3) Backend fallback + diagnostics hardening
File: `model/ssm_kernels.py`
- Added structured warning helper that logs:
  - requested backend
  - resolved backend
  - fallback reason
- Added runtime try/except fallback for Mamba2 Pallas failures to reference.
- Added analogous runtime fallback guard for Mamba1 Pallas (robustness only; no Mamba1 speed work yet).
- Replaced Mamba2 naive smem guard with a tiled-footprint estimate function.

### 4) Benchmark utility improvements
File: `experiment/utils/bench_ssm_backends.py`
- Extended benchmark to support both model families:
  - `--model-family {mamba,mamba2}` (default `mamba2`)
- Added Mamba2-specific knobs:
  - `--n-heads`
  - `--scan-chunk-len`
- Added speed-gate reporting:
  - `--speed-gate` (default `1.5`)
  - prints `pallas/reference speedup` and `PASS/FAIL`.

### 5) Test coverage updates
File: `tests/test_ssm_kernels.py`
- Added Mamba2 chunked parity test for non-multiple sequence lengths.
- Added Mamba2 gradient parity smoke test (`pallas` vs `reference`).
- Added fallback warning + forced-budget fallback test.
- Added tiled smem-estimator behavior test.
- Existing parity tests retained.

## Validation Run
- `./.venv/bin/python -m pytest tests/test_ssm_kernels.py -v --tb=short`
  - **8 passed**
- `./.venv/bin/python -m pytest tests/test_ssm.py -v --tb=short`
  - **33 passed**
- Updated benchmark script executes successfully for both `mamba` and `mamba2`.

## Benchmark Outcome (Local GPU, current implementation)
- Mamba2 speed gate still fails on tested local shapes.
- Representative run:
  - `reference`: ~22.94 steps/s
  - `pallas`: ~19.56 steps/s
  - speedup: **0.85x** (gate target was `>=1.5x`)
- Smaller-shape run also failed gate (`~0.87x`).

## Current Status
- Correctness and robustness improved materially:
  - tiled Mamba2 kernel path exists,
  - gradients remain stable via reference backward,
  - runtime fallback behavior is explicit and safe.
- Performance target is **not met yet**; Mamba2 tiled Pallas path is currently slower than reference in local benchmarks.

## Next Steps (Priority Order)
1. **Tune tiled kernel mapping for throughput**
   - Revisit grid decomposition and `BlockSpec` layout to improve memory access/coalescing.
   - Sweep `scan_chunk_len` and shape-dependent tile strategies rather than fixed chunking.
2. **Profile kernel-level bottlenecks**
   - Capture forward-pass profile to identify dominant costs (summary pass vs output pass vs launch overhead).
   - Separate compile-time from steady-state timings.
3. **Run a GPU benchmark matrix on representative sweep shapes**
   - Use the updated benchmark script across arch/layer sweep-like configs.
   - Record tokens/sec/steps/sec and fallback incidence.
4. **Only promote `scan_backend='auto'` after repeated speed-gate pass**
   - Keep default `reference` until Mamba2 Pallas is consistently faster.
5. **Defer Mamba1 optimization**
   - Keep Mamba1 on robust fallback path for now; resume only after Mamba2 meets speed gate.

---

## 2026-02-23 - Bonsai Mamba2 Adaptation Pass (Separate Implementation)

## Task Summary
Set aside the custom Mamba2 kernel track and adapt a fast JAX reference Mamba2 SSM implementation from Bonsai into this repo as a separate model path, preserving chunkwise SSD and cache-based decoding. Benchmark against the current `model/ssm.py` Mamba2 implementation for both training and inference.

## Changes Implemented

### 1) Added separate Bonsai-adapted Mamba2 module
File: `model/ssm_bonsai.py` (new)
- Added:
  - `Mamba2BonsaiConfig`
  - `Mamba2Bonsai`
  - `Mamba2BonsaiCache`
  - `create_empty_cache(...)`
- Implemented Bonsai-style chunkwise SSD core:
  - `_segsum`
  - `_ssd_forward`
  - chunk padding/chunked recurrence composition
- Preserved cache-aware decode path:
  - per-layer `conv_states` and `ssm_states`
  - incremental decode with `cache=` and `return_cache=True`
- Kept repo-facing model behavior aligned with existing SSM interfaces:
  - `to_model(rngs=...)`
  - token embedding optional (`n_vocab`)
  - output modes: `last_token`, `full_sequence`, `last_nonpad`
  - `n_pred_tokens` and `n_out` shaping behavior
  - optional `MuReadout` when `use_mup=True`

### 2) Added focused test coverage for Bonsai path
File: `tests/test_ssm_bonsai.py` (new)
- Added tests for:
  - config defaults and `to_model`
  - forward shape checks with and without embedding
  - cache tensor shapes from `create_empty_cache`
  - incremental cached forward vs full-sequence forward equivalence (numerical tolerance)
  - `last_nonpad` token requirement behavior
  - train-step integration (output changes after one optimizer step)

### 3) Added implementation benchmark script (original vs bonsai)
File: `experiment/utils/bench_mamba2_impls.py` (new)
- Benchmarks two implementations:
  - `original`: `model.ssm.Mamba2` (configurable `scan_backend`)
  - `bonsai`: `model.ssm_bonsai.Mamba2Bonsai`
- Includes two workloads:
  - training throughput (`train_step`)
  - inference throughput:
    - original: repeated full-context forward (`inference_no_cache`)
    - bonsai: cached 1-token decode steps (`inference_cached`)
- Prints per-impl mean/p95 step time, steps/s, and speedup:
  - `training speedup (bonsai/original)`
  - `inference speedup (bonsai/original)`

## Validation Run
- `./.venv/bin/python -m py_compile model/ssm_bonsai.py tests/test_ssm_bonsai.py experiment/utils/bench_mamba2_impls.py`
- `./.venv/bin/python -m pytest tests/test_ssm_bonsai.py -v --tb=short`
  - **8 passed**
- `./.venv/bin/python -m pytest tests/test_ssm.py -v --tb=short`
  - **33 passed**

## Benchmark Results (Local GPU)

### Run A
Command:
`./.venv/bin/python experiment/utils/bench_mamba2_impls.py --batch-size 64 --n-seq 64 --n-vocab 512 --n-layers 4 --n-hidden 128 --n-heads 8 --d-state 16 --scan-chunk-len 64 --original-backend reference --train-warmup-steps 4 --train-timed-steps 16 --infer-warmup-steps 8 --infer-timed-steps 64`

Results:
- training:
  - original: **12.44 steps/s**
  - bonsai: **36.63 steps/s**
  - speedup: **2.95x**
- inference:
  - original (`inference_no_cache`): **84.24 steps/s**
  - bonsai (`inference_cached`): **202.09 steps/s**
  - speedup: **2.40x**

### Run B
Command:
`./.venv/bin/python experiment/utils/bench_mamba2_impls.py --batch-size 32 --n-seq 128 --n-vocab 512 --n-layers 4 --n-hidden 256 --n-heads 8 --d-state 16 --scan-chunk-len 64 --original-backend reference --train-warmup-steps 4 --train-timed-steps 12 --infer-warmup-steps 8 --infer-timed-steps 48`

Results:
- training:
  - original: **6.46 steps/s**
  - bonsai: **22.97 steps/s**
  - speedup: **3.56x**
- inference:
  - original (`inference_no_cache`): **48.27 steps/s**
  - bonsai (`inference_cached`): **239.95 steps/s**
  - speedup: **4.97x**

## Current Status
- Separate Bonsai-adapted Mamba2 path is implemented and tested.
- Existing `model/ssm.py` Mamba2 path remains unchanged for low-risk A/B.
- On this GPU, Bonsai path is substantially faster than original in both measured workloads.

## Next Steps
1. Add this implementation as an optional family in remote sweeps (`mamba2_bonsai`) for controlled experiment rollout.
2. Add benchmark CSV/JSON output mode so speed results can be tracked in CI/nightly regressions.
3. Evaluate memory usage and stability at larger sweep-scale shapes before adopting as default Mamba2 path.
