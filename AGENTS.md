# Repository Guidelines

## Project Overview
This is a JAX/Flax NNX-based deep learning research codebase for studying neural network dynamics on synthetic reasoning tasks. In particular, the project focuses in theorem proving using Gentzen's NJ proof system, applied to propositional logic. The goal is study different training data sets, and how they influence generalization.

## Project Structure & Module Organization
- `model/` defines Flax NNX architectures (`mlp.py`, `transformer.py`) and their `*Config.to_model(rngs=...)` builders.
- `task/` contains iterator-style data generators (`__next__()` returns `(xs, ys)`; `batch_size` required).
- `train.py` is the core training loop (`train(config, train_iter, ...)`), `common.py` has shared helpers like `collate_dfs`.
- `experiment/` holds numbered experiment scripts; `experiment/remote/` contains SLURM array jobs and `sb_*.sh` wrappers.
- `experiment/interactive/` mirrors `experiment/remote/` experiment IDs for local train/debug/inspection scripts that are meant to run on a workstation.
- `tests/` contains pytest suites (e.g., `tests/test_model.py`, `tests/test_train.py`).
- Propositional data pipeline: `task/prop_gen/generate_imply.py` → ArrayRecord shards → `task/prop.py` (`ImplySizeTask` via grain).

## Build, Test, and Development Commands
- `source ./.venv/bin/activate` activates the typical local venv (or call `./.venv/bin/python`).
- `./.venv/bin/python -m pytest tests/ -v --tb=short` runs the full test suite with repo defaults.
- `./.venv/bin/python -m pytest tests/test_model.py -v` runs a single module.
  - Many scripts assume local data and external deps; prefer reading a script before running.

## Testing Guidelines
- Framework: pytest (see `pyproject.toml` for config).
- Conventions: test files are `tests/test_*.py`, test functions are `test_*`.
- Coverage config targets `model`, `train`, and `common`, and omits `experiment/` by default.
- Always run tests after significant changes (code, logic, or behavior). Default command: `./.venv/bin/python -m pytest tests/ -v --tb=short`.
- Keep tests fast: small configs (`n_hidden=32-64`, `n_layers=1-2`) and fixed seeds (`nnx.Rngs(42)`).
- For Transformers, ensure `n_hidden` is divisible by `n_heads`.
- If a test is inherently long-running and cannot be made fast, mark it with `@pytest.mark.slow` (most slow tests involve coordinate checking).
- Run slow tests only when the relevant code has changed (e.g., coordinate-checking logic); otherwise prefer `./.venv/bin/python -m pytest tests/ -v --tb=short -m "not slow"`.
- To run only slow tests: `./.venv/bin/python -m pytest tests/ -v --tb=short -m slow`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; keep lines reasonably short and readable.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Follow existing patterns in `model/` and `task/`; add docstrings for new public helpers.
- Prefer writing code using small, interpretable, modular helper functions rather
than large, monolithic functions. Prefer code reuse and abstraction rather than copying, where possible.
- NNX modules are stateful; use `nnx.state(model)` when you need to serialize state.
- The `# <codecell>` tags and `# %%` tags are used to dilineate code blocks that can be executed in the style of Jupyter notebook cells. Keep these intact where possible, and add them after each plot is made in experiment plotting scripts.

## Experiment Workflow (New Runs)
- Place sweep logic in `experiment/remote/<id_name>/run.py` and Slurm submission in `experiment/remote/<id_name>/sb_*.sh`.
- Place local interactive/debug scripts in `experiment/interactive/<id_name>/` (mirroring the corresponding `remote` ID when applicable).
- For interactive scripts, prefer a single editable in-file config object over CLI parsing, so local notebook-style iteration only requires modifying the script.
- For interactive scripts, keep execution at top level (avoid a `main()` wrapper) to simplify stepwise runs in codecell/Jupyter-like workflows.
- Put plotting/analysis in `experiment/<id_name>.py`, using `common.collate_dfs` to load `remote/<id_name>` results.
- Always include a clearly delineated, commented-out **TEST CONFIGS** section in `run.py` with minimal configs for quick local testing before cluster submission (e.g., small sweep lists, short `train_iters`, `RUN_SPLIT=1`).
- For full runs, keep `RUN_SPLIT` aligned with the Slurm `--array` size.
- Use a placeholder dataset path in `run.py` when data is not yet available, and update it before cluster runs.
- Save outputs from local interactive scripts under `experiment/interactive/<id_name>/set/` by default.
