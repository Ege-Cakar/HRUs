"""Factory for creating paired FOL task conditions (internalized + ICL).

Provides a single entry point for constructing both FOL reasoning conditions
with a shared rule bank, tokenizer, and consistent vocabulary dimensions.

Usage::

    factory = FOLTaskFactory(
        rule_bank_seed=42,
        d_train_max=5,
        d_eval_max=15,
    )

    # Internalized rules: no demos, model must recall rules from training
    train_task = factory.make_internalized_task(batch_size=64)
    eval_tasks = factory.make_internalized_eval_tasks(batch_size=64)

    # ICL rules: demos provided in prompt, model retrieves + applies
    train_task_icl = factory.make_icl_task(batch_size=64)
    eval_tasks_icl = factory.make_icl_eval_tasks(batch_size=64)

    # Shared dimensions for model config
    n_vocab = factory.n_vocab
    n_seq_int = factory.dims_internalized.n_seq_ar
    n_seq_icl = factory.dims_icl.n_seq_ar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from task.layer_fol.common import compute_fol_dims, _build_tokenizer_for_fresh_icl
from task.layer_fol.task import FOLLayerTask
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    HybridICLBank,
    HybridICLSampledProblem,
    build_hybrid_icl_bank,
    build_random_fol_rule_bank,
    load_fol_rule_bank,
    sample_hybrid_icl_problem,
    save_fol_rule_bank,
    save_hybrid_icl_bank,
    FOLSequent,
)
from task.layer_gen.util.fol_completion import sampled_completion_texts
from task.layer_fol.demos._core import (
    augment_prompt_with_demos,
    _prepend_demo_statements_to_prompt,
    _instantiate_demo_schema_with_random_constants,
)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleBankConfig:
    """Parameters for constructing the shared FOL rule bank."""

    n_layers: int = 16
    predicates_per_layer: int | tuple[int, ...] = 8
    rules_per_transition: int | tuple[int, ...] = 32
    arity_min: int = 1
    arity_max: int = 3
    vars_per_rule_max: int = 4
    constants: tuple[str, ...] = ("a", "b", "c", "d")
    k_in_min: int = 1
    k_in_max: int = 3
    k_out_min: int = 1
    k_out_max: int = 3
    initial_ant_max: int = 3


@dataclass(frozen=True)
class ICLConfig:
    """Parameters for the ICL (in-context learning) demo distribution."""

    max_n_demos: int = 16
    min_n_demos: int = 1
    demo_distribution: str = "zipf_per_rule"
    demo_distribution_alpha: float = 1.0
    demo_ranked: bool = True
    demo_ranking_beta: float = float("inf")
    demo_all: bool = False
    demo_unique: bool = True
    include_oracle: bool = False


@dataclass(frozen=True)
class HybridICLConfig:
    """Parameters for the hybrid ICL condition."""

    fresh_predicates_per_layer: int = 8
    fresh_rules_per_transition: int = 8
    pred_train_frac: float = 0.5
    p_fresh: float = 0.5
    predicate_name_len: int = 4
    # Demo distribution for the in-context demonstrations.
    max_n_demos: int = 16
    min_n_demos: int = 1
    demo_distribution: str = "zipf_per_rule"
    demo_distribution_alpha: float = 1.0
    demo_ranked: bool = True
    demo_ranking_beta: float = float("inf")
    demo_unique: bool = True
    include_oracle: bool = False


@dataclass(frozen=True)
class FOLConditionDims:
    """Computed dimensions for a single task condition."""

    n_vocab: int
    n_seq_ar: int
    max_prompt_len: int
    max_completion_len: int
    max_atom_len: int


# ---------------------------------------------------------------------------
# Depth curriculum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthCurriculumPhase:
    """A single phase of the depth curriculum."""

    start_step: int
    d_max: int


class DepthCurriculum:
    """Depth curriculum that increases max derivation depth over training.

    Supports three scheduling modes:

    - **linear**: ``d_max`` increases by 1 every ``steps_per_depth`` steps.
    - **exponential**: ``d_max`` increases by 1 at exponentially growing
      intervals (``steps_per_depth * growth_factor^k`` for the k-th increase).
    - **manual**: Explicit list of ``(start_step, d_max)`` phases.

    Usage::

        # Linear: start at D=1, add 1 depth every 5000 steps, cap at D=8
        curriculum = DepthCurriculum.linear(
            d_start=1, d_max=8, steps_per_depth=5000,
        )

        # In training loop:
        current_d = curriculum.max_depth(step)
        if current_d != prev_d:
            task = factory.make_internalized_task(distance_range=(1, current_d))

    Parameters
    ----------
    phases : list[DepthCurriculumPhase]
        Ordered list of phases. Each phase specifies the training step at
        which it begins and the max depth for that phase.
    """

    def __init__(self, phases: list[DepthCurriculumPhase]) -> None:
        if not phases:
            raise ValueError("Curriculum must have at least one phase.")
        self.phases = sorted(phases, key=lambda p: p.start_step)
        if self.phases[0].start_step != 0:
            raise ValueError("First phase must start at step 0.")

    def max_depth(self, step: int) -> int:
        """Return the max derivation depth for the given training step."""
        result = self.phases[0].d_max
        for phase in self.phases:
            if step >= phase.start_step:
                result = phase.d_max
            else:
                break
        return result

    def phase_boundaries(self) -> list[tuple[int, int]]:
        """Return ``[(start_step, d_max), ...]`` for all phases."""
        return [(p.start_step, p.d_max) for p in self.phases]

    @classmethod
    def linear(
        cls,
        *,
        d_start: int = 1,
        d_max: int,
        steps_per_depth: int,
    ) -> "DepthCurriculum":
        """Create a linear curriculum: add 1 depth every ``steps_per_depth`` steps.

        Parameters
        ----------
        d_start : int
            Starting max depth (default 1).
        d_max : int
            Final max depth.
        steps_per_depth : int
            Training steps before increasing depth by 1.
        """
        phases = []
        for d in range(d_start, d_max + 1):
            start_step = (d - d_start) * steps_per_depth
            phases.append(DepthCurriculumPhase(start_step=start_step, d_max=d))
        return cls(phases)

    @classmethod
    def exponential(
        cls,
        *,
        d_start: int = 1,
        d_max: int,
        steps_per_depth: int,
        growth_factor: float = 2.0,
    ) -> "DepthCurriculum":
        """Create an exponential curriculum: intervals grow by ``growth_factor``.

        The k-th depth increase happens after
        ``steps_per_depth * growth_factor^k`` steps from the previous increase.

        Parameters
        ----------
        d_start : int
            Starting max depth.
        d_max : int
            Final max depth.
        steps_per_depth : int
            Steps for the first depth increase.
        growth_factor : float
            Multiplicative factor for subsequent intervals.
        """
        phases = []
        cumulative = 0
        for k, d in enumerate(range(d_start, d_max + 1)):
            phases.append(DepthCurriculumPhase(start_step=int(cumulative), d_max=d))
            cumulative += steps_per_depth * (growth_factor ** k)
        return cls(phases)

    @classmethod
    def manual(
        cls,
        phases: list[tuple[int, int]],
    ) -> "DepthCurriculum":
        """Create a curriculum from explicit ``(start_step, d_max)`` pairs.

        Parameters
        ----------
        phases : list[tuple[int, int]]
            List of ``(start_step, d_max)`` pairs. Must include step 0.
        """
        return cls([
            DepthCurriculumPhase(start_step=int(s), d_max=int(d))
            for s, d in phases
        ])

    def to_dict(self) -> dict:
        """Serialize for W&B / JSON logging."""
        return {
            "phases": [
                {"start_step": p.start_step, "d_max": p.d_max}
                for p in self.phases
            ],
        }

    def __repr__(self) -> str:
        phase_strs = [f"step {p.start_step}: D≤{p.d_max}" for p in self.phases]
        return f"DepthCurriculum([{', '.join(phase_strs)}])"


class CurriculumTaskManager:
    """Manages task lifecycle under a depth curriculum.

    Wraps a factory + curriculum + offline ``ds_path``. On each call to
    ``step()``, checks whether the current max depth has changed. If so,
    closes the old task and creates a new one with the updated distance
    range — pointing at the **same** pre-generated offline shards (which
    contain all depths up to ``d_eval_max``).

    Usage::

        curriculum = DepthCurriculum.linear(d_start=1, d_max=8, steps_per_depth=5000)
        manager = CurriculumTaskManager(
            factory=factory,
            curriculum=curriculum,
            condition="internalized",       # or "icl", "hybrid_icl"
            ds_path="data/fol_seed42/internalized",
            batch_size=64,
        )

        for step in range(total_steps):
            xs, ys = manager.next_batch(step)
            # ... train ...

    Parameters
    ----------
    factory : FOLTaskFactory
        The task factory (provides rule bank, tokenizer, dims).
    curriculum : DepthCurriculum
        Depth schedule mapping step → max depth.
    condition : str
        Task condition: ``"internalized"``, ``"icl"``, or
        ``"hybrid_icl"`` (with ``eval_mode`` kwarg).
    ds_path : str | Path | None
        Path to offline shards. If None, uses online mode.
    batch_size : int
        Batch size for the task iterator.
    **task_kwargs
        Additional kwargs passed to the factory's ``make_*_task()`` method
        (e.g. ``icl_config``, ``eval_mode``, ``hybrid_config``).
    """

    def __init__(
        self,
        *,
        factory: "FOLTaskFactory",
        curriculum: DepthCurriculum,
        condition: str = "internalized",
        ds_path: str | Path | None = None,
        batch_size: int = 64,
        **task_kwargs: Any,
    ) -> None:
        self.factory = factory
        self.curriculum = curriculum
        self.condition = str(condition)
        self.ds_path = ds_path
        self.batch_size = int(batch_size)
        self.task_kwargs = task_kwargs

        self._current_d_max: int | None = None
        self._task: Any = None  # FOLLayerTask or HybridICLTask

    @property
    def current_d_max(self) -> int | None:
        """The current maximum depth, or None if no step has been taken."""
        return self._current_d_max

    def next_batch(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Get the next batch for the given training step.

        Rebuilds the task if the curriculum's max depth has changed.
        """
        d_max = self.curriculum.max_depth(step)
        if d_max != self._current_d_max:
            self._rebuild_task(d_max)
        return next(self._task)

    def _rebuild_task(self, d_max: int) -> None:
        """Close the current task and create a new one with updated depth."""
        if self._task is not None and hasattr(self._task, "close"):
            self._task.close()

        mode = "offline" if self.ds_path is not None else "online"
        common = dict(
            batch_size=self.batch_size,
            distance_range=(1, d_max),
            mode=mode,
            ds_path=self.ds_path,
        )
        common.update(self.task_kwargs)

        if self.condition == "internalized":
            self._task = self.factory.make_internalized_task(**common)
        elif self.condition == "icl":
            self._task = self.factory.make_icl_task(**common)
        elif self.condition.startswith("hybrid_icl"):
            self._task = self.factory.make_hybrid_icl_task(**common)
        else:
            raise ValueError(f"Unknown condition: {self.condition!r}")

        self._current_d_max = d_max

    def close(self) -> None:
        """Close the current task and release resources."""
        if self._task is not None and hasattr(self._task, "close"):
            self._task.close()
            self._task = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@dataclass
class FOLTaskFactory:
    """Factory for paired internalized / ICL FOL task conditions.

    Builds a single shared rule bank from a fixed seed and computes
    consistent dimensions across both conditions.  Then provides
    ``make_*_task()`` methods that instantiate :class:`FOLLayerTask`
    with the correct parameters.

    Parameters
    ----------
    rule_bank_seed : int
        Seed for constructing the shared rule bank (deterministic).
    d_train_max : int
        Maximum derivation depth for training (inclusive).
        Training tasks sample depths ``1..d_train_max``.
    d_eval_max : int
        Maximum derivation depth for evaluation (inclusive).
    bank_config : RuleBankConfig
        Rule bank construction parameters.
    icl_config : ICLConfig
        Demo distribution parameters for the ICL condition.
    internalized_completion_format : str
        ``"single"`` (rollout eval) or ``"full"`` (multi-step output).
    icl_completion_format : str
        ``"single"`` or ``"full"`` for the ICL condition.
    sample_max_attempts : int
        Max sampling attempts during online generation.
    max_unify_solutions : int
        Max unification solutions during sampling.
    online_prefetch_backend : str
        Prefetch backend for online tasks.
    """

    rule_bank_seed: int
    d_train_max: int
    d_eval_max: int = 20
    bank_config: RuleBankConfig = field(default_factory=RuleBankConfig)
    icl_config: ICLConfig = field(default_factory=ICLConfig)
    internalized_completion_format: str = "single"
    icl_completion_format: str = "single"
    sample_max_attempts: int = 4096
    max_unify_solutions: int = 128
    online_prefetch_backend: str = "sync"

    # -- Computed (set in __post_init__) ------------------------------------
    _rule_bank: FOLRuleBank = field(init=False, repr=False)
    _tokenizer: tokenize_layer_fol.FOLLayerTokenizer = field(init=False, repr=False)
    _dims_internalized: dict[str, int] = field(init=False, repr=False)
    _dims_icl: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bc = self.bank_config

        # 1. Build the shared rule bank.
        rng = np.random.default_rng(self.rule_bank_seed)
        self._rule_bank = build_random_fol_rule_bank(
            n_layers=bc.n_layers,
            predicates_per_layer=bc.predicates_per_layer,
            rules_per_transition=bc.rules_per_transition,
            arity_min=bc.arity_min,
            arity_max=bc.arity_max,
            vars_per_rule_max=bc.vars_per_rule_max,
            constants=bc.constants,
            k_in_min=bc.k_in_min,
            k_in_max=bc.k_in_max,
            k_out_min=bc.k_out_min,
            k_out_max=bc.k_out_max,
            rng=rng,
        )

        # 2. Build the tokenizer.
        self._tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(
            self._rule_bank
        )

        # 3. Compute dims for each condition.
        self._dims_internalized = compute_fol_dims(
            rule_banks=[self._rule_bank],
            tokenizer=self._tokenizer,
            initial_ant_max=bc.initial_ant_max,
            max_n_demos=0,
            completion_format=self.internalized_completion_format,
            completion_steps_max=self.d_eval_max,
        )
        self._dims_icl = compute_fol_dims(
            rule_banks=[self._rule_bank],
            tokenizer=self._tokenizer,
            initial_ant_max=bc.initial_ant_max,
            max_n_demos=self.icl_config.max_n_demos,
            completion_format=self.icl_completion_format,
            completion_steps_max=self.d_eval_max,
        )

    # -- Properties ---------------------------------------------------------

    @property
    def rule_bank(self) -> FOLRuleBank:
        """The shared rule bank (deterministic from seed)."""
        return self._rule_bank

    @property
    def tokenizer(self) -> tokenize_layer_fol.FOLLayerTokenizer:
        """The shared tokenizer built from the rule bank."""
        return self._tokenizer

    @property
    def dims_internalized(self) -> FOLConditionDims:
        """Computed dimensions for the internalized-rules condition."""
        return _dict_to_dims(self._dims_internalized)

    @property
    def dims_icl(self) -> FOLConditionDims:
        """Computed dimensions for the ICL condition."""
        return _dict_to_dims(self._dims_icl)

    @property
    def n_vocab(self) -> int:
        """Shared vocabulary size (identical across conditions)."""
        return int(self._dims_internalized["n_vocab"])

    @property
    def max_n_seq(self) -> int:
        """Maximum sequence length across both conditions."""
        return max(
            int(self._dims_internalized["n_seq_ar"]),
            int(self._dims_icl["n_seq_ar"]),
        )

    # -- Task builders (shared kwargs) --------------------------------------

    def _shared_task_kwargs(self) -> dict[str, Any]:
        """Return kwargs common to all FOLLayerTask instantiations."""
        bc = self.bank_config
        return dict(
            mode="online",
            task_split="none",
            n_layers=bc.n_layers,
            predicates_per_layer=bc.predicates_per_layer,
            rules_per_transition=bc.rules_per_transition,
            arity_min=bc.arity_min,
            arity_max=bc.arity_max,
            vars_per_rule_max=bc.vars_per_rule_max,
            constants=bc.constants,
            k_in_min=bc.k_in_min,
            k_in_max=bc.k_in_max,
            k_out_min=bc.k_out_min,
            k_out_max=bc.k_out_max,
            initial_ant_max=bc.initial_ant_max,
            sample_max_attempts=self.sample_max_attempts,
            max_unify_solutions=self.max_unify_solutions,
            prediction_objective="autoregressive",
            fixed_length_mode="global_max",
            online_prefetch_backend=self.online_prefetch_backend,
        )

    # -- Internalized condition ---------------------------------------------

    def make_internalized_task(
        self,
        *,
        batch_size: int = 64,
        distance_range: tuple[int, int] | None = None,
        mode: str = "online",
        ds_path: str | Path | None = None,
        fixed_length_n_seq: int | None = None,
        completion_format: str | None = None,
        seed: int | None = None,
        **overrides: Any,
    ) -> FOLLayerTask:
        """Create an internalized-rules task (no demos in prompt).

        Parameters
        ----------
        batch_size : int
            Batch size for the task iterator.
        distance_range : tuple[int, int] | None
            ``(min_depth, max_depth)`` for sampling. Defaults to
            ``(1, d_train_max)``.
        fixed_length_n_seq : int | None
            Override padded sequence length. Defaults to computed value.
        completion_format : str | None
            ``"single"`` or ``"full"``. Defaults to factory setting.
        seed : int | None
            RNG seed. Defaults to ``rule_bank_seed + 1``.
        **overrides
            Additional kwargs passed to :class:`FOLLayerTask`.
        """
        if distance_range is None:
            distance_range = (1, self.d_train_max)
        if completion_format is None:
            completion_format = self.internalized_completion_format
        if fixed_length_n_seq is None:
            fixed_length_n_seq = int(self._dims_internalized["n_seq_ar"])
        if seed is None:
            seed = self.rule_bank_seed + 1

        kwargs = self._shared_task_kwargs()
        kwargs.update(
            mode=mode,
            batch_size=batch_size,
            distance_range=distance_range,
            max_n_demos=0,
            completion_format=completion_format,
            fixed_length_n_seq=fixed_length_n_seq,
            seed=seed,
        )
        if ds_path is not None:
            kwargs["ds_path"] = ds_path
        kwargs.update(overrides)
        return FOLLayerTask(**kwargs)

    def make_internalized_eval_tasks(
        self,
        *,
        batch_size: int = 64,
        depths: list[int] | None = None,
        fixed_length_n_seq: int | None = None,
        completion_format: str | None = None,
        **overrides: Any,
    ) -> dict[int, FOLLayerTask]:
        """Create per-depth eval tasks for the internalized condition.

        Returns ``{depth: FOLLayerTask}`` for each requested depth.
        """
        if depths is None:
            depths = list(range(1, self.d_eval_max + 1))

        tasks: dict[int, FOLLayerTask] = {}
        for d in depths:
            # Each depth gets a unique seed derived from rule_bank_seed.
            eval_seed = self.rule_bank_seed + 1000 + d
            tasks[d] = self.make_internalized_task(
                batch_size=batch_size,
                distance_range=(d, d),
                fixed_length_n_seq=fixed_length_n_seq,
                completion_format=completion_format,
                seed=eval_seed,
                **overrides,
            )
        return tasks

    # -- ICL condition ------------------------------------------------------

    def make_icl_task(
        self,
        *,
        batch_size: int = 64,
        distance_range: tuple[int, int] | None = None,
        mode: str = "online",
        ds_path: str | Path | None = None,
        fixed_length_n_seq: int | None = None,
        completion_format: str | None = None,
        icl_config: ICLConfig | None = None,
        seed: int | None = None,
        **overrides: Any,
    ) -> FOLLayerTask:
        """Create an ICL-rules task (demos in prompt).

        Parameters
        ----------
        batch_size : int
            Batch size for the task iterator.
        distance_range : tuple[int, int] | None
            ``(min_depth, max_depth)`` for sampling. Defaults to
            ``(1, d_train_max)``.
        fixed_length_n_seq : int | None
            Override padded sequence length. Defaults to computed value.
        completion_format : str | None
            ``"single"`` or ``"full"``. Defaults to factory setting.
        icl_config : ICLConfig | None
            Override demo config. Defaults to factory setting.
        seed : int | None
            RNG seed. Defaults to ``rule_bank_seed + 2``.
        **overrides
            Additional kwargs passed to :class:`FOLLayerTask`.
        """
        if distance_range is None:
            distance_range = (1, self.d_train_max)
        if completion_format is None:
            completion_format = self.icl_completion_format
        if icl_config is None:
            icl_config = self.icl_config
        if fixed_length_n_seq is None:
            fixed_length_n_seq = int(self._dims_icl["n_seq_ar"])
        if seed is None:
            seed = self.rule_bank_seed + 2

        kwargs = self._shared_task_kwargs()
        kwargs.update(
            mode=mode,
            batch_size=batch_size,
            distance_range=distance_range,
            max_n_demos=icl_config.max_n_demos,
            min_n_demos=icl_config.min_n_demos,
            demo_distribution=icl_config.demo_distribution,
            demo_distribution_alpha=icl_config.demo_distribution_alpha,
            demo_ranked=icl_config.demo_ranked,
            demo_ranking_beta=icl_config.demo_ranking_beta,
            demo_all=icl_config.demo_all,
            demo_unique=icl_config.demo_unique,
            include_oracle=icl_config.include_oracle,
            completion_format=completion_format,
            fixed_length_n_seq=fixed_length_n_seq,
            seed=seed,
        )
        if ds_path is not None:
            kwargs["ds_path"] = ds_path
        kwargs.update(overrides)
        return FOLLayerTask(**kwargs)

    def make_icl_eval_tasks(
        self,
        *,
        batch_size: int = 64,
        depths: list[int] | None = None,
        fixed_length_n_seq: int | None = None,
        completion_format: str | None = None,
        icl_config: ICLConfig | None = None,
        **overrides: Any,
    ) -> dict[int, FOLLayerTask]:
        """Create per-depth eval tasks for the ICL condition.

        Returns ``{depth: FOLLayerTask}`` for each requested depth.
        """
        if depths is None:
            depths = list(range(1, self.d_eval_max + 1))

        tasks: dict[int, FOLLayerTask] = {}
        for d in depths:
            eval_seed = self.rule_bank_seed + 2000 + d
            tasks[d] = self.make_icl_task(
                batch_size=batch_size,
                distance_range=(d, d),
                fixed_length_n_seq=fixed_length_n_seq,
                completion_format=completion_format,
                icl_config=icl_config,
                seed=eval_seed,
                **overrides,
            )
        return tasks

    # -- Hybrid ICL condition -----------------------------------------------

    def _get_or_build_hybrid_bank(
        self,
        hybrid_config: HybridICLConfig | None = None,
    ) -> HybridICLBank:
        """Build a HybridICLBank from the base bank."""
        if hybrid_config is None:
            hybrid_config = HybridICLConfig()
        return build_hybrid_icl_bank(
            base_bank=self._rule_bank,
            fresh_predicates_per_layer=hybrid_config.fresh_predicates_per_layer,
            fresh_rules_per_transition=hybrid_config.fresh_rules_per_transition,
            pred_train_frac=hybrid_config.pred_train_frac,
            p_fresh=hybrid_config.p_fresh,
            predicate_name_len=hybrid_config.predicate_name_len,
            rng=np.random.default_rng(self.rule_bank_seed + 5000),
        )

    def make_hybrid_icl_task(
        self,
        *,
        batch_size: int = 64,
        distance_range: tuple[int, int] | None = None,
        eval_mode: str = "train",
        hybrid_config: HybridICLConfig | None = None,
        hybrid_bank: HybridICLBank | None = None,
        seed: int | None = None,
        **overrides: Any,
    ) -> "HybridICLTask":
        """Create a hybrid ICL task (internalized + in-context rules).

        Parameters
        ----------
        batch_size : int
            Batch size for the task iterator.
        distance_range : tuple[int, int] | None
            ``(min_depth, max_depth)`` for sampling. Defaults to
            ``(1, d_train_max)``.
        eval_mode : str
            ``"train"``, ``"rule_gen"``, or ``"pred_gen"``.
        hybrid_config : HybridICLConfig | None
            Config for hybrid bank construction. Defaults to
            ``HybridICLConfig()``.
        hybrid_bank : HybridICLBank | None
            Pre-built hybrid bank. If provided, ``hybrid_config`` is
            ignored for bank construction (but still used for demo params).
        seed : int | None
            RNG seed. Defaults to ``rule_bank_seed + 3``.
        """
        if hybrid_config is None:
            hybrid_config = HybridICLConfig()
        if distance_range is None:
            distance_range = (1, self.d_train_max)
        if seed is None:
            seed = self.rule_bank_seed + 3
        if hybrid_bank is None:
            hybrid_bank = self._get_or_build_hybrid_bank(hybrid_config)

        # Build an extended tokenizer that covers fresh predicate chars.
        hybrid_tokenizer = _build_tokenizer_for_fresh_icl(
            base_bank=self._rule_bank,
            predicate_name_len=hybrid_config.predicate_name_len,
        )

        return HybridICLTask(
            hybrid_bank=hybrid_bank,
            tokenizer=hybrid_tokenizer,
            batch_size=batch_size,
            distance_range=distance_range,
            eval_mode=eval_mode,
            seed=seed,
            initial_ant_max=self.bank_config.initial_ant_max,
            sample_max_attempts=self.sample_max_attempts,
            max_unify_solutions=self.max_unify_solutions,
            max_n_demos=hybrid_config.max_n_demos,
            min_n_demos=hybrid_config.min_n_demos,
            demo_distribution=hybrid_config.demo_distribution,
            demo_distribution_alpha=hybrid_config.demo_distribution_alpha,
            demo_ranked=hybrid_config.demo_ranked,
            demo_ranking_beta=hybrid_config.demo_ranking_beta,
            demo_unique=hybrid_config.demo_unique,
            include_oracle=hybrid_config.include_oracle,
            completion_format=self.icl_completion_format,
        )

    def make_hybrid_icl_eval_tasks(
        self,
        *,
        batch_size: int = 64,
        depths: list[int] | None = None,
        eval_mode: str = "train",
        hybrid_config: HybridICLConfig | None = None,
        hybrid_bank: HybridICLBank | None = None,
        **overrides: Any,
    ) -> dict[int, "HybridICLTask"]:
        """Create per-depth eval tasks for the hybrid ICL condition."""
        if depths is None:
            depths = list(range(1, self.d_eval_max + 1))
        if hybrid_config is None:
            hybrid_config = HybridICLConfig()
        if hybrid_bank is None:
            hybrid_bank = self._get_or_build_hybrid_bank(hybrid_config)

        tasks: dict[int, HybridICLTask] = {}
        for d in depths:
            eval_seed = self.rule_bank_seed + 3000 + d
            tasks[d] = self.make_hybrid_icl_task(
                batch_size=batch_size,
                distance_range=(d, d),
                eval_mode=eval_mode,
                hybrid_config=hybrid_config,
                hybrid_bank=hybrid_bank,
                seed=eval_seed,
                **overrides,
            )
        return tasks

    # -- Persistence --------------------------------------------------------

    def save_rule_bank(self, path: str | Path) -> Path:
        """Persist the shared rule bank to disk for reproducibility."""
        p = Path(path)
        save_fol_rule_bank(p, self._rule_bank)
        return p

    @classmethod
    def from_rule_bank_path(
        cls,
        path: str | Path,
        *,
        d_train_max: int = 4,
        d_eval_max: int = 20,
        bank_config_overrides: dict | None = None,
        icl_config: ICLConfig | None = None,
        **kwargs,
    ) -> "FOLTaskFactory":
        """Load a rule bank from disk and build a factory around it.

        Rule bank parameters are derived from the loaded bank.
        ICL and other config can be overridden.
        """
        path = Path(path)
        bank = load_fol_rule_bank(path)

        # Infer rules_per_transition from the loaded bank.
        rpt_counts = tuple(
            len(bank.transition_rules(layer))
            for layer in sorted(bank.transitions)
        )
        rules_per_transition: int | tuple[int, ...] = rpt_counts[0] if len(set(rpt_counts)) == 1 else rpt_counts

        bc = RuleBankConfig(
            n_layers=bank.n_layers,
            predicates_per_layer=bank.predicates_per_layer,
            rules_per_transition=rules_per_transition,
            arity_min=bank.arity_min,
            arity_max=bank.arity_max,
            vars_per_rule_max=bank.vars_per_rule_max,
            constants=bank.constants,
            **(bank_config_overrides or {}),
        )

        factory = cls.__new__(cls)
        factory.rule_bank_seed = 0  # placeholder — bank already built
        factory.d_train_max = int(d_train_max)
        factory.d_eval_max = int(d_eval_max)
        factory.bank_config = bc
        factory.icl_config = icl_config or ICLConfig()
        factory.internalized_completion_format = kwargs.get(
            "internalized_completion_format", "single"
        )
        factory.icl_completion_format = kwargs.get("icl_completion_format", "single")
        factory.sample_max_attempts = kwargs.get("sample_max_attempts", 4096)
        factory.max_unify_solutions = kwargs.get("max_unify_solutions", 128)
        factory.online_prefetch_backend = kwargs.get("online_prefetch_backend", "sync")

        factory._rule_bank = bank
        factory._tokenizer = tokenize_layer_fol.build_tokenizer_from_rule_bank(bank)
        factory._dims_internalized = compute_fol_dims(
            rule_banks=[bank],
            tokenizer=factory._tokenizer,
            initial_ant_max=bc.initial_ant_max,
            max_n_demos=0,
            completion_format=factory.internalized_completion_format,
            completion_steps_max=factory.d_eval_max,
        )
        factory._dims_icl = compute_fol_dims(
            rule_banks=[bank],
            tokenizer=factory._tokenizer,
            initial_ant_max=bc.initial_ant_max,
            max_n_demos=factory.icl_config.max_n_demos,
            completion_format=factory.icl_completion_format,
            completion_steps_max=factory.d_eval_max,
        )
        return factory

    # -- Summary ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for W&B logging."""
        return {
            "rule_bank_seed": self.rule_bank_seed,
            "d_train_max": self.d_train_max,
            "d_eval_max": self.d_eval_max,
            "n_vocab": self.n_vocab,
            "n_seq_internalized": self.dims_internalized.n_seq_ar,
            "n_seq_icl": self.dims_icl.n_seq_ar,
            "max_n_seq": self.max_n_seq,
            "bank_n_layers": self.bank_config.n_layers,
            "bank_rules_per_transition": self.bank_config.rules_per_transition,
            "bank_predicates_per_layer": self.bank_config.predicates_per_layer,
            "bank_arity_range": (self.bank_config.arity_min, self.bank_config.arity_max),
            "internalized_completion_format": self.internalized_completion_format,
            "icl_completion_format": self.icl_completion_format,
            "icl_max_n_demos": self.icl_config.max_n_demos,
            "icl_min_n_demos": self.icl_config.min_n_demos,
            "icl_demo_distribution": self.icl_config.demo_distribution,
            "icl_demo_alpha": self.icl_config.demo_distribution_alpha,
            "icl_demo_beta": self.icl_config.demo_ranking_beta,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hybrid ICL task iterator
# ---------------------------------------------------------------------------

class HybridICLTask:
    """Online iterator for hybrid ICL problems (internalized + fresh rules).

    Each ``next()`` call samples a batch of hybrid ICL problems, tokenizes
    them with fresh-rule demos prepended, and returns ``(xs, ys)`` arrays.

    This is a standalone iterator (not a ``FOLLayerTask``) because the
    sampling logic is fundamentally different — it mixes two rule pools
    per-transition rather than using a single bank.
    """

    def __init__(
        self,
        *,
        hybrid_bank: HybridICLBank,
        tokenizer: tokenize_layer_fol.FOLLayerTokenizer,
        batch_size: int,
        distance_range: tuple[int, int],
        eval_mode: str,
        seed: int,
        initial_ant_max: int,
        sample_max_attempts: int,
        max_unify_solutions: int,
        max_n_demos: int,
        min_n_demos: int,
        demo_distribution: str,
        demo_distribution_alpha: float,
        demo_ranked: bool,
        demo_ranking_beta: float,
        demo_unique: bool,
        include_oracle: bool,
        completion_format: str,
    ) -> None:
        self.hybrid_bank = hybrid_bank
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.eval_mode = str(eval_mode)
        self.initial_ant_max = int(initial_ant_max)
        self.sample_max_attempts = int(sample_max_attempts)
        self.max_unify_solutions = int(max_unify_solutions)
        self.max_n_demos = int(max_n_demos)
        self.min_n_demos = int(min_n_demos)
        self.demo_distribution = str(demo_distribution)
        self.demo_distribution_alpha = float(demo_distribution_alpha)
        self.demo_ranked = bool(demo_ranked)
        self.demo_ranking_beta = float(demo_ranking_beta)
        self.demo_unique = bool(demo_unique)
        self.include_oracle = bool(include_oracle)
        self.completion_format = str(completion_format)

        # Parse distance range.
        if isinstance(distance_range, (list, tuple)) and len(distance_range) == 2:
            self._distances = list(range(int(distance_range[0]), int(distance_range[1]) + 1))
        else:
            self._distances = [int(d) for d in distance_range]

        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        records = []
        for _ in range(self.batch_size):
            record = self._sample_one()
            records.append(record)

        # Batch: pad to max length in batch.
        max_len = max(
            len(r["prompt"]) + len(r["completion"]) - 1
            for r in records
        )

        xs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        ys = np.zeros((self.batch_size, max_len), dtype=np.int32)

        for i, r in enumerate(records):
            prompt = r["prompt"]
            completion = r["completion"]
            seq_len = len(prompt) + len(completion) - 1
            # AR format: xs = prompt + completion[:-1], ys = prompt[1:] + completion
            # But the standard convention is:
            # xs[prompt_len-1:prompt_len-1+comp_len] = completion tokens
            # ys shifts by 1
            full_seq = list(prompt) + list(completion)
            # xs = full_seq[:-1], ys = full_seq[1:]
            xs_seq = full_seq[:-1]
            ys_seq = full_seq[1:]
            xs[i, :len(xs_seq)] = xs_seq
            ys[i, :len(ys_seq)] = ys_seq

        return xs, ys

    def _sample_one(self) -> dict:
        """Sample one hybrid ICL problem and tokenize it."""
        distance = int(self._rng.choice(self._distances))

        problem = sample_hybrid_icl_problem(
            hybrid_bank=self.hybrid_bank,
            distance=distance,
            eval_mode=self.eval_mode,
            initial_ant_max=self.initial_ant_max,
            rng=self._rng,
            max_attempts=self.sample_max_attempts,
            max_unify_solutions=self.max_unify_solutions,
        )

        # Pick a random step to predict.
        step_idx = int(self._rng.integers(0, len(problem.step_rules)))
        src_layer = problem.step_layers[step_idx]
        ants = problem.step_ants[step_idx]
        sequent = FOLSequent(ants=ants, cons=problem.goal_atom)

        # Tokenize prompt and completion.
        prompt = self.tokenizer.tokenize_prompt(sequent)
        completion_texts = sampled_completion_texts(
            sampled=problem.to_fol_sampled_problem(),
            step_idx=step_idx,
            completion_format=self.completion_format,
        )
        completion = self.tokenizer.encode_completion_texts(completion_texts)

        # Build demo statements: fresh rules used + noise from base bank.
        demo_statements: list[str] = []

        # 1. Required fresh rule demos (instantiated with random constants).
        for rule in problem.fresh_rules_used:
            demo_statements.append(
                _instantiate_demo_schema_with_random_constants(
                    rule=rule,
                    constants=self.hybrid_bank.base_bank.constants,
                    rng=self._rng,
                )
            )

        # 2. Additional noise demos from base bank via augment_prompt_with_demos.
        remaining_demos = max(0, self.max_n_demos - len(demo_statements))
        if remaining_demos > 0:
            augmented = augment_prompt_with_demos(
                prompt_tokens=prompt,
                rule_bank=self.hybrid_bank.base_bank,
                tokenizer=self.tokenizer,
                rng=self._rng,
                src_layer=int(src_layer),
                ants=ants,
                max_n_demos=remaining_demos,
                min_n_demos=0,
                max_unify_solutions=self.max_unify_solutions,
                include_oracle=False,
                demo_distribution=self.demo_distribution,
                demo_distribution_alpha=self.demo_distribution_alpha,
                goal_atom=problem.goal_atom,
                demo_ranked=self.demo_ranked,
                demo_ranking_beta=self.demo_ranking_beta,
                demo_unique=self.demo_unique,
            )
            # Extract the noise demo statements from the augmented prompt.
            for inst_text in augmented.demo_instances:
                demo_statements.append(str(inst_text))

        # Prepend all demos to the prompt.
        if demo_statements:
            prompt = _prepend_demo_statements_to_prompt(
                prompt_tokens=prompt,
                demo_statements=demo_statements,
                tokenizer=self.tokenizer,
            )

        return {
            "prompt": prompt,
            "completion": completion,
            "distance": distance,
            "transition_sources": problem.transition_sources,
            "n_fresh_rules": len(problem.fresh_rules_used),
        }


def _dict_to_dims(d: dict[str, int]) -> FOLConditionDims:
    return FOLConditionDims(
        n_vocab=int(d["n_vocab"]),
        n_seq_ar=int(d["n_seq_ar"]),
        max_prompt_len=int(d["max_prompt_len"]),
        max_completion_len=int(d["max_completion_len"]),
        max_atom_len=int(d["max_atom_len"]),
    )
