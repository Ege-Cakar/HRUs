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

from task.layer_fol.common import compute_fol_dims
from task.layer_fol.task import FOLLayerTask
from task.layer_gen.util import tokenize_layer_fol
from task.layer_gen.util.fol_rule_bank import (
    FOLRuleBank,
    build_random_fol_rule_bank,
    load_fol_rule_bank,
    save_fol_rule_bank,
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
class FOLConditionDims:
    """Computed dimensions for a single task condition."""

    n_vocab: int
    n_seq_ar: int
    max_prompt_len: int
    max_completion_len: int
    max_atom_len: int


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

def _dict_to_dims(d: dict[str, int]) -> FOLConditionDims:
    return FOLConditionDims(
        n_vocab=int(d["n_vocab"]),
        n_seq_ar=int(d["n_seq_ar"]),
        max_prompt_len=int(d["max_prompt_len"]),
        max_completion_len=int(d["max_completion_len"]),
        max_atom_len=int(d["max_atom_len"]),
    )
