"""Tests for task.fol_task_factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from task.fol_task_factory import (
    FOLConditionDims,
    FOLTaskFactory,
    ICLConfig,
    RuleBankConfig,
)


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------

def _small_factory(**overrides) -> FOLTaskFactory:
    """Build a factory with small configs suitable for unit tests."""
    defaults = dict(
        rule_bank_seed=42,
        d_train_max=2,
        d_eval_max=4,
        bank_config=RuleBankConfig(
            n_layers=6,
            predicates_per_layer=4,
            rules_per_transition=8,
            arity_min=0,
            arity_max=2,
            vars_per_rule_max=3,
            constants=("a", "b"),
            k_in_min=1,
            k_in_max=2,
            k_out_min=1,
            k_out_max=2,
            initial_ant_max=2,
        ),
        icl_config=ICLConfig(
            max_n_demos=4,
            min_n_demos=1,
            demo_distribution="zipf_per_rule",
            demo_distribution_alpha=1.0,
        ),
        sample_max_attempts=512,
        online_prefetch_backend="sync",
    )
    defaults.update(overrides)
    return FOLTaskFactory(**defaults)


# ---------------------------------------------------------------------------
# Rule bank construction
# ---------------------------------------------------------------------------

class TestFactoryConstruction:
    def test_builds_rule_bank(self):
        factory = _small_factory()
        bank = factory.rule_bank
        assert bank.n_layers == 6
        assert len(bank.transitions) > 0
        # Should have transitions for layers 0..4 (n_layers-1 transitions)
        assert len(bank.transitions) == 5

    def test_deterministic_from_seed(self):
        f1 = _small_factory(rule_bank_seed=99)
        f2 = _small_factory(rule_bank_seed=99)
        # Same seed → same predicate arities
        assert f1.rule_bank.predicate_arities == f2.rule_bank.predicate_arities

    def test_tokenizer_built(self):
        factory = _small_factory()
        tok = factory.tokenizer
        assert tok.vocab_size > 0
        from task.layer_gen.util.tokenize_layer_fol import pad_idx
        assert tok.token_to_id["<PAD>"] == pad_idx
        assert tok.eot_token_id > 0


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_dims_internalized_type(self):
        factory = _small_factory()
        dims = factory.dims_internalized
        assert isinstance(dims, FOLConditionDims)
        assert dims.n_vocab > 0
        assert dims.n_seq_ar > 0

    def test_dims_icl_type(self):
        factory = _small_factory()
        dims = factory.dims_icl
        assert isinstance(dims, FOLConditionDims)
        assert dims.n_vocab > 0
        assert dims.n_seq_ar > 0

    def test_icl_seq_longer_than_internalized(self):
        factory = _small_factory()
        # ICL has demos prepended → longer sequences.
        assert factory.dims_icl.n_seq_ar > factory.dims_internalized.n_seq_ar

    def test_n_vocab_consistent(self):
        factory = _small_factory()
        assert factory.dims_internalized.n_vocab == factory.dims_icl.n_vocab
        assert factory.n_vocab == factory.dims_internalized.n_vocab

    def test_max_n_seq(self):
        factory = _small_factory()
        assert factory.max_n_seq == max(
            factory.dims_internalized.n_seq_ar,
            factory.dims_icl.n_seq_ar,
        )


# ---------------------------------------------------------------------------
# Internalized task
# ---------------------------------------------------------------------------

class TestInternalizedTask:
    def test_produces_batches(self):
        factory = _small_factory()
        task = factory.make_internalized_task(batch_size=4)
        xs, ys = next(task)
        assert xs.shape[0] == 4
        assert ys.shape[0] == 4
        assert xs.dtype == np.int32
        assert ys.dtype == np.int32

    def test_correct_seq_length(self):
        factory = _small_factory()
        n_seq = factory.dims_internalized.n_seq_ar
        task = factory.make_internalized_task(batch_size=2)
        xs, ys = next(task)
        assert xs.shape[1] == n_seq
        assert ys.shape[1] == n_seq

    def test_no_demos_in_prompt(self):
        factory = _small_factory()
        tok = factory.tokenizer
        task = factory.make_internalized_task(batch_size=8)
        xs, _ys = next(task)

        sep_id = int(tok.sep_token_id)
        start_id = int(tok.start_token_id)

        for i in range(xs.shape[0]):
            row = xs[i]
            # Find START token position.
            start_positions = np.where(row == start_id)[0]
            assert len(start_positions) >= 1, "No START token in prompt"
            first_start = int(start_positions[0])
            # No SEP tokens should appear before START (SEP = demo separator).
            prompt_prefix = row[:first_start]
            sep_count = int(np.sum(prompt_prefix == sep_id))
            assert sep_count == 0, (
                f"Found {sep_count} SEP tokens before START at position "
                f"{first_start} in internalized task (should be 0)"
            )

    def test_custom_distance_range(self):
        factory = _small_factory()
        task = factory.make_internalized_task(
            batch_size=2,
            distance_range=(1, 1),
        )
        xs, ys = next(task)
        assert xs.shape[0] == 2


# ---------------------------------------------------------------------------
# ICL task
# ---------------------------------------------------------------------------

class TestICLTask:
    def test_produces_batches(self):
        factory = _small_factory()
        task = factory.make_icl_task(batch_size=4)
        xs, ys = next(task)
        assert xs.shape[0] == 4
        assert ys.shape[0] == 4
        assert xs.dtype == np.int32
        assert ys.dtype == np.int32

    def test_correct_seq_length(self):
        factory = _small_factory()
        n_seq = factory.dims_icl.n_seq_ar
        task = factory.make_icl_task(batch_size=2)
        xs, ys = next(task)
        assert xs.shape[1] == n_seq
        assert ys.shape[1] == n_seq

    def test_has_demos_in_prompt(self):
        factory = _small_factory()
        tok = factory.tokenizer
        # Force at least 1 demo.
        icl_cfg = ICLConfig(max_n_demos=4, min_n_demos=1)
        task = factory.make_icl_task(batch_size=8, icl_config=icl_cfg)
        xs, _ys = next(task)

        sep_id = int(tok.sep_token_id)
        start_id = int(tok.start_token_id)

        found_sep = False
        for i in range(xs.shape[0]):
            row = xs[i]
            start_positions = np.where(row == start_id)[0]
            if len(start_positions) == 0:
                continue
            first_start = int(start_positions[0])
            prompt_prefix = row[:first_start]
            if int(np.sum(prompt_prefix == sep_id)) > 0:
                found_sep = True
                break

        assert found_sep, (
            "Expected at least one example with SEP tokens (demos) before START"
        )

    def test_override_icl_config(self):
        factory = _small_factory()
        custom_icl = ICLConfig(max_n_demos=2, min_n_demos=2, demo_distribution="uniform")
        task = factory.make_icl_task(batch_size=2, icl_config=custom_icl)
        xs, ys = next(task)
        assert xs.shape[0] == 2


# ---------------------------------------------------------------------------
# Eval tasks
# ---------------------------------------------------------------------------

class TestEvalTasks:
    def test_internalized_eval_tasks_per_depth(self):
        factory = _small_factory(d_eval_max=4)
        tasks = factory.make_internalized_eval_tasks(batch_size=2)
        assert set(tasks.keys()) == {1, 2, 3, 4}
        for d, task in tasks.items():
            xs, ys = next(task)
            assert xs.shape[0] == 2

    def test_icl_eval_tasks_per_depth(self):
        factory = _small_factory(d_eval_max=3)
        tasks = factory.make_icl_eval_tasks(batch_size=2)
        assert set(tasks.keys()) == {1, 2, 3}
        for d, task in tasks.items():
            xs, ys = next(task)
            assert xs.shape[0] == 2

    def test_custom_depths(self):
        factory = _small_factory(d_eval_max=4)
        tasks = factory.make_internalized_eval_tasks(
            batch_size=2, depths=[1, 3]
        )
        assert set(tasks.keys()) == {1, 3}

    def test_eval_seq_length_consistent(self):
        factory = _small_factory(d_eval_max=3)
        tasks = factory.make_internalized_eval_tasks(batch_size=2)
        # All eval tasks should have the same n_seq.
        shapes = {next(t)[0].shape[1] for t in tasks.values()}
        assert len(shapes) == 1


# ---------------------------------------------------------------------------
# Persistence & summary
# ---------------------------------------------------------------------------

class TestPersistenceAndSummary:
    def test_save_rule_bank(self, tmp_path: Path):
        factory = _small_factory()
        path = tmp_path / "bank.json"
        factory.save_rule_bank(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_summary_keys(self):
        factory = _small_factory()
        s = factory.summary()
        expected_keys = {
            "rule_bank_seed",
            "d_train_max",
            "d_eval_max",
            "n_vocab",
            "n_seq_internalized",
            "n_seq_icl",
            "max_n_seq",
            "bank_n_layers",
            "bank_rules_per_transition",
            "bank_predicates_per_layer",
            "bank_arity_range",
            "internalized_completion_format",
            "icl_completion_format",
            "icl_max_n_demos",
            "icl_min_n_demos",
            "icl_demo_distribution",
            "icl_demo_alpha",
            "icl_demo_beta",
        }
        assert set(s.keys()) == expected_keys

    def test_from_rule_bank_path(self, tmp_path: Path):
        factory = _small_factory()
        path = tmp_path / "bank.json"
        factory.save_rule_bank(path)

        factory2 = FOLTaskFactory.from_rule_bank_path(
            path,
            d_train_max=2,
            d_eval_max=4,
            icl_config=ICLConfig(max_n_demos=4, min_n_demos=1),
        )
        # Should have the same rule bank structure.
        assert factory2.rule_bank.n_layers == factory.rule_bank.n_layers
        assert factory2.rule_bank.predicate_arities == factory.rule_bank.predicate_arities
        # Should produce valid batches.
        task = factory2.make_internalized_task(batch_size=2)
        xs, ys = next(task)
        assert xs.shape[0] == 2

    def test_summary_values_match_factory(self):
        factory = _small_factory()
        s = factory.summary()
        assert s["rule_bank_seed"] == factory.rule_bank_seed
        assert s["d_train_max"] == factory.d_train_max
        assert s["n_vocab"] == factory.n_vocab
        assert s["n_seq_internalized"] == factory.dims_internalized.n_seq_ar
        assert s["n_seq_icl"] == factory.dims_icl.n_seq_ar
