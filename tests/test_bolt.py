"""Tests for bolt layout computation and worker GPU allocation."""

import sys
from pathlib import Path

import pytest

# Make bolt/ importable
BOLT_DIR = Path(__file__).resolve().parent.parent / "bolt"
sys.path.insert(0, str(BOLT_DIR))

from layout import (
    GPUS_PER_NODE,
    compute_child_split_ranges,
    compute_gpu_ids,
    compute_num_children,
    splits_per_child,
)


# ── splits_per_child ────────────────────────────────────────────────────

class TestSplitsPerChild:
    def test_full_node(self):
        assert splits_per_child(8) == 1

    def test_half_node(self):
        assert splits_per_child(4) == 2

    def test_single_gpu(self):
        assert splits_per_child(1) == 8

    def test_two_gpus(self):
        assert splits_per_child(2) == 4

    def test_custom_node_size(self):
        assert splits_per_child(2, gpus_per_node=16) == 8


# ── compute_num_children ────────────────────────────────────────────────

class TestComputeNumChildren:
    def test_2_splits_8_gpus(self):
        """2 splits × 8 GPUs each = 2 children (1 split per child)."""
        assert compute_num_children(2, 8) == 2

    def test_2_splits_4_gpus(self):
        """2 splits × 4 GPUs each = 1 child (both fit on one node)."""
        assert compute_num_children(2, 4) == 1

    def test_8_splits_1_gpu(self):
        """8 splits × 1 GPU each = 1 child (all fit on one 8-GPU node)."""
        assert compute_num_children(8, 1) == 1

    def test_9_splits_1_gpu(self):
        """9 splits × 1 GPU each = 2 children (9 doesn't fit on 8)."""
        assert compute_num_children(9, 1) == 2

    def test_1_split_8_gpus(self):
        assert compute_num_children(1, 8) == 1

    def test_3_splits_4_gpus(self):
        """3 splits × 4 GPUs = ceil(3/2) = 2 children."""
        assert compute_num_children(3, 4) == 2

    def test_4_splits_2_gpus(self):
        """4 splits × 2 GPUs = 4/4 = 1 child."""
        assert compute_num_children(4, 2) == 1

    def test_5_splits_2_gpus(self):
        """5 splits × 2 GPUs = ceil(5/4) = 2 children."""
        assert compute_num_children(5, 2) == 2

    def test_formula_matches(self):
        """Verify: num_children = ceil(num_splits * gpus_per_split / gpus_per_node)."""
        import math
        for n in range(1, 17):
            for g in [1, 2, 4, 8]:
                expected = math.ceil(n * g / GPUS_PER_NODE)
                assert compute_num_children(n, g) == expected, (
                    f"Mismatch for num_splits={n}, gpus_per_split={g}"
                )


# ── compute_child_split_ranges ──────────────────────────────────────────

class TestComputeChildSplitRanges:
    def test_2_splits_8_gpus(self):
        """Each split on its own child."""
        ranges = compute_child_split_ranges(2, 8)
        assert ranges == [(1, 1), (2, 2)]

    def test_2_splits_4_gpus(self):
        """Both splits on one child."""
        ranges = compute_child_split_ranges(2, 4)
        assert ranges == [(1, 2)]

    def test_8_splits_1_gpu(self):
        """All 8 splits on one child."""
        ranges = compute_child_split_ranges(8, 1)
        assert ranges == [(1, 8)]

    def test_9_splits_1_gpu(self):
        """8 on first child, 1 on second."""
        ranges = compute_child_split_ranges(9, 1)
        assert ranges == [(1, 8), (9, 9)]

    def test_3_splits_4_gpus(self):
        """2 fit on first child, 1 on second."""
        ranges = compute_child_split_ranges(3, 4)
        assert ranges == [(1, 2), (3, 3)]

    def test_single_split(self):
        ranges = compute_child_split_ranges(1, 8)
        assert ranges == [(1, 1)]

    def test_ranges_cover_all_splits(self):
        """Every split index from 1..num_splits appears in exactly one range."""
        for n in range(1, 17):
            for g in [1, 2, 4, 8]:
                ranges = compute_child_split_ranges(n, g)
                all_indices = []
                for start, end in ranges:
                    all_indices.extend(range(start, end + 1))
                assert all_indices == list(range(1, n + 1)), (
                    f"Ranges don't cover 1..{n} for gpus_per_split={g}: {ranges}"
                )

    def test_no_overlap(self):
        """Ranges don't overlap."""
        for n in range(1, 17):
            for g in [1, 2, 4, 8]:
                ranges = compute_child_split_ranges(n, g)
                for i in range(len(ranges) - 1):
                    assert ranges[i][1] < ranges[i + 1][0]


# ── compute_gpu_ids ─────────────────────────────────────────────────────

class TestComputeGpuIds:
    def test_single_gpu_per_split(self):
        """gpus_per_split=1: each split gets one GPU."""
        assert compute_gpu_ids(1, 1, 1) == [0]
        assert compute_gpu_ids(2, 1, 1) == [1]
        assert compute_gpu_ids(8, 1, 1) == [7]

    def test_four_gpus_per_split(self):
        """gpus_per_split=4: first split gets 0-3, second gets 4-7."""
        assert compute_gpu_ids(1, 1, 4) == [0, 1, 2, 3]
        assert compute_gpu_ids(2, 1, 4) == [4, 5, 6, 7]

    def test_eight_gpus_per_split(self):
        """gpus_per_split=8: single split gets all GPUs."""
        assert compute_gpu_ids(1, 1, 8) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_nonzero_start_index(self):
        """start_index > 1: GPU assignment is relative to start."""
        assert compute_gpu_ids(3, 3, 4) == [0, 1, 2, 3]
        assert compute_gpu_ids(4, 3, 4) == [4, 5, 6, 7]

    def test_two_gpus_per_split(self):
        assert compute_gpu_ids(1, 1, 2) == [0, 1]
        assert compute_gpu_ids(2, 1, 2) == [2, 3]
        assert compute_gpu_ids(3, 1, 2) == [4, 5]
        assert compute_gpu_ids(4, 1, 2) == [6, 7]

    def test_no_gpu_overlap_within_child(self):
        """GPU allocations for splits in the same child never overlap."""
        for g in [1, 2, 4, 8]:
            spc = GPUS_PER_NODE // g
            for start in [1, 5, 9]:
                all_gpus = []
                for idx in range(start, start + spc):
                    gpus = compute_gpu_ids(idx, start, g)
                    assert len(gpus) == g
                    all_gpus.extend(gpus)
                assert len(set(all_gpus)) == len(all_gpus), (
                    f"GPU overlap for gpus_per_split={g}, start={start}"
                )


# ── End-to-end layout scenarios ─────────────────────────────────────────

class TestEndToEndLayout:
    """Verify full scheduling for the scenarios in the user's request."""

    def test_experiment_12_full_node(self):
        """RUN_SPLIT=2, --gpus 8 → 2 children, each running 1 split on 8 GPUs."""
        num_splits, gpus_per_split = 2, 8
        assert compute_num_children(num_splits, gpus_per_split) == 2

        ranges = compute_child_split_ranges(num_splits, gpus_per_split)
        assert ranges == [(1, 1), (2, 2)]

        # Child 0: split 1 gets all 8 GPUs
        assert compute_gpu_ids(1, 1, 8) == list(range(8))
        # Child 1: split 2 gets all 8 GPUs
        assert compute_gpu_ids(2, 2, 8) == list(range(8))

    def test_experiment_12_half_node(self):
        """RUN_SPLIT=2, --gpus 4 → 1 child with 2 splits, each on 4 GPUs."""
        num_splits, gpus_per_split = 2, 4
        assert compute_num_children(num_splits, gpus_per_split) == 1

        ranges = compute_child_split_ranges(num_splits, gpus_per_split)
        assert ranges == [(1, 2)]

        # Both splits on same child, non-overlapping GPUs
        assert compute_gpu_ids(1, 1, 4) == [0, 1, 2, 3]
        assert compute_gpu_ids(2, 1, 4) == [4, 5, 6, 7]

    def test_large_sweep_single_gpu(self):
        """RUN_SPLIT=16, --gpus 1 → 2 children, 8 splits each, 1 GPU per split."""
        num_splits, gpus_per_split = 16, 1
        assert compute_num_children(num_splits, gpus_per_split) == 2

        ranges = compute_child_split_ranges(num_splits, gpus_per_split)
        assert ranges == [(1, 8), (9, 16)]

        # Child 0: split i gets GPU i-1
        for idx in range(1, 9):
            assert compute_gpu_ids(idx, 1, 1) == [idx - 1]

        # Child 1: split 9 gets GPU 0, split 10 gets GPU 1, etc.
        for idx in range(9, 17):
            assert compute_gpu_ids(idx, 9, 1) == [idx - 9]

    def test_odd_split_count(self):
        """RUN_SPLIT=3, --gpus 4 → 2 children (child 2 has 1 split, wastes 4 GPUs)."""
        num_splits, gpus_per_split = 3, 4
        assert compute_num_children(num_splits, gpus_per_split) == 2

        ranges = compute_child_split_ranges(num_splits, gpus_per_split)
        assert ranges == [(1, 2), (3, 3)]

        # Child 0: 2 splits
        assert compute_gpu_ids(1, 1, 4) == [0, 1, 2, 3]
        assert compute_gpu_ids(2, 1, 4) == [4, 5, 6, 7]

        # Child 1: 1 split on first 4 GPUs
        assert compute_gpu_ids(3, 3, 4) == [0, 1, 2, 3]


# ── Worker command generation ───────────────────────────────────────────

class TestWorkerCommandGeneration:
    """Verify that parent.py would generate correct worker commands."""

    def _worker_cmd(self, experiment, start, end, gpus_per_split, distributed=False):
        cmd = (f".venv/bin/python bolt/worker.py {experiment}"
               f" {start} {end} {gpus_per_split}")
        if distributed:
            cmd += " --distributed"
        return cmd

    def test_2_splits_8_gpus_distributed(self):
        ranges = compute_child_split_ranges(2, 8)
        cmds = [self._worker_cmd("12_hf_fol", s, e, 8, distributed=True)
                for s, e in ranges]
        assert cmds == [
            ".venv/bin/python bolt/worker.py 12_hf_fol 1 1 8 --distributed",
            ".venv/bin/python bolt/worker.py 12_hf_fol 2 2 8 --distributed",
        ]

    def test_2_splits_4_gpus_distributed(self):
        ranges = compute_child_split_ranges(2, 4)
        cmds = [self._worker_cmd("12_hf_fol", s, e, 4, distributed=True)
                for s, e in ranges]
        assert cmds == [
            ".venv/bin/python bolt/worker.py 12_hf_fol 1 2 4 --distributed",
        ]

    def test_8_splits_1_gpu_independent(self):
        ranges = compute_child_split_ranges(8, 1)
        cmds = [self._worker_cmd("8_fol_icl", s, e, 1)
                for s, e in ranges]
        assert cmds == [
            ".venv/bin/python bolt/worker.py 8_fol_icl 1 8 1",
        ]
