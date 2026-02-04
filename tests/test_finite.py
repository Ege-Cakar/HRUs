import json
from pathlib import Path
import pickle

import numpy as np
from array_record.python import array_record_module

from task.prop import Finite, ImplySizeTask


def _write_array_record(path: Path, records) -> None:
    writer = array_record_module.ArrayRecordWriter(str(path), "group_size:1")
    for record in records:
        writer.write(pickle.dumps(record, protocol=5))
    writer.close()


class DummyTask:
    def __init__(self, batch_size, batches) -> None:
        self.batch_size = batch_size
        self._iter = iter(batches)

    def __next__(self):
        return next(self._iter)

    def __iter__(self):
        return self


def test_finite_caches_and_shuffles() -> None:
    xs1 = np.array([0, 1, 2, 3], dtype=np.int32)
    ys1 = np.array([10, 11, 12, 13], dtype=np.int32)
    xs2 = np.array([4, 5, 6, 7], dtype=np.int32)
    ys2 = np.array([14, 15, 16, 17], dtype=np.int32)
    task = DummyTask(batch_size=3, batches=[(xs1, ys1), (xs2, ys2)])

    finite = Finite(task, k=5, seed=0)

    assert np.array_equal(
        finite.data[0], np.array([0, 1, 2, 3, 4], dtype=np.int32)
    )
    assert np.array_equal(
        finite.data[1], np.array([10, 11, 12, 13, 14], dtype=np.int32)
    )

    rng = np.random.default_rng(0)
    order1 = rng.permutation(5)
    order2 = rng.permutation(5)
    idxs1 = order1[:3]
    idxs2 = np.concatenate([order1[3:], order2[:1]])

    batch1 = next(finite)
    batch2 = next(finite)

    assert np.array_equal(batch1[0], finite.data[0][idxs1])
    assert np.array_equal(batch1[1], finite.data[1][idxs1])
    assert np.array_equal(batch2[0], finite.data[0][idxs2])
    assert np.array_equal(batch2[1], finite.data[1][idxs2])


def test_finite_batch_size_gt_k() -> None:
    xs = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    ys = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    task = DummyTask(batch_size=5, batches=[(xs, ys)])

    finite = Finite(task, k=3, seed=1)
    batch = next(finite)

    assert finite.data[0].shape == (3,)
    assert batch[0].shape == (5,)
    assert batch[1].shape == (5,)
    assert np.all(np.isin(batch[0], finite.data[0]))
    assert np.all(np.isin(batch[1], finite.data[1]))


def test_finite_with_imply_size_task(tmp_path: Path) -> None:
    size_dir = tmp_path / "size_02"
    size_dir.mkdir(parents=True)
    shard_path = size_dir / "shard_00000.array_record"

    rec1 = {
        "sequent": np.array([1, 2, 3], dtype=np.int32),
        "rule": np.array([[1, 0]], dtype=np.int32),
    }
    rec2 = {
        "sequent": np.array([4], dtype=np.int32),
        "rule": np.array([[2, 0]], dtype=np.int32),
    }
    rec3 = {
        "sequent": np.array([5, 6], dtype=np.int32),
        "rule": np.array([[3, 0]], dtype=np.int32),
    }
    _write_array_record(shard_path, [rec1, rec2, rec3])

    metadata = {
        "sizes": {
            "2": {
                "examples": 3,
                "shards": 1,
                "seconds": 0.0,
                "stats": {
                    "max_token": 6,
                    "max_pos": 0,
                    "max_seq": 3,
                },
            }
        }
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    task = ImplySizeTask(
        tmp_path,
        size_range=(2, 2),
        batch_size=2,
        shuffle=False,
        worker_count=0,
        drop_remainder=False,
    )
    finite = Finite(task, k=3, seed=0)

    assert finite.data[0].shape == (3, 3)
    assert finite.data[1].shape == (3, 2)
    assert np.array_equal(
        finite.data[0],
        np.array(
            [
                [1, 2, 3],
                [4, 0, 0],
                [5, 6, 0],
            ],
            dtype=np.int32,
        ),
    )
