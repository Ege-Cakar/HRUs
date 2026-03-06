"""Tests for HuggingFace training utilities."""

from contextlib import contextmanager
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")

import train_hf as train_hf_module
import wandb_utils
from train_hf import HFTrainConfig, train_hf
from wandb_utils import WandbConfig


class _SimpleIterator:
    def __init__(self, xs, ys):
        self.xs = np.asarray(xs, dtype=np.int64)
        self.ys = np.asarray(ys, dtype=np.int64)
        self.batch_size = int(self.xs.shape[0])

    def __iter__(self):
        return self

    def __next__(self):
        return self.xs, self.ys


class _FakeWandbModule:
    def __init__(self):
        self.login_calls = []
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0

    def login(self, **kwargs):
        self.login_calls.append(kwargs)

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return object()

    def log(self, payload, step=None):
        self.log_calls.append({"payload": payload, "step": step})

    def finish(self):
        self.finish_calls += 1


class _FakeScheduler:
    def step(self):
        return None


class _FakeAccelerator:
    def __init__(self, *, is_main_process=True):
        self.is_main_process = bool(is_main_process)
        self.sync_gradients = True
        self.device = torch.device("cpu")
        self.num_processes = 1
        self.process_index = 0

    def prepare(self, *args):
        return args

    @contextmanager
    def accumulate(self, model):
        yield

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)

    def gather_for_metrics(self, tensor):
        return tensor

    def unwrap_model(self, model):
        return model


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)
        self.config = types.SimpleNamespace(model_type="gpt2")

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return types.SimpleNamespace(logits=logits)


def test_train_hf_logs_metrics_to_wandb(monkeypatch, tmp_path):
    fake_wandb = _FakeWandbModule()
    key_path = tmp_path / "wandb.txt"
    key_path.write_text("hf-key\n", encoding="utf-8")

    monkeypatch.setattr(
        wandb_utils.importlib,
        "import_module",
        lambda name: fake_wandb,
    )
    monkeypatch.setattr(
        train_hf_module,
        "_build_model",
        lambda config, vocab_size=None: _TinyLM(vocab_size=vocab_size or 16),
    )
    monkeypatch.setattr(
        train_hf_module,
        "get_cosine_schedule_with_warmup",
        lambda optimizer, num_warmup_steps, num_training_steps: _FakeScheduler(),
    )

    xs = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int64)
    ys = np.array([[2, 3, 4, 5], [3, 4, 5, 6]], dtype=np.int64)
    train_iter = _SimpleIterator(xs, ys)
    test_iter = _SimpleIterator(xs, ys)
    accelerator = _FakeAccelerator(is_main_process=True)

    model, hist = train_hf(
        HFTrainConfig(
            model_name_or_path="stub",
            from_scratch=True,
            tokenize_mode="direct",
            lr=1e-3,
            train_iters=2,
            test_every=1,
            test_iters=1,
            grad_accum_steps=1,
            mixed_precision="no",
            use_tqdm=False,
        ),
        train_iter,
        test_iter=test_iter,
        vocab_size=16,
        accelerator=accelerator,
        print_fn=lambda step, hist: None,
        wandb_cfg=WandbConfig(
            enabled=True,
            project="hf-test",
            name="hf-run",
            api_key_path=key_path,
        ),
    )

    assert len(hist["train"]) == 2
    assert fake_wandb.login_calls == [{"key": "hf-key", "relogin": True}]
    assert fake_wandb.init_calls[0]["project"] == "hf-test"
    assert fake_wandb.log_calls[0]["step"] == 1
    assert fake_wandb.log_calls[1]["step"] == 2
    assert "train/loss" in fake_wandb.log_calls[0]["payload"]
    assert "test/acc" in fake_wandb.log_calls[0]["payload"]
    assert fake_wandb.finish_calls == 1
    assert isinstance(model, _TinyLM)


def test_train_hf_skips_wandb_on_non_main_process(monkeypatch, tmp_path):
    key_path = tmp_path / "wandb.txt"
    key_path.write_text("hf-key\n", encoding="utf-8")

    monkeypatch.setattr(
        wandb_utils.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected import: {name}")),
    )
    monkeypatch.setattr(
        train_hf_module,
        "_build_model",
        lambda config, vocab_size=None: _TinyLM(vocab_size=vocab_size or 16),
    )
    monkeypatch.setattr(
        train_hf_module,
        "get_cosine_schedule_with_warmup",
        lambda optimizer, num_warmup_steps, num_training_steps: _FakeScheduler(),
    )

    xs = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int64)
    ys = np.array([[2, 3, 4, 5], [3, 4, 5, 6]], dtype=np.int64)
    accelerator = _FakeAccelerator(is_main_process=False)

    _, hist = train_hf(
        HFTrainConfig(
            model_name_or_path="stub",
            from_scratch=True,
            tokenize_mode="direct",
            lr=1e-3,
            train_iters=1,
            test_every=1,
            test_iters=1,
            grad_accum_steps=1,
            mixed_precision="no",
            use_tqdm=False,
        ),
        _SimpleIterator(xs, ys),
        test_iter=_SimpleIterator(xs, ys),
        vocab_size=16,
        accelerator=accelerator,
        print_fn=lambda step, hist: None,
        wandb_cfg=WandbConfig(
            enabled=True,
            project="hf-test",
            name="hf-run",
            api_key_path=key_path,
        ),
    )

    assert len(hist["train"]) == 1
