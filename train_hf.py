"""HuggingFace training utilities (peer to train.py).

Provides a reusable training loop for HuggingFace causal language models
using Accelerate for multi-GPU distribution with gradient accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from wandb_utils import WandbConfig, log_wandb_metrics, wandb_run_context


_SSM_MODEL_TYPES = {"mamba", "mamba2"}


def _uses_attention_mask(model):
    """Check whether the model accepts attention_mask (False for SSMs like Mamba)."""
    model_type = getattr(model.config, "model_type", "")
    return model_type not in _SSM_MODEL_TYPES


@dataclass
class HFTrainConfig:
    model_name_or_path: str = "gpt2"
    from_scratch: bool = False
    tokenize_mode: Literal["native", "direct"] = "direct"

    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    max_grad_norm: float = 1.0

    train_iters: int = 10_000
    test_every: int = 1_000
    test_iters: int = 1
    grad_accum_steps: int = 1
    mixed_precision: str = "bf16"
    use_tqdm: bool = False
    max_seq_len: int | None = None


# ---------------------------------------------------------------------------
# Tokenization adapters
# ---------------------------------------------------------------------------

def _make_direct_adapter():
    """Adapter for direct mode: numpy int arrays -> torch batch dicts.

    FOLLayerTask produces (xs, ys) where xs and ys are int32 numpy arrays
    with 0 as the pad token.  We map pad positions in labels to -100 so
    that they are ignored by cross-entropy.
    """
    def adapt(xs, ys):
        input_ids = torch.from_numpy(np.asarray(xs)).long()
        labels = torch.from_numpy(np.asarray(ys)).long()
        labels = labels.masked_fill(labels == 0, -100)
        return {"input_ids": input_ids, "labels": labels}
    return adapt


def _make_native_adapter(fol_tokenizer, hf_tokenizer, max_seq_len):
    """Adapter for native mode: decode FOL tokens, re-tokenize with HF tokenizer.

    Useful for finetuning pretrained models with their own tokenizer.
    """
    def adapt(xs, ys):
        # Decode the full sequences (xs is the input, ys is the target)
        # We reconstruct the full sequence from xs (prompt + all but last)
        # and ys (shifted targets)
        texts = fol_tokenizer.decode_batch_ids(xs, skip_pad=True)

        encoding = hf_tokenizer(
            texts,
            padding="max_length" if max_seq_len else "longest",
            truncation=bool(max_seq_len),
            max_length=max_seq_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Labels = input_ids shifted; mask padding to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return adapt


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_model(config: HFTrainConfig, vocab_size: int | None = None):
    """Build or load a HuggingFace causal LM based on config."""
    if config.from_scratch:
        model_cfg = AutoConfig.from_pretrained(config.model_name_or_path)
        if config.tokenize_mode == "direct" and vocab_size is not None:
            model_cfg.vocab_size = vocab_size
        model = AutoModelForCausalLM.from_config(model_cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    return model


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

def _eval_batches(model, adapter, test_iter, test_iters, accelerator, uses_attn=True):
    """Run test_iters batches and return mean loss and token accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(test_iters):
            xs, ys = next(test_iter)
            batch = adapter(xs, ys)
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

            attn_mask = batch.get("attention_mask") if uses_attn else None
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=attn_mask,
            ).logits

            labels = batch["labels"]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

            mask = labels != -100
            preds = logits.argmax(dim=-1)
            correct = ((preds == labels) & mask).sum()
            n_tokens = mask.sum()

            # Gather across ranks
            gathered = accelerator.gather_for_metrics(
                torch.stack([loss * n_tokens.float(), correct.float(), n_tokens.float()])
            )
            if gathered.dim() > 1:
                gathered = gathered.sum(dim=0)

            total_loss += gathered[0].item()
            total_correct += gathered[1].item()
            total_tokens += gathered[2].item()

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return {"loss": avg_loss, "acc": avg_acc}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _should_eval(step, train_iters, test_every):
    return ((step + 1) % test_every == 0) or ((step + 1) == train_iters)


def _default_print_fn(step, hist):
    train = hist["train"][-1]
    test = hist["test"][-1]
    print(
        f"ITER {step}:  train_loss={train['loss']:.4f}  "
        f"train_acc={train['acc']:.4f}  "
        f"test_loss={test['loss']:.4f}  "
        f"test_acc={test['acc']:.4f}"
    )


def train_hf(
    config: HFTrainConfig,
    train_iter,
    test_iter=None,
    *,
    fol_tokenizer=None,
    vocab_size: int | None = None,
    accelerator: Accelerator | None = None,
    print_fn=None,
    wandb_cfg: WandbConfig | None = None,
    seed: int = 42,
) -> tuple:
    """Train a HuggingFace causal LM on iterator-style data.

    Args:
        config: Training configuration.
        train_iter: Iterator yielding (xs, ys) numpy batches.
        test_iter: Optional eval iterator (defaults to train_iter).
        fol_tokenizer: FOLLayerTokenizer, needed for tokenize_mode="native".
        vocab_size: Vocab size for tokenize_mode="direct" + from_scratch.
        accelerator: HF Accelerator instance (created if None).
        print_fn: Custom print function(step, hist).
        seed: Random seed for reproducibility.

    Returns:
        (unwrapped_model, hist) where hist = {"train": [...], "test": [...]}.
    """
    if test_iter is None:
        test_iter = train_iter
    if print_fn is None:
        print_fn = _default_print_fn

    # --- Accelerator ---
    own_accelerator = accelerator is None
    if own_accelerator:
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.grad_accum_steps,
        )

    torch.manual_seed(seed)

    # --- Tokenization adapter ---
    if config.tokenize_mode == "direct":
        adapter = _make_direct_adapter()
    else:
        if fol_tokenizer is None:
            raise ValueError("fol_tokenizer required for tokenize_mode='native'")
        hf_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        adapter = _make_native_adapter(fol_tokenizer, hf_tokenizer, config.max_seq_len)

    # --- Model ---
    model = _build_model(config, vocab_size=vocab_size)
    uses_attn = _uses_attention_mask(model)
    model.train()

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.train_iters * config.warmup_frac)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.train_iters,
    )

    # --- Accelerate prepare ---
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # --- Training ---
    hist = {"train": [], "test": []}
    active_wandb_cfg = wandb_cfg if accelerator.is_main_process else None

    with wandb_run_context(active_wandb_cfg) as wandb:
        steps = range(config.train_iters)
        if config.use_tqdm and accelerator.is_main_process:
            steps = tqdm(steps, total=config.train_iters)

        for step in steps:
            # --- Forward / backward with gradient accumulation ---
            with accelerator.accumulate(model):
                xs, ys = next(train_iter)
                batch = adapter(xs, ys)
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}

                attn_mask = batch.get("attention_mask") if uses_attn else None
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=attn_mask,
                ).logits

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                )

                accelerator.backward(loss)

                if config.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # --- Eval ---
            if _should_eval(step, config.train_iters, config.test_every):
                train_metrics = _eval_batches(
                    model, adapter, train_iter, config.test_iters, accelerator,
                    uses_attn=uses_attn,
                )
                test_metrics = _eval_batches(
                    model, adapter, test_iter, config.test_iters, accelerator,
                    uses_attn=uses_attn,
                )

                hist["train"].append(train_metrics)
                hist["test"].append(test_metrics)

                if accelerator.is_main_process:
                    log_wandb_metrics(
                        wandb,
                        step=step + 1,
                        train=train_metrics,
                        test=test_metrics,
                    )
                    print_fn(step + 1, hist)

    # --- Unwrap and return ---
    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model, hist
