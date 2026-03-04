"""Training utilities for Flax NNX."""

from dataclasses import dataclass, field
from typing import Callable, Iterable

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from common import new_seed, merge_dicts


def warmup_cosine_schedule(peak_lr, train_iters, warmup_frac=0.1, end_lr=0.0):
    """Linear warmup then cosine decay to end_lr."""
    warmup_steps = int(train_iters * warmup_frac)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=train_iters,
        end_value=end_lr,
    )


def warmup_constant_schedule(peak_lr, train_iters, warmup_frac=0.1):
    """Linear warmup then constant at peak_lr."""
    warmup_steps = int(train_iters * warmup_frac)
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=peak_lr,
                                  transition_steps=warmup_steps),
            optax.constant_schedule(peak_lr),
        ],
        boundaries=[warmup_steps],
    )


def linear_decay_schedule(init_lr, train_iters, end_lr=0.0):
    """Simple linear decay from init_lr to end_lr."""
    return optax.linear_schedule(
        init_value=init_lr,
        end_value=end_lr,
        transition_steps=train_iters,
    )


def create_optimizer(model: nnx.Module,
                     lr: float | Callable = 1e-4,
                     optim=optax.adamw,
                     clip: float | None = None,
                     wrt: nnx.filterlib.Filter = nnx.Param,
                     **opt_kwargs) -> nnx.Optimizer:
    """Create an NNX optimizer for the model.

    Args:
        lr: Learning rate. Either a float for constant LR or a callable
            schedule (e.g. from warmup_cosine_schedule).
    """
    tx = optim(learning_rate=lr, **opt_kwargs)

    if clip is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(clip),
            tx
        )

    return nnx.ModelAndOptimizer(model, tx, wrt=wrt)


def ce_mask(logits, labels):
    assert logits.shape[:2] == labels.shape, f'logit shape {logits.shape} not compatible with label shape {labels.shape}'

    out = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mask = (labels != 0).astype(int)

    res = jnp.sum(out * mask)
    total = jnp.sum(mask)
    return res / total


_LOSS_MAP = {
    'bce': optax.sigmoid_binary_cross_entropy,
    'ce': optax.softmax_cross_entropy_with_integer_labels,
    'ce_mask': ce_mask,
    'mse': optax.squared_error,
}


def parse_loss_name(loss):
    if callable(loss):
        return loss

    if loss not in _LOSS_MAP:
        raise ValueError(f'unrecognized loss name: {loss}')
    return _LOSS_MAP[loss]


def _make_train_step_fn(loss_func):
    """Create a JIT-compiled train step function for a specific loss function."""
    @nnx.jit
    def _train_step_jit(optimizer: nnx.Optimizer, x, labels):
        def loss_fn(model):
            logits = model(x)
            train_loss = loss_func(logits, labels)

            if len(labels.shape) > 1 and logits.shape == train_loss.shape:
                train_loss = train_loss.mean(axis=-1)

            return train_loss.mean()

        loss_val, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, optimizer.wrt)
        )(optimizer.model)
        optimizer.update(grads)
        return loss_val
    
    return _train_step_jit


# Cache of JIT-compiled train step functions per loss function
_train_step_cache: dict[Callable, Callable] = {}


def _get_train_step(loss_func):
    if loss_func not in _train_step_cache:
        _train_step_cache[loss_func] = _make_train_step_fn(loss_func)
    return _train_step_cache[loss_func]


def train_step(optimizer: nnx.Optimizer, batch, loss_func):
    """Perform a single training step (JIT-compiled, cached per loss function)."""
    x, labels = batch
    return _get_train_step(loss_func)(optimizer, x, labels)


def _preds_from_logits(logits):
    if len(logits.shape) == 1:
        return logits > 0
    return logits.argmax(axis=-1)


def _accuracy_from_preds(preds, labels):
    if len(labels.shape) == 2:
        # autoregressive branch
        mask = labels != 0
        res = jnp.sum((preds == labels) & mask)
        total = jnp.sum(mask)
        return res / total
    if len(preds.shape) == 2:
        return jnp.mean(preds[:, -1] == labels)
    return jnp.mean(preds == labels)


def loss_and_acc(optimizer: nnx.Optimizer, batch, loss='bce'):
    x, labels = batch
    logits = optimizer.model(x)
    loss_func = parse_loss_name(loss)
    loss_val = loss_func(logits, labels).mean()

    preds = _preds_from_logits(logits)
    acc = _accuracy_from_preds(preds, labels)

    return {'loss': loss_val, 'acc': acc}


def _resolve_seed(seed):
    return new_seed() if seed is None else seed


def _resolve_iterators(train_iter, test_iter):
    return train_iter, train_iter if test_iter is None else test_iter


def _resolve_eval_fns(eval_fns):
    return [loss_and_acc] if eval_fns is None else eval_fns


def _resolve_print_fn(print_fn):
    return _print_status if print_fn is None else print_fn


def _build_optimizer(config, seed, optim, opt_kwargs):
    if isinstance(config, nnx.Optimizer):
        return config
    rngs = nnx.Rngs(seed)
    model = config.to_model(rngs=rngs)
    return create_optimizer(model, optim=optim, **opt_kwargs)


def _init_history(optimizer, save_params):
    return {
        'train': [],
        'test': [],
        'params': [nnx.state(optimizer.model)] if save_params else [],
        'summary': []
    }


def _iter_steps(train_iter, train_iters, use_tqdm):
    it = zip(range(train_iters), train_iter)
    return tqdm(it, total=train_iters) if use_tqdm else it


def _eval_batch(optimizer, batch, eval_fns, loss):
    return merge_dicts([fn(optimizer, batch, loss=loss) for fn in eval_fns])


def _aggregate_metrics(metrics_list):
    if not metrics_list:
        return {}
    return jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *metrics_list)


def _collect_eval(optimizer, train_iter, test_iter, eval_fns, loss, test_iters):
    all_train = []
    all_test = []
    for _, train_batch, test_batch in zip(range(test_iters), train_iter, test_iter):
        all_train.append(_eval_batch(optimizer, train_batch, eval_fns, loss))
        all_test.append(_eval_batch(optimizer, test_batch, eval_fns, loss))
    return _aggregate_metrics(all_train), _aggregate_metrics(all_test)


def _should_eval(step, train_iters, test_every):
    return ((step + 1) % test_every == 0) or ((step + 1) == train_iters)


def train(config, train_iter,
          test_iter=None,
          loss='ce',
          eval_fns: Iterable = None, print_fn=None,
          summary_fn=None,
          train_iters=10_000, test_iters=1, test_every=1_000, save_params=False,
          optim=optax.adamw,
          seed=None, use_tqdm=False,
          **opt_kwargs):

    seed = _resolve_seed(seed)
    train_iter, test_iter = _resolve_iterators(train_iter, test_iter)
    eval_fns = _resolve_eval_fns(eval_fns)
    print_fn = _resolve_print_fn(print_fn)
    loss_func = parse_loss_name(loss)

    optimizer = _build_optimizer(config, seed, optim, opt_kwargs)
    hist = _init_history(optimizer, save_params)

    for step, batch in _iter_steps(train_iter, train_iters, use_tqdm):
        train_step(optimizer, batch, loss_func)

        if _should_eval(step, train_iters, test_every):
            all_train, all_test = _collect_eval(
                optimizer, train_iter, test_iter, eval_fns, loss, test_iters
            )

            hist['train'].append(all_train)
            hist['test'].append(all_test)

            if summary_fn is not None:
                hist['summary'].append(summary_fn(optimizer))

            print_fn((step + 1), hist)

            if save_params:
                hist['params'].append(nnx.state(optimizer.model))

    return optimizer, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1]["loss"]:.4f}   train_acc={hist["train"][-1]["acc"]:.4f}   test_loss={hist["test"][-1]["loss"]:.4f}   test_acc={hist["test"][-1]["acc"]:.4f}')


@dataclass
class Case:
    name: str
    config: dataclass
    train_task: Iterable | None = None
    test_task: Iterable | None = None
    train_args: dict = field(default_factory=dict)
    optimizer: nnx.Optimizer | None = None
    hist: list = None
    info: dict = field(default_factory=dict)

    def run(self):
        self.optimizer, self.hist = train(self.config, train_iter=self.train_task, test_iter=self.test_task, **self.train_args)
    
    def eval(self, task, eval_fns, n_iters=1, prefix=None):
        loss = self.train_args.get('loss', None)
        all_res = []
        for _ in range(n_iters):
            batch = next(task)
            all_res.append(_eval_batch(self.optimizer, batch, eval_fns, loss))

        all_res = _aggregate_metrics(all_res)
        if prefix is not None:
            all_res = {prefix: all_res}

        self.info = self.info | all_res
