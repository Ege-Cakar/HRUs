"""
Training utilities for Flax NNX
"""

# <codecell>
from dataclasses import dataclass, field
from functools import partial
import itertools
from typing import Any, Iterable

from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from common import new_seed, merge_dicts, gen1, gen2


def create_optimizer(model: nnx.Module,
                     lr: float = 1e-4,
                     optim=optax.adamw,
                     clip: float | None = None,
                     wrt: nnx.filterlib.Filter = nnx.Param,
                     **opt_kwargs) -> nnx.Optimizer:
    """Create an NNX optimizer for the model."""
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


def mse_mask(logits, labels):
    assert logits.shape[:2] == labels.shape, f'logit shape {logits.shape} not compatible with label shape {labels.shape}'

    targets = 2 * jax.nn.one_hot(labels, logits.shape[-1]) - 1
    out = ((targets - logits)**2).mean(axis=-1)
    mask = (labels != 0).astype(int)

    res = jnp.sum(out * mask)
    total = jnp.sum(mask)
    return res / total


def parse_loss_name(loss):
    if callable(loss):
        return loss

    loss_func = None
    if loss == 'bce':
        loss_func = optax.sigmoid_binary_cross_entropy
    elif loss == 'ce':
        loss_func = optax.softmax_cross_entropy_with_integer_labels
    elif loss == 'ce_mask':
        loss_func = ce_mask
    elif loss == 'mse':
        loss_func = optax.squared_error
    elif loss == 'mse_mask':
        loss_func = mse_mask
    else:
        raise ValueError(f'unrecognized loss name: {loss}')
    return loss_func


# TODO: simplify with static arg nums to cache loss_func (fix agent issue)
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
_train_step_cache: dict = {}


def train_step(optimizer: nnx.Optimizer, batch, loss_func):
    """Perform a single training step (JIT-compiled, cached per loss function)."""
    x, labels = batch
    
    # Get or create the JIT-compiled function for this loss
    if loss_func not in _train_step_cache:
        _train_step_cache[loss_func] = _make_train_step_fn(loss_func)
    
    return _train_step_cache[loss_func](optimizer, x, labels)



def loss_and_acc(optimizer: nnx.Optimizer, batch, loss='bce'):
    x, labels = batch
    logits = optimizer.model(x)
    loss_func = parse_loss_name(loss)
    loss_val = loss_func(logits, labels).mean()

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=-1)
    
    if len(labels.shape) == 2:
        # autoregressive branch
        mask = labels != 0
        res = jnp.sum((preds == labels) & mask)
        total = jnp.sum(mask)
        acc = res / total
    elif len(preds.shape) == 2:
        acc = jnp.mean(preds[:,-1] == labels)
    else:
        acc = jnp.mean(preds == labels)

    return {'loss': loss_val, 'acc': acc}


def decomp_flat_acc(optimizer: nnx.Optimizer, batch, loss=None):
    x, labels = batch
    logits = optimizer.model(x)

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=-1)
    
    true_pos = labels * preds
    true_neg = (1 - labels) * (1 - preds)
    false_pos = (1 - labels) * preds
    false_neg = labels * (1 - preds)

    return {
        'true_pos': jnp.mean(true_pos),
        'true_neg': jnp.mean(true_neg),
        'false_pos': jnp.mean(false_pos),
        'false_neg': jnp.mean(false_neg),
    }


def gen_acc_cot2(optimizer: nnx.Optimizer, batch, loss=None):
    xs, ys = batch
    ys = ys[:,2:]
    ans_idx = jnp.sum(ys != 0, axis=-1) - 1
    ans = ys[jnp.arange(len(ys)), ans_idx]

    traj = gen2(optimizer, xs)
    preds = extract_pred(traj)

    return {'gen_acc': jnp.mean(preds == ans)}


def gen_acc_cot1(optimizer: nnx.Optimizer, batch, loss=None):
    xs, ys = batch
    ys = ys[:,2:]
    ans_idx = jnp.sum(ys != 0, axis=-1) - 1
    ans = ys[jnp.arange(len(ys)), ans_idx]

    traj = gen1(optimizer, xs)
    preds = extract_pred(traj)

    return {'gen_acc': jnp.mean(preds == ans)}


def gen_acc_rl(optimizer: nnx.Optimizer, batch, loss=None):
    xs, ys = batch

    traj = gen2(optimizer, xs)
    preds = extract_pred(traj)

    return {'gen_acc': jnp.mean(preds == ys)}


def extract_pred(traj):
    # assumes no/yes classification offset by 1 for padding
    no_occ = jnp.argmax(traj == 1, axis=1)
    no_occ = jnp.where(no_occ == 0, jnp.inf, no_occ)
    yes_occ = jnp.argmax(traj == 2, axis=1)
    yes_occ = jnp.where(yes_occ == 0, jnp.inf, yes_occ)

    preds = jnp.argmin(jnp.stack((no_occ, yes_occ), axis=1), axis=1) + 1
    preds = jnp.where(no_occ != yes_occ, preds, jnp.inf)
    return preds


def print_gen(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1]["loss"]:.4f}   train_acc={hist["train"][-1]["gen_acc"]:.4f}   test_loss={hist["test"][-1]["loss"]:.4f}   test_acc={hist["test"][-1]["gen_acc"]:.4f}')


def train(config, train_iter, 
          test_iter=None, 
          loss='ce',
          eval_fns: Iterable=None, print_fn=None,
          summary_fn=None,
          train_iters=10_000, test_iters=1, test_every=1_000, save_params=False,
          early_stop_n=None, early_stop_key='loss', early_stop_decision='min',
          optim=optax.adamw,
          seed=None, use_tqdm=False, wdb=None,
          **opt_kwargs):

    if seed is None:
        seed = new_seed()

    if test_iter is None:
        test_iter = train_iter
    
    if eval_fns is None:
        eval_fns = [loss_and_acc]
    
    if print_fn is None:
        print_fn = _print_status
    
    loss_func = parse_loss_name(loss)

    if isinstance(config, nnx.Optimizer):
        optimizer = config
    else:
        rngs = nnx.Rngs(seed)
        model = config.to_model(rngs=rngs)
        optimizer = create_optimizer(model, optim=optim, **opt_kwargs)

    hist = {
        'train': [],
        'test': [],
        'params': [nnx.state(optimizer.model)] if save_params else [],
        'summary': []
    }

    it = zip(range(train_iters), train_iter)
    if use_tqdm:
        it = tqdm(it, total=train_iters)

    for step, batch in it:
        train_step(optimizer, batch, loss_func)

        if ((step + 1) % test_every == 0) or ((step + 1) == train_iters):
            all_train = []
            all_test = []

            for _, train_batch, test_batch in zip(range(test_iters), train_iter, test_iter):
                all_train.append(merge_dicts([fn(optimizer, train_batch, loss=loss) for fn in eval_fns]))
                all_test.append(merge_dicts([fn(optimizer, test_batch, loss=loss) for fn in eval_fns]))
            
            all_train = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *all_train)
            all_test = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *all_test)

            hist['train'].append(all_train)
            hist['test'].append(all_test)

            if summary_fn is not None:
                summ = summary_fn(optimizer)
                hist['summary'].append(summ)

            print_fn((step + 1), hist)

            if wdb is not None:
                train_obj = hist['train'][-1]
                test_obj = hist['test'][-1]
                
                train_log = {f'train_{k}': v for k, v in train_obj.items()}
                test_log = {f'test_{k}': v for k, v in test_obj.items()}
                log = train_log | test_log

                if summary_fn is not None:
                    summ_obj = hist['summary'][-1]
                    summ_log = {f'summary_{k}': v for k, v in summ_obj.items()}
                    log = log | summ_log
                
                wdb.log(log)

            if save_params:
                hist['params'].append(nnx.state(optimizer.model))
        
            if early_stop_n is not None and len(hist['train']) > early_stop_n:
                last_n_metrics = np.array([m[early_stop_key] for m in hist['train'][-early_stop_n - 1:]])
                if early_stop_decision == 'min' and np.all(last_n_metrics[0] < last_n_metrics[1:]) \
                or early_stop_decision == 'max' and np.all(last_n_metrics[0] > last_n_metrics[1:]):
                    print(f'info: stopping early with {early_stop_key} =', last_n_metrics[-1])
                    break
    
    return optimizer, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1]["loss"]:.4f}   train_acc={hist["train"][-1]["acc"]:.4f}   test_loss={hist["test"][-1]["loss"]:.4f}   test_acc={hist["test"][-1]["acc"]:.4f}')


def reinforce(optimizer: nnx.Optimizer, train_iter, 
              action_fn, reward_fn, rl_loss_fn,
              train_iters=10_000, 
              test_iter=None, test_iters=10, test_every=1000, loss=None,
              eval_fns=None, print_fn=None,
              save_params=False,
              use_tqdm=False):

    if test_iter is None:
        test_iter = train_iter

    if eval_fns is None:
        eval_fns = [loss_and_acc]
    
    if print_fn is None:
        print_fn = _print_rl_status

    it = zip(range(train_iters), train_iter)
    if use_tqdm:
        it = tqdm(it, total=train_iters)

    hist = {
        'rew': [],
        'test': [],
        'params': []
    }
    
    for step, batch in it:
        rl_step(optimizer, batch, action_fn, reward_fn, rl_loss_fn)

        if ((step + 1) % test_every == 0) or ((step + 1) == train_iters):
            avg_rew = 0
            all_test = []

            for _, test_batch in zip(range(test_iters), test_iter):
                all_test.append(merge_dicts([fn(optimizer, test_batch, loss=loss) for fn in eval_fns]))
                rew = compute_reward(optimizer, batch, action_fn, reward_fn)
                avg_rew += np.mean(rew) / test_iters
            
            all_test = jax.tree.map(lambda *xs: np.mean(xs), *all_test)
            hist['test'].append(all_test)
            hist['rew'].append(avg_rew)

            print_fn(step+1, hist)

            if save_params:
                hist['params'].append(nnx.state(optimizer.model))
    
    return optimizer, hist


def _print_rl_status(step, hist):
    print(f'ITER {step}:  test_rew={hist["rew"][-1]:.4f}   test_acc={hist["test"][-1]["gen_acc"]:.4f}')


@nnx.jit
def rl_step(optimizer: nnx.Optimizer, batch, act_fn, rew_fn, rl_loss_fn):
    xs, ys = batch
    traj = act_fn(optimizer, xs)
    rew = rew_fn(traj, ys)
    
    def loss_fn(model):
        return rl_loss_fn(model, traj, rew)
    
    grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, optimizer.wrt))(optimizer.model)
    optimizer.update(grads)


def compute_reward(optimizer: nnx.Optimizer, batch, act_fn, rew_fn):
    xs, ys = batch
    traj = act_fn(optimizer, xs)
    rew = rew_fn(traj, ys)
    return rew


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
    wdb_proj: str = None

    def run(self):
        if self.wdb_proj is not None:
            from dataclasses import asdict
            # wdb = wandb.init(project=self.wdb_proj, config=asdict(self.config))
        else:
            wdb = None

        self.optimizer, self.hist = train(self.config, train_iter=self.train_task, test_iter=self.test_task, wdb=wdb, **self.train_args)
    
    def eval(self, task, eval_fns, n_iters=1, prefix=None):
        all_res = []
        loss = self.train_args.get('loss', None)

        for _ in range(n_iters):
            batch = next(task)
            all_res.append(merge_dicts([fn(self.optimizer, batch, loss=loss) for fn in eval_fns]))

        all_res = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *all_res)
        if prefix is not None:
            all_res = {prefix: all_res}

        self.info = self.info | all_res
