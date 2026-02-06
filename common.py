"""Common utilities"""

from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.special import logsumexp
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')

def set_theme():
    sns.set_theme(style='ticks', font_scale=1.25, rc={
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.figsize': (4, 3)
    })


def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)


def t(xs):
    return np.swapaxes(xs, -2, -1)


def split_batch(batch_size, max_size):
    fac = np.ceil(batch_size / max_size)
    batch_size = np.round(batch_size / fac)
    return int(batch_size), int(fac)


def split_cases(all_cases, run_split, shuffle_seed=None):
    run_idx = sys.argv[1]
    try:
        run_idx = int(run_idx) % run_split
    except ValueError:
        print(f'warn: unable to parse index {run_idx}, setting run_idx=0')
        run_idx = 0

    print('RUN IDX', run_idx)

    all_cases = np.array(all_cases)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(all_cases)

    all_cases = np.array_split(all_cases, run_split)[run_idx]
    return list(all_cases)


def summon_dir(path: str, clear_if_exists=False):
    new_dir = Path(path)
    if new_dir.exists() and clear_if_exists:
        shutil.rmtree(new_dir)

    new_dir.mkdir(parents=True)
    return new_dir


def collate_dfs(df_dir, show_progress=False, concat=True):
    pkl_path = Path(df_dir)
    dfs = []

    it = pkl_path.iterdir()
    if show_progress:
        it = tqdm(list(it))
    
    for f in it:
        if f.suffix == '.pkl':
            try:
                df = pd.read_pickle(f)
                dfs.append(df)
            except Exception as e:
                print(f'warn: fail to read {f}')
                print(e)

    if concat:
        dfs = pd.concat(dfs)

    return dfs


def merge_dicts(dicts):
    all_dicts = dicts[0]
    for d in dicts[1:]:
        all_dicts = all_dicts | d
    
    return all_dicts


def rule_membership_accuracy(pred_rules, rule_set_batch, rule_set_mask):
    """Compute accuracy where a prediction is correct if it appears in the rule set."""
    preds = jnp.asarray(pred_rules)
    rule_set = jnp.asarray(rule_set_batch)
    mask = jnp.asarray(rule_set_mask).astype(bool)
    matches = jnp.all(preds[:, None, :] == rule_set, axis=-1)
    matches = jnp.where(mask, matches, False)
    return jnp.mean(jnp.any(matches, axis=-1))


def multiclass_nll_from_logits(logits, labels):
    """Mean negative log-likelihood for integer labels."""
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels).astype(jnp.int32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    return -jnp.mean(gathered)


def expected_calibration_error_from_logits(logits, labels, n_bins=10):
    """Expected calibration error (ECE) for multiclass logits."""
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels).astype(jnp.int32)
    probs = jax.nn.softmax(logits, axis=-1)
    conf = jnp.max(probs, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    correct = preds == labels

    bins = jnp.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = conf.size
    for idx in range(n_bins):
        low = bins[idx]
        high = bins[idx + 1]
        if idx == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        count = jnp.sum(mask)
        if count == 0:
            continue
        bin_acc = jnp.mean(correct[mask])
        bin_conf = jnp.mean(conf[mask])
        ece += jnp.abs(bin_acc - bin_conf) * (count / total)
    return ece


def gen1(optimizer, xs, **kwargs):
    return generate(optimizer, xs, idx=1, **kwargs)


def gen2(optimizer, xs, **kwargs):
    return generate(optimizer, xs, idx=2, **kwargs)


def generate(optimizer, xs, idx, beta=1, seed=None):
    """Autoregressive generation using the model in optimizer."""
    if seed is None:
        seed = new_seed()

    source = jax.random.key(seed)
    while idx < xs.shape[1] - 1:
        key, source = jax.random.split(source)
        xs = _gen_pass(key, optimizer, xs, idx, beta)
        idx += 1
    
    return xs


def _gen_pass(key, optimizer, xs, idx, beta):
    logits = optimizer.model(xs)
    preds = jax.random.categorical(key, beta * logits)
    xs = xs.at[:,idx+1].set(preds[:,idx])
    return xs
