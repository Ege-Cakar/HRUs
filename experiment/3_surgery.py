"""Playground: small Transformer on ImplySizeTask."""

#<codecell>
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.transformer import TransformerConfig
from task.prop import Finite, ImplySizeTask
from task.prop_gen.util.tokenize import char_to_id, id_to_char, id_to_rule_type, tokenize
from train import train
from common import rule_membership_accuracy, new_seed


SEED = new_seed()
RULE_CLASSES = 12  # rule_type_to_id max (11) + 1 for pad


def make_metrics_fn():
    def _metrics(optimizer, batch, loss=None):
        if len(batch) == 2:
            xs, labels = batch
            rule_set_batch = None
            rule_set_mask = None
        elif len(batch) == 4:
            xs, labels, rule_set_batch, rule_set_mask = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        logits = optimizer.model(xs)
        loss_val = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        preds = jnp.argmax(logits, axis=-1)
        rule_pred = preds[:, 0]
        pos_pred = preds[:, 1]
        rule_true = labels[:, 0]
        pos_true = labels[:, 1]
        rule_acc = jnp.mean(rule_pred == rule_true)
        pos_acc = jnp.mean(pos_pred == pos_true)
        joint_acc = jnp.mean((rule_pred == rule_true) & (pos_pred == pos_true))
        metrics = {
            "loss": loss_val,
            "rule_acc": rule_acc,
            "pos_acc": pos_acc,
            "joint_acc": joint_acc,
        }
        if rule_set_batch is not None and rule_set_mask is not None:
            pred_rules = jnp.stack([rule_pred, pos_pred], axis=-1)
            metrics["rule_membership_acc"] = rule_membership_accuracy(
                pred_rules, rule_set_batch, rule_set_mask
            )
        return metrics

    return _metrics


def make_print_fn():
    def _print(step, hist):
        train_metrics = hist["train"][-1]
        test_metrics = hist["test"][-1]
        msg = (
            "ITER {}: train_loss={:.4f} train_joint={:.4f} "
            "test_loss={:.4f} test_joint={:.4f}"
        ).format(
            step,
            train_metrics["loss"],
            train_metrics["joint_acc"],
            test_metrics["loss"],
            test_metrics["joint_acc"],
        )
        if "rule_membership_acc" in test_metrics:
            msg += " test_member={:.4f}".format(test_metrics["rule_membership_acc"])
        print(msg)

    return _print


def eval_on_sizes(
    optimizer,
    ds_path: Path,
    sizes,
    batch_size: int,
    n_iters: int = 5,
):
    metrics_fn = make_metrics_fn()
    for size in sizes:
        task = ImplySizeTask(
            ds_path,
            size_range=(size, size),
            batch_size=batch_size,
            shuffle=True,
            drop_remainder=False,
            worker_count=0,
            return_rule_sets=True,
        )
        metrics = []
        for _ in range(n_iters):
            metrics.append(metrics_fn(optimizer, next(task)))
        avg = {
            k: float(np.mean([float(m[k]) for m in metrics])) for k in metrics[0]
        }
        msg = (
            "SIZE {:02d}: loss={:.4f} rule_acc={:.4f} pos_acc={:.4f} joint_acc={:.4f}"
        ).format(
            size,
            avg["loss"],
            avg["rule_acc"],
            avg["pos_acc"],
            avg["joint_acc"],
        )
        if "rule_membership_acc" in avg:
            msg += " rule_member={:.4f}".format(avg["rule_membership_acc"])
        print(msg)


DS_PATH = ROOT / "task" / "prop_gen" / "data" / "toy_imply"
TRAIN_SIZES = (2, 5)
TEST_SIZES = (6, 12)

BATCH_SIZE = 64
TRAIN_ITERS = 1000
TEST_EVERY = 200
TEST_ITERS = 1


train_task = ImplySizeTask(
    DS_PATH,
    size_range=TRAIN_SIZES,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_remainder=True,
    worker_count=0,
)

test_task = ImplySizeTask(
    DS_PATH,
    size_range=TEST_SIZES,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_remainder=False,
    worker_count=0,
    return_rule_sets=True,
)

train_stats = train_task.stats
test_stats = test_task.stats

max_token = max(train_stats["max_token"], test_stats["max_token"])
max_pos = max(train_stats["max_pos"], test_stats["max_pos"])
max_seq = max(train_stats["max_seq"], test_stats["max_seq"])

n_vocab = max(128, max_token + 1)
n_seq = max(128, max_seq)
n_out = max(RULE_CLASSES, max_pos + 1, TEST_SIZES[1] + 1)

print(
    "DATA STATS: max_token={} max_pos={} max_seq={} n_vocab={} n_out={} n_seq={}".format(
        max_token, max_pos, max_seq, n_vocab, n_out, n_seq
    )
)

# NOTE: testing with finite dataset
train_task = Finite(train_task, k=1000)


config = TransformerConfig(
    n_vocab=n_vocab,
    n_seq=n_seq,
    n_layers=2,
    n_hidden=64,
    n_heads=1,
    n_out=n_out,
    n_pred_tokens=2,
    pos_encoding="none",
    layer_norm=True,
    use_bias=True,
    dropout_rate=0.0,
    output_mode="last_nonpad",
    pad_token_id=0,
    use_swiglu=False,
    use_sow=False
)

metrics_fn = make_metrics_fn()

optimizer, hist = train(
    config,
    train_iter=train_task,
    test_iter=test_task,
    loss="ce",
    eval_fns=[metrics_fn],
    print_fn=make_print_fn(),
    train_iters=TRAIN_ITERS,
    test_iters=TEST_ITERS,
    test_every=TEST_EVERY,
    lr=1e-3,
    seed=SEED,
    use_tqdm=False,
)

print("\nEval on longer sizes:")
eval_on_sizes(
    optimizer,
    DS_PATH,
    range(TEST_SIZES[0], TEST_SIZES[1] + 1),
    BATCH_SIZE,
    n_iters=TEST_ITERS,
)

# <codecell>
### IMPLEMENTING COMPARISON TO SIMPLE MATCH
xs_train, ys_train = train_task.data
model = optimizer.model

E = np.array(model.embed.embedding[:])
E[0] = 0
xs_train_emb = E[xs_train]
xs_train_feat = xs_train_emb.sum(axis=1)

# <codecell>
last_tokens = xs_train[jnp.arange(xs_train.shape[0]), jnp.sum(xs_train != 0, axis=1) - 1]
last_tokens

# <codecell>
custom_ex = '( p2 → p1 ) , ⊥ , p3 , ( p1 → p2 ) ⊢ p3'
ids = [char_to_id(c) for c in custom_ex.split(' ')]
xs_pred = jnp.array([ids])

def pred_single(xs_pred, readable=True):
    xs_train, ys_train = train_task.data
    model = optimizer.model

    E = np.array(model.embed.embedding[:])
    E[0] = 0
    xs_train_emb = E[xs_train]
    xs_train_feat = xs_train_emb.sum(axis=1)
    last_tokens = xs_train[jnp.arange(xs_train.shape[0]), jnp.sum(xs_train != 0, axis=1) - 1]
    xs_pred_feat = E[xs_pred].sum(axis=1)

    last_tok_pred = xs_pred[0,-1]
    xs_sel = xs_train_feat[last_tokens == last_tok_pred]
    ys_sel = ys_train[last_tokens == last_tok_pred]

    weights = jax.nn.softmax((xs_sel @ xs_pred_feat.T).squeeze())

    all_res = {}
    for cat in [0, 1]:
        opts = np.unique(ys_sel[:,cat])
        res = {}
        for opt in opts:
            opt_weight = weights[ys_sel[:,cat] == opt].sum()
            res[opt.item()] = opt_weight

        if cat == 0:
            if readable:
                res = {id_to_rule_type[k].__name__: v for k, v in res.items()}
            else:
                res = {k: v for k, v in res.items()}

        all_res[cat] = res

    return all_res

pred_single(xs_pred)

# %%
### Visualization attention weights and predictions
xs, _, ys, ys_mask = next(test_task)


# <codecell>
optimizer.model.blocks[0].attn.use_sow = True
optimizer.model.blocks[1].attn.use_sow = True

# <codecell>
# custom_ex = '( p1 → p2 ) , p3 , ( p1 → p3 ) ⊢ ( p1 → p3 )'
# custom_ex = 'p1 , ( p1 → p2 ) , ( p1 → p3 ) ⊢ p3'
# custom_ex = '( p2 → p1 ) , ⊥ , p3 , ( p1 → p2 ) ⊢ p3'
custom_ex = 'p1 , ( p1 → ⊤ ) , ⊥ ⊢ ( p2 → ( ( ( ⊥ → p1 ) → ⊤ ) → ⊥ ) ) → p1 )'
ids = [char_to_id(c) for c in custom_ex.split(' ')]
xs = jnp.array([ids])
xs.shape

# <codecell>
logits = optimizer.model(xs)

# <codecell>
ex_idx = 0
ex = xs[ex_idx]
ex_len = int(jnp.sum(ex != 0))
ex = ex[:ex_len]

ex_names = [id_to_char(int(i)) for i in ex]

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for layer in range(2):
    for head in range(4):
        if head >= 1:
            break
        attn = optimizer.model.blocks[layer].attn.attn_weights[-1]
        ax = axs[layer, head]
        ax.imshow(attn[ex_idx, head, :ex_len, :ex_len])
        ax.set_title(f"Layer {layer+1} Head {head+1}")
        ax.set_xticks(range(len(ex)))
        ax.set_xticklabels(ex_names)
        ax.set_yticks(range(len(ex)))
        ax.set_yticklabels(ex_names)
plt.show()

print(''.join(ex_names))

ex_rule = ys[ex_idx]
for y in ex_rule:
    if y[0] != 0:
        print(id_to_rule_type[int(y[0])].__name__, int(y[1]))


ex_probs = jax.nn.softmax(logits[ex_idx], axis=-1)
ids = range(1, ex_probs.shape[1] - 1)
ids_names = ["0"] + [id_to_rule_type[i].__name__ for i in ids]

ids_names += ["0"] * (ex_probs.shape[1] - len(ids_names))

fig, axs = plt.subplots(2, 1, figsize=(14, 5))
axs[0].bar(ids_names, ex_probs[0,:])
axs[1].bar(np.arange(ex_probs.shape[1]), ex_probs[1,:])
plt.show()

xs_single = xs[ex_idx]
xs_single = xs_single[None, xs_single != 0]
pred_single(xs_single)


# <codecell>
### ESTIMATE ACCURACY OF SIMPLE PREDICTOR
xs, _, ys, ys_mask = next(test_task)
logits = optimizer.model(xs)
preds = logits.argmax(axis=-1)
preds

# <codecell>
all_simple_preds = []

for x in xs:
    x = x[x != 0]
    simple_pred = pred_single(x[None, :], readable=False)
    simple_rule = jnp.array([list(simple_pred[0].keys())[jnp.argmax(jnp.array(list(simple_pred[0].values())))],
                            list(simple_pred[1].keys())[jnp.argmax(jnp.array(list(simple_pred[1].values())))]])

    all_simple_preds.append(simple_rule)

all_simple_preds = np.array(all_simple_preds)
all_simple_preds

# <codecell>
rules = np.unique(preds[:,0])
rule = rules[2]

sel = preds[:,0] == rule
acc = jnp.mean(preds[sel,0] == all_simple_preds[sel,0])
acc


# <codecell>
all_simple_preds[sel,0]


# %%
### Visualizing embeddings

e = optimizer.model.embed.embedding
e = e[:(max_token + 1), :]  # only logic tokens and p-variables

plt.imshow(e @ e.T, cmap='bwr', vmin=-1, vmax=1)
plt.colorbar()