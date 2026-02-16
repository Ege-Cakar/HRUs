# <codecell>
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from layer import LayerTask
from layer_gen.util.rule_bank import *

rule_bank = build_random_rule_bank(n_layers=10, props_per_layer=8, rules_per_transition=32, k_in_max=3, k_out_max=3, rng=np.random.default_rng())
problem = sample_problem(bank=rule_bank, distance=4, initial_ant_max=3, rng=np.random.default_rng())
problem

# <codecell>
problem.step_ants
problem.step_rules

# <codecell>
a = set({'1', '2', '3'})
a.update({'4', '5'})
a
# <codecell>
task = LayerTask(ds_path="layer_gen/data/toy_layer", batch_size=8, mode='online')
tokenizer = task.tokenizer

xs, ys = next(task)

# <codecell>
print(xs[0])
print(ys[0])

# ''.join([tokenizer.id_to_char(int(x)) for x in xs[3] if int(x) != tokenizer.sep_token_id and int(x) != 0])
task.tokenizer.decode_batch_ids(xs)

# <codecell>
task.rule_bank.transitions[1]

# <codecell>
import time

DS_PATH = "layer_gen/data/toy_layer"
BATCH_SIZE = 8
N_BATCHES = 10_000

results = {}
for mode in ("online", "offline"):
    task = LayerTask(ds_path=DS_PATH, batch_size=BATCH_SIZE, mode=mode)
    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        next(task)
    elapsed = time.perf_counter() - t0
    results[mode] = elapsed
    print(f"{mode:>8s}: {elapsed:.2f}s  ({N_BATCHES / elapsed:.0f} batches/s)")

print(f"\nonline/offline ratio: {results['online'] / results['offline']:.2f}x")

# <codecell>

