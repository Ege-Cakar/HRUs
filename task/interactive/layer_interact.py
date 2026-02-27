# <codecell>
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task.layer import LayerTask
from task.layer_gen.util.rule_bank import *

rule_bank = build_random_rule_bank(n_layers=24, props_per_layer=16, rules_per_transition=32, k_in_max=3, k_out_max=6, rng=np.random.default_rng())
problem = sample_problem(max_attempts=1024, bank=rule_bank, distance=20, initial_ant_max=3, rng=np.random.default_rng())
problem.step_rules

# <codecell>
problem.step_ants
problem.step_rules

# <codecell>
a = set({'1', '2', '3'})
a.update({'4', '5'})
a
# <codecell>
task = LayerTask(
    ds_path=ROOT / "task" / "layer_gen" / "data" / "toy_layer",
    batch_size=8,
    mode='online',
)
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

DS_PATH = ROOT / "task" / "layer_gen" / "data" / "toy_layer"
BATCH_SIZE = 64
N_BATCHES = 500

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
