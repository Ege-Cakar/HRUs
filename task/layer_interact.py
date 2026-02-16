# <codecell>
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from layer_axiom import LayerAxiomTask


task = LayerAxiomTask(ds_path="layer_gen/data/toy_layer", batch_size=8)
tokenizer = task.tokenizer

xs, ys = next(task)

# <codecell>
print(xs[0])
print(ys[0])

# ''.join([tokenizer.id_to_char(int(x)) for x in xs[3] if int(x) != tokenizer.sep_token_id and int(x) != 0])
task.tokenizer.decode_batch_ids(xs)

# <codecell>
task.rule_bank.transitions[1]
