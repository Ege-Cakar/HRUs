"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
from concurrent.futures import ThreadPoolExecutor 
import itertools
import json
from pathlib import Path
import uuid
import sys

from tqdm import tqdm

from util.data import *
from util.proof import prove
from util.out import format_example 
from util.sample import *


def split(it, run_split):
    run_idx = sys.argv[1]
    try:
        run_idx = int(run_idx) % run_split
        print(f'info: run_idx = {run_idx}')
    except ValueError:
        print(f'warn: unable to parse index {run_idx}, setting run_idx=0')
        run_idx = 0
    
    div = run_idx % run_split

    for i, item in enumerate(it):
        if i % run_split == div:
            yield item


run_id = uuid.uuid4()
print('RUN_ID', run_id)

# total number of tokens with this configuration is 1559127906 (about 1.6 billion): TODO: update
# n_atoms = 3
# max_nodes = 5
# n_cores = 16 * 16
# keep = 'until_success'


# total number of tokens with this configuration is ?
# n_atoms = 3
# max_nodes = 5
# n_cores = 16 * 16
# keep = 'simplest'

# total number of tokens with this configuration is ?
# NOTE: generation ops set to Implies only
# n_atoms = 3
# max_nodes = 7
# n_cores = 16 * 16
# keep = 'until_success'

# total number of tokens with this configuration is ?
# NOTE: generation ops set to Or only
n_atoms = 3
max_nodes = 8
n_cores = 16 * 16
keep = 'until_success'

run_split = 72


### START TEST CONFIG
# out_dir = Path('test_data')
# n_cores = 16
# n_atoms = 2
# max_nodes = 4
### END TEST CONFIG

# all_ex = itertools.chain(*[gen_batch(n_atoms, n) for n in range(1, max_nodes + 1)])
# all_ex = itertools.chain(*[gen_batch_or(n_atoms, n) for n in range(1, max_nodes + 1)])

# all_ex = gen_php(seed=3011)
# total_ex = len(all_ex)

# total_ex = 2_500
# all_ex = gen_or(n_exs_per_set=total_ex)
run_idx = int(sys.argv[1])
all_ex = gen_php(seed=1130, do_start=True)
# n_sets = 28 * 2

all_ex = list(split(all_ex, run_split))
# total_ex = sum(n_combo(n_atoms, n) for n in range(1, max_nodes + 1))
# total_ex = sum(n_combo_or(n_atoms, n) for n in range(1, max_nodes + 1))

# pbar = tqdm(total=total_ex // run_split)
pbar = tqdm(total=len(all_ex))

# out_dir = Path('/n/netscratch/pehlevan_lab/Lab/wlt/prop/or_big')
out_dir = Path('/n/netscratch/pehlevan_lab/Lab/wlt/prop/php_enum')
# out_dir = Path('/n/netscratch/pehlevan_lab/Lab/wlt/prop/or_par')


if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=False)

out_path = out_dir / f'{run_id}.json'

def write_example(prop):
    proof = prove(prop, keep=keep)
    ex = format_example(n_atoms, prop, proof)
    pbar.update(1)

    with open(out_path, 'a') as fp:
        json.dump(ex, fp)
        fp.write('\n')

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    itr = all_ex
    executor.map(write_example, itr)

pbar.close()
print('done')

