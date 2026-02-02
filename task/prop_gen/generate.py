"""
Generate propositional logic dataset
"""

# <codecell>
from util.sample import sample_imply, list_sequents

p = sample_imply(3, 5)
list_sequents(p)

def gen_imply_exs(n_exs, n_vars, n_size):
    total = 0
    while total < n_exs:
        p = sample_imply(n_vars, n_size)
        exs = list_sequents(p)
        for ex in exs:
            yield ex
            total += 1


for p in gen_imply_exs(10, 3, 5):
    print(p)