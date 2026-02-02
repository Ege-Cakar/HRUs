"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
import itertools as it
import json
import xml.etree.ElementTree as et

import numpy as np
from typing import Optional, Sequence

from util.elem import *
from util.proof import *


from tqdm import tqdm
from functools import lru_cache


p = Atom('p')
q = Atom('q')
r = Atom('r')

# Example: (p ∧ q), (p ∧ q) → r, p → r ⊢ r
# 
# Both elimination orderings succeed:
# 1. Use (p ∧ q) → r first: prove p ∧ q by Axiom (2 steps total)
# 2. Use p → r first: need AndLeft to extract p from p ∧ q (3 steps total)
#
# Different orderings give correct but differently-sized proofs.

conj = And(p, q)
prop1 = Implies(conj, r)  # (p ∧ q) → r
prop2 = Implies(p, r)     # p → r

seq = Sequent(ants=[conj, prop1, prop2], cons=r)

t = build_proof_tree(seq)
print(t)
print(f"\nTree size: {t.size}, depth: {t.depth}, provable: {t.is_provable}")

# <codecell>
t.branches


### OLD CODE BELOW
# # <codecell>
# prop = Implies(Implies(Atom('p1'), Atom('p2')), Implies(Atom('p1'), Atom('p2')))

# print('PROP', prop)
# # proof = prove(prop, keep='until_success')
# proof = prove(prop, keep='simplest')
# print('PROOF', proof)

# ex = format_example(3, prop, proof, proof_to_string=False)

# print(ex['input'])

# all_proof_lines = []
# for elem in ex['proof']:
#     proof_line = et.tostring(elem, encoding='utf-8').decode('utf-8')
#     all_proof_lines.append(proof_line)

#     et.indent(elem)
#     proof_line = et.tostring(elem, encoding='utf-8').decode('utf-8')
#     print(proof_line)

# # <codecell>
# inp = et.fromstring(ex['input'])
# et.indent(inp)
# print(et.tostring(inp, encoding='utf-8').decode('utf-8'))
