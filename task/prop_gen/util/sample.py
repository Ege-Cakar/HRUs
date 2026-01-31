"""Enumerate new propositions"""

# <codecell>
from functools import lru_cache
import itertools as it
import math

import numpy as np

try:
    from .data import *
except ImportError:
    from data import *


def new_seed(rng):
    return rng.integers(0, np.iinfo(np.int32).max)


@lru_cache(maxsize=None)
def _count_dyck_suffixes(n_pairs: int, open_used: int, close_used: int) -> int:
    if open_used == n_pairs:
        return 1

    total = 0
    if open_used < n_pairs:
        total += _count_dyck_suffixes(n_pairs, open_used + 1, close_used)
    if close_used < open_used:
        total += _count_dyck_suffixes(n_pairs, open_used, close_used + 1)

    return total


def sample_dyck_word(n_pairs: int, seed=None) -> str:
    if n_pairs <= 0:
        return ''
    rng = np.random.default_rng(seed)

    word = []
    open_used = 0
    close_used = 0
    total_len = 2 * n_pairs

    while len(word) < total_len:
        n_left = 0
        n_right = 0

        if open_used < n_pairs:
            n_left = _count_dyck_suffixes(n_pairs, open_used + 1, close_used)
        if close_used < open_used:
            n_right = _count_dyck_suffixes(n_pairs, open_used, close_used + 1)

        p_left = n_left / (n_left + n_right) if (n_left + n_right) > 0 else 0
        is_left = rng.random() < p_left
        
        if is_left:
            word.append('(')
            open_used += 1
        else:
            word.append(')')
            close_used += 1

    return ''.join(word)


def _match_parentheses(word: str) -> dict[int, int]:
    stack = []
    matches: dict[int, int] = {}
    for idx, char in enumerate(word):
        if char == '(':
            stack.append(idx)
        elif char == ')':
            if not stack:
                raise ValueError('Invalid Dyck word: unmatched closing parenthesis.')
            start = stack.pop()
            matches[start] = idx
        else:
            raise ValueError('Dyck word must consist only of "(" and ")" characters.')
    if stack:
        raise ValueError('Invalid Dyck word: unmatched opening parenthesis.')
    return matches


def group_by_samp_dyck(op, leaves, word):
    leaves = tuple(leaves)
    if not leaves:
        raise ValueError('Cannot build expression with no atoms.')

    expected_pairs = len(leaves) - 1
    if expected_pairs == 0:
        if word:
            raise ValueError('Dyck word should be empty when only one atom is provided.')
        return leaves[0]
    if len(word) != 2 * expected_pairs:
        raise ValueError('Dyck word length does not match number of atoms.')

    matches = _match_parentheses(word)
    leaf_iter = iter(leaves)

    def consume(start: int, end: int):
        if start >= end:
            try:
                return next(leaf_iter)
            except StopIteration as exc:
                raise ValueError('Dyck word consumed more atoms than provided.') from exc

        if word[start] != '(':
            raise ValueError('Invalid Dyck word structure.')

        try:
            split = matches[start]
        except KeyError as exc:
            raise ValueError('Invalid Dyck word: missing matching parenthesis.') from exc

        if split >= end:
            raise ValueError('Invalid Dyck word segmentation.')

        left = consume(start + 1, split)
        right = consume(split + 1, end)
        return op(left, right)

    tree = consume(0, len(word))
    try:
        next(leaf_iter)
    except StopIteration:
        return tree
    raise ValueError('Unused atoms remain after constructing the OR tree.')


def random_group(op, atoms, seed=None):
    dyck_word = sample_dyck_word(len(atoms) - 1, seed=seed)
    out = group_by_samp_dyck(op, atoms, dyck_word)
    return out


def catalan(nodes):
    if len(nodes) == 1:
        yield nodes[0]
        return

    if len(nodes) == 2:
        for op in ops:
            yield op(nodes[0], nodes[1])
        return

    for idx, _ in enumerate(nodes[:-1]):
        left_branch = nodes[:(idx+1)]
        right_branch = nodes[(idx+1):]

        for left, right in it.product(catalan(left_branch), catalan(right_branch)):
            for op in ops:
                yield op(left, right)

# ops = [And, Or, Implies]
# ops = [Implies]
ops = [Or]

def gen_batch_or(n_atoms, n_nodes):
    assert len(ops) == 1 and ops[0] == Or
    
    atoms = [Atom(f'p{i+1}') for i in range(n_atoms)]
    for target_atom in atoms:
        # rem_atoms = [a for a in atoms if a is not target_atom]
        for node_set in it.product(*it.repeat(atoms, n_nodes)):
            for example in catalan(node_set):
                formula = Implies(target_atom, example)
                yield formula


def n_combo_or(n_atoms, n_nodes):
    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = len(ops)**(n) * cat
    by_atoms = n_atoms**n_nodes * by_ops
    return n_atoms * by_atoms


def gen_batch(n_atoms, n_nodes):
    atoms = [PFalse(), PTrue()] + [Atom(f'p{i+1}') for i in range(n_atoms)]
    
    for node_set in it.product(*it.repeat(atoms, n_nodes)):
        for example in catalan(node_set):
            yield example


def n_combo(n_atoms, n_nodes):
    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = len(ops)**(n) * cat
    by_atoms = (n_atoms + 2)**n_nodes * by_ops
    return by_atoms
    

### pigeon mania
def prod(a, b):
    for x in a:
        for y in b:
            yield (x, y)


def catalan_adv(nodes, op):
    if len(nodes) == 1:
        try:
            for n in nodes[0]:
                yield n
        except TypeError:
            yield nodes[0]
        return

    if len(nodes) == 2:
        left, right = nodes
        try:
            for l, r in prod(left, right):
                yield op(l, r)
        except TypeError:
            yield op(left, right)
        return

    for idx, _ in enumerate(nodes[:-1]):
        left_branch = nodes[:(idx+1)]
        right_branch = nodes[(idx+1):]
        
        for left, right in prod(catalan_adv(left_branch, op), catalan_adv(right_branch, op)):
            try:
                for l, r in prod(left, right):
                    yield op(l, r)
            except TypeError:
                yield op(left, right)


def pigeon_set(repeats=1, *args, **kwargs):
    all_pigeons = []
    count = 0
    seed = kwargs.pop('seed', None)
    rng = np.random.default_rng(seed)
    
    while count < repeats:
        pigeons = list(pigeon(rng, *args, **kwargs))
        if len(pigeons) != 0:
            all_pigeons.extend(pigeons)
            count += 1
    
    return all_pigeons


def pigeon(rng, n_pigeons, n_holes, pigeon_occupation_ablation_prop=None, roommate_ablation_prop=None):
    atoms = [[Atom(f'p{i}{j}') for j in range(n_holes)] for i in range(n_pigeons)]

    pigeons = []
    for i in range(n_pigeons):
        curr_p = atoms[i]
        if pigeon_occupation_ablation_prop is not None:
            keep_idxs = rng.binomial(1, pigeon_occupation_ablation_prop, size=len(curr_p)).astype(bool)
            if np.sum(keep_idxs) == 0:
                keep_idxs[rng.integers(0, len(keep_idxs))] = True
            curr_p = [curr_p[j] for j, keep in enumerate(keep_idxs) if keep]

        pigeons.append(catalan_adv(curr_p, Or))

    pigeons_in_a_hole = catalan_adv(pigeons, And)

    all_pigeon_roommates = []
    for i1 in range(n_pigeons):
        for i2 in range(n_pigeons):
            if i1 != i2:
                statements = [And(atoms[i1][j], atoms[i2][j]) for j in range(n_holes)]
                all_pigeon_roommates.extend(statements)

    if roommate_ablation_prop is not None:
        keep_idxs = rng.binomial(1, roommate_ablation_prop, size=len(all_pigeon_roommates)).astype(bool)
        if np.sum(keep_idxs) == 0:
            keep_idxs[rng.integers(0, len(keep_idxs))] = True
        all_pigeon_roommates = [all_pigeon_roommates[i] for i, keep in enumerate(keep_idxs) if keep]

    all_pigeon_roommates = catalan_adv(all_pigeon_roommates, Or)
    php = catalan_adv([pigeons_in_a_hole, all_pigeon_roommates], Implies)
    return php


# def gen_php(seed=5):
#     n_pigeons = 4
#     n_holes = 4

#     param_sets = []
#     for n_p in range(2, n_pigeons + 1):
#         for n_h in range(1, n_holes + 1):
#             pigeon_occupation_ablation_prop = None
#             roommate_ablation_prop = None
#             repeats = 1

#             if n_p == 2 and n_h == 4:
#                 pigeon_occupation_ablation_prop = 0.75
#                 roommate_ablation_prop = 0.75
#                 repeats = 5
            
#             elif n_p == 3 and n_h >= 3:
#                 pigeon_occupation_ablation_prop = 2 / n_h
#                 roommate_ablation_prop = 2 / n_h
#                 repeats = 5

#             elif n_p == 4 and n_h >= 2:
#                 pigeon_occupation_ablation_prop = 1 / n_h
#                 roommate_ablation_prop = 1 / n_h
#                 repeats = 5
            
#             param_sets.append({'n_pigeons': n_p,
#                             'n_holes': n_h,
#                             'pigeon_occupation_ablation_prop': pigeon_occupation_ablation_prop,
#                             'roommate_ablation_prop': roommate_ablation_prop,
#                             'repeats': repeats,
#                             'seed': seed})

#     from tqdm import tqdm
#     all_pigeons = list(it.chain.from_iterable([pigeon_set(**params) for params in tqdm(param_sets)]))
#     return all_pigeons

# def pigeon_set(repeats=1, *args, **kwargs):
#     all_pigeons = []
#     seed = kwargs.pop('seed', None)
#     rng = np.random.default_rng(seed)
    
#     all_pigeons = [pigeon(rng, *args, **kwargs) for _ in range(repeats)]
#     return all_pigeons


# def pigeon(rng, n_pigeons, n_holes, pigeon_occupation_ablation_prop=None, roommate_ablation_prop=None):
#     atoms = [[Atom(f'p{i}{j}') for j in range(n_holes)] for i in range(n_pigeons)]

#     pigeons = []
#     for i in range(n_pigeons):
#         curr_p = atoms[i]
#         if pigeon_occupation_ablation_prop is not None:
#             keep_idxs = rng.binomial(1, pigeon_occupation_ablation_prop, size=len(curr_p)).astype(bool)

#             if np.sum(keep_idxs) == 0:
#                 keep_idxs[rng.integers(0, len(keep_idxs))] = True

#             curr_p = [curr_p[j] for j, keep in enumerate(keep_idxs) if keep]

#         pigeons.append(random_group(Or, curr_p, seed=new_seed(rng)))

#     pigeons_in_a_hole = random_group(And, pigeons, seed=new_seed(rng))

#     all_pigeon_roommates = []
#     for i1 in range(n_pigeons):
#         for i2 in range(n_pigeons):
#             if i1 != i2:
#                 statements = [And(atoms[i1][j], atoms[i2][j]) for j in range(n_holes)]
#                 all_pigeon_roommates.extend(statements)

#     if roommate_ablation_prop is not None:
#         keep_idxs = rng.binomial(1, roommate_ablation_prop, size=len(all_pigeon_roommates)).astype(bool)

#         if np.sum(keep_idxs) == 0:
#             keep_idxs[rng.integers(0, len(keep_idxs))] = True

#         all_pigeon_roommates = [all_pigeon_roommates[i] for i, keep in enumerate(keep_idxs) if keep]

#     all_pigeon_roommates = random_group(Or, all_pigeon_roommates, seed=new_seed(rng))
#     php = Implies(pigeons_in_a_hole, all_pigeon_roommates)
#     return php


def gen_php(seed=None, do_start=False):
    n_pigeons = 4
    n_holes = 4
    reps_per_six_case = 16

    param_sets = []
    for n_p in range(2, n_pigeons + 1):
        for n_h in range(1, n_holes + 1):
            pigeon_occupation_ablation_prop = None
            roommate_ablation_prop = None
            repeats = 1
            skip = not do_start

            if n_p == 2 and n_h == 4:
                pigeon_occupation_ablation_prop = 0.75
                roommate_ablation_prop = 0.75
                repeats = reps_per_six_case
                skip = False
            
            elif n_p == 3 and n_h >= 3:
                pigeon_occupation_ablation_prop = 2 / n_h
                roommate_ablation_prop = 2 / n_h
                repeats = reps_per_six_case
                skip = False

            elif n_p == 4 and n_h >= 2:
                pigeon_occupation_ablation_prop = 1 / n_h
                roommate_ablation_prop = 1 / n_h
                repeats = reps_per_six_case
                skip = False
            
            if not skip:
                param_sets.append({'n_pigeons': n_p,
                                'n_holes': n_h,
                                'pigeon_occupation_ablation_prop': pigeon_occupation_ablation_prop,
                                'roommate_ablation_prop': roommate_ablation_prop,
                                'repeats': repeats,
                                'seed': seed})

    all_pigeons = it.chain.from_iterable([pigeon_set(**params) for params in param_sets])
    return all_pigeons

# all_pigeons = gen_php(seed=1130, do_start=True)
# all_pigeons = list(all_pigeons)
# print(all_pigeons[-1])
# len(all_pigeons)

# <codecell>

### or what?
def gen_or(n_exs_per_set=10_000, seed=None):
    n_prop_set = np.arange(2, 30)
    switch = [False, True]

    max_pid = 100_000
    global_rng = np.random.default_rng(seed)

    ### START TEST CONFIG
    # max_pid = 10
    # n_prop_set = np.arange(2, 5)
    # switch = [False, True]
    # n_exs_per_set = 3
    ### END TEST CONFIG

    for n_props, is_true in it.product(n_prop_set, switch):
        for _ in range(n_exs_per_set):
            pids = global_rng.choice(max_pid, size=n_props + (not is_true), replace=False)

            if is_true:
                target_pid = global_rng.choice(pids)
                pids = np.append(pids, target_pid)

            atoms = [Atom(f'p{pid}') for pid in pids]
            target_atom = atoms[-1]
            all_atoms = atoms[:-1]

            seed = global_rng.integers(0, np.iinfo(np.int32).max)
            cons = random_group(Or, all_atoms, seed=seed)
            prop = Implies(target_atom, cons)
            yield prop

