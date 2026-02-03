"""Micro-benchmarks for proposition sampling and proof search."""

from __future__ import annotations

import argparse
from functools import lru_cache
import math
import random
import time

from .elem import (
    Atom,
    PTrue,
    PFalse,
    Implies,
    Sequent,
    Rule,
    ImpliesLeft,
    AndLeft,
    OrLeft,
)
from .proof import (
    CompletedProofNode,
    FailedProofNode,
    InternalProofNode,
    get_next_rules,
)
from .sample import (
    _dyck_count_table,
    _group_by_dyck,
    _sample_dyck_word,
    list_sequents,
    sample_imply,
)

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax import random as jrandom

    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False


@lru_cache(maxsize=None)
def _baseline_count_dyck_suffixes(n_pairs: int, open_used: int, close_used: int) -> int:
    if open_used == n_pairs:
        return 1

    total = 0
    if open_used < n_pairs:
        total += _baseline_count_dyck_suffixes(n_pairs, open_used + 1, close_used)
    if close_used < open_used:
        total += _baseline_count_dyck_suffixes(n_pairs, open_used, close_used + 1)
    return total


def _baseline_sample_dyck_word(n_pairs: int, rng: random.Random | None = None) -> str:
    if n_pairs <= 0:
        return ""

    rng = rng or random
    word = []
    open_used = 0
    close_used = 0
    total_len = 2 * n_pairs

    while len(word) < total_len:
        n_left = 0
        n_right = 0

        if open_used < n_pairs:
            n_left = _baseline_count_dyck_suffixes(n_pairs, open_used + 1, close_used)
        if close_used < open_used:
            n_right = _baseline_count_dyck_suffixes(n_pairs, open_used, close_used + 1)

        total = n_left + n_right
        p_left = n_left / total if total else 0.0
        if rng.random() < p_left:
            word.append("(")
            open_used += 1
        else:
            word.append(")")
            close_used += 1

    return "".join(word)


def _baseline_sample_imply(n_vars: int, size: int, rng: random.Random | None = None) -> Implies:
    rng = rng or random
    leaves = [_sample_atom(n_vars, rng=rng) for _ in range(size)]
    dyck_word = _baseline_sample_dyck_word(len(leaves) - 1, rng=rng)
    return _group_by_dyck(Implies, leaves, dyck_word)


def _sample_atom(n_vars: int, rng: random.Random | None = None):
    rng = rng or random
    pick = rng.randrange(n_vars + 2)
    if pick < n_vars:
        return Atom(f"p{pick + 1}")
    if pick == n_vars:
        return PTrue()
    return PFalse()


def _list_sequents_tree(prop) -> list[tuple[Sequent, list[Rule]]]:
    tree = _build_proof_tree_baseline(Sequent(ants=[], cons=prop))
    if not tree.is_provable:
        return []

    entries: dict[Sequent, list[Rule]] = {}
    rule_keys: dict[Sequent, set[tuple]] = {}

    def rule_key(rule: Rule) -> tuple:
        if isinstance(rule, ImpliesLeft):
            return (ImpliesLeft, rule.implication)
        if isinstance(rule, AndLeft):
            return (AndLeft, rule.conjunction)
        if isinstance(rule, OrLeft):
            return (OrLeft, rule.disjunction)
        return (type(rule),)

    def add_rule(sequent: Sequent, rule: Rule) -> None:
        key = rule_key(rule)
        if sequent not in entries:
            entries[sequent] = []
            rule_keys[sequent] = set()
        if key not in rule_keys[sequent]:
            entries[sequent].append(rule)
            rule_keys[sequent].add(key)

    def traverse(node) -> None:
        if isinstance(node, CompletedProofNode):
            add_rule(node.sequent, node.rule)
            return
        if isinstance(node, InternalProofNode):
            successful = [
                (rule, children)
                for rule, children in node.branches
                if all(child.is_provable for child in children)
            ]
            for rule, _ in successful:
                add_rule(node.sequent, rule)
            for _, children in successful:
                for child in children:
                    traverse(child)

    traverse(tree)
    return [(sequent, list(rules)) for sequent, rules in entries.items()]


def _build_proof_tree_baseline(
    sequent: Sequent,
    max_depth: int = 10_000,
):
    if max_depth <= 0:
        return FailedProofNode(sequent, reason="depth limit")

    branches = []

    for rule in get_next_rules(sequent):
        result = rule.apply(sequent)
        if result is None:
            continue
        if len(result) == 0:
            return CompletedProofNode(sequent, rule)
        children = [_build_proof_tree_baseline(subgoal, max_depth - 1) for subgoal in result]
        branches.append((rule, children))

    if not branches:
        return FailedProofNode(sequent, reason="no applicable rules")

    return InternalProofNode(sequent, branches)


def _timeit(label: str, fn, loops: int) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        fn()
    elapsed = time.perf_counter() - start
    per = elapsed / max(loops, 1)
    print(f"{label:28s} {elapsed:8.3f}s  ({per * 1e3:7.2f} ms/iter)")
    return elapsed


def _baseline_batch_dyck(n_pairs: int, batch: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [_baseline_sample_dyck_word(n_pairs, rng=rng) for _ in range(batch)]


def _baseline_batch_atom_ids(n_vars: int, batch: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(n_vars + 2) for _ in range(batch)]


def _jax_batch_dyck_sampler(n_pairs: int, batch: int):
    counts = _dyck_count_table(n_pairs)
    log_counts = [
        [math.log(value) if value > 0 else -math.inf for value in row]
        for row in counts
    ]
    log_counts = jnp.asarray(log_counts, dtype=jnp.float32)

    def step(carry, _):
        open_used, close_used, key = carry
        key, subkey = jrandom.split(key)

        log_left = jnp.where(
            open_used < n_pairs,
            log_counts[open_used + 1, close_used],
            -jnp.inf,
        )
        log_right = jnp.where(
            close_used < open_used,
            log_counts[open_used, close_used + 1],
            -jnp.inf,
        )
        log_denom = jsp.special.logsumexp(
            jnp.stack([log_left, log_right], axis=0),
            axis=0,
        )
        p_left = jnp.exp(log_left - log_denom)
        p_left = jnp.where(jnp.isfinite(p_left), p_left, 0.0)
        pick_left = jrandom.uniform(subkey, shape=(batch,), dtype=p_left.dtype) < p_left

        open_used = open_used + pick_left.astype(jnp.int32)
        close_used = close_used + (~pick_left).astype(jnp.int32)

        return (open_used, close_used, key), pick_left

    def sample(key):
        open_used = jnp.zeros((batch,), dtype=jnp.int32)
        close_used = jnp.zeros((batch,), dtype=jnp.int32)
        (_, _, _), picks = jax.lax.scan(
            step,
            (open_used, close_used, key),
            xs=None,
            length=2 * n_pairs,
        )
        return picks

    return jax.jit(sample)


def _jax_batch_atom_ids(n_vars: int, batch: int):
    def sample(key):
        return jrandom.randint(key, shape=(batch,), minval=0, maxval=n_vars + 2)

    return jax.jit(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--dyck-loops", type=int, default=300)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--vars", type=int, default=3)
    parser.add_argument("--sample-loops", type=int, default=300)
    parser.add_argument("--proof-loops", type=int, default=30)
    parser.add_argument("--jax-batch", type=int, default=2048)
    parser.add_argument("--jax-loops", type=int, default=50)
    args = parser.parse_args()

    print("Dyck word sampling:")
    rng = random.Random(0)
    _timeit(
        "baseline dyck",
        lambda: _baseline_sample_dyck_word(args.pairs, rng=rng),
        args.dyck_loops,
    )
    rng = random.Random(0)
    _timeit(
        "optimized dyck",
        lambda: _sample_dyck_word(args.pairs, rng=rng),
        args.dyck_loops,
    )

    if _JAX_AVAILABLE:
        print("\nJAX batched dyck:")
        sampler = _jax_batch_dyck_sampler(args.pairs, args.jax_batch)
        key = jrandom.PRNGKey(0)
        sampler(key).block_until_ready()
        _timeit(
            "jax batch dyck",
            lambda: sampler(jrandom.PRNGKey(0)).block_until_ready(),
            args.jax_loops,
        )
        _timeit(
            "baseline batch dyck",
            lambda: _baseline_batch_dyck(args.pairs, args.jax_batch, seed=0),
            max(1, args.jax_loops // 5),
        )

    print("\nFull implication sampling:")
    rng = random.Random(1)
    _timeit(
        "baseline sample_imply",
        lambda: _baseline_sample_imply(args.vars, args.size, rng=rng),
        args.sample_loops,
    )
    rng = random.Random(1)
    _timeit(
        "optimized sample_imply",
        lambda: sample_imply(args.vars, args.size, rng=rng),
        args.sample_loops,
    )

    if _JAX_AVAILABLE:
        print("\nJAX batched atom ids:")
        sampler = _jax_batch_atom_ids(args.vars, args.jax_batch)
        key = jrandom.PRNGKey(1)
        sampler(key).block_until_ready()
        _timeit(
            "jax batch atom ids",
            lambda: sampler(jrandom.PRNGKey(1)).block_until_ready(),
            args.jax_loops,
        )
        _timeit(
            "baseline batch atom ids",
            lambda: _baseline_batch_atom_ids(args.vars, args.jax_batch, seed=1),
            max(1, args.jax_loops // 5),
        )

    print("\nProof sampling (list_sequents):")
    rng = random.Random(2)
    props = [sample_imply(args.vars, args.size, rng=rng) for _ in range(args.proof_loops)]
    baseline_props = list(props)
    optimized_props = list(props)
    _timeit(
        "baseline list_sequents",
        lambda: _list_sequents_tree(baseline_props.pop()),
        args.proof_loops,
    )
    _timeit(
        "optimized list_sequents",
        lambda: list_sequents(optimized_props.pop()),
        args.proof_loops,
    )


if __name__ == "__main__":
    main()
