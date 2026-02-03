"""Sampling different propositions."""

from __future__ import annotations

from functools import lru_cache
import random

from .elem import *
from .proof import build_proof_tree, CompletedProofNode, InternalProofNode


def _validate_imply_args(n_vars: int, size: int) -> None:
    if n_vars < 0:
        raise ValueError("n_vars must be non-negative")
    if size < 1:
        raise ValueError("size must be at least 1")


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


def _sample_dyck_word(n_pairs: int, rng: random.Random | None = None) -> str:
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
            n_left = _count_dyck_suffixes(n_pairs, open_used + 1, close_used)
        if close_used < open_used:
            n_right = _count_dyck_suffixes(n_pairs, open_used, close_used + 1)

        total = n_left + n_right
        p_left = n_left / total if total else 0.0
        if rng.random() < p_left:
            word.append("(")
            open_used += 1
        else:
            word.append(")")
            close_used += 1

    return "".join(word)


def _match_parentheses(word: str) -> dict[int, int]:
    stack = []
    matches: dict[int, int] = {}
    for idx, char in enumerate(word):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            if not stack:
                raise ValueError("Invalid Dyck word: unmatched closing parenthesis")
            start = stack.pop()
            matches[start] = idx
        else:
            raise ValueError('Dyck word must consist only of "(" and ")" characters')
    if stack:
        raise ValueError("Invalid Dyck word: unmatched opening parenthesis")
    return matches


def _group_by_dyck(op, leaves: list[Proposition], word: str) -> Proposition:
    leaves = tuple(leaves)
    if not leaves:
        raise ValueError("Cannot build expression with no atoms")

    expected_pairs = len(leaves) - 1
    if expected_pairs == 0:
        if word:
            raise ValueError("Dyck word should be empty when only one atom is provided")
        return leaves[0]
    if len(word) != 2 * expected_pairs:
        raise ValueError("Dyck word length does not match number of atoms")

    matches = _match_parentheses(word)
    leaf_iter = iter(leaves)

    def consume(start: int, end: int) -> Proposition:
        if start >= end:
            try:
                return next(leaf_iter)
            except StopIteration as exc:
                raise ValueError("Dyck word consumed more atoms than provided") from exc

        if word[start] != "(":
            raise ValueError("Invalid Dyck word structure")

        try:
            split = matches[start]
        except KeyError as exc:
            raise ValueError("Invalid Dyck word: missing matching parenthesis") from exc

        if split >= end:
            raise ValueError("Invalid Dyck word segmentation")

        left = consume(start + 1, split)
        right = consume(split + 1, end)
        return op(left, right)

    tree = consume(0, len(word))
    try:
        next(leaf_iter)
    except StopIteration:
        return tree
    raise ValueError("Unused atoms remain after constructing the tree")


def _random_group(op, leaves: list[Proposition], rng: random.Random | None = None) -> Proposition:
    if len(leaves) == 1:
        return leaves[0]
    dyck_word = _sample_dyck_word(len(leaves) - 1, rng=rng)
    return _group_by_dyck(op, leaves, dyck_word)


def _sample_atom(n_vars: int, rng: random.Random | None = None) -> Proposition:
    rng = rng or random
    pick = rng.randrange(n_vars + 2)
    if pick < n_vars:
        return Atom(f"p{pick + 1}")
    if pick == n_vars:
        return PTrue()
    return PFalse()


def sample_imply(
    n_vars: int,
    size: int,
    rng: random.Random | None = None,
) -> Proposition:
    """Sample a random proposition using only implications and atoms.

    Args:
        n_vars: Number of unique propositional variables available.
        size: Number of atoms (variables or constants) in the proposition.
    """
    _validate_imply_args(n_vars, size)
    leaves = [_sample_atom(n_vars, rng=rng) for _ in range(size)]
    return _random_group(Implies, leaves, rng=rng)


def list_sequents(
    prop: Proposition,
    rng: random.Random | None = None,
) -> list[Example]:
    """List sequents in the proof tree for `prop` with all applicable rules
    per sequent. If `prop` cannot be proved, return [].
    """

    tree = build_proof_tree(Sequent(ants=[], cons=prop))
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

    _ = rng
    collected: list[Example] = []
    for sequent, rules in entries.items():
        if not rules:
            continue
        collected.append((sequent, list(rules)))
    return collected
