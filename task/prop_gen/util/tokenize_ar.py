"""Autoregressive tokenization utilities for implicational sequents."""

from __future__ import annotations

from .elem import (
    And,
    AndLeft,
    AndRight,
    Axiom,
    Atom,
    FalseLeft,
    Implies,
    ImpliesLeft,
    ImpliesRight,
    NegationRight,
    Or,
    OrLeft,
    OrRight1,
    OrRight2,
    PFalse,
    PTrue,
    Rule,
    Sequent,
    TrueRight,
    Unprovable,
)

pad_idx = 0

logic_char_to_id = {
    "⊢": 1,
    "∧": 2,
    "∨": 3,
    "¬": 4,
    "→": 5,
    "(": 6,
    ")": 7,
    "⊤": 8,
    "⊥": 9,
    ",": 10,
}

id_to_logic_char = {v: k for k, v in logic_char_to_id.items()}

_RULE_BASE = 32
rule_type_to_id = {
    Axiom: _RULE_BASE + 0,
    ImpliesRight: _RULE_BASE + 1,
    ImpliesLeft: _RULE_BASE + 2,
    AndRight: _RULE_BASE + 3,
    AndLeft: _RULE_BASE + 4,
    OrRight1: _RULE_BASE + 5,
    OrRight2: _RULE_BASE + 6,
    OrLeft: _RULE_BASE + 7,
    TrueRight: _RULE_BASE + 8,
    FalseLeft: _RULE_BASE + 9,
    NegationRight: _RULE_BASE + 10,
    Unprovable: _RULE_BASE + 11,
}

id_to_rule_type = {v: k for k, v in rule_type_to_id.items()}

sep_token_id = _RULE_BASE + 12
start_token_id = _RULE_BASE + 13
eot_token_id = _RULE_BASE + 14

var_base = 128


def char_to_id(c: str) -> int:
    if c == "<PAD>":
        return pad_idx
    if c == "<SEP>":
        return sep_token_id
    if c == "<START>":
        return start_token_id
    if c == "<EOT>":
        return eot_token_id
    if c in logic_char_to_id:
        return logic_char_to_id[c]
    if c.startswith("p") and c[1:].isdigit():
        return var_base + int(c[1:]) - 1
    raise ValueError(f"Unknown symbol: {c}")


def id_to_char(i: int) -> str:
    if i == pad_idx:
        return "<PAD>"
    if i == sep_token_id:
        return "<SEP>"
    if i == start_token_id:
        return "<START>"
    if i == eot_token_id:
        return "<EOT>"
    if i in id_to_logic_char:
        return id_to_logic_char[i]
    if i >= var_base:
        return f"p{i - var_base + 1}"
    raise ValueError(f"Invalid logic token id: {i}")


def _tokenize_prop_text(text: str) -> list[int]:
    tokens: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "p":
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            if j == i + 1:
                raise ValueError(f"Invalid atom token at index {i}: {text[i:i+1]}")
            tokens.append(char_to_id(text[i:j]))
            i = j
            continue
        tokens.append(char_to_id(ch))
        i += 1
    return tokens


def tokenize_prop(prop) -> list[int]:
    return _tokenize_prop_text(str(prop))


def tokenize_prompt(sequent: Sequent) -> list[int]:
    return _tokenize_prop_text(str(sequent)) + [start_token_id]


def _rule_target_prop(rule: Rule):
    if isinstance(rule, ImpliesLeft):
        return rule.implication
    if isinstance(rule, AndLeft):
        return rule.conjunction
    if isinstance(rule, OrLeft):
        return rule.disjunction
    return None


def encode_completion(sequent: Sequent, rule: Rule) -> list[int]:
    _ = sequent
    rule_type = type(rule)
    if rule_type not in rule_type_to_id:
        raise ValueError(f"Unknown rule type for AR encoding: {rule_type}")

    out = [rule_type_to_id[rule_type]]
    target = _rule_target_prop(rule)
    if target is not None:
        out.extend(tokenize_prop(target))
    out.append(eot_token_id)
    return out


def tokenize(example: tuple[Sequent, list[Rule]]) -> tuple[list[int], list[list[int]]]:
    sequent, rules = example
    prompt = tokenize_prompt(sequent)
    completions = [encode_completion(sequent, rule) for rule in rules]
    return prompt, completions


def _parse_prop(tokens: list[str], start: int) -> tuple[object, int]:
    if start >= len(tokens):
        raise ValueError("Unexpected end of tokens while parsing proposition.")
    tok = tokens[start]
    if tok == "(":
        left, idx = _parse_prop(tokens, start + 1)
        if idx >= len(tokens):
            raise ValueError("Unexpected end of tokens after left operand.")
        op = tokens[idx]
        idx += 1
        right, idx = _parse_prop(tokens, idx)
        if idx >= len(tokens) or tokens[idx] != ")":
            raise ValueError("Expected ')' after binary proposition.")
        idx += 1
        if op == "∧":
            return And(left, right), idx
        if op == "∨":
            return Or(left, right), idx
        if op == "→":
            return Implies(left, right), idx
        raise ValueError(f"Unknown binary operator: {op}")
    if tok == "⊤":
        return PTrue(), start + 1
    if tok == "⊥":
        return PFalse(), start + 1
    if tok == "¬":
        operand, idx = _parse_prop(tokens, start + 1)
        return Implies(operand, PFalse()), idx
    if tok.startswith("p"):
        return Atom(tok), start + 1
    raise ValueError(f"Unknown proposition token: {tok}")


def _parse_prop_list(tokens: list[str]) -> list[object]:
    props: list[object] = []
    idx = 0
    while idx < len(tokens):
        prop, idx = _parse_prop(tokens, idx)
        props.append(prop)
        if idx < len(tokens):
            if tokens[idx] != ",":
                raise ValueError(f"Expected ',' between antecedents, found {tokens[idx]}")
            idx += 1
    return props


def decode_prompt(prompt_tokens: list[int]) -> Sequent:
    if not prompt_tokens:
        raise ValueError("Prompt cannot be empty.")
    nonpad = [int(tok) for tok in prompt_tokens if int(tok) != pad_idx]
    start_positions = [idx for idx, tok in enumerate(nonpad) if tok == start_token_id]
    if len(start_positions) != 1:
        raise ValueError("Prompt must contain exactly one START token.")

    start_idx = int(start_positions[0])
    sequent_tokens = nonpad[:start_idx]
    sep_positions = [idx for idx, tok in enumerate(sequent_tokens) if tok == sep_token_id]
    body_start = int(sep_positions[-1]) + 1 if sep_positions else 0
    sequent_tokens = sequent_tokens[body_start:]
    if not sequent_tokens:
        raise ValueError("Prompt body cannot be empty.")
    symbols = [id_to_char(tok) for tok in sequent_tokens]
    if "⊢" not in symbols:
        raise ValueError("Prompt sequent missing turnstile '⊢'.")

    turnstile_idx = symbols.index("⊢")
    ants_tokens = symbols[:turnstile_idx]
    cons_tokens = symbols[turnstile_idx + 1 :]
    if not cons_tokens:
        raise ValueError("Prompt sequent missing consequent.")

    ants = _parse_prop_list(ants_tokens) if ants_tokens else []
    cons, end_idx = _parse_prop(cons_tokens, 0)
    if end_idx != len(cons_tokens):
        raise ValueError("Extra tokens after parsing consequent.")

    return Sequent(ants, cons)


def _decode_rule_with_target(sequent: Sequent, rule_type, target_tokens: list[int]) -> Rule:
    if not target_tokens:
        raise ValueError(f"Rule {rule_type.__name__} requires a target proposition.")
    target_symbols = [id_to_char(tok) for tok in target_tokens]
    target, idx = _parse_prop(target_symbols, 0)
    if idx != len(target_symbols):
        raise ValueError("Invalid target proposition encoding.")
    if target not in sequent.ants:
        raise ValueError("Target proposition is not present in antecedents.")

    if rule_type is ImpliesLeft:
        if not isinstance(target, Implies):
            raise ValueError("ImpliesLeft target must be an implication.")
        return ImpliesLeft(target)
    if rule_type is AndLeft:
        if not isinstance(target, And):
            raise ValueError("AndLeft target must be a conjunction.")
        return AndLeft(target)
    if rule_type is OrLeft:
        if not isinstance(target, Or):
            raise ValueError("OrLeft target must be a disjunction.")
        return OrLeft(target)

    raise ValueError(f"Unsupported targeted rule type: {rule_type}")


def decode_completion(sequent: Sequent, completion_tokens: list[int]) -> Rule:
    if len(completion_tokens) < 2:
        raise ValueError("Completion must contain at least rule token and EOT.")
    if completion_tokens[-1] != eot_token_id:
        raise ValueError("Completion must terminate with EOT token.")

    rule_token = completion_tokens[0]
    target_tokens = completion_tokens[1:-1]
    if rule_token not in id_to_rule_type:
        raise ValueError(f"Unknown rule token id: {rule_token}")

    rule_type = id_to_rule_type[rule_token]
    if rule_type in (ImpliesLeft, AndLeft, OrLeft):
        return _decode_rule_with_target(sequent, rule_type, target_tokens)

    if target_tokens:
        raise ValueError(f"Rule {rule_type.__name__} should not have target tokens.")
    return rule_type()


def decode(tokens: tuple[list[int], list[int]]) -> tuple[Sequent, Rule]:
    prompt_tokens, completion_tokens = tokens
    sequent = decode_prompt(prompt_tokens)
    rule = decode_completion(sequent, completion_tokens)
    return sequent, rule
