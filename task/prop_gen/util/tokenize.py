"""
Generate propositional logic dataset
"""

# <codecell>
from .elem import *
from .sample import sample_imply, list_sequents

type Example = tuple[Sequent, list[Rule]]
type RuleToken = tuple[int, int]
type TokenizedExample = tuple[list[int], list[RuleToken]]

pad_idx = 0

logic_char_to_id = {
    '⊢': 1,
    '∧': 2,
    '∨': 3,
    '¬': 4,
    '→': 5,
    '(': 6,
    ')': 7,
    '⊤': 8,
    '⊥': 9,
    ',': 10
}

id_to_logic_char = {v: k for k, v in logic_char_to_id.items()}

rule_type_to_id = {
    Axiom: 1,
    ImpliesRight: 2,
    ImpliesLeft: 3,
    AndRight: 4,
    AndLeft: 5,
    OrRight1: 6,
    OrRight2: 7,
    OrLeft: 8,
    TrueRight: 9,
    FalseLeft: 10,
    NegationRight: 11,
}

id_to_rule_type = {v: k for k, v in rule_type_to_id.items()}

def char_to_id(c: str) -> int:
    if c in logic_char_to_id:
        return logic_char_to_id[c]
    else:
        if c[0] == 'p':
            return len(logic_char_to_id) + int(c[1:])

def id_to_char(i: int) -> str:
    if i in id_to_logic_char:
        return id_to_logic_char[i]
    else:
        n_fixed = len(logic_char_to_id)
        if i > n_fixed:
            return 'p' + str(i - n_fixed)
        else:
            raise ValueError(f"Invalid id: {i}")


def _tokenize_sequent(sequent: Sequent) -> list[int]:
    text = str(sequent)
    tokens: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == 'p':
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            if j == i + 1:
                raise ValueError(f"Invalid atom token starting at index {i}: {text[i:i+1]}")
            tokens.append(char_to_id(text[i:j]))
            i = j
            continue
        tokens.append(char_to_id(ch))
        i += 1
    return tokens


def _find_assumption_position(ants: tuple[Proposition, ...], target: Proposition) -> int:
    for idx, ant in enumerate(ants, start=1):
        if ant == target:
            return idx
    raise ValueError(f"Target assumption {target} not found in antecedents: {ants}")


def tokenize(example: Example) -> TokenizedExample:
    """Tokenizes the sequent and inference rules into ids. Inference rules are tokenized as
    two indices: rule type and (if relevant) position of the target assumption in the
    antecedent (with position starting at 1). If the rule does not require an assumption,
    then the second index is 0.
    """
    sequent, rules = example
    sequent_tokens = _tokenize_sequent(sequent)
    rule_tokens: list[RuleToken] = []
    for rule in rules:
        rule_type = type(rule)
        if rule_type not in rule_type_to_id:
            raise ValueError(f"Unknown rule type: {rule_type}")
        rule_id = rule_type_to_id[rule_type]
        pos = 0
        if isinstance(rule, ImpliesLeft):
            pos = _find_assumption_position(sequent.ants, rule.implication)
        elif isinstance(rule, AndLeft):
            pos = _find_assumption_position(sequent.ants, rule.conjunction)
        elif isinstance(rule, OrLeft):
            pos = _find_assumption_position(sequent.ants, rule.disjunction)
        rule_tokens.append((rule_id, pos))
    return sequent_tokens, rule_tokens


def _parse_prop(tokens: list[str], start: int) -> tuple[Proposition, int]:
    if start >= len(tokens):
        raise ValueError("Unexpected end of tokens while parsing proposition.")
    tok = tokens[start]
    if tok == '(':
        left, idx = _parse_prop(tokens, start + 1)
        if idx >= len(tokens):
            raise ValueError("Unexpected end of tokens after left operand.")
        op = tokens[idx]
        idx += 1
        right, idx = _parse_prop(tokens, idx)
        if idx >= len(tokens) or tokens[idx] != ')':
            raise ValueError("Expected ')' after binary proposition.")
        idx += 1
        if op == '∧':
            return And(left, right), idx
        if op == '∨':
            return Or(left, right), idx
        if op == '→':
            return Implies(left, right), idx
        raise ValueError(f"Unknown binary operator: {op}")
    if tok == '⊤':
        return PTrue(), start + 1
    if tok == '⊥':
        return PFalse(), start + 1
    if tok == '¬':
        operand, idx = _parse_prop(tokens, start + 1)
        return Implies(operand, PFalse()), idx
    if tok.startswith('p'):
        return Atom(tok), start + 1
    raise ValueError(f"Unknown proposition token: {tok}")


def _parse_prop_list(tokens: list[str]) -> list[Proposition]:
    props: list[Proposition] = []
    idx = 0
    while idx < len(tokens):
        prop, idx = _parse_prop(tokens, idx)
        props.append(prop)
        if idx < len(tokens):
            if tokens[idx] != ',':
                raise ValueError(f"Expected ',' between antecedents, found {tokens[idx]}")
            idx += 1
    return props


def decode(tokens: TokenizedExample) -> Example:
    """Decodes the tokenized sequent and inference rules back into their original forms."""
    sequent_tokens, rule_tokens = tokens
    symbols = [id_to_char(tok) for tok in sequent_tokens]
    if '⊢' not in symbols:
        raise ValueError("Tokenized sequent missing turnstile '⊢'.")
    turnstile_idx = symbols.index('⊢')
    ants_tokens = symbols[:turnstile_idx]
    cons_tokens = symbols[turnstile_idx + 1:]
    if not cons_tokens:
        raise ValueError("Tokenized sequent missing consequent.")
    ants = _parse_prop_list(ants_tokens) if ants_tokens else []
    cons, end_idx = _parse_prop(cons_tokens, 0)
    if end_idx != len(cons_tokens):
        raise ValueError("Extra tokens after parsing consequent.")
    sequent = Sequent(ants, cons)

    rules: list[Rule] = []
    for rule_id, pos in rule_tokens:
        if rule_id not in id_to_rule_type:
            raise ValueError(f"Unknown rule id: {rule_id}")
        rule_type = id_to_rule_type[rule_id]
        if rule_type in (ImpliesLeft, AndLeft, OrLeft):
            if pos <= 0 or pos > len(sequent.ants):
                raise ValueError(f"Invalid antecedent position {pos} for {rule_type}.")
            target = sequent.ants[pos - 1]
            if rule_type is ImpliesLeft:
                if not isinstance(target, Implies):
                    raise ValueError("ImpliesLeft requires an implication antecedent.")
                rules.append(ImpliesLeft(target))
            elif rule_type is AndLeft:
                if not isinstance(target, And):
                    raise ValueError("AndLeft requires a conjunction antecedent.")
                rules.append(AndLeft(target))
            else:
                if not isinstance(target, Or):
                    raise ValueError("OrLeft requires a disjunction antecedent.")
                rules.append(OrLeft(target))
        else:
            if pos != 0:
                raise ValueError(f"Rule {rule_type} should not have antecedent position.")
            rules.append(rule_type())
    return sequent, rules
