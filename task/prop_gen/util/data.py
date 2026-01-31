
"""Basic types and operations"""

from __future__ import annotations
from dataclasses import dataclass
from typing import *


@dataclass(frozen=True)
class Atom:
    name: str

    def __str__(self):
        return self.name
    
@dataclass(frozen=True)
class PosAtom:
    name: str

    def __str__(self):
        return f"{self.name}⁺"

@dataclass(frozen=True)
class NegAtom:
    name: str

    def __str__(self):
        return f"{self.name}⁻"

@dataclass(frozen=True)
class And:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

@dataclass(frozen=True)
class PosAnd:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} ∧⁺ {self.right})"

@dataclass(frozen=True)
class NegAnd:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} ∧⁻ {self.right})"

@dataclass(frozen=True)
class Or:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} ∨ {self.right})"

@dataclass(frozen=True)
class Implies:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} → {self.right})"

@dataclass(frozen=True)  
class PTrue:
    def __str__(self):
        return "⊤"

@dataclass(frozen=True)
class PFalse:
    def __str__(self):
        return "⊥"
    
@dataclass(frozen=True)
class Upshift:
    operand: Proposition

    def __str__(self):
        return f"⇑{self.operand}"

@dataclass(frozen=True)
class Downshift:
    operand: Proposition

    def __str__(self):
        return f"⇓{self.operand}"

Proposition = And | PosAnd | NegAnd | Or | Implies | PTrue | PFalse | Atom | PosAtom | NegAtom | Upshift | Downshift # up and down shifts are used in proof search only

@dataclass(frozen=True)
class ImpliesR:
    sub: Tuple[str,Proof]

@dataclass(frozen=True)
class NegAndR:
    left: Proof
    right: Proof

@dataclass(frozen=True)
class NegTrueR:
    pass

@dataclass(frozen=True)
class NegAtomR:
    sub: Proof

@dataclass(frozen=True)
class UpshiftR:
    sub: Proof

@dataclass(frozen=True)
class OrL:
    name: str
    left: Tuple[str, Proof]
    right: Tuple[str, Proof]

@dataclass(frozen=True)
class FalseL:
    name: str
    pass

@dataclass(frozen=True)
class PosAndL:
    name: str
    sub: Tuple[str, str, Proof]

@dataclass(frozen=True)
class PosTrueL:
    name: str
    sub: Proof

@dataclass(frozen=True)
class PosAtomL:
    name: str
    sub: Proof

@dataclass(frozen=True)
class DownshiftL:
    name: str
    sub: Proof

@dataclass(frozen=True)
class StableSeq:
    sub: Proof

@dataclass(frozen=True)
class FocusR:
    sub: Proof


@dataclass(frozen=True)
class FocusL:
    name: str
    sub: Proof

@dataclass(frozen=True)
class FocusChoice:
    choices: List[FocusR | FocusL]

@dataclass(frozen=True)
class ProofFailed:
    reason : Optional[str] = None
    

@dataclass(frozen=True)
class PosAndR:
    left: Proof
    right: Proof

@dataclass(frozen=True)
class OrR_left:
    sub: Proof

@dataclass(frozen=True)
class OrR_right:
    sub: Proof

@dataclass(frozen=True)
class OrR_choice:
    l: OrR_left | OrR_right
    r: OrR_right | OrR_left

@dataclass(frozen=True)
class PosTrueR:
    pass

@dataclass(frozen=True)
class PosAtomR:
    name: str
    pass

@dataclass(frozen=True)
class DownshiftR:
    sub: Proof

@dataclass(frozen=True)
class ImpliesL:
    name: str
    prop: Proposition
    left: Proof
    right: Tuple[str, Proof]
    additional_name: str # used purely in lean

@dataclass(frozen=True)
class NegAndL_left:
    name: str
    sub: Tuple[str, Proof]

@dataclass(frozen=True)
class NegAndL_right:
    name: str
    sub: Tuple[str, Proof]

@dataclass(frozen=True)
class NegAndL_choice:
    l: NegAndL_left | NegAndL_right
    r: NegAndL_right | NegAndL_left

@dataclass(frozen=True)
class NegAtomL:
    name: str

@dataclass(frozen=True)
class UpshiftL:
    name: str
    sub: Proof


Proof = (ImpliesR | NegAndR | NegTrueR | NegAtomR | UpshiftR | OrL | FalseL |
PosAndL | PosTrueL | PosAtomL | DownshiftL | StableSeq | FocusR | PosAndR | OrR_left 
| OrR_right | PosTrueR | PosAtomR | DownshiftR | ImpliesL | NegAndL_left | NegAndL_right 
| NegAtomL | UpshiftL | FocusL| FocusChoice | NegAndL_choice | OrR_choice| ProofFailed)


def map_proof(f: Callable[[Proof], Optional[Proof]], p : Proof) -> Proof:
    tentative = f(p) 
    if tentative is not None:
        return tentative
    else:
        match p:
            case ImpliesR((name, subproof)):
                return ImpliesR((name, map_proof(f, subproof)))
            case NegAndR(left, right):
                return NegAndR(map_proof(f, left), map_proof(f, right))
            case NegTrueR():
                return p
            case NegAtomR(subproof):
                return NegAtomR(map_proof(f, subproof))
            case UpshiftR(subproof):
                return UpshiftR(map_proof(f, subproof))
            case OrL(name, (left_name, left_proof), (right_name, right_proof)):
                return OrL(name,
                    (left_name, map_proof(f, left_proof)),
                    (right_name, map_proof(f, right_proof))
                    )
            case FalseL(name):
                return p
            case PosAndL(name, (left_name, right_name, subproof)):
                return PosAndL(name,
                    (left_name, right_name, map_proof(f, subproof))
                    )
            case PosTrueL(name, subproof):
                return PosTrueL(name, map_proof(f, subproof))
            case PosAtomL(name, subproof):
                return PosAtomL(name, map_proof(f, subproof))
            case DownshiftL(name, subproof):
                return DownshiftL(name, map_proof(f, subproof))
            case StableSeq(subproof):
                return StableSeq(map_proof(f, subproof))
            case FocusR(subproof):
                return FocusR(map_proof(f, subproof))
            case FocusL(name, subproof):
                return FocusL(name, map_proof(f, subproof))
            case OrR_left(subproof):
                return OrR_left(map_proof(f, subproof))
            case OrR_right(subproof):
                return OrR_right(map_proof(f, subproof))
            case PosAndR(left, right):
                return PosAndR(map_proof(f, left), map_proof(f, right))
            case PosTrueR():
                return p
            case PosAtomR(name):
                return p
            case DownshiftR(subproof):
                return DownshiftR(map_proof(f, subproof))
            case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
                return ImpliesL(name, prop, map_proof(f, left), (right_name, map_proof(f, right_proof)), additional_name)
            case NegAndL_left(name, (left_name, left_proof)):
                return NegAndL_left(name, (left_name, map_proof(f, left_proof)))
            case NegAndL_right(name, (right_name, right_proof)):
                return NegAndL_right(name, (right_name, map_proof(f, right_proof)))
            case NegAtomL(name):
                return p
            case UpshiftL(name, subproof):
                return UpshiftL(name, map_proof(f, subproof))
            case ProofFailed():
                return p
            case FocusChoice(choices):
                return FocusChoice([map_proof(f, c) for c in choices])
            case NegAndL_choice(l, r):
                return NegAndL_choice(map_proof(f, l), map_proof(f, r))
            case OrR_choice(l, r):
                return OrR_choice(map_proof(f, l), map_proof(f, r))
            case _:
                raise ValueError(f"Invalid proof: {p}")

