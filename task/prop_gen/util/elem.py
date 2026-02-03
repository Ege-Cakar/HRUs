
"""Basic types and operations"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Atom:
    name: str

    def __str__(self):
        return self.name

    __repr__ = __str__

@dataclass(frozen=True)
class And:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    __repr__ = __str__

@dataclass(frozen=True)
class Or:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} ∨ {self.right})"

    __repr__ = __str__

@dataclass(frozen=True)
class Implies:
    left: Proposition
    right: Proposition

    def __str__(self):
        return f"({self.left} → {self.right})"

    __repr__ = __str__

@dataclass(frozen=True)  
class PTrue:
    def __str__(self):
        return "⊤"

    __repr__ = __str__

@dataclass(frozen=True)
class PFalse:
    def __str__(self):
        return "⊥"

    __repr__ = __str__


Proposition = And | Or | Implies | PTrue | PFalse | Atom

@dataclass
class Sequent:
    """A sequent with antecedents and consequent. Ants stored as tuple for immutability."""
    ants: Tuple[Proposition, ...]
    cons: Proposition
    
    def __init__(self, ants: Sequence[Proposition], cons: Proposition):
        """Create a sequent, converting ants to tuple if needed."""
        object.__setattr__(self, 'ants', tuple(ants))
        object.__setattr__(self, 'cons', cons)
    
    def __hash__(self):
        return hash((self.ants, self.cons))
    
    def __eq__(self, other):
        if not isinstance(other, Sequent):
            return NotImplemented
        return self.ants == other.ants and self.cons == other.cons

    def __str__(self):
        ants_str = ', '.join(str(ant) for ant in self.ants)
        return f"{ants_str} ⊢ {self.cons}"

    __repr__ = __str__

class Rule:
    """Base class for inference rules in Gentzen's NJ system.
    
    Rules return a list of subgoals (sequents that must be proven).
    - Empty list [] means the rule succeeded with no remaining subgoals (proven).
    - None means the rule is not applicable.
    - Non-empty list means those sequents must be proven to complete the proof.
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        raise NotImplementedError


Example = tuple[Sequent, list[Rule]]
RuleToken = tuple[int, int]
RuleTokens = list[RuleToken]
TokenizedExample = tuple[list[int], RuleTokens]


# =============================================================================
# Axiom (Identity)
# =============================================================================

class Axiom(Rule):
    """Axiom: If the consequent appears in the antecedents, the sequent is proven.
    
    Γ, A ⊢ A
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        return [] if sequent.cons in sequent.ants else None
    
    def __str__(self) -> str:
        return "Ax"

    __repr__ = __str__


# =============================================================================
# Implication Rules
# =============================================================================

class ImpliesRight(Rule):
    """→I (Implication Introduction): To prove A → B, assume A and prove B.
    
        Γ, A ⊢ B
       ──────────
        Γ ⊢ A → B
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if isinstance(sequent.cons, Implies):
            return [Sequent(sequent.ants + (sequent.cons.left,), sequent.cons.right)]
        return None
    
    def __str__(self) -> str:
        return "→R"

    __repr__ = __str__


class ImpliesLeft(Rule):
    """→E (Implication Elimination / Modus Ponens): From A → B in context, prove A to get B.
    
       Γ ⊢ A    Γ, B ⊢ C
      ────────────────────  (where A → B ∈ Γ)
            Γ ⊢ C
    """
    def __init__(self, implication: Implies):
        self.implication = implication
    
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if self.implication not in sequent.ants:
            return None
        new_ants = tuple(a for a in sequent.ants if a != self.implication)
        return [
            Sequent(new_ants, self.implication.left),
            Sequent(new_ants + (self.implication.right,), sequent.cons),
        ]
    
    def __str__(self) -> str:
        return f"→L ({self.implication})"

    __repr__ = __str__


# =============================================================================
# Conjunction Rules
# =============================================================================

class AndRight(Rule):
    """∧I (Conjunction Introduction): To prove A ∧ B, prove both A and B.
    
       Γ ⊢ A    Γ ⊢ B
      ────────────────
         Γ ⊢ A ∧ B
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if isinstance(sequent.cons, And):
            return [
                Sequent(sequent.ants, sequent.cons.left),
                Sequent(sequent.ants, sequent.cons.right),
            ]
        return None
    
    def __str__(self) -> str:
        return "∧R"

    __repr__ = __str__


class AndLeft(Rule):
    """∧E (Conjunction Elimination): From A ∧ B in context, extract A and B.
    
       Γ, A, B ⊢ C
      ──────────────  (where A ∧ B ∈ Γ)
       Γ, A ∧ B ⊢ C
    
    This rule is invertible: applying it never loses provability.
    """
    def __init__(self, conjunction: And):
        self.conjunction = conjunction
    
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if self.conjunction not in sequent.ants:
            return None
        new_ants = tuple(a for a in sequent.ants if a != self.conjunction)
        return [Sequent(new_ants + (self.conjunction.left, self.conjunction.right), sequent.cons)]
    
    def __str__(self) -> str:
        return f"∧L ({self.conjunction})"

    __repr__ = __str__


# =============================================================================
# Disjunction Rules
# =============================================================================

class OrRight1(Rule):
    """∨I₁ (Disjunction Introduction Left): To prove A ∨ B, prove A.
    
         Γ ⊢ A
       ──────────
        Γ ⊢ A ∨ B
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if isinstance(sequent.cons, Or):
            return [Sequent(sequent.ants, sequent.cons.left)]
        return None
    
    def __str__(self) -> str:
        return "∨R₁"

    __repr__ = __str__


class OrRight2(Rule):
    """∨I₂ (Disjunction Introduction Right): To prove A ∨ B, prove B.
    
         Γ ⊢ B
       ──────────
        Γ ⊢ A ∨ B
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if isinstance(sequent.cons, Or):
            return [Sequent(sequent.ants, sequent.cons.right)]
        return None
    
    def __str__(self) -> str:
        return "∨R₂"

    __repr__ = __str__


class OrLeft(Rule):
    """∨E (Disjunction Elimination): From A ∨ B in context, prove goal from both cases.
    
       Γ, A ⊢ C    Γ, B ⊢ C
      ────────────────────────  (where A ∨ B ∈ Γ)
            Γ, A ∨ B ⊢ C
    
    This rule is invertible: applying it never loses provability.
    """
    def __init__(self, disjunction: Or):
        self.disjunction = disjunction
    
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if self.disjunction not in sequent.ants:
            return None
        new_ants = tuple(a for a in sequent.ants if a != self.disjunction)
        return [
            Sequent(new_ants + (self.disjunction.left,), sequent.cons),
            Sequent(new_ants + (self.disjunction.right,), sequent.cons),
        ]
    
    def __str__(self) -> str:
        return f"∨L ({self.disjunction})"

    __repr__ = __str__


# =============================================================================
# Truth Rules
# =============================================================================

class TrueRight(Rule):
    """⊤I (Truth Introduction): ⊤ is always provable.
    
       ─────────
        Γ ⊢ ⊤
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        return [] if isinstance(sequent.cons, PTrue) else None
    
    def __str__(self) -> str:
        return "⊤R"

    __repr__ = __str__


# =============================================================================
# Falsity Rules
# =============================================================================

_PFALSE = PFalse()  # Singleton for comparisons


class FalseLeft(Rule):
    """⊥E (Falsity Elimination / Ex Falso Quodlibet): From ⊥, derive anything.
    
       ─────────────  (where ⊥ ∈ Γ)
         Γ ⊢ C
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        return [] if _PFALSE in sequent.ants else None
    
    def __str__(self) -> str:
        return "⊥L"

    __repr__ = __str__


# =============================================================================
# Negation (derived - encoded as A → ⊥)
# =============================================================================

class NegationRight(Rule):
    """¬I (Negation Introduction): To prove ¬A (encoded as A → ⊥), assume A and derive ⊥.
    
    This is ImpliesRight specialized for implications to ⊥; exists for clarity.
    
        Γ, A ⊢ ⊥
       ──────────
        Γ ⊢ A → ⊥
    """
    def apply(self, sequent: Sequent) -> Optional[List[Sequent]]:
        if isinstance(sequent.cons, Implies) and isinstance(sequent.cons.right, PFalse):
            return [Sequent(sequent.ants + (sequent.cons.left,), _PFALSE)]
        return None
    
    def __str__(self) -> str:
        return "¬R"

    __repr__ = __str__
