"""Construct a proof using the focusing + chaining method"""

from .data import *

use_complete_proof_search = True
count = 0

def next_assump_name():
    global count
    count += 1
    return f"assump_{count}"

Ctx = List[Tuple[str, Proposition]]
Sequent = Tuple[Set[Proposition], Proposition]
RepCtx = List[Sequent]


def prove(p : Proposition, keep='simplest') -> Proof:
    polarized = negative_polarize({}, p)
    raw_proof =  (right_inverse([], [], [], polarized))
    proof = refine_choice(raw_proof)

    if keep == 'simplest':
        proof = keep_simplest(proof)
    elif keep == 'until_success':
        proof = keep_until_success(proof)
    elif keep is None:
        pass
    else:
        raise ValueError(f'unrecognized keep option: keep={keep}')

    proof = rename_assumptions(proof)
    return proof


def negative_polarize(atoms: Dict[str, Proposition], p : Proposition) -> Proposition:
    match p:
        case Atom(name):
            if name in atoms:
                if isinstance(atoms[name], NegAtom):
                    return atoms[name]
                else:
                    return Upshift(atoms[name])
            else:
                atoms[name] = NegAtom(name)
                return NegAtom(name)
        case And(left, right):
            return NegAnd(negative_polarize(atoms, left), negative_polarize(atoms, right))
        case Or(left, right):
            return Upshift(positive_polarize(atoms, p))
        case Implies(left, right):
            return Implies(positive_polarize(atoms, left), negative_polarize(atoms, right))
        case PTrue():
            return p
        case PFalse():
            return Upshift(positive_polarize(atoms, p))
        case _:
            raise ValueError(f"Invalid proposition: {p}")


def positive_polarize(atoms: Dict[str, Proposition], p : Proposition) -> Proposition:
    match p:
        case Atom(name):
            if name in atoms:
                if isinstance(atoms[name], PosAtom):
                    return atoms[name]
                else:
                    return Downshift(atoms[name])
            else:
                atoms[name] = PosAtom(name)
                return PosAtom(name)
        case And(left, right):
            return PosAnd(positive_polarize(atoms, left), positive_polarize(atoms, right))
        case Or(left, right):
            return Or(positive_polarize(atoms, left), positive_polarize(atoms, right))
        case Implies(left, right):
            return Downshift(negative_polarize(atoms, p))
        case PTrue():
            return p
        case PFalse():
            return p
        case _:
            raise ValueError(f"Invalid proposition: {p}")


def right_inverse(rep_ctx: RepCtx, stable_ctx : Ctx, unstable : Ctx, p : Proposition) -> Proof:
    match p:
        case Implies(left, right):
            new_name = next_assump_name()
            return ImpliesR(
                (new_name, right_inverse(rep_ctx, stable_ctx, [(new_name, left), *unstable], right))
                )
        case NegAnd(left, right):
            return NegAndR(
                right_inverse(rep_ctx, stable_ctx, unstable, left),
                right_inverse(rep_ctx, stable_ctx, unstable, right)
                )
        case PTrue():
            return NegTrueR()
        case NegAtom(name):
            return NegAtomR(left_inverse(rep_ctx, stable_ctx, unstable, p))
        case Upshift(operand):
            return UpshiftR(left_inverse(rep_ctx, stable_ctx, unstable, operand))
        case _:
            raise ValueError(f"Invalid proposition in right inverse: {p}")


def left_inverse(rep_ctx: RepCtx, stable_ctx : Ctx, unstable : Ctx, p : Proposition) -> Proof:
    match unstable:
        case []:
            return StableSeq(chaining(rep_ctx, stable_ctx, p))
        case [(name, prop), *rest]:
            new_name_1 = next_assump_name()
            new_name_2 = next_assump_name()
            match prop:
                case Or(left, right):
                    return OrL(name,
                        (new_name_1, left_inverse(rep_ctx, stable_ctx, [(new_name_1, left), *rest], p)),
                        (new_name_2, left_inverse(rep_ctx, stable_ctx, [(new_name_2, right), *rest], p))
                        )
                case PFalse():
                    return FalseL(name)
                case PosAnd(left, right):
                    return PosAndL(name,
                        (new_name_1, new_name_2, 
                         left_inverse(rep_ctx, stable_ctx, [(new_name_1, left), (new_name_2, right), *rest], p))
                        )
                case PTrue():
                    return PosTrueL(name, left_inverse(rep_ctx, stable_ctx, rest, p))
                case PosAtom(_):
                    return left_inverse(rep_ctx, [(name, prop), *stable_ctx], rest, p)
                case Downshift(operand):
                    return DownshiftL(name, 
                        left_inverse(rep_ctx, [(name, operand), *stable_ctx], rest, p))
                case _:
                    raise ValueError(f"Invalid proposition in left inverse: {p}")
        case _:
            raise ValueError(f"Invalid unstable context: {unstable}")


def ctx_to_sequent(ctx: Ctx, right: Proposition) -> Sequent:
    return (set([p for (_, p) in ctx]), right)


def chaining(rep_ctx: RepCtx, stable_ctx: Ctx, p : Proposition) -> Proof:
    if use_complete_proof_search:
        this_sequent = ctx_to_sequent(stable_ctx, p)
        if this_sequent in rep_ctx:
            return ProofFailed()
        rep_ctx = [*rep_ctx, this_sequent]
    else:
        pass
    
    results: List[FocusL | FocusR] = []
    
    if not isinstance(p, NegAtom):
        proof_after_right_focus = right_focus(rep_ctx, stable_ctx, p)
        r_1 = FocusR(proof_after_right_focus)
        results.append(r_1)
    
    for i,(name, prop) in enumerate(stable_ctx):
        if isinstance(prop, PosAtom):
            continue

        if use_complete_proof_search:
            new_stable_ctx = stable_ctx[:i] + stable_ctx[i:]
        else:
            new_stable_ctx = stable_ctx[:i] + stable_ctx[i+1:]
        rl = FocusL(name, left_focus(rep_ctx, new_stable_ctx, name, prop, p))
        results.append(rl)
    
    return FocusChoice(results)


def right_focus(rep_ctx: RepCtx, stable_ctx: Ctx, p : Proposition) -> Proof:
    match p:
        case Or(left, right):
                return OrR_choice(
                    OrR_left(right_focus(rep_ctx, stable_ctx, left)),
                    OrR_right(right_focus(rep_ctx, stable_ctx, right)))
        case PFalse():
            return ProofFailed()
        case PosAnd(left, right):
            return PosAndR(right_focus(rep_ctx, stable_ctx, left), right_focus(rep_ctx, stable_ctx, right))
        case PTrue():
            return PosTrueR()
        case PosAtom(name):
            match [assump_name for (assump_name, assump) in stable_ctx if assump == p]:
                case []:
                    return ProofFailed()
                case [assump_name, *rest]:
                    return PosAtomR(assump_name)
                case _:
                    raise ValueError("unrecognized value")
        case Downshift(operand):
            return DownshiftR(right_inverse(rep_ctx, stable_ctx, [], operand))
        case _:
            raise ValueError(f"invalid proposition in right focus: {p}")


def left_focus(rep_ctx: RepCtx, stable_ctx: Ctx, name: str, prop: Proposition, p : Proposition) -> Proof:
    new_name = next_assump_name()
    new_name_2 = next_assump_name()
    match prop:
        case Implies(left, right):
            return ImpliesL(name, 
                left,
                right_focus(rep_ctx, stable_ctx, left),
                (new_name, left_focus(rep_ctx, stable_ctx, new_name, right, p)), 
                next_assump_name()
            )
        case NegAnd(left, right):
            return NegAndL_choice(
                NegAndL_left(name, (new_name, left_focus(rep_ctx, stable_ctx, new_name, left, p))),
                NegAndL_right(name, (new_name_2, left_focus(rep_ctx, stable_ctx, new_name_2, right, p)))
            )
        case PTrue():
            return ProofFailed()
        case NegAtom(fname):
            if p == prop:
                return NegAtomL(name)
            else:
                return ProofFailed()
        case Upshift(operand):
            return UpshiftL(name, left_inverse(rep_ctx, stable_ctx, [(name, operand)], p))
        case _:
            raise ValueError(f"Invalid proposition in left focus: {prop}")


def collect_top_level_choices(p: Proof) -> Sequence[Proof]:
    match p:
        case FocusChoice(choices):
            all_choices : List[Proof] = []
            for c in choices:
                all_choices.extend(collect_top_level_choices(c))
            return all_choices
        case NegAndL_choice(l, r):
            return [l, r]
        case OrR_choice(l, r):
            return [l, r]
        case FocusR(sub):
            return collect_top_level_choices(sub)
        case FocusL(name, sub):
            return collect_top_level_choices(sub)
        case ProofFailed():
            return []
        case _:
            return [p]

        
def refine_choice(proof: Proof) -> Proof:
    def consolidate_proof_choice(proof: Proof) -> Optional[Proof]:
        match proof:
            case FocusChoice(choices):
                all_choices = collect_top_level_choices(proof)
                refined = [refine_choice(c) for c in all_choices]
                return FocusChoice(refined)
            case _:
                return None
    return map_proof(consolidate_proof_choice, proof)



def keep_simplest(p: Proof) -> Proof:
    if proof_has_failed(p):
        return keep_until_success(p)

    def make_choice(p: Proof) -> Optional[Proof]:
        match p:
            case FocusChoice(choices):
                success_choices = [keep_simplest(c) for c in choices if not proof_has_failed(c)]
                return keep_simplest(min(success_choices, key=calculate_complexity))
            case NegAndL_choice(l, r):
                success_choices = [keep_simplest(c) for c in [l, r] if not proof_has_failed(c)]
                return keep_simplest(min(success_choices, key=calculate_complexity))
            case OrR_choice(l, r):
                success_choices = [keep_simplest(c) for c in [l, r] if not proof_has_failed(c)]
                return keep_simplest(min(success_choices, key=calculate_complexity))
            case _:
                return None

    return map_proof(make_choice, p)


def keep_until_success(p: Proof) -> Proof:
    def make_choice(p: Proof) -> Optional[Proof]:
        match p:
            case FocusChoice(choices):
                new_choices = []
                for c in choices:
                    if not proof_has_failed(c):
                        new_choices.append(keep_until_success(c))
                        break
                    else:
                        new_choices.append(keep_until_success(c))
                return FocusChoice(new_choices)
            case NegAndL_choice(l, r):
                if not proof_has_failed(l):
                    return keep_until_success(l)
                else:
                    return NegAndL_choice(
                        keep_until_success(l),
                        keep_until_success(r)
                    )
            case OrR_choice(l, r):
                if not proof_has_failed(l):
                    return keep_until_success(l)
                else:
                    return OrR_choice(
                        keep_until_success(l),
                        keep_until_success(r)
                    )
            case _:
                return None

    result =  map_proof(make_choice, p)    
    return result


def proof_has_failed(p: Proof) -> bool:
    class ProofHasFailed(Exception): pass

    def check_failure(p: Proof) -> Optional[Proof]:
        match p:
            case ProofFailed():
                raise ProofHasFailed()
            case FocusChoice(choices):
                if all([proof_has_failed(c) for c in choices]):
                    raise ProofHasFailed()
                else:
                    return "FAIL CHECK STUB" # cut off checking here
            case NegAndL_choice(l, r):
                if proof_has_failed(l) and proof_has_failed(r):
                    raise ProofHasFailed()
                else:
                    return "FAIL CHECK STUB"
            case OrR_choice(l, r):
                if proof_has_failed(l) and proof_has_failed(r):
                    raise ProofHasFailed()
                else:
                    return "FAIL CHECK STUB"
            case _:
                return None
    
    try:
        map_proof(check_failure, p)
        return False

    except ProofHasFailed:
        return True


def calculate_complexity(p: Proof) -> int:
    match p:
        case ImpliesR((name, subproof)):
            return 1 + calculate_complexity(subproof)
        case NegAndR(left, right):
            return 1 + calculate_complexity(left) + calculate_complexity(right)
        case NegTrueR():
            return 1
        case NegAtomR(subproof):
            return 1 + calculate_complexity(subproof)
        case UpshiftR(subproof):
            return 1 + calculate_complexity(subproof)
        case OrL(name, (left_name, left_proof), (right_name, right_proof)):
            return 1 + calculate_complexity(left_proof) + calculate_complexity(right_proof)
        case FalseL(name):
            return 1
        case PosAndL(name, (left_name, right_name, subproof)):
            return 1 + calculate_complexity(subproof)
        case PosTrueL(name, subproof):
            return 1 + calculate_complexity(subproof)
        case PosAtomL(name, subproof):
            return 1 + calculate_complexity(subproof)
        case DownshiftL(name, subproof):
            return 1 + calculate_complexity(subproof)
        case StableSeq(subproof):
            return 1 + calculate_complexity(subproof)
        case FocusR(subproof):
            return 1 + calculate_complexity(subproof)
        case FocusL(name, subproof):
            return 1 + calculate_complexity(subproof)
        case OrR_left(subproof):
            return 1 + calculate_complexity(subproof)
        case OrR_right(subproof):
            return 1 + calculate_complexity(subproof)
        case PosAndR(left, right):
            return 1 + calculate_complexity(left) + calculate_complexity(right)
        case PosTrueR():
            return 1
        case PosAtomR(name):
            return 1
        case DownshiftR(subproof):
            return 1 + calculate_complexity(subproof)
        case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
            return 1 + calculate_complexity(left) + calculate_complexity(right_proof)
        case NegAndL_left(name, (left_name, left_proof)):
            return 1 + calculate_complexity(left_proof)
        case NegAndL_right(name, (right_name, right_proof)):
            return 1 + calculate_complexity(right_proof)
        case NegAtomL(name):
            return 1
        case UpshiftL(name, subproof):
            return 1 + calculate_complexity(subproof)
        case FocusChoice(choices):
            return 1 + sum([calculate_complexity(c) for c in choices])
        case OrR_choice(l, r):
            return 1 + calculate_complexity(l) + calculate_complexity(r)
        case NegAndL_choice(l, r):
            return 1 + calculate_complexity(l) + calculate_complexity(r)
        case ProofFailed():
            return 1
        case _:
            raise ValueError(f"Invalid proof: {p}")


def rename_assumptions(p: Proof) -> Proof:
    return _rename(AssumptionCtx(), p)


class AssumptionCtx():
    def __init__(self):
        self.existing = {}

    def add(self, name) -> str:
        assert name not in self.existing
        self.existing[name] = f"h{len(self.existing.keys()) + 1}"
        return self.existing[name]

    def get(self, name) -> str:
        return self.existing[name]

def _rename(ctx: AssumptionCtx, p: Proof) -> Proof:
    match p:
        case ImpliesR((name, subproof)):
            return ImpliesR((ctx.add(name), _rename(ctx, subproof)))
        case NegAndR(left, right):
            return NegAndR(_rename(ctx, left), _rename(ctx, right))
        case NegTrueR():
            return NegTrueR()
        case NegAtomR(subproof):
            return NegAtomR(_rename(ctx, subproof))
        case UpshiftR(subproof):
            return UpshiftR(_rename(ctx, subproof))
        case OrL(name, (left_name, left_proof), (right_name, right_proof)):
            return OrL(ctx.get(name),
                (ctx.add(left_name), _rename(ctx, left_proof)),
                (ctx.add(right_name), _rename(ctx, right_proof))
                )
        case FalseL(name):
            return FalseL(ctx.get(name))
        case PosAndL(name, (left_name, right_name, subproof)):
            return PosAndL(ctx.get(name),
                (ctx.add(left_name), ctx.add(right_name), _rename(ctx, subproof))
                )
        case PosTrueL(name, subproof):
            return PosTrueL(ctx.get(name), _rename(ctx, subproof))
        case PosAtomL(name, subproof):
            return PosAtomL(ctx.get(name), _rename(ctx, subproof))
        case DownshiftL(name, subproof):
            return DownshiftL(ctx.get(name), _rename(ctx, subproof))
        case StableSeq(subproof):
            return StableSeq(_rename(ctx, subproof))
        case FocusR(subproof):
            return FocusR(_rename(ctx, subproof))
        case FocusL(name, subproof):
            return FocusL(ctx.get(name), _rename(ctx, subproof))
        case OrR_left(subproof):
            return OrR_left(_rename(ctx, subproof))
        case OrR_right(subproof):
            return OrR_right(_rename(ctx, subproof))
        case PosAndR(left, right):
            return PosAndR(_rename(ctx, left), _rename(ctx, right))
        case PosTrueR():
            return PosTrueR()
        case PosAtomR(name):
            return PosAtomR(ctx.get(name))
        case DownshiftR(subproof):
            return DownshiftR(_rename(ctx, subproof))
        case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
            ctx.add(additional_name)
            return ImpliesL(ctx.get(name), prop,
                _rename(ctx, left),
                (ctx.add(right_name), _rename(ctx, right_proof)),
                ctx.get(additional_name)
                )
        case NegAndL_left(name, (left_name, left_proof)):
            return NegAndL_left(ctx.get(name),
                (ctx.add(left_name), _rename(ctx, left_proof))
                )
        case NegAndL_right(name, (right_name, right_proof)):
            return NegAndL_right(ctx.get(name),
                (ctx.add(right_name), _rename(ctx, right_proof))
                )
        case NegAtomL(name):
            return NegAtomL(ctx.get(name))
        case UpshiftL(name, subproof):
            return UpshiftL(ctx.get(name), _rename(ctx, subproof))
        case FocusChoice(choices):
            return FocusChoice([_rename(ctx, choice) for choice in choices])
        case NegAndL_choice(l, r):
            return NegAndL_choice(_rename(ctx, l), _rename(ctx, r))
        case OrR_choice(l, r):
            return OrR_choice(_rename(ctx, l), _rename(ctx, r))
        case ProofFailed():
            return ProofFailed()
        case _:
            raise ValueError(f"Invalid proof: {p}")