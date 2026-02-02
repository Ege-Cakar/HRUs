"""Formatting proof for output"""

import json
import subprocess
import xml.etree.ElementTree as et

from .data import *

# lean_repl_path = r'/home/grandpaa/workspace/imply/imply/task/prop_gen/old/propositional_logic/random_gen/lean-repl'
lean_repl_path = r'util/repl'


def format_example(n_atoms: int, prop: Proposition, proof: Proof, proof_to_string=True):
    ctx = StateTrackingCtx(n_atoms, prop)

    init_state = build_state(ctx.get_cur_tactic_state(), 0)

    is_succ = traverse_proof(ctx, proof, 0)
    result = ctx.result_buffer

    if is_succ:
        result.append(et.Element('success'))
    else:
        result.append(et.Element('failure'))
    
    
    proof_states = [int(e.get('id')) for e in result if e.tag == 'state']
    proof_length = max(proof_states) if len(proof_states) > 0 else 0
    uniq_ops = set([standardize(e.text) for e in result if e.tag == 'tactic'])

    if proof_to_string:
        proof = ''.join([et.tostring(r, encoding='utf-8').decode('utf-8') for r in result])
    else:
        proof = result
    
    ex = {
        'input': et.tostring(init_state, encoding='utf-8').decode('utf-8'),
        'is_true': is_succ,
        'proof': proof,
        'length': proof_length,
        'ops': list(uniq_ops)
    }

    return ex


def build_state(tactic_state, state_id):
    # lines = tactic_state.split('\n')
    premise, conclusion = tactic_state.split('⊢')
    lines = premise.split('\n')
    conclusion = conclusion.strip().replace('\n', ' ').split()
    conclusion = ' '.join(conclusion)

    state = et.Element('state', id=str(state_id))
    for line in lines:
        if len(line) > 0:
            if_elem = et.SubElement(state, 'if')
            if_elem.text = line

    then_elem = et.SubElement(state, 'then')
    then_elem.text = conclusion
    return state


class StateTrackingCtx:
    def __init__(self, num_vars: int, prop: Proposition, 
                 initial_indent:str = "  "):
        self.num_vars = num_vars
        self.prop = prop
        lean_text = to_lean_decl(num_vars, prop)
        self.states = [lean_text]
        self.cur_state_num = 0
        self.current_indent = initial_indent
        self.result_buffer : List[et.Element] = []
        self.state_current_choices : Dict[int, int] = {}

    def push_state(self, state: str) -> None:
        """State should be a partial lean proof"""
        self.states.append(state)
        self.cur_state_num = len(self.states) - 1

    def get_cur_state_num(self) -> int:
        return self.cur_state_num

    def set_cur_state_num(self, num: int) -> None:
        self.cur_state_num = num

    def get_cur_state_text(self) -> str:
        return self.states[self.cur_state_num]

    def increment_indent(self) -> None:
        self.current_indent += "  "
    
    def decrement_indent(self) -> None:
        self.current_indent = self.current_indent[:-2]
    
    def get_cur_tactic_state(self) -> str:
        return get_lean_tactic_state(self.get_cur_state_text() + "\n" + self.current_indent + "sorry")


def traverse_proof(ctx: StateTrackingCtx, proof: Proof, next_tactic_num: int) -> bool:
    result = ctx.result_buffer
    def recurse(p: Proof):
        return traverse_proof(ctx, p, 0)
    def show_tactic(tactic: str, tactic_num: int) -> None:
        tac = et.Element('tactic')
        tac.text = tactic
        result.append(tac)
    def new_state_with_new_tactic(tactic: str) -> None:
        ctx.push_state(ctx.get_cur_state_text() + "\n" + ctx.current_indent + tactic)
    def repeat_previous_state() -> None:
        ctx.push_state(ctx.get_cur_state_text())
    def show_current_tactic_state() -> None:
        tactic_state = ctx.get_cur_tactic_state()
        state = et.Element('state', id=str(ctx.get_cur_state_num()))

        if tactic_state == 'goal complete':
            et.SubElement(state, 'complete')
        else:
            premise, conclusion = tactic_state.split('⊢')
            lines = premise.split('\n')
            conclusion = conclusion.strip().replace('\n', ' ').split()
            conclusion = ' '.join(conclusion)

            for line in lines:
                if len(line) > 0:
                    if_elem = et.SubElement(state, 'if')
                    if_elem.text = line

            then_elem = et.SubElement(state, 'then')
            then_elem.text = conclusion

        result.append(state)
        
    
    def inversion_one_step(tactic: str, tactic_num: int) -> None:
        show_tactic(tactic, tactic_num)
        new_state_with_new_tactic(tactic)
        show_current_tactic_state()
    def one_step_with_increment(tactic: str, tactic_num: int) -> None:
        show_tactic(tactic, tactic_num)
        new_state_with_new_tactic(tactic)
        ctx.increment_indent()
        show_current_tactic_state()
    def decrement_and_repeat_previous_state() -> None:
        ctx.decrement_indent()
        repeat_previous_state()
        show_current_tactic_state()

    def process_choices(choices: List[Proof]) -> bool:
        cur_state_num = ctx.get_cur_state_num()
        if cur_state_num not in ctx.state_current_choices:
            ctx.state_current_choices[cur_state_num] = 0

        cur_indent = ctx.current_indent
        for k, c in enumerate(choices):
            # result.append(f'--- CHOICE: {c} of {choices} ------------------\n')
            # if len(choices) > 1:
            #     result.append(f'choice {k+1} of {len(choices)} in state {cur_state_num}\n')  # TODO: devise choice encoding

            i = ctx.state_current_choices[cur_state_num]

            if traverse_proof(ctx, c, i):
                return True
            else:
                bt = et.Element('backtrack', to=str(cur_state_num))
                result.append(bt)

                ctx.set_cur_state_num(cur_state_num)
                ctx.current_indent = cur_indent # reset indent
                show_current_tactic_state()
                ctx.state_current_choices[cur_state_num] += 1

        # if len(choices) > 1:
        #     result.append("all choices exhausted\n")

        return False

    top_level_tactics = to_lean_tactic("", proof)
    if len(top_level_tactics) == 0:
        match proof:
            case NegAtomR(subproof):
                return recurse(subproof)
            case UpshiftR(subproof):
                return recurse(subproof)
            case PosTrueL(name, subproof):
                return recurse(subproof)
            case PosAtomL(name, subproof):
                return recurse(subproof)
            case DownshiftL(name, subproof):
                return recurse(subproof)
            case StableSeq(subproof):
                return recurse(subproof)
            case FocusR(subproof):
                return recurse(subproof)
            case FocusL(name, subproof):
                return recurse(subproof)
            case UpshiftL(name, subproof):
                return recurse(subproof)
            case DownshiftR(subproof):
                return recurse(subproof)
            case ProofFailed():
                return False
            case FocusChoice(choices):
                return process_choices(choices)
            case OrR_choice(l, r):
                return process_choices([l, r])
            case NegAndL_choice(l, r):
                return process_choices([l, r])
            case _:
                raise ValueError(f"invalid proof: {proof}")

    if len(top_level_tactics) == 1:
        inversion_one_step(top_level_tactics[0], next_tactic_num)
        match proof:
            case ImpliesR((name, subproof)):
                return recurse(subproof)
            case NegAndR(left, right):
                return (recurse(left) and recurse(right))
            case NegTrueR():
                return True
            case FalseL(name):
                return True
            case OrR_left(subproof):
                return recurse(subproof)
            case OrR_right(subproof):
                return recurse(subproof)
            case PosAndR(left, right):
                return (recurse(left) and recurse(right))
            case PosTrueR():
                return True
            case PosAtomR(name):
                return True
            case NegAndL_left(name, (left_name, left_proof)):
                return recurse(left_proof)
            case NegAndL_right(name, (right_name, right_proof)):
                return recurse(right_proof)
            case NegAtomL(name):
                return True # TODO: check
            case _:
                raise ValueError(f"invalid proof: {proof}")
    else:
        match proof:
            case OrL(name, (left_name, left_proof), (right_name, right_proof)):
                inversion_one_step(top_level_tactics[0], next_tactic_num)
                one_step_with_increment(top_level_tactics[1], 0)
                first_cond = recurse(left_proof)
                decrement_and_repeat_previous_state()
                one_step_with_increment(top_level_tactics[2], 0)
                second_cond = recurse(right_proof)
                decrement_and_repeat_previous_state()
                return (first_cond and second_cond)
            case PosAndL(name, (left_name, right_name, subproof)):
                inversion_one_step(top_level_tactics[0], next_tactic_num)
                inversion_one_step(top_level_tactics[1], 0)
                return recurse(subproof)
            case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
                one_step_with_increment(top_level_tactics[0], next_tactic_num)
                first_cond = recurse(left)
                decrement_and_repeat_previous_state()
                inversion_one_step(top_level_tactics[1], 0)
                second_cond = recurse(right_proof)
                return (first_cond and second_cond)
            case _:
                raise ValueError(f"Invalid proof: {proof}")


def start_lean(cwd=lean_repl_path):
    process = subprocess.Popen(["lake", "exec", "repl"], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               cwd=cwd)
    return process


def get_lean_tactic_state(lean_text: str) -> str:
    input_str = json.dumps({"cmd": lean_text})

    with start_lean() as lean_proc:
        lean_proc.stdin.write(input_str + '\n\n')
        lean_proc.stdin.flush()

        out = ''
        line = lean_proc.stdout.readline()
        while line != '\n':
            out = out + line
            line = lean_proc.stdout.readline()

    output_json = json.loads(out)

    if 'sorries' not in output_json or len(output_json["sorries"]) == 0:
        return 'goal complete'
    elif len(output_json["sorries"]) == 1:
        ret_text = output_json["sorries"][0]["goal"]
        assert ret_text != 'unknown goal'
        return ret_text
    else:
        raise ValueError(f"more than one goal: {output_json['sorries']}")


def to_lean_decl(n_atoms : int, prop: Proposition) -> str:
    vars_decl = f"variable ({' '.join(f'p{i+1}' for i in range(n_atoms))} : Prop)\n" 
    theorem_text = f"example : {to_lean_theorem_text(prop)} := by\n"

    return vars_decl + theorem_text


def to_lean_theorem_text(p : Proposition):
        match p:
            case Atom(name):
                return name
            case PosAtom(name):
                return name
            case NegAtom(name):
                return name
            case And(left, right):
                return f"({to_lean_theorem_text(left)} ∧ {to_lean_theorem_text(right)})"
            case PosAnd(left, right):
                return f"({to_lean_theorem_text(left)} ∧ {to_lean_theorem_text(right)})"
            case NegAnd(left, right):
                return f"({to_lean_theorem_text(left)} ∧ {to_lean_theorem_text(right)})"
            case Or(left, right):
                return f"({to_lean_theorem_text(left)} ∨ {to_lean_theorem_text(right)})"
            case Implies(left, right):
                return f"({to_lean_theorem_text(left)} → {to_lean_theorem_text(right)})"
            case PTrue():
                return "True"
            case PFalse():
                return "False"
            case Downshift(operand):
                return to_lean_theorem_text(operand)
            case Upshift(operand):
                return to_lean_theorem_text(operand)
            case _:
                raise ValueError(f"Invalid proposition: {p}")


def to_lean_tactic(indent: str, proof: Proof) -> List[str]:
    match proof:
        case ImpliesR((name, subproof)):
            return [f"{indent}intro {name}"]
        case NegAndR(left, right):
            return [f"{indent}apply And.intro"]
        case NegTrueR():
            return [f"{indent}apply True.intro"]
        case NegAtomR(subproof):
            return []
        case UpshiftR(subproof):
            return []
        case OrL(name, (left_name, left_proof), (right_name, right_proof)):
            return [f"{indent}cases {name}",
                   f"{indent}case inl {left_name} =>",
                     f"{indent}case inr {right_name} =>"]
        case FalseL(name):
            return [f"{indent}apply False.elim {name}"]
        case PosAndL(name, (left_name, right_name, subproof)):
            return [f"{indent}have {left_name} := {name}.left",
                   f"{indent}have {right_name} := {name}.right"]
        case PosTrueL(name, subproof):
            return []
        case PosAtomL(name, subproof):
            return []
        case DownshiftL(name, subproof):
            return []
        case StableSeq(subproof):
            return []
        case FocusR(subproof):
            return []
        case FocusL(name, subproof):
            return []
        case OrR_left(subproof):
            return [f"{indent}apply Or.inl"]
        case OrR_right(subproof):
            return [f"{indent}apply Or.inr"]
        case PosAndR(left, right):
            return [f"{indent}apply And.intro"]
        case PosTrueR():
            return [f"{indent}apply True.intro"]
        case PosAtomR(name):
            return [f"{indent}exact {name}"]
        case DownshiftR(subproof):
            return []
        case ImpliesL(name, prop, left, (right_name, right_proof), additional_name):
            return [f"{indent}have {additional_name} : {to_lean_theorem_text(prop)} := by",
                   f"{indent}have {right_name} := {name} {additional_name}"]
        case NegAndL_left(name, (left_name, left_proof)):
            return [f"{indent}have {left_name} := {name}.left"]
        case NegAndL_right(name, (right_name, right_proof)):
            return [f"{indent}have {right_name} := {name}.right"]
        case NegAtomL(name):
            return [f"{indent}exact {name}"]
        case UpshiftL(name, subproof):
            return []
        case ProofFailed():
            return []
        case FocusChoice(_):
            return []
        case NegAndL_choice(_, _):
            return []
        case OrR_choice(_, _):
            return []
        case _:
            raise ValueError(f"Invalid proof: {proof}")


def standardize(tactic: str) -> str:
    if tactic.startswith('intro'):
        return 'intro h'
    elif tactic == 'apply And.intro':
        return 'apply And'
    elif tactic == 'apply True.intro':
        return 'apply True'
    elif tactic.startswith('apply Or.'):
        return 'apply Or'
    elif tactic.startswith('case'):
        return 'cases Or'
    elif tactic.startswith('apply False.elim'):
        return 'efq'
    elif tactic.endswith('.left') or tactic.endswith('.right'):
        return 'split And'
    elif tactic.startswith('have'):
        return 'split Imply'
    elif tactic.startswith('exact'):
        return 'exact'
    else:
        print(f'warn: unrecognized tactic: {tactic}')
        return tactic
