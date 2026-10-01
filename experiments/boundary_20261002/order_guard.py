"""Keep all F100 arithmetic; distinguish exact softmax ties from order reversal."""
import ast
import numpy as np
import torch
from experiments.v1_f100.historical import REFERENCE
from prprompt.train import tensor_hash
from prprompt.protocol import sha_file

POLICY = 'exact_maximum_set_intersection_without_tolerance'


def check_order(teacher, probability):
    assert teacher.ndim == 2 and teacher.shape == probability.shape
    assert teacher.dtype == probability.dtype
    assert torch.isfinite(teacher).all() and torch.isfinite(probability).all()
    mismatch = teacher.argmax(1) != probability.argmax(1)
    common = ((teacher == teacher.amax(1, keepdim=True)) &
              (probability == probability.amax(1, keepdim=True))).any(1)
    record = dict(policy=POLICY, original_argmax_assertion_passed=bool(~mismatch.any()),
        original_argmax_mismatch_rows=int(mismatch.sum()),
        exact_maximum_intersection_all_rows=bool(common.all()),
        genuine_order_reversal_rows=int((~common).sum()),
        accepted_exact_tie_rows=int((mismatch & common).sum()),
        base_teacher_sha256=tensor_hash(teacher), gate_probability_sha256=tensor_hash(probability))
    assert common.all(), 'F100 pre-fit probability maxima are disjoint; no retry or tolerance'
    return record


def install_order_guard(namespace):
    """Replace only optimizer_prompt's diagnostic assertion, before token allocation."""
    path = REFERENCE / 'run_round83_graft.py'
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'optimizer_prompt')
    expression = "torch.equal(gate_info['probability'].argmax(1), teacher.argmax(1))"
    positions = [i for i, n in enumerate(function.body)
                 if isinstance(n, ast.Assert) and ast.unparse(n.test) == expression]
    assert len(positions) == 1, 'Refuse an unidentified historical assertion'
    record = {}

    def checked(teacher, probability):
        record.update(check_order(teacher, probability))
        record['base_teacher'] = teacher.detach().cpu().clone()
        record['gate_probability'] = probability.detach().cpu().clone()

    replacement = ast.parse("_checked_prefit_order(teacher, gate_info['probability'])").body[0]
    ast.copy_location(replacement, function.body[positions[0]])
    function.body[positions[0]] = replacement
    namespace['_checked_prefit_order'] = checked
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), namespace)
    return record


def finalize_order_record(record, prompt):
    base = record.pop('base_teacher')
    probability = record.pop('gate_probability')
    assert torch.equal(probability, prompt['gate_probability'])
    prompt['prefit_base_teacher'] = base
    return dict(record)


def save_order_diagnostic(folder, state, audit):
    if 'prefit_order_guard' not in audit:
        return
    prompt = state['prompt']
    path = folder / 'prefit_order.npz'
    temporary = folder / 'prefit_order.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, base_teacher=prompt['prefit_base_teacher'].numpy(),
                            gate_probability=prompt['gate_probability'].numpy())
    temporary.replace(path)
    audit['prefit_order_sha256'] = sha_file(path)


def verify_order_diagnostic(folder, state, audit):
    assert sha_file(folder / 'prefit_order.npz') == audit['prefit_order_sha256']
    with np.load(folder / 'prefit_order.npz') as data:
        base = torch.from_numpy(data['base_teacher'])
        probability = torch.from_numpy(data['gate_probability'])
    assert torch.equal(base, state['prompt']['prefit_base_teacher'])
    assert torch.equal(probability, state['prompt']['gate_probability'])
    assert check_order(base, probability) == audit['prefit_order_guard']
    return audit['prefit_order_guard']
