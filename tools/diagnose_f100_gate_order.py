"""Inspect the original pre-fit F100 gate assertion; never fit or score queries."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import torch
from experiments.v1_f100.historical import load
from experiments.boundary_20261002.filters import construct_filters_once
from prprompt.model import normalized_adjacency
from prprompt.protocol import sample_support


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent / 'source'
    for name, expected in plan['source'].items():
        assert sha(root / name) == expected, name
    spec = plan['datasets'][args.dataset]
    assert sha(spec['snapshot']) == spec['snapshot_sha256']
    data = torch.load(spec['snapshot'], map_location='cpu', weights_only=True)
    x, edges = data['x'].to(args.device), data['edges'].to(args.device)
    adjacency = normalized_adjacency(edges, len(x), x.dtype)
    representation = torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x)).detach()
    old = load()
    filters, filter_audit = construct_filters_once(x, edges, old['make_filters'])
    reference = root / 'experiments/v1_f100/reference/run_round83_graft.py'
    tree = ast.parse(reference.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'optimizer_prompt')
    expression = "torch.equal(gate_info['probability'].argmax(1), teacher.argmax(1))"
    indexes = [i for i, n in enumerate(function.body) if isinstance(n, ast.Assert) and ast.unparse(n.test) == expression]
    assert len(indexes) == 1
    index = indexes[0]
    original_line = function.body[index].lineno
    capture = ast.parse('_capture(teacher, raw_scores, gate_info)').body[0]
    # Keep the exact original assertion, then return before allocating tokens or fitting.
    function.body = function.body[:index] + [capture, function.body[index], ast.Return(value=ast.Constant(None))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    original_rectified = old['rectified_teacher']
    teacher_calls, rows, active = [], [], {}

    def rectified(*values):
        result = original_rectified(*values)
        teacher_calls.append(result[0].detach().clone())
        return result

    def capture(teacher, raw, info):
        probability = info['probability']
        mismatched = (teacher.argmax(1) != probability.argmax(1)).nonzero().flatten()
        first = teacher_calls[0]
        second = teacher_calls[-1]
        from_same_scores = raw.softmax(-1)
        common = ((teacher == teacher.amax(1, keepdim=True)) &
                  (probability == probability.amax(1, keepdim=True))).any(1)
        active.update(mismatch_rows=len(mismatched),
            all_rows_exact_maximum_intersection=bool(common.all()),
            mismatch_rows_without_exact_maximum_intersection=int((~common[mismatched]).sum()),
            repeated_teacher_scores_exact=bool(torch.equal(first, second)),
            repeated_teacher_scores_max_error=float((first-second).abs().max()),
            repeated_teacher_score_argmax_changes=int((first.argmax(1) != second.argmax(1)).sum()),
            temperature_only_argmax_changes=int((from_same_scores.argmax(1) != probability.argmax(1)).sum()),
            finite=bool(torch.isfinite(teacher).all() and torch.isfinite(raw).all() and torch.isfinite(probability).all()),
            examples=[dict(node=int(node), teacher_scores=first[node].tolist(), gate_scores=raw[node].tolist(),
                teacher_probability=teacher[node].tolist(), gate_probability=probability[node].tolist(),
                same_scores_base_probability=from_same_scores[node].tolist(),
                teacher_argmax=int(teacher[node].argmax()), gate_argmax=int(probability[node].argmax()))
                for node in mismatched[:8]])

    old.update(rectified_teacher=rectified, _capture=capture)
    exec(compile(module, str(reference), 'exec'), old)
    started = time.time()
    for split in args.splits:
        for shots in args.shots:
            support = sample_support(data['pool'], data['labels'][data['pool']], spec['classes'], shots, split)
            sy = data['labels'][support]
            for repetition in range(args.repetitions):
                teacher_calls, active = [], dict(split_seed=split, shots=shots, repetition=repetition)
                try:
                    old['optimizer_prompt'](None, representation, adjacency, None, edges, support.to(args.device),
                        sy.to(args.device), spec['classes'], 0, 'token_ridge_kd10plain', 100,
                        filters, 'none', .1, 'raw', 'adam', .01)
                    active['original_assertion_passed'] = True
                except AssertionError:
                    assert 'mismatch_rows' in active, 'An earlier assertion failed; do not reinterpret it'
                    active['original_assertion_passed'] = False
                rows.append(active)
    report = dict(plan_sha256=plan['plan_sha256'], source_sha256=sha(reference), assertion_line=original_line,
        diagnostic_tool_sha256=sha(__file__), dataset=args.dataset, snapshot_sha256=spec['snapshot_sha256'],
        graph_sha256=spec['graph_sha256'], device=args.device, torch=str(torch.__version__),
        source_arithmetic_and_original_assertion_unchanged=True, stopped_before_token_allocation=True,
        no_fit_or_query_scoring=True, original_failed_forward_not_bitwise_reconstructed=True,
        train_seed=0, filter_audit=filter_audit, started=started, finished=time.time(),
        original_assertion_failures=sum(not r['original_assertion_passed'] for r in rows), rows=rows)
    assert not args.output.exists(), 'Never replace a diagnostic'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['rows', 'filter_audit']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--dataset', default='Chameleon')
    parser.add_argument('--splits', type=int, nargs='+', default=[30093009])
    parser.add_argument('--shots', type=int, nargs='+', default=[1])
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    diagnose(parser.parse_args())
