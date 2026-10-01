"""Measure the unchanged historical filter assertion without fitting or scoring.

Instrument only the assertion to retain its filter name and exact tested error.
The original construction arithmetic and 2e-6 threshold remain unchanged.
"""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F


def diagnose(plan_path, device, repetitions, output):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent / 'source'
    reference = root / 'experiments/v1_f100/reference/run_round83_graft.py'
    source = reference.read_bytes()
    expected = plan['source']['experiments/v1_f100/reference/run_round83_graft.py']
    assert hashlib.sha256(source).hexdigest() == expected
    tree = ast.parse(source.decode())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_filters')
    loop = next(n for n in function.body if isinstance(n, ast.For))
    index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assert))
    check = loop.body[index]
    assert isinstance(check.test, ast.Compare) and isinstance(check.test.ops[0], ast.Lt)
    assert check.test.comparators[0].value == 2e-6
    assignment = ast.Assign(targets=[ast.Name(id='_tested_error', ctx=ast.Store())], value=check.test.left)
    recorder = ast.parse('record(name, float(_tested_error), bool(_tested_error < 2e-6))').body[0]
    check.test.left = ast.Name(id='_tested_error', ctx=ast.Load())
    loop.body[index:index + 1] = [assignment, recorder, check]
    instrumented = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    spec = plan['datasets']['Computers']
    snapshot = Path(spec['snapshot'])
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == spec['snapshot_sha256']
    data = torch.load(snapshot, map_location='cpu', weights_only=True)
    x, edges = data['x'].to(device), data['edges'].to(device)
    del data
    rows = []
    active = []
    namespace = {'torch': torch, 'F': F, 'record': lambda name, error, passed: active.append(dict(filter=name, tested_error=error, passed=passed))}
    exec(compile(instrumented, str(reference), 'exec'), namespace)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
    started = time.time()
    for repetition in range(repetitions):
        active = []
        try:
            result = namespace['make_filters'](x, edges)
            del result
            passed = True
        except AssertionError:
            passed = False
        rows.append(dict(repetition=repetition, passed=passed, filters=active))
    rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        rng_unchanged = rng_unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    report = dict(plan_sha256=plan['plan_sha256'], reference_sha256=expected,
                  snapshot_sha256=spec['snapshot_sha256'], device=device, repetitions=repetitions,
                  original_threshold=2e-6, assertion_predicate='strictly less than',
                  construction_arithmetic_unchanged=True, no_fit_or_query_scoring=True,
                  rng_unchanged=bool(rng_unchanged), started=started, finished=time.time(),
                  passed=sum(row['passed'] for row in rows), rows=rows)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.plan, args.device, args.repetitions, args.output)
