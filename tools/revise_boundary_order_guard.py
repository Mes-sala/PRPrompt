"""Seal R3 with exact R2 inputs, exact-tie diagnostics and zero reused formal units."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def revise(args):
    parent, workspace = args.parent.resolve(), args.workspace.resolve()
    assert not (workspace/'plan.json').exists() and not (workspace/'results').exists()
    previous = json.loads((parent/'plan.json').read_text())
    assert previous['plan_sha256'] == args.expected_parent == digest({k:v for k,v in previous.items() if k != 'plan_sha256'})
    for name, expected in previous['source'].items():
        assert sha(parent/'source'/name) == expected, name
    plan = json.loads(json.dumps(previous))
    del plan['plan_sha256']
    for job in plan['jobs']:
        original = Path(job['checkpoint'])
        assert sha(original) == job['checkpoint_sha256']
        target = workspace/'checkpoints'/original.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, target)
        assert sha(target) == job['checkpoint_sha256']; job['checkpoint'] = str(target)
    for spec in plan['datasets'].values():
        original = Path(spec['snapshot'])
        assert sha(original) == spec['snapshot_sha256']
        target = workspace/'dataset_snapshots'/original.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(original, target)
        assert sha(target) == spec['snapshot_sha256']; spec['snapshot'] = str(target)
    source = {}
    for dirname in ['prprompt', 'experiments/v1_f100', 'experiments/boundary_20261002']:
        for path in sorted((ROOT/dirname).rglob('*')):
            if path.is_file() and path.suffix in ['.py', '.md']:
                source[str(path.relative_to(ROOT))] = sha(path)
    for name in ['tools/metric_utils.py', 'tools/revise_boundary_filter_cache.py',
                 'tools/revise_boundary_order_guard.py', 'tools/audit_boundary_delivery.py',
                 'tools/validate_f100_order_revision.py']:
        source[name] = sha(ROOT/name)
    changed = {name for name in previous['source'] if source[name] != previous['source'][name]}
    allowed = {'experiments/boundary_20261002/'+name for name in
               ['PLAN.md','model.py','prepare.py','run.py','audit.py','summarize.py']}
    assert changed == allowed, changed
    assert all(source[name] == expected for name, expected in previous['source'].items()
               if name.startswith(('prprompt/', 'experiments/v1_f100/')))
    plan['source'] = source
    plan['prefit_order_guard'] = dict(policy='exact_maximum_set_intersection_without_tolerance',
        retain_original_argmax_failure=True, probabilities_and_training_unchanged=True)
    plan['engineering_revision'] = dict(name='R3', parent_workspace=str(parent),
        parent_plan_sha256=args.expected_parent, parent_plan_file_sha256=sha(parent/'plan.json'),
        changed_parent_source_files=sorted(changed), all_inputs_byte_identical=True,
        legacy_reference_file_bytes_and_prprompt_source_unchanged=True,
        active_optimizer_AST_change='Only pre-fit argmax diagnostic assertion; exact maximum-set intersection, no tolerance',
        formal_results_reused=0, reason='Reproducible float32 softmax exact ties; no probability, route, loss, optimizer, support, seed, selection or statistical change')
    assert plan['expected_units'] == 31500 and plan['expected_probe_units'] == 15120
    assert plan['shots'] == [1,3,5] and plan['split_seeds'] == previous['split_seeds']
    assert plan['train_seeds'] == previous['train_seeds'] and plan['config'] == previous['config']
    plan['plan_sha256'] = digest(plan)
    (workspace/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(plan_sha256=plan['plan_sha256'], source_files=len(source), formal_results_reused=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--expected-parent', required=True)
    revise(parser.parse_args())
