"""Seal an isolated R2 plan using exact R1 inputs and no reused formal units."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def revise(parent, workspace, expected_parent):
    assert not (workspace / 'plan.json').exists(), 'Never replace a sealed plan'
    assert not (workspace / 'results').exists(), 'R2 formal units must start from zero'
    old = json.loads((parent / 'plan.json').read_text())
    assert old['plan_sha256'] == expected_parent == digest({k: v for k, v in old.items() if k != 'plan_sha256'})
    for name, expected in old['source'].items():
        assert sha(parent / 'source' / name) == expected, name
    plan = json.loads(json.dumps(old))
    del plan['plan_sha256']
    for job in plan['jobs']:
        original = Path(job['checkpoint'])
        assert sha(original) == job['checkpoint_sha256']
        target = workspace / 'checkpoints' / original.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(original, target)
        assert sha(target) == job['checkpoint_sha256']
        job['checkpoint'] = str(target)
    for name, spec in plan['datasets'].items():
        original = Path(spec['snapshot'])
        assert sha(original) == spec['snapshot_sha256']
        target = workspace / 'dataset_snapshots' / original.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(original, target)
        assert sha(target) == spec['snapshot_sha256']
        spec['snapshot'] = str(target)
    source = {}
    for dirname in ['prprompt', 'experiments/v1_f100', 'experiments/boundary_20261002']:
        for path in sorted((ROOT / dirname).rglob('*')):
            if path.is_file() and path.suffix in ['.py', '.md']:
                source[str(path.relative_to(ROOT))] = sha(path)
    for name in ['tools/metric_utils.py', 'tools/revise_boundary_filter_cache.py']:
        source[name] = sha(ROOT / name)
    changed = [name for name in old['source'] if source[name] != old['source'][name]]
    assert set(changed) == {'experiments/boundary_20261002/PLAN.md',
                            'experiments/boundary_20261002/model.py',
                            'experiments/boundary_20261002/prepare.py'}
    assert all(source[name] == expected for name, expected in old['source'].items()
               if name.startswith(('prprompt/', 'experiments/v1_f100/')))
    plan['source'] = source
    plan['filter_construction_retry'] = dict(max_attempts=3, only_before_fit=True,
        original_assertion_unchanged=True, cache='exact immutable input tensor objects per worker')
    plan['engineering_revision'] = dict(name='R2', parent_workspace=str(parent),
        parent_plan_sha256=expected_parent, parent_plan_file_sha256=sha(parent / 'plan.json'),
        changed_parent_source_files=changed, all_inputs_byte_identical=True,
        original_legacy_AST_and_prprompt_source_unchanged=True,
        formal_results_reused=0, reason='Two historical pre-fit filter assertions; bounded construction and per-worker immutable graph cache; no query-driven changes')
    assert plan['expected_units'] == 31500 and plan['expected_probe_units'] == 15120
    plan['plan_sha256'] = digest(plan)
    (workspace / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(plan_sha256=plan['plan_sha256'], formal_results_reused=0,
                         source_files=len(source), changed_parent_source_files=changed)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--expected-parent-plan', required=True)
    args = parser.parse_args()
    revise(args.parent.resolve(), args.workspace.resolve(), args.expected_parent_plan)
