"""Independent immutable supplement; copy the exact inputs and frozen parent source."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from prprompt.protocol import sha_file, save_json
from experiments.f100_boundary_20261002.model import CONTROLS

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_plan(path):
    p = json.loads(Path(path).read_text())
    assert digest({k: v for k, v in p.items() if k != 'plan_sha256'}) == p['plan_sha256']
    for name, expected in p['source'].items():
        assert sha_file(ROOT / name) == expected, name
    return p


def prepare(workspace, parent):
    assert not (workspace / 'plan.json').exists(), 'Never replace a frozen plan'
    original = json.loads((parent / 'plan.json').read_text())
    assert digest({k: v for k, v in original.items() if k != 'plan_sha256'}) == original['plan_sha256']
    for name, expected in original['source'].items():
        assert sha_file(ROOT / name) == expected, name
    jobs = []
    datasets = {}
    for j in original['jobs']:
        assert j['family'] == 'Edgepred_Gprompt'
        assert sha_file(j['checkpoint']) == j['checkpoint_sha256']
        target = workspace / 'checkpoints' / Path(j['checkpoint']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(j['checkpoint'], target)
        jobs.append(dict(j, checkpoint=str(target)))
    for name, spec in original['datasets'].items():
        assert sha_file(spec['snapshot']) == spec['snapshot_sha256']
        target = workspace / 'dataset_snapshots' / Path(spec['snapshot']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec['snapshot'], target)
        datasets[name] = dict(spec, snapshot=str(target))
    source = dict(original['source'])
    for path in sorted((ROOT / 'experiments/f100_boundary_20261002').rglob('*')):
        if path.is_file() and path.suffix in ['.py', '.md']:
            source[str(path.relative_to(ROOT))] = sha_file(path)
    p = dict(schema='f100-paired-routing-boundary-v1', jobs=jobs, datasets=datasets, source=source,
        shots=original['shots'], split_seeds=original['split_seeds'], train_seeds=original['train_seeds'],
        split_shards=3, controls=CONTROLS, steps=100, smoke_steps=2,
        expected_units=3780, expected_reference_units=2520, expected_smoke=63,
        parent_workspace=str(parent), parent_plan_sha256=original['plan_sha256'],
        parent_plan_file_sha256=sha_file(parent / 'plan.json'), bootstrap_seed=20261002,
        audit_rules=original['audit_rules'],
        filter_construction_retry=dict(max_attempts=3, only_before_fit=True, original_assertion_unchanged=True),
        recipe='Exact historical optimizer and readout; only returned gate changes; one-shot inclusive training adaptation named explicitly',
        reference_policy='Copy sealed parent F100 and None-CenteredPrototype metrics, predictions, seals and selection after own predictions are sealed; references are not new fits')
    p['plan_sha256'] = digest(p)
    save_json(workspace / 'plan.json', p)
    print(json.dumps(dict(plan_sha256=p['plan_sha256'], expected_units=3780)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('workspace', type=Path)
    p.add_argument('--parent', type=Path, required=True)
    a = p.parse_args()
    prepare(a.workspace.resolve(), a.parent.resolve())
