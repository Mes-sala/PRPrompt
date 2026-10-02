"""Read-only Linux observation required before accepting a complete boundary run.

Checks actual processes, including workers omitted from the status file. Never
starts training, copies model bytes or accepts server_complete before exit.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def processes():
    result = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = p.joinpath('cmdline').read_bytes().split(b'\0')
            args = [v.decode(errors='replace') for v in args if v]
            if not args:
                continue
            try:
                cwd = str(p.joinpath('cwd').resolve(strict=True))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                cwd = None
            result[int(p.name)] = dict(args=args, cwd=cwd)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def observe(root, expected_plan, actual):
    root = root.resolve()
    status = json.loads((root / 'pipeline_status.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    assert status['plan_sha256'] == plan['plan_sha256'] == expected_plan
    assert status['phase'] == 'server_complete', 'Server stage is not complete'
    assert status['pid'] not in actual, 'Controller is alive or its PID has been reused'
    assert not status['active'] and status['pending'] == 0 and not status.get('failed')
    family = ('experiments.f100_boundary_20261002' if
              plan['schema'] == 'f100-paired-routing-boundary-v1' else
              'experiments.boundary_20261002')
    related = []
    for pid, entry in actual.items():
        args = entry['args']
        is_family = any(family in v or family.replace('.', '/') in v for v in args)
        if not is_family:
            continue
        targets = {root, root / 'plan.json'}
        is_workspace = any(Path(v).resolve() in targets for v in args if Path(v).is_absolute())
        cwd = entry.get('cwd')
        is_workspace |= bool(cwd and Path(cwd).resolve() == root / 'source')
        if is_family and is_workspace:
            related.append(dict(pid=pid, args=args))
    assert not related, 'Actual experiment worker remains: ' + str(related)
    manifest = root / 'result_files_sha256.json'
    hashes = json.loads(manifest.read_text())
    assert isinstance(hashes, dict) and hashes, 'Missing or empty final manifest'
    for name, expected in hashes.items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert isinstance(expected, str) and len(expected) == 64
        int(expected, 16)
    extra = {}
    for name in ['pipeline.log', 'pipeline.pid']:
        path = root / name
        if path.is_file():
            extra[name] = sha(path)
    if (root / 'pipeline.pid').is_file():
        assert int((root / 'pipeline.pid').read_text().strip()) == status['pid']
    return dict(workspace=str(root), phase=status['phase'], plan_sha256=expected_plan,
        pid=status['pid'], actual_controller_alive=False, actual_worker_count=0,
        result_manifest_sha256=sha(manifest), extra_files_sha256=extra,
        actual_process_scan_performed=True, observed_at=time.time(),
        observer_source_sha256=sha(Path(__file__)),
        qualification='Fresh terminal observation only; full hashes, coverage, metrics and paired statistics still require independent delivery audit.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('workspace', type=Path)
    p.add_argument('--expected-plan', required=True)
    args = p.parse_args()
    print(json.dumps(observe(args.workspace, args.expected_plan, processes()), indent=2))
