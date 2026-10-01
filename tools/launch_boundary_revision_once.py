"""Launch isolated R2 once after actual R1 exit; never restart an unknown failure."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def command(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return ''


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def verify_plan(workspace, expected):
    plan = json.loads((workspace / 'plan.json').read_text())
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == expected == plan['plan_sha256']
    for name, digest in plan['source'].items():
        assert sha(workspace / 'source' / name) == digest, name
    for j in plan['jobs']:
        assert sha(Path(j['checkpoint'])) == j['checkpoint_sha256']
    for spec in plan['datasets'].values():
        assert sha(Path(spec['snapshot'])) == spec['snapshot_sha256']
    return plan


def launch(args):
    workspace, parent, supplement = args.workspace.resolve(), args.parent.resolve(), args.supplement.resolve()
    folder = workspace / 'engineering_diagnostics'
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    outcome = folder / 'revision_launch_once.json'
    assert not outcome.exists() and not (workspace / 'pipeline_status.json').exists()
    save(folder / 'revision_launch_wait.json', dict(pid=os.getpid(), phase='waiting_for_actual_predecessor_exit',
         predecessor_pid=args.parent_pid, expected_plan=args.expected_plan, started=time.time()))
    while True:
        cmd = command(args.parent_pid)
        if not cmd:
            break
        assert 'experiments.boundary_20261002.pipeline' in cmd and str(parent) in cmd, 'Predecessor PID identity changed'
        time.sleep(30)
    old_status = json.loads((parent / 'pipeline_status.json').read_text())
    assert old_status['pid'] == args.parent_pid and old_status['phase'] == 'failed'
    assert not old_status.get('active'), 'Predecessor still records active workers'
    for p in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            assert not ('experiments.boundary_20261002.' in cmd and (p / 'cwd').resolve() == parent / 'source'), 'Predecessor worker exists'
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    failures = old_status.get('failed', [])
    assert len(failures) >= 2
    failure_evidence = []
    for failed in failures:
        assert failed['job'] == 4 and failed['module'] == 'experiments.boundary_20261002.run' and failed['exit_code'] == 1
        log = Path(failed['logfile'])
        text = log.read_text()
        assert text.rstrip().endswith('AssertionError') and 'in make_filters' in text and 'run_round83_graft.py' in text
        assert "torch.sparse.mm(op, torch.ones(n, 1, device=x.device))" in text
        failure_evidence.append(dict(failed, log_sha256=sha(log)))
    plan = verify_plan(workspace, args.expected_plan)
    previous = verify_plan(parent, plan['engineering_revision']['parent_plan_sha256'])
    auxiliary = verify_plan(supplement, args.expected_supplement_plan)
    assert auxiliary['parent_workspace'] == str(workspace) and auxiliary['parent_plan_sha256'] == args.expected_plan
    assert plan['engineering_revision']['formal_results_reused'] == 0
    assert not (workspace / 'results').exists() and not (supplement / 'results').exists()
    assert all(plan['datasets'][n]['snapshot_sha256'] == v['snapshot_sha256'] for n, v in previous['datasets'].items())
    assert [j['checkpoint_sha256'] for j in plan['jobs']] == [j['checkpoint_sha256'] for j in previous['jobs']]
    save(folder / 'predecessor_terminal_status.json', dict(status=old_status, failure_evidence=failure_evidence,
         actual_exit_and_worker_absence_verified=True, time=time.time()))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    save(outcome, dict(phase='launching_isolated_revision_once', plan_sha256=args.expected_plan,
         predecessor_pid=args.parent_pid, launcher_sha256=sha(Path(__file__)), time=time.time()))
    with (workspace / 'pipeline.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.boundary_20261002.pipeline', str(workspace)],
            cwd=workspace / 'source', env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (workspace / 'pipeline.pid').write_text(str(process.pid))
    save(outcome, dict(phase='isolated_main_launched_once', main_pid=process.pid, plan_sha256=args.expected_plan,
         all_predecessor_units_excluded=True, launcher_sha256=sha(Path(__file__)), time=time.time()))
    while not (workspace / 'pipeline_status.json').exists():
        assert command(process.pid), 'R2 exited before publishing status; do not relaunch'
        time.sleep(2)
    status = json.loads((workspace / 'pipeline_status.json').read_text())
    assert status['pid'] == process.pid and status['plan_sha256'] == args.expected_plan
    assert not (supplement / 'pipeline_status.json').exists()
    with (supplement / 'pipeline.log').open('x') as stream:
        aux = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.f100_boundary_20261002.pipeline', str(supplement)],
            cwd=supplement / 'source', env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (supplement / 'pipeline.pid').write_text(str(aux.pid))
    save(outcome, dict(phase='isolated_main_and_waiting_supplement_launched_once', main_pid=process.pid,
         supplement_pid=aux.pid, plan_sha256=args.expected_plan, supplement_plan_sha256=args.expected_supplement_plan,
         all_predecessor_units_excluded=True, launcher_sha256=sha(Path(__file__)), time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--expected-plan', required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--expected-supplement-plan', required=True)
    launch(parser.parse_args())
