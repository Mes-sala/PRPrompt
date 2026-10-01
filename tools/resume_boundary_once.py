"""One bounded retry of the frozen controller after the observed pre-fit filter assertion.

Run on the experiment server. Never kill a process, alter the plan or change numerical
thresholds. A timeout, stale status or unidentified failure cannot trigger a retry.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def process_command(pid):
    p = Path('/proc') / str(pid) / 'cmdline'
    try:
        return p.read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def workspace_workers(workspace):
    source = workspace / 'source'
    workers = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'experiments.boundary_20261002.' in cmd and str((p / 'cwd').resolve()) == str(source):
                workers.append({'pid': int(p.name), 'command': cmd})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return workers


def retry(workspace, parent_pid):
    folder = workspace / 'engineering_diagnostics'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    outcome = folder / 'resume_once.json'
    assert not outcome.exists(), 'A bounded retry has already been resolved'
    pending = folder / 'resume_wait.json'
    save(pending, dict(pid=os.getpid(), parent_pid=parent_pid, phase='waiting_for_actual_exit', started=time.time()))
    while True:
        command = process_command(parent_pid)
        if not command:
            break
        assert 'experiments.boundary_20261002.pipeline' in command, 'PID was reused'
        time.sleep(30)
    status = json.loads((workspace / 'pipeline_status.json').read_text())
    assert status['pid'] == parent_pid, 'Controller identity changed; do not duplicate it'
    assert not workspace_workers(workspace), 'Workers still exist; do not restart'
    if status['phase'] == 'server_complete':
        save(outcome, dict(phase='not_needed', status=status, time=time.time()))
        return
    assert status['phase'] == 'failed', 'No confirmed terminal failure'
    failed = status.get('failed', [])
    assert len(failed) == 1, 'Only the single observed filter assertion is eligible'
    failure = failed[0]
    assert failure['job'] == 4 and failure['shots'] == 1 and failure['shard'] == 0
    assert failure['module'] == 'experiments.boundary_20261002.run' and failure['exit_code'] == 1
    log = Path(failure['logfile']).read_text()
    assert log.rstrip().endswith('AssertionError')
    assert 'in make_filters' in log and 'run_round83_graft.py' in log
    assert "torch.sparse.mm(op, torch.ones(n, 1, device=x.device))" in log
    plan = json.loads((workspace / 'plan.json').read_text())
    assert plan['plan_sha256'] == status['plan_sha256']
    body = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == plan['plan_sha256']
    for name, expected in plan['source'].items():
        assert hashlib.sha256((workspace / 'source' / name).read_bytes()).hexdigest() == expected, name
    # Keep the complete terminal snapshot before a new controller writes status.
    save(folder / 'failed_controller_before_same_source_retry.json', status)
    save(outcome, dict(phase='launching_once', parent_pid=parent_pid, plan_sha256=plan['plan_sha256'],
                      unchanged_source_verified=True, original_failure=failure, numerical_thresholds_unchanged=True,
                      existing_units_retained=True, reason='Observed assertion before fitting; exact same source retry', time=time.time()))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (folder / 'same_source_retry_pipeline.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'experiments.boundary_20261002.pipeline', str(workspace)],
                                   cwd=workspace / 'source', env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    save(workspace / 'engineering_diagnostics/retry_controller_pid.json', dict(pid=process.pid, time=time.time()))
    save(outcome, dict(phase='launched_once', parent_pid=parent_pid, retry_pid=process.pid,
                      plan_sha256=plan['plan_sha256'], unchanged_source_verified=True, original_failure=failure,
                      numerical_thresholds_unchanged=True, existing_units_retained=True, time=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    retry(args.workspace.resolve(), args.parent_pid)
