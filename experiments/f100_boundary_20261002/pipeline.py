"""Wait for verified parent completion, then run all controls without overlapping GPUs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from prprompt.protocol import save_json, sha_file
from experiments.f100_boundary_20261002.prepare import read_plan

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'experiments.f100_boundary_20261002.'


def command(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, ProcessLookupError):
        return ''


def pipeline(workspace):
    plan = read_plan(workspace / 'plan.json')
    logs = workspace / 'logs'
    logs.mkdir(exist_ok=True)
    lock = (workspace / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(pid=os.getpid(), plan_sha256=plan['plan_sha256'], started=time.time(), completed_tasks=[])
    launch = str(int(time.time()))
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def update(**values):
        status.update(values, updated_at=time.time())
        save_json(workspace / 'pipeline_status.json', status)

    def wait_parent():
        parent = Path(plan['parent_workspace'])
        while True:
            parent_status = json.loads((parent / 'pipeline_status.json').read_text())
            assert parent_status['plan_sha256'] == plan['parent_plan_sha256']
            cmd = command(parent_status['pid'])
            if cmd:
                assert 'experiments.boundary_20261002.pipeline' in cmd and str(parent) in cmd, 'Parent PID identity changed'
            update(phase='waiting_for_parent', parent_phase=parent_status['phase'], parent_pid=parent_status['pid'],
                   parent_process_alive=bool(cmd), active=[], pending=0)
            if not cmd and parent_status['phase'] == 'server_complete' and (parent / 'result_files_sha256.json').exists():
                assert not parent_status['active'] and parent_status['pending'] == 0
                audit = json.loads((parent / 'analysis/scientific_audit.json').read_text())
                assert audit['main_present'] == 31500 and audit['probe_present'] == 15120
                assert audit['saved_readout_exact'] and audit['nested_shots_passed']
                assert sha_file(parent / 'plan.json') == plan['parent_plan_file_sha256']
                save_json(workspace / 'parent_completion.json', dict(parent_status=parent_status, scientific_audit=audit,
                    result_manifest_sha256=sha_file(parent / 'result_files_sha256.json'), actual_exit_verified=True,
                    time=time.time(), numerical_qualification_preserved=True))
                return
            # A failed or unobservable parent never authorizes starting GPU work or restarting the parent.
            time.sleep(30)

    def dispatch(tasks, phase, budgets=None):
        pending, active, failures = list(tasks), [], []
        update(phase=phase, pending=len(pending), active=[], failed=[])
        while pending or active:
            if not failures:
                for gpu in [0, 1]:
                    limit = 4 if budgets else 1
                    while sum(t['gpu'] == gpu for t in active) < limit:
                        used = sum(t['budget_bytes'] for t in active if t['gpu'] == gpu)
                        index = next((i for i, t in enumerate(pending) if used+(budgets[t['job']] if budgets else 0) <= 28*1024**3), None)
                        if index is None: break
                        task = pending.pop(index)
                        args = [sys.executable, '-m', task['module'], '--device', f'cuda:{gpu}']
                        if task['module'] != MODULE+'model':
                            args += ['--plan', str(workspace / 'plan.json'), '--job-index', str(task['job']), '--shots', str(task['shots'])]
                            if 'shard' in task: args += ['--shard', str(task['shard'])]
                            if phase.startswith('smoke'): args += ['--smoke']
                        path = logs / f"{launch}_{phase}_{task.get('job',gpu)}_{task.get('shots',0)}_{task.get('shard',0)}.log"
                        stream = path.open('x')
                        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                        active.append(dict(task, gpu=gpu, pid=process.pid, process=process, stream=stream,
                            logfile=str(path), budget_bytes=budgets[task['job']] if budgets else 0))
            remaining = []
            for t in active:
                code = t['process'].poll()
                if code is None:
                    remaining.append(t)
                    continue
                t['stream'].close()
                record = {k: v for k, v in t.items() if k not in ['process', 'stream']}
                record.update(exit_code=code, finished=time.time())
                (failures if code else status['completed_tasks']).append(record)
            active = remaining
            update(active=[{k: v for k, v in t.items() if k not in ['process', 'stream']} for t in active], pending=len(pending), failed=failures)
            if failures and not active: raise RuntimeError('Failed stage; logs and sealed units retained: '+str(failures))
            if pending and not active and budgets and min(budgets[t['job']] for t in pending) > 28*1024**3:
                raise RuntimeError('Task exceeds memory budget')
            if pending or active: time.sleep(5)

    try:
        wait_parent()
        update(phase='operator_cpu')
        with (logs / (launch+'_operator_cpu.log')).open('x') as stream:
            subprocess.run([sys.executable, '-m', MODULE+'model', '--device', 'cpu'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        dispatch([dict(module=MODULE+'model') for _ in range(2)], 'operator_gpu')
        tasks = [dict(job=i, shots=k) for i in range(len(plan['jobs'])) for k in plan['shots']]
        dispatch([dict(t, module=MODULE+'run', shard=0) for t in tasks], 'smoke_fit')
        dispatch([dict(t, module=MODULE+'audit') for t in tasks], 'smoke_audit')
        smoke = [json.loads(p.read_text()) for p in (workspace / 'smoke').glob('*/*/*/*/*/metrics.json')]
        audits = [json.loads(p.read_text()) for p in (workspace / 'smoke_analysis').glob('*_audit.json')]
        assert len(smoke) == 63 and len(audits) == 21 and sum(a['reference_units'] for a in audits) == 42
        budgets, calibration = {}, []
        for i, job in enumerate(plan['jobs']):
            rows = [r for r in smoke if r['checkpoint_id'] == job['id']]
            peak = max(max(r['peak_cuda_bytes'], r['peak_reserved_cuda_bytes']) for r in rows)
            budgets[i] = int(1.4*peak+2*1024**3)
            calibration.append(dict(job=i, smoke_peak_bytes=peak, budget_bytes=budgets[i], max_workers_per_gpu=4,
                smoke_state_bytes=sum((p.parent / 'state.pt.gz').stat().st_size for p in (workspace / 'smoke' / job['id']).glob('*/*/*/*/metrics.json'))))
        save_json(workspace / 'memory_calibration.json', calibration)
        save_json(workspace / 'smoke_acceptance.json', dict(control_units=63, reference_units=42,
            engineering_audits_passed=True, numerical_failures=sum(a['predeclared_gate_forward_failures'] for a in audits),
            smoke=True, not_formal_results=True))
        tasks.sort(key=lambda t: -budgets[t['job']])
        dispatch([dict(t, module=MODULE+'run', shard=s) for t in tasks for s in range(3)], 'formal_fit', budgets)
        dispatch([dict(t, module=MODULE+'audit') for t in tasks], 'formal_audit', budgets)
        update(phase='summarize', active=[], pending=0)
        with (logs / (launch+'_summary.log')).open('x') as stream:
            subprocess.run([sys.executable, '-m', MODULE+'summarize', str(workspace)], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        for name, spec in plan['datasets'].items():
            d = torch.load(spec['snapshot'], weights_only=True, map_location='cpu')
            output = workspace / 'audit_reference' / (name+'.npz')
            output.parent.mkdir(exist_ok=True)
            np.savez_compressed(output, labels=d['labels'].numpy(), pool=d['pool'].numpy(), query=d['query'].numpy())
        with (workspace / 'environment_freeze.txt').open('w') as stream:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
        update(phase='server_complete', active=[], pending=0, controls=3780, references=2520, finished=time.time())
        files = {}
        for p in sorted(workspace.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.name in ['result_files_sha256.json', 'pipeline.log', 'controller.lock', 'pipeline.pid'] or '__pycache__' in p.parts:
                continue
            assert p.suffix not in ['.lock', '.tmp'], str(p)
            files[str(p.relative_to(workspace))] = sha_file(p)
        save_json(workspace / 'result_files_sha256.json', files)
    except Exception as error:
        update(phase='failed', error=repr(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('workspace', type=Path)
    pipeline(p.parse_args().workspace.resolve())
