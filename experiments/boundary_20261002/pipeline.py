"""Detached two-GPU controller with memory limits, immutable inputs and resumable units."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from prprompt.protocol import save_json, sha_file
from experiments.boundary_20261002.prepare import read_plan

ROOT = Path(__file__).resolve().parents[2]

def pipeline(workspace):
    plan = read_plan(workspace/'plan.json')
    logs = workspace/'logs'; logs.mkdir(exist_ok=True)
    lockstream = (workspace/'controller.lock').open('a'); fcntl.flock(lockstream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(pid=os.getpid(), plan_sha256=plan['plan_sha256'], started_at=time.time(), completed_tasks=[])
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    launch_id = str(int(time.time()))
    def update(**fields):
        status.update(fields, updated_at=time.time()); save_json(workspace/'pipeline_status.json', status)
    def dispatch(tasks, phase, budgets=None):
        pending = list(tasks); active = []; failures = []; update(phase=phase, pending=len(pending), active=[], failed=[])
        while pending or active:
            if not failures:
                for gpu in [0, 1]:
                    limit = 6 if budgets else 1
                    while sum(t['gpu'] == gpu for t in active) < limit:
                        used = sum(t['budget_bytes'] for t in active if t['gpu'] == gpu)
                        index = next((i for i, t in enumerate(pending) if used+(budgets[t['job']] if budgets else 0) <= 28*1024**3), None)
                        if index is None: break
                        task = pending.pop(index)
                        args = [sys.executable, '-m', task['module']]
                        if task['module'].endswith('test_operators'):
                            args += ['--device', f'cuda:{gpu}']
                        else:
                            args += ['--plan', str(workspace/'plan.json'), '--job-index', str(task['job']), '--shots', str(task['shots']), '--device', f'cuda:{gpu}']
                            if 'shard' in task: args += ['--shard', str(task['shard'])]
                            if phase.startswith('smoke'): args += ['--smoke']
                        logfile = logs/f"{launch_id}_{phase}_{task.get('job',gpu)}_{task.get('shots',0)}_{task.get('shard',0)}.log"
                        stream = logfile.open('x')
                        process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                        active.append(dict(task, gpu=gpu, pid=process.pid, process=process, stream=stream, logfile=str(logfile),
                                           budget_bytes=budgets[task['job']] if budgets else 0))
            done = []
            for task in active:
                code = task['process'].poll()
                if code is None: continue
                task['stream'].close(); done.append(task)
                record = {k: v for k, v in task.items() if k not in ['stream', 'process']}; record.update(exit_code=code, finished_at=time.time())
                if code: failures.append(record)
                else: status['completed_tasks'].append(record)
            active = [t for t in active if t not in done]
            update(active=[{k: v for k, v in t.items() if k not in ['stream', 'process']} for t in active], pending=len(pending), failed=failures)
            if failures and not active: raise RuntimeError('Stage failed; all logs and completed units retained: '+str(failures))
            if pending and not active and not failures and budgets and min(budgets[t['job']] for t in pending) > 28*1024**3:
                raise RuntimeError('A task exceeds the memory budget')
            if pending or active: time.sleep(5)
    try:
        update(phase='operator_tests')
        with (logs/(launch_id+'_operator_cpu.log')).open('x') as f:
            subprocess.run([sys.executable, '-m', 'experiments.boundary_20261002.test_operators', '--device', 'cpu'], cwd=ROOT, env=environment, stdout=f, stderr=subprocess.STDOUT, check=True)
        dispatch([dict(module='experiments.boundary_20261002.test_operators') for _ in range(2)], 'operator_gpu')
        tasks = [dict(job=i, shots=k) for i in range(len(plan['jobs'])) for k in plan['shots']]
        dispatch([dict(t, module='experiments.boundary_20261002.run', shard=0) for t in tasks], 'smoke_fit')
        dispatch([dict(t, module='experiments.boundary_20261002.probes') for t in tasks], 'smoke_probes')
        dispatch([dict(t, module='experiments.boundary_20261002.audit') for t in tasks], 'smoke_audit')
        smoke = [json.loads(p.read_text()) for p in (workspace/'smoke').glob('*/*/*/*/*/metrics.json')]
        assert len(smoke) == 525
        audits = [json.loads(p.read_text()) for p in (workspace/'smoke_analysis').glob('*_audit.json')]
        assert len(audits) == 21 and sum(a['probe_units'] for a in audits) == 252
        budgets, calibration = {}, []
        for i, j in enumerate(plan['jobs']):
            rows = [r for r in smoke if r['checkpoint_id'] == j['id']]
            peak = max(max(r['peak_cuda_bytes'], r['peak_reserved_cuda_bytes']) for r in rows)
            budget = int(1.4*peak+2*1024**3); budgets[i] = budget
            calibration.append(dict(job=i, checkpoint_id=j['id'], smoke_peak_bytes=peak, budget_bytes=budget,
                                    smoke_state_bytes=sum((p.parent/'state.pt.gz').stat().st_size for p in (workspace/'smoke'/j['id']).glob('*/*/*/*/metrics.json')),
                                    smoke_seconds=sum(r['total_seconds'] for r in rows), max_workers_per_gpu=6, gpu_budget_bytes=28*1024**3))
        save_json(workspace/'memory_calibration.json', calibration)
        save_json(workspace/'smoke_acceptance.json', dict(main_units=525, probe_units=252, engineering_audits_passed=True,
            numerical_failures=sum(a['predeclared_gate_forward_failures'] for a in audits), smoke=True, not_formal_results=True))
        tasks.sort(key=lambda t: -budgets[t['job']])
        dispatch([dict(t, module='experiments.boundary_20261002.run', shard=s) for t in tasks for s in range(3)], 'formal_fit', budgets)
        dispatch([dict(t, module='experiments.boundary_20261002.probes') for t in tasks], 'formal_probes', budgets)
        dispatch([dict(t, module='experiments.boundary_20261002.audit') for t in tasks], 'formal_audit', budgets)
        update(phase='summarize', active=[], pending=0)
        with (logs/(launch_id+'_summarize.log')).open('x') as f:
            subprocess.run([sys.executable, '-m', 'experiments.boundary_20261002.summarize', str(workspace)], cwd=ROOT, env=environment, stdout=f, stderr=subprocess.STDOUT, check=True)
        provenance = []
        for name, spec in plan['datasets'].items():
            data = torch.load(spec['snapshot'], map_location='cpu', weights_only=True)
            path = workspace/'audit_reference'/(name+'.npz'); path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, labels=data['labels'].numpy(), pool=data['pool'].numpy(), query=data['query'].numpy())
            provenance.append(dict(dataset=name, sha256=sha_file(path), exported_after_sealing=True))
        save_json(workspace/'audit_reference/provenance.json', provenance)
        with (workspace/'environment_freeze.txt').open('w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        update(phase='server_complete', active=[], pending=0, finished_at=time.time(), main_units=31500, probe_units=15120,
               next_action='Verify local delivery hashes and metrics; compare results and choose thesis proposal; update GitHub')
        files = {}
        for p in sorted(workspace.rglob('*')):
            if not p.is_file() or p.is_symlink() or p.name in ['result_files_sha256.json', 'pipeline.log', 'pipeline.pid', 'controller.lock'] or '__pycache__' in p.parts: continue
            assert p.suffix not in ['.lock', '.tmp'], str(p)
            files[str(p.relative_to(workspace))] = sha_file(p)
        save_json(workspace/'result_files_sha256.json', files)
    except Exception as error:
        update(phase='failed', error=repr(error)); raise
    finally:
        lockstream.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('workspace', type=Path); pipeline(p.parse_args().workspace)
