"""Wait for actual R2 exit, retain failure evidence, then launch isolated R3 once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def command(pid):
    try: return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError,ProcessLookupError): return ''


def workers(root):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            cmd = command(int(p.name))
            if (p/'cwd').resolve() == root/'source' and '-m experiments.' in cmd and '.pipeline ' not in cmd:
                found.append(dict(pid=int(p.name),command=cmd))
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    return found


def verify(root, expected):
    p = json.loads((root/'plan.json').read_text())
    digest = hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='plan_sha256'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert p['plan_sha256'] == expected == digest
    for name, h in p['source'].items(): assert sha(root/'source'/name)==h,name
    for spec in p['datasets'].values(): assert sha(spec['snapshot'])==spec['snapshot_sha256']
    for job in p['jobs']: assert sha(job['checkpoint'])==job['checkpoint_sha256']
    return p


def retain(root, status):
    target = root/'retained_failed_revision_manifest.json'
    assert not target.exists()
    files = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.name in ['pipeline.pid','controller.lock']:
            continue
        assert p.suffix not in ['.tmp','.lock'],str(p)
        before = p.stat(); digest=sha(p); after=p.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
        files[str(p.relative_to(root))]=digest
    record = dict(phase='failed_revision_retained_not_formal_completion', status=status,
        actual_controller_exit_and_worker_absence_verified=True,
        formal_present=len(list((root/'results').glob('*/*/*/*/*/metrics.json'))),
        formal_completion_claimed=False, formal_results_reused_by_R3=0, files=files, time=time.time())
    save(target,record)
    return dict(manifest_sha256=sha(target),file_count=len(files),formal_present=record['formal_present'])


def launch(args):
    root,aux,parent,old_aux=(p.resolve() for p in [args.workspace,args.supplement,args.parent,args.old_supplement])
    folder=root/'engineering_diagnostics';folder.mkdir(exist_ok=True)
    outcome=folder/'revision_launch_once.json'
    assert not outcome.exists(),'Never launch a duplicate revision'
    outcome.open('x').close()
    save(outcome,dict(phase='waiting_for_actual_predecessor_exit',pid=os.getpid(),parent_pid=args.parent_pid,time=time.time()))
    while command(args.parent_pid):
        assert 'experiments.boundary_20261002.pipeline' in command(args.parent_pid) and str(parent) in command(args.parent_pid)
        print(json.dumps(dict(event='waiting_for_R2_controller_exit',pid=args.parent_pid,time=time.time())),flush=True)
        time.sleep(55)
    assert not workers(parent)
    previous=verify(parent,args.expected_parent)
    old_status=json.loads((parent/'pipeline_status.json').read_text())
    assert old_status['phase']=='failed' and not old_status['active']
    failures=old_status['failed'];assert failures
    failure=next(f for f in failures if f['job']==6 and f['shots']==1 and f['shard']==0)
    assert failure['exit_code']==1
    log=Path(failure['logfile']);text=log.read_text()
    assert 'in optimizer_prompt' in text and "gate_info['probability'].argmax(1),teacher.argmax(1)" in text
    assert sha(log)==args.failure_log_sha256
    prefix=json.loads((parent/'engineering_diagnostics/order_prefix_grid/summary.json').read_text())
    assert prefix['no_fitting_or_query_scoring'] and prefix['checks']==420 and prefix['train_seed']==0
    assert all(r['maximum_intersection_failures']==0 for r in prefix['rows'])
    plan=verify(root,args.expected_plan);supplement=verify(aux,args.expected_supplement)
    assert plan['engineering_revision']['name']=='R3' and plan['engineering_revision']['formal_results_reused']==0
    assert plan['engineering_revision']['parent_plan_sha256']==args.expected_parent
    assert supplement['parent_workspace']==str(root) and supplement['parent_plan_sha256']==args.expected_plan
    assert not (root/'results').exists() and not (aux/'results').exists()
    assert all(plan['datasets'][name]['snapshot_sha256']==spec['snapshot_sha256'] for name,spec in previous['datasets'].items())
    assert [j['checkpoint_sha256'] for j in plan['jobs']]==[j['checkpoint_sha256'] for j in previous['jobs']]
    for device in ['cuda0','cuda1']:
        validation=json.loads((folder/f'failed_cell_{device}/validation.json').read_text())
        assert validation['no_query_scoring'] and validation['steps']==100 and validation['saved_readout_exact']
        assert validation['audit']['prefit_order_guard']['exact_maximum_intersection_all_rows']
        for name,h in validation['source_sha256'].items(): assert plan['source'][name]==h,name
    save(outcome,dict(phase='retaining_R2_failure_before_R3_launch',pid=os.getpid(),
        actual_predecessor_exit_and_worker_absence_verified=True,time=time.time()))
    retained=retain(parent,old_status)
    cmd=command(args.old_supplement_pid)
    if cmd:
        assert 'experiments.f100_boundary_20261002.pipeline' in cmd and str(old_aux) in cmd
        old_aux_status=json.loads((old_aux/'pipeline_status.json').read_text())
        assert old_aux_status['phase']=='waiting_for_parent' and not old_aux_status['active'] and not workers(old_aux)
        os.kill(args.old_supplement_pid,signal.SIGTERM)
        while command(args.old_supplement_pid): time.sleep(1)
        old_aux_status.update(phase='R2_wait_cancelled_for_R3',actual_wait_controller_exit_verified=True,time=time.time())
        save(old_aux/'pipeline_status.json',old_aux_status)
        retain(old_aux,old_aux_status)
    save(folder/'predecessor_terminal_status.json',dict(status=old_status,retention=retained,
        failure_log_sha256=args.failure_log_sha256,formal_results_reused=0,time=time.time()))
    for workspace,p in [(root,plan),(aux,supplement)]:
        save(workspace/'execution_provenance.json',dict(source_commit=args.source_commit,plan_sha256=p['plan_sha256'],
            frozen_source_files=p['source'],all_frozen_source_hashes_verified=True,all_inputs_verified=True,
            R2_formal_units_reused=0,launcher_sha256=sha(__file__),time=time.time()))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    save(outcome,dict(phase='launching_R3_once',retained_predecessor=retained,time=time.time()))
    with (root/'pipeline.log').open('x') as stream:
        process=subprocess.Popen([sys.executable,'-u','-m','experiments.boundary_20261002.pipeline',str(root)],
            cwd=root/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'pipeline.pid').write_text(str(process.pid))
    save(outcome,dict(phase='R3_main_launched_once',main_pid=process.pid,time=time.time()))
    while not (root/'pipeline_status.json').exists():
        assert command(process.pid),'R3 exited before status; do not relaunch'
        time.sleep(2)
    with (aux/'pipeline.log').open('x') as stream:
        waiting=subprocess.Popen([sys.executable,'-u','-m','experiments.f100_boundary_20261002.pipeline',str(aux)],
            cwd=aux/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (aux/'pipeline.pid').write_text(str(waiting.pid))
    save(outcome,dict(phase='R3_main_and_waiting_supplement_launched_once',main_pid=process.pid,
        supplement_pid=waiting.pid,plan_sha256=args.expected_plan,supplement_plan_sha256=args.expected_supplement,
        R2_formal_units_reused=0,retained_predecessor=retained,launcher_sha256=sha(__file__),time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['workspace','supplement','parent','old-supplement']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['expected-parent','expected-plan','expected-supplement','failure-log-sha256','source-commit']:p.add_argument('--'+name,required=True)
    p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--old-supplement-pid',type=int,required=True)
    launch(p.parse_args())
