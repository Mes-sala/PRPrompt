"""Continue the frozen broad grid with route-stratified new-support confirmation."""
from pathlib import Path
import os,sys,time,json,subprocess,fcntl,shutil
import run_study as r
ROOT=r.ROOT

def main():
    out=ROOT/'results/expanded_confirmation_dispatch';out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()))
    upstream=ROOT/'results/expanded_grid_dispatch'
    while not (upstream/'complete.json').exists():
        meta=json.loads((upstream/'process.json').read_text());pid=meta['pid']
        try:
            os.kill(pid,0)
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            assert b'schedule_expanded_grid_v6.py' in cmd
        except (ProcessLookupError,FileNotFoundError,AssertionError):
            r.save(out/'needs_attention.json',dict(reason='Broad grid process ended without completion; inspect upstream before resuming',pid=pid));return
        r.save(out/'current.json',dict(phase='waiting_for_broad_grid',verified_upstream_pid=pid,time=time.time()))
        time.sleep(30)
    assert json.loads((upstream/'complete.json').read_text())['units']==157
    for script in ['analyze_expanded_grid.py','prepare_expanded_confirmation.py']:
        with (out/(script+'.log')).open('a') as f:
            code=subprocess.run([sys.executable,str(ROOT/'code'/script),str(ROOT)],stdout=f,stderr=subprocess.STDOUT).returncode
        if code:r.save(out/'needs_attention.json',dict(reason='Preparation failed',script=script,exit_code=code));return
    folder=ROOT/'design/expanded_confirmation';manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['selection_uses_query_labels'] is False and manifest['validation_folds']==5
    assert manifest['seeds']==list(range(271400,271420))
    for i,item in enumerate(manifest['plans']):
        if shutil.disk_usage(ROOT).free<12*1024**3:
            r.save(out/'needs_attention.json',dict(reason='Disk below12GiB; sync before continuing'));return
        plan=folder/item['plan'];q=json.loads(plan.read_text());runner=Path(q['runner_script']);assert r.sha(runner)==q['runner_sha256']
        r.save(out/'current.json',dict(phase='confirmation',index=i,total=len(manifest['plans']),unit=item['unit'],time=time.time()))
        with (out/(item['unit']+'.log')).open('a') as f:
            code=subprocess.run([sys.executable,str(runner),'--plan',str(plan)],stdout=f,stderr=subprocess.STDOUT).returncode
        if code:r.save(out/'needs_attention.json',dict(reason='Confirmation failed',unit=item['unit'],exit_code=code));return
    r.save(out/'complete.json',dict(units=len(manifest['plans']),rows=manifest['rows'],manifest_sha256=r.sha(folder/'manifest.json'),finished=time.time()))
if __name__=='__main__':main()
