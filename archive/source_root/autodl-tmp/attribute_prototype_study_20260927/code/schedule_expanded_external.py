"""Run equal-budget external readouts after expanded confirmation releases the GPU."""
from pathlib import Path
import os,sys,time,json,subprocess,fcntl,shutil
import run_study as r

def main():
    root=r.ROOT;state=root/'results/expanded_external_dispatch';state.mkdir(parents=True,exist_ok=True)
    lock=open(state/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r.save(state/'process.json',dict(pid=os.getpid(),started=time.time()))
    audit=json.loads((root/'provenance/expanded_external_cpu_regression.json').read_text());assert audit['cpu_regression_outputs']==8 and audit['selected_configs_and_argmax_equal']
    upstream=root/'results/expanded_confirmation_dispatch'
    while not (upstream/'complete.json').exists():
        pid=json.loads((upstream/'process.json').read_text())['pid']
        try:
            os.kill(pid,0);assert b'schedule_expanded_confirmation.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except (ProcessLookupError,FileNotFoundError,AssertionError):
            r.save(state/'needs_attention.json',dict(reason='Confirmation ended without completion',pid=pid));return
        r.save(state/'current.json',dict(phase='waiting_for_confirmation',verified_upstream_pid=pid,time=time.time()));time.sleep(30)
    if shutil.disk_usage(root).free<12*1024**3:r.save(state/'needs_attention.json',dict(reason='Low disk; sync first'));return
    plan=root/'design/expanded_external14_plan.json';q=json.loads(plan.read_text());assert q['seeds']==list(range(271600,271620)) and len(q['datasets'])==14
    r.save(state/'current.json',dict(phase='external_readouts',plan_sha256=r.sha(plan),time=time.time()))
    with (state/'run.log').open('a') as log:
        code=subprocess.run([sys.executable,str(root/'code/run_external_expanded.py'),'--plan',str(plan)],stdout=log,stderr=subprocess.STDOUT).returncode
    if code:r.save(state/'needs_attention.json',dict(reason='External readouts failed',exit_code=code));return
    done=json.loads((root/'results/expanded_external14/complete.json').read_text());assert done['completed']==2240
    r.save(state/'complete.json',dict(rows=2240,finished=time.time()))
if __name__=='__main__':main()
