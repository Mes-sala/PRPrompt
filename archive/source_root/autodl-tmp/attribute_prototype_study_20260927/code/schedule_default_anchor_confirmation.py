"""Recheck the prespecified four baseline readouts on the new confirmation draws."""
from pathlib import Path
import json,os,sys,time,subprocess,fcntl,shutil
import run_study as r

def main():
 root=r.ROOT;state=root/'results/default_anchor_dispatch';state.mkdir(parents=True,exist_ok=True);lock=open(state/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(state/'process.json',dict(pid=os.getpid(),started=time.time()))
 upstream=root/'results/expanded_external_dispatch'
 while not (upstream/'complete.json').exists():
  pid=json.loads((upstream/'process.json').read_text())['pid']
  try:os.kill(pid,0);assert b'schedule_expanded_external.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
  except (ProcessLookupError,FileNotFoundError,AssertionError):r.save(state/'needs_attention.json',dict(reason='External stage ended without completion',pid=pid));return
  r.save(state/'current.json',dict(phase='waiting_for_external',verified_upstream_pid=pid,time=time.time()));time.sleep(30)
 units=json.loads((root/'results/expanded_grid_dispatch/coverage_verified.json').read_text())['units'];folder=root/'design/default_anchor_confirmation';folder.mkdir(parents=True,exist_ok=True)
 for i,u in enumerate(units):
  source=root/'design/grid_units'/('backbone_value_'+u['id']+'.json');q=json.loads(source.read_text());assert len(q['config_ids'])==4;q.update(round='default_anchor_'+u['id'],seeds=list(range(271400,271420)))
  f=folder/(q['round']+'.json')
  if f.exists():assert json.loads(f.read_text())==q
  else:r.save(f,q)
  assert shutil.disk_usage(root).free>12*1024**3
  r.save(state/'current.json',dict(phase='baseline_confirmation',unit=u['id'],index=i,total=len(units),time=time.time()))
  with (state/(u['id']+'.log')).open('a') as log:code=subprocess.run([sys.executable,q['runner_script'],'--plan',str(f)],stdout=log,stderr=subprocess.STDOUT).returncode
  if code:r.save(state/'needs_attention.json',dict(reason='Baseline confirmation failed',unit=u['id']));return
 r.save(state/'complete.json',dict(units=len(units),rows=len(units)*80,finished=time.time()))
if __name__=='__main__':main()
