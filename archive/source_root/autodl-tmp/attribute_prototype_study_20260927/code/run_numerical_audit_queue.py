"""Serial numerical sensitivity audit; preserves sealed labels and reports differences."""
from pathlib import Path
import json,sys,os,time,subprocess,fcntl,shutil
import run_study as r

def main():
 root=r.ROOT;folder=root/'design/numerical_probability_audit';manifest=json.loads((folder/'manifest.json').read_text());state=root/'results/numerical_probability_dispatch';state.mkdir(parents=True,exist_ok=True);lock=open(state/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(state/'process.json',dict(pid=os.getpid(),started=time.time(),manifest_sha256=r.sha(folder/'manifest.json')))
 for i,item in enumerate(manifest['plans']):
  assert shutil.disk_usage(root).free>8*1024**3
  r.save(state/'current.json',dict(index=i,total=len(manifest['plans']),unit=item['unit'],source=item['source'],time=time.time()))
  with (state/(item['plan']+'.log')).open('a') as log:code=subprocess.run([sys.executable,str(root/'code/replay_numerical_audit.py'),'--plan',str(folder/item['plan'])],stdout=log,stderr=subprocess.STDOUT).returncode
  if code:r.save(state/'needs_attention.json',dict(reason='Numerical audit execution failed; inspect before continuing',plan=item['plan'],exit_code=code));return
 r.save(state/'complete.json',dict(plans=len(manifest['plans']),predictions=manifest['unique_predictions'],selection_rows=manifest['selection_rows'],finished=time.time()))
if __name__=='__main__':main()
