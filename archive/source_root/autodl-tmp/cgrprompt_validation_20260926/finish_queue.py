from pathlib import Path
import os,time,json,subprocess,fcntl
R=Path(__file__).resolve().parent
lock=open(R/'finish.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (R/'tuning_complete.json').exists():
 s=json.loads((R/'supervisor_status.json').read_text())
 if s['status']=='failed':raise SystemExit('previous stage failed')
 stat=Path('/proc/4370/stat').read_text();assert stat.split()[2]!='Z'
 time.sleep(30)
if not (R/'cost_complete.json').exists():
 with open(R/'cost.log','a') as log:
  subprocess.run(['/root/autodl-tmp/prog_env/bin/python','-u',str(R/'run_cost.py')],cwd=R,stdout=log,stderr=log,check=True)
(R/'all_remote_stages_complete.json').write_text(json.dumps(dict(finished=time.time(),remaining='local full verification, interpretation, narrative update; further experiments if needed')))
