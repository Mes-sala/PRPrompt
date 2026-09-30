from pathlib import Path
import time,json,subprocess,fcntl
R=Path(__file__).resolve().parent
lock=open(R/'readout_queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (R/'all_remote_stages_complete.json').exists():
 s=json.loads((R/'supervisor_status.json').read_text())
 if s['status']=='failed':raise SystemExit('previous stage failed')
 assert Path('/proc/5231/stat').read_text().split()[2]!='Z'
 time.sleep(30)
with open(R/'readout_confirmation.log','a') as log:subprocess.run(['/root/autodl-tmp/prog_env/bin/python','-u',str(R/'run_readout_confirmation.py')],cwd=R,stdout=log,stderr=log,check=True)
