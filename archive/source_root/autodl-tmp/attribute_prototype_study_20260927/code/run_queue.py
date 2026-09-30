import subprocess,sys,fcntl,json,time,os
from pathlib import Path
p=Path(__file__).resolve().parent.parent
lock=open(p/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for script,plan,tag in [('run_study.py','r03_plan.json','r03_confirmation'),('run_selection.py','r04_plan.json','r04_support_selection')]:
 state=dict(queue_pid=os.getpid(),stage=tag,time=time.time());(p/'queue_status.json').write_text(json.dumps(state))
 with (p/f'{tag}.log').open('a') as f:
  code=subprocess.call([sys.executable,str(p/'code'/script),'--plan',str(p/'design'/plan)],stdout=f,stderr=subprocess.STDOUT)
 if code!=0:raise RuntimeError(f'{tag} failed with {code}; queue stopped')
 assert (p/'results'/tag/'complete.json').exists()
(p/'queue_status.json').write_text(json.dumps(dict(status='complete',time=time.time())))
