import os,json,subprocess,time,shutil,fcntl
from pathlib import Path
from summarize import summarize,dump
ROOT=Path(__file__).resolve().parent;PY='/root/autodl-tmp/prog_env/bin/python'
lock=(ROOT/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
audit=json.loads((ROOT/'smoke_audit.json').read_text());assert audit['complete'] and audit['passed']
plan=json.loads((ROOT/'plan.json').read_text());(ROOT/'logs').mkdir(exist_ok=True);(ROOT/'failures').mkdir(exist_ok=True)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');dump(ROOT/'scheduler.json',dict(pid=os.getpid(),started=time.time(),status='running'))
for job in plan['jobs']:
 if (ROOT/'results'/job['id']/'complete.json').exists():continue
 if shutil.disk_usage(ROOT).free<5*1024**3:
  dump(ROOT/'scheduler.json',dict(pid=os.getpid(),status='blocked_disk',job=job['id'],time=time.time()));raise SystemExit(2)
 dump(ROOT/'scheduler.json',dict(pid=os.getpid(),status='running',job=job['id'],time=time.time()))
 with (ROOT/'logs'/f"{job['id']}.log").open('a') as log:
  proc=subprocess.Popen([PY,str(ROOT/'grid.py'),'run','--job',job['id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
  last=0
  while proc.poll() is None:
   time.sleep(2)
   if time.time()-last>=20:summarize();last=time.time()
 if proc.returncode:
  dump(ROOT/'failures'/f"{job['id']}.json",dict(returncode=proc.returncode,time=time.time()))
 else:
  (ROOT/'failures'/f"{job['id']}.json").unlink(missing_ok=True)
 summarize()
s=summarize();dump(ROOT/'scheduler.json',dict(pid=os.getpid(),status='complete' if s['complete'] else 'finished_with_failures',time=time.time(),completed_rows=s['completed_rows'],expected_rows=s['expected_rows']))
