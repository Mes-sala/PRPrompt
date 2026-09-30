import os,sys,json,subprocess,time,fcntl
from pathlib import Path
r=Path(__file__).resolve().parent;gpu=int(sys.argv[1]);lock=open(r/f'gpu{gpu}.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((r/'plan.json').read_text());jobs=plan['jobs'][gpu::2]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
def status(s,**kw):(r/f'queue_gpu{gpu}_status.json').write_text(json.dumps(dict(status=s,gpu=gpu,pid=os.getpid(),time=time.time(),**kw),indent=2))
# Do not compete for an occupied device. Retry until genuinely idle.
while True:
 raw=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 mem,util=map(int,raw.strip().split(','))
 if mem<200 and util<5:break
 status('waiting_for_idle',memory_mib=mem,utilization=util);time.sleep(30)
for smoke in [True,False]:
 for j in jobs:
  status('smoke' if smoke else 'running',job=j['id']);cmd=[sys.executable,'-u',str(r/'run_softening.py'),'--job',j['id']]+(['--smoke'] if smoke else [])
  with open(r/f'{j["id"]}_{"smoke" if smoke else "run"}.log','a') as f:
   ret=subprocess.call(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
  if ret:status('failed',job=j['id'],returncode=ret,smoke=smoke);sys.exit(ret)
status('complete')
