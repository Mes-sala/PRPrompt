import os,json,subprocess,time,fcntl,shutil,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/root/autodl-tmp/prog_env/bin/python'
def dump(n,d):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((R/'plan.json').read_text());assert json.loads((R/'operator_audit.json').read_text())['passed'];(R/'logs').mkdir(exist_ok=True)
def work(job,smoke):
 if shutil.disk_usage(R).free<4*1024**3:raise RuntimeError('disk reserve')
 suffix='smoke' if smoke else 'formal'
 with (R/'logs'/f"{job['id']}_{suffix}.log").open('a') as log:
  ret=subprocess.run([PY,str(R/'grid.py'),'run','--job',job['id']]+(['--smoke'] if smoke else []),cwd=R,stdout=log,stderr=subprocess.STDOUT)
 return job['id'],ret.returncode
def summary(stage):
 rows=list((R/stage).glob('*/*/result.json'));dump('summary.json',{'stage':stage,'completed_rows':len(rows),'expected_rows':12 if stage=='smoke' else 240,'updated':time.time()})
try:
 for smoke in [True,False]:
  stage='smoke' if smoke else 'results';dump('pipeline_status.json',{'stage':stage,'status':'running','pid':os.getpid(),'time':time.time()})
  with concurrent.futures.ThreadPoolExecutor(1) as pool:
   fs=[pool.submit(work,j,smoke) for j in plan['jobs']]
   while any(not f.done() for f in fs):summary(stage);time.sleep(10)
   outcomes=[f.result() for f in fs];assert all(code==0 for _,code in outcomes),outcomes
  summary(stage);subprocess.run([PY,str(R/'audit.py')]+(['--smoke'] if smoke else []),check=True,cwd=R)
 dump('pipeline_status.json',{'stage':'complete','status':'complete_audited','time':time.time()})
except Exception as e:
 dump('pipeline_status.json',{'status':'failed','error':repr(e),'time':time.time()});raise
