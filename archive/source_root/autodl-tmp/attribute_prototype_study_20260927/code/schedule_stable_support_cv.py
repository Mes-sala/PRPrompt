"""Serial numerical support-selection correction; no new candidate search."""
from pathlib import Path
import json,sys,os,time,subprocess,fcntl
import run_study as r

def main():
 folder=r.ROOT/'design/stable_support_cv_Amazon';mf=folder/'manifest.json';m=json.loads(mf.read_text());out=r.ROOT/'results/stable_support_cv_dispatch';out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(out/'process.json',dict(pid=os.getpid(),manifest_sha256=r.sha(mf),started=time.time()))
 for i,item in enumerate(m['plans']):
  r.save(out/'current.json',dict(index=i,total=len(m['plans']),unit=item['unit'],time=time.time()))
  with (out/(item['unit']+'.log')).open('a') as log:code=subprocess.run([sys.executable,str(r.ROOT/'code/run_stable_support_cv.py'),'--plan',str(folder/item['plan'])],stdout=log,stderr=subprocess.STDOUT).returncode
  if code:r.save(out/'needs_attention.json',dict(unit=item['unit'],exit_code=code));return
 r.save(out/'complete.json',dict(units=len(m['plans']),rows=m['rows'],manifest_sha256=r.sha(mf),finished=time.time()))
if __name__=='__main__':main()
