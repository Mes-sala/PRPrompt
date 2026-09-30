from pathlib import Path
import json,subprocess,time,sys,fcntl,os
R=Path(__file__).resolve().parent
lock=(R/'metrics_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def dump(obj):
 p=R/'metrics_pipeline_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));os.replace(q,p)
try:
 while True:
  states=[json.loads((root/'pipeline_status.json').read_text()) for root in [R,R/'extension2026']]
  if any(s['status']=='failed' for s in states):raise RuntimeError(states)
  if all(s['status']=='complete_audited' for s in states):break
  dump({'status':'waiting_for_formal_runs','time':time.time(),'pid':os.getpid()});time.sleep(30)
 for root in [R,R/'extension2026']:
  a=json.loads((root/'final_audit.json').read_text());assert a['passed'] and a['count']==a['expected']
 dump({'status':'computing','time':time.time(),'pid':os.getpid()})
 for script in ['metrics.py','analyze_metrics.py']:subprocess.run([sys.executable,str(R/script)],cwd=R,check=True)
 dump({'status':'complete','rows':1680,'time':time.time()})
except Exception as e:
 dump({'status':'failed','error':repr(e),'time':time.time()});raise
