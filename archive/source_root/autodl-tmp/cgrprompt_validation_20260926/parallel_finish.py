from pathlib import Path
import sys,subprocess,json,time,fcntl,shutil,hashlib
R=Path(__file__).resolve().parent;PY='/root/autodl-tmp/prog_env/bin/python'
lockfile=open(R/'parallel_finish.lock','w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(name,obj):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def state(**kw):save('supervisor_status.json',dict(time=time.time(),**kw))
try:
 jobs=[j for j in json.loads((R/'plan.json').read_text())['jobs'] if j['phase']=='main'];waiting=list(jobs);active=[]
 while waiting or active:
  assert shutil.disk_usage(R).free>4*1024**3
  while waiting and len(active)<3:
   j=waiting.pop(0);log=open(R/('tuning_'+j['id']+'.log'),'a');p=subprocess.Popen([PY,'-u',str(R/'tuning_worker.py'),j['id']],cwd=R,stdout=log,stderr=log);active.append((j,p,log))
  state(status='running',stage='tuning',mode='3 independent jobs',workers=[dict(job=j['id'],pid=p.pid) for j,p,l in active],queued=[j['id'] for j in waiting])
  time.sleep(10)
  for j,p,l in active[:]:
   ret=p.poll()
   if ret is None:continue
   l.close();assert ret==0,(j['id'],ret)
   files=list((R/'tuning'/j['id']).glob('*/result.json'));assert len(files)==140,(j['id'],len(files));active.remove((j,p,l))
 expected={(j['id'],seed,m) for j in jobs for seed in j['seeds'] for m in ['fixed_adam','fixed_gshampoo','fixed_ungrafted','learned_hidden','learned_teacher','GPF+','GPrompt']};rows=[json.loads(f.read_text()) for f in (R/'tuning').glob('*/*/result.json')];assert len(rows)==840 and {(x['job'],x['seed'],x['method']) for x in rows}==expected
 save('tuning_complete.json',dict(finished=time.time(),rows=840,mode='3 independent jobs; unchanged scientific runners'))
 for stage,script in [('cost','run_cost.py'),('readout_confirmation','run_readout_confirmation.py')]:
  if (R/(stage+'_complete.json')).exists():continue
  assert shutil.disk_usage(R).free>4*1024**3
  with open(R/(stage+'.log'),'a') as log:
   p=subprocess.Popen([PY,'-u',str(R/script)],cwd=R,stdout=log,stderr=log);state(status='running',stage=stage,pid=p.pid);ret=p.wait();assert ret==0,(stage,ret)
  assert (R/(stage+'_complete.json')).exists()
 save('all_remote_stages_complete.json',dict(finished=time.time(),stages=['core','allocation','tuning','cost','readout_confirmation'],remaining='local full audit and narrative delivery'))
 state(status='complete',stage='all_remote',remaining='local full audit and narrative delivery')
except Exception as e:
 state(status='failed',stage='parallel_finish',error=repr(e));raise
