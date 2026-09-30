import os,sys,json,time,hashlib,signal,subprocess,shutil,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
D=R/'downstream'; W=R/'parallel_execution'; M=R/'parallel_migration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));os.replace(t,p)
def main():
 plan=read(R/'parallel_execution_plan.json');audit=read(R/'parallel_execution_replay_audit.json');cfg=read(R/'downstream_freeze.json')
 assert audit['status']=='exact_execution_only_replay_passed'
 assert sha(R/'run_downstream41_shard.py')==plan['shard_runner_sha256']==audit['shard_runner_sha256']
 assert sha(R/'downstream_freeze.json')==plan['scientific_freeze_sha256']==audit['scientific_freeze_sha256']
 assert not M.exists(),'Migration already exists; inspect before resuming'
 assert not (D/'complete.json').exists()
 pid=3294
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 assert str(R/'run_downstream41.py').encode() in cmd,cmd
 M.mkdir();write(M/'start.json',{'time':time.time(),'old_pid':pid,'reason':'Exact execution replay passed; disjoint cell concurrency only','audit_sha256':sha(R/'parallel_execution_replay_audit.json')})
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  if not Path(f'/proc/{pid}').exists():break
  time.sleep(.1)
 assert not Path(f'/proc/{pid}').exists(),'Old producer must exit before shards start'
 for n in ['downstream_production_process.json','downstream_production_terminal.json','downstream_production.log']:
  if (R/n).exists():shutil.copy2(R/n,M/n)
 if (D/'progress.json').exists():shutil.copy2(D/'progress.json',M/'old_progress.json')
 kept={};quarantine=[]
 for cell in cfg['cells']:
  folder=D/cell['dataset']/cell['encoder']
  if not folder.exists():continue
  for job in folder.iterdir():
   if not job.is_dir():continue
   if not (job/'record.json').exists():
    dest=R/'parallel_interrupted'/job.relative_to(D);dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(job),str(dest));quarantine.append(str(dest.relative_to(R)));continue
   rec=read(job/'record.json');assert rec['freeze_sha256']==sha(R/'downstream_freeze.json')
   assert sha(job/'sealed.npz')==rec['sealed_sha256']
   if 'state_sha256' in rec:assert sha(job/'prompt_states.pt')==rec['state_sha256']
   for p in job.iterdir():
    if p.is_file():kept[str(p.relative_to(R))]=sha(p)
 write(M/'preserved.json',{'files':kept,'quarantined':quarantine,'time':time.time()})
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 workers=[];W.mkdir(exist_ok=True)
 for i in range(4):
  log=open(W/f'shard{i}.log','ab');p=subprocess.Popen([sys.executable,str(R/'run_downstream41_shard.py'),'--shard',str(i),'--shards','4'],env=env,stdout=log,stderr=subprocess.STDOUT);workers.append(p)
 write(W/'processes.json',{'driver_pid':os.getpid(),'workers':[{'shard':i,'pid':p.pid} for i,p in enumerate(workers)],'time':time.time()})
 while any(p.poll() is None for p in workers):
  count=sum(1 for p in D.glob('*/*/episode*/record.json'));write(W/'progress.json',{'status':'running','completed_jobs':count,'total_jobs':cfg['total_jobs'],'workers':[{'shard':i,'pid':p.pid,'returncode':p.poll()} for i,p in enumerate(workers)],'time':time.time()});time.sleep(10)
 codes=[p.returncode for p in workers];write(W/'terminal.json',{'returncodes':codes,'time':time.time()});assert codes==[0]*4,codes
 owned=[];count=0
 for i in range(4):
  c=read(W/str(i)/'complete.json');assert c['execution_source_sha256']==plan['shard_runner_sha256'];assert c['freeze_sha256']==plan['scientific_freeze_sha256'];owned+=c['owned_cells'];count+=c['count']
 assert len(owned)==len(set(owned))==42 and set(owned)=={c['dataset']+'/'+c['encoder'] for c in cfg['cells']}
 assert count==cfg['total_jobs']==5040
 for cell in cfg['cells']:
  folder=D/cell['dataset']/cell['encoder'];jobs=[folder/'no_prompt']+[folder/f'episode{ei:02d}__{method}__lr{lr:g}' for ei in cfg['episode_indices'] for method in cfg['prompt_methods'] for lr in cfg['learning_rates']]
  for job in jobs:
   rec=read(job/'record.json');assert rec['freeze_sha256']==plan['scientific_freeze_sha256'];assert sha(job/'sealed.npz')==rec['sealed_sha256']
   if 'state_sha256' in rec:assert sha(job/'prompt_states.pt')==rec['state_sha256']
 for name,h in kept.items():assert sha(R/name)==h,name
 write(W/'final_audit.json',{'status':'all_expected_records_validated','preserved_files':len(kept),'count':count,'cells':42,'query_labels_read':False,'time':time.time()})
 write(D/'complete.json',{'status':'sealed_unscored','count':count,'cells':42,'freeze_sha256':plan['scientific_freeze_sha256'],'query_labels_read':False,'execution_audit_sha256':sha(W/'final_audit.json')})
if __name__=='__main__':
 try:main()
 except Exception:
  write(W/'driver_failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
