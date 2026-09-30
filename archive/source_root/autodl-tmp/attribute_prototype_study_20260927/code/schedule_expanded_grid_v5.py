"""Prepare/review expansion gate, pilot, then sequential immutable grid unit plans."""
from pathlib import Path
import json,time,subprocess,sys,os,fcntl,hashlib
import run_study as r
ROOT=r.ROOT

def main():
 state=ROOT/'results/expanded_grid_dispatch';state.mkdir(parents=True,exist_ok=True);lock=open(state/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r.save(state/'process.json',dict(pid=os.getpid(),started=time.time()));marker=ROOT/'results/expanded_pretraining/complete.json'
 # This process observes the existing pretraining queue. It never restarts it.
 while not marker.exists():
  meta=json.loads((ROOT/'results/expanded_pretraining/process.json').read_text());pid=meta['pid']
  try:os.kill(pid,0)
  except ProcessLookupError:r.save(state/'needs_attention.json',dict(reason='Pretraining queue missing without completion marker',pid=pid));return
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'expand_pretraining.py' in cmd
  time.sleep(30)
 completed=json.loads(marker.read_text());assert completed['total']==117
 exclusions=json.loads((ROOT/'provenance/expanded_capability_exclusions.json').read_text());excluded={x['id'] for x in exclusions};assert len(excluded)==28
 for x in exclusions:assert x['job']['family']=='Bandana' and x['job']['architecture'] in ['GraphSAGE','GAT'] and r.sha(x['source'])==x['source_sha256']
 retry_folder=ROOT/'checkpoints/CS__GraphMAE2__GAT__seed42_norm'
 if (retry_folder/'failure.json').exists() and not (retry_folder/'ready.json').exists():
  with (retry_folder/'isolated_retry.log').open('w') as log:code=subprocess.run([sys.executable,str(ROOT/'code/expand_pretraining.py'),'--job',str(retry_folder/'job.json')],stdout=log,stderr=subprocess.STDOUT).returncode
  if code==0:
   m=json.loads((retry_folder/'ready.json').read_text());m.update(memory_execution='same_recipe_isolated_gpu_retry',repair_runner_sha256=r.sha(ROOT/'code/expand_pretraining.py'));r.save(retry_folder/'ready.json',m)
 jobs=[j for j in json.loads((ROOT/'design/expanded_pretraining_plan.json').read_text())['jobs'] if j['id'] not in excluded];missing=[];repaired=[]
 for job in jobs:
  target=ROOT/'checkpoints'/job['id'];ready=target/'ready.json'
  if not ready.exists():missing.append(job['id']);continue
  meta=json.loads(ready.read_text());assert r.sha(meta['path'])==meta['sha256'] and meta['labels_used'] is False
  if (target/'failure.json').exists():
   repair_sources={'GCNConv decomposed_layers=8':'repair_pretrain_memory.py','SAGE mean aggregation feature_chunks=16':'repair_sage_memory.py','same_recipe_isolated_gpu_retry':'expand_pretraining.py'}
   assert meta.get('memory_execution') in repair_sources and meta['repair_runner_sha256']==r.sha(ROOT/'code'/repair_sources[meta['memory_execution']])
   repaired.append(job['id'])
 r.save(state/'repair_reconciliation.json',dict(original_queue_ready=completed['ready'],verified_repairs=repaired,missing=missing))
 if missing:r.save(state/'needs_attention.json',dict(reason='Expansion failures require review before full grid',missing=missing));return
 legacy=json.loads((ROOT/'provenance/checkpoint_inventory.json').read_text());units=[]
 from expanded_data import ALL
 for name in ALL:units.append(dict(id=name+'_attributes',dataset=name,kind='attributes'))
 for x in legacy:
  assert x['family']!='DGI' and r.sha(x['path'])==x['sha256'];units.append(dict(x,kind='frozen',architecture='GCN',legacy=True,source_root='/root/autodl-tmp/unified_three_backbone_20260924/prog_src'))
 jobs=[j for j in json.loads((ROOT/'design/expanded_pretraining_plan.json').read_text())['jobs'] if j['id'] not in excluded]
 for job in jobs:
  x=json.loads((ROOT/'checkpoints'/job['id']/'ready.json').read_text());assert r.sha(x['path'])==x['sha256'];units.append(dict(x,kind='frozen',legacy=False,source_root=str(ROOT/'source/prog_expansion')))
 assert len(units)==157 and len({u['id'] for u in units})==157
 r.save(state/'coverage_verified.json',dict(datasets=len(ALL),attribute_units=14,frozen_units=143,unsupported_combinations=exclusions,units=units))
 bank=ROOT/'design/expanded_grid_candidate_bank.json';b=json.loads(bank.read_text());runner=ROOT/'code/expanded_grid_v2.py';hashes={}
 for root in {u['source_root'] for u in units if u['kind']=='frozen'}:
  hashes[root]={str(f):r.sha(f) for f in (Path(root)/'prompt_graph').rglob('*.py')}
 def plan(unit,round_name,config_ids=None,seeds=None,validation_folds=None,runner_script=None):
  q=dict(round=round_name,units=[unit],seeds=seeds or b['screen_seeds'],shots=5,runner_sha256=r.sha(Path(runner_script) if runner_script else runner),loader_sha256=r.sha(ROOT/'code/expanded_data.py'),operator_sha256=r.sha(ROOT/'code/run_study.py'),bank_path=str(bank),bank_sha256=r.sha(bank),source_hashes=hashes.get(unit.get('source_root'),{}))
  if config_ids:q['config_ids']=config_ids
  if validation_folds:q['validation_folds']=validation_folds
  if runner_script:q['runner_script']=str(runner_script)
  f=ROOT/'design/grid_units'/(round_name+'.json');f.parent.mkdir(parents=True,exist_ok=True)
  if f.exists():assert json.loads(f.read_text())==q
  else:r.save(f,q)
  return f
 def execute(q,logname):
  with (state/logname).open('a') as log:return subprocess.run([sys.executable,json.loads(q.read_text()).get('runner_script',str(runner)),'--plan',str(q)],stdout=log,stderr=subprocess.STDOUT).returncode
 # Large attribute matrices, long propagation and iterative readouts.
 pilot_ids=[c['id'] for c in b['attributes'] if c['hops'] in [0,4] and c['geometry'] in ['raw','diag'] and c['beta']==.5 and c['tau']==.05 and c['rounds']==10 and c['module'] in ['none','neighbor']]
 for name in ['Physics','CS','Computers']:
  unit=next(u for u in units if u['id']==name+'_attributes');round_name='resource_pilot_'+name;q=plan(unit,round_name,pilot_ids,[279996]);code=execute(q,round_name+'.log')
  if code:r.save(state/'needs_attention.json',dict(reason='Resource pilot failed',dataset=name,exit_code=code));return
 r.save(state/'resource_pilots_passed.json',dict(datasets=['Physics','CS','Computers'],configs_per_dataset=len(pilot_ids),finished=time.time()))
 # User-priority matched comparison precedes the broad parameter grid.
 for i,unit in enumerate(units):
  configs=b[unit['kind']];ids=[c['id'] for c in configs if c.get('hops',2)==2 and c['geometry'] in ['raw','center'] and c['beta']==.5 and c['tau']==.05 and c['rounds'] in [0,3] and c['module']=='none'];assert len(ids)==4
  round_name='backbone_value_'+unit['id'];q=plan(unit,round_name,ids,list(range(271500,271520)),5,ROOT/'code/expanded_grid_cv.py');r.save(state/'current.json',dict(phase='backbone_value',unit=unit['id'],index=i,total=len(units),time=time.time()))
  if execute(q,round_name+'.log'):r.save(state/'needs_attention.json',dict(reason='Matched backbone value comparison failed',unit=unit['id']));return
 r.save(state/'backbone_value_complete.json',dict(units=len(units),rows=len(units)*4*20,finished=time.time()))
 # Individual units use independent Python processes so old/new model sources never collide.
 for i,unit in enumerate(units):
  round_name='grid_'+unit['id'];q=plan(unit,round_name);r.save(state/'current.json',dict(unit=unit['id'],index=i,total=len(units),time=time.time()))
  if __import__('shutil').disk_usage(ROOT).free<12*1024**3:r.save(state/'needs_attention.json',dict(reason='Disk below 12 GiB; sync and archive completed units'));return
  code=execute(q,round_name+'.log')
  if code:r.save(state/'needs_attention.json',dict(reason='Grid unit failed',unit=unit['id'],exit_code=code));return
 r.save(state/'complete.json',dict(units=len(units),finished=time.time()))
if __name__=='__main__':main()
