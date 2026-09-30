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
 if completed['ready']!=117:r.save(state/'needs_attention.json',dict(reason='Expansion failures require review before full grid',summary=completed));return
 legacy=json.loads((ROOT/'provenance/checkpoint_inventory.json').read_text());units=[]
 from expanded_data import ALL
 for name in ALL:units.append(dict(id=name+'_attributes',dataset=name,kind='attributes'))
 for x in legacy:
  assert x['family']!='DGI' and r.sha(x['path'])==x['sha256'];units.append(dict(x,kind='frozen',architecture='GCN',legacy=True,source_root='/root/autodl-tmp/unified_three_backbone_20260924/prog_src'))
 jobs=json.loads((ROOT/'design/expanded_pretraining_plan.json').read_text())['jobs']
 for job in jobs:
  x=json.loads((ROOT/'checkpoints'/job['id']/'ready.json').read_text());assert r.sha(x['path'])==x['sha256'];units.append(dict(x,kind='frozen',legacy=False,source_root=str(ROOT/'source/prog_expansion')))
 assert len(units)==185 and len({u['id'] for u in units})==185
 r.save(state/'coverage_verified.json',dict(datasets=len(ALL),attribute_units=14,frozen_units=171,units=units))
 bank=ROOT/'design/expanded_grid_candidate_bank.json';b=json.loads(bank.read_text());runner=ROOT/'code/expanded_grid_v2.py';hashes={}
 for root in {u['source_root'] for u in units if u['kind']=='frozen'}:
  hashes[root]={str(f):r.sha(f) for f in (Path(root)/'prompt_graph').rglob('*.py')}
 def plan(unit,round_name,config_ids=None,seeds=None):
  q=dict(round=round_name,units=[unit],seeds=seeds or b['screen_seeds'],shots=5,runner_sha256=r.sha(runner),loader_sha256=r.sha(ROOT/'code/expanded_data.py'),operator_sha256=r.sha(ROOT/'code/run_study.py'),bank_path=str(bank),bank_sha256=r.sha(bank),source_hashes=hashes.get(unit.get('source_root'),{}))
  if config_ids:q['config_ids']=config_ids
  f=ROOT/'design/grid_units'/(round_name+'.json');f.parent.mkdir(parents=True,exist_ok=True)
  if f.exists():assert json.loads(f.read_text())==q
  else:r.save(f,q)
  return f
 def execute(q,logname):
  with (state/logname).open('a') as log:return subprocess.run([sys.executable,str(runner),'--plan',str(q)],stdout=log,stderr=subprocess.STDOUT).returncode
 # Large attribute matrices, long propagation and iterative readouts.
 pilot_ids=[c['id'] for c in b['attributes'] if c['hops'] in [0,4] and c['geometry'] in ['raw','diag'] and c['beta']==.5 and c['tau']==.05 and c['rounds']==10 and c['module'] in ['none','neighbor']]
 for name in ['Physics','CS','Computers']:
  unit=next(u for u in units if u['id']==name+'_attributes');round_name='resource_pilot_'+name;q=plan(unit,round_name,pilot_ids,[279996]);code=execute(q,round_name+'.log')
  if code:r.save(state/'needs_attention.json',dict(reason='Resource pilot failed',dataset=name,exit_code=code));return
 r.save(state/'resource_pilots_passed.json',dict(datasets=['Physics','CS','Computers'],configs_per_dataset=len(pilot_ids),finished=time.time()))
 # Individual units use independent Python processes so old/new model sources never collide.
 for i,unit in enumerate(units):
  round_name='grid_'+unit['id'];q=plan(unit,round_name);r.save(state/'current.json',dict(unit=unit['id'],index=i,total=len(units),time=time.time()))
  if __import__('shutil').disk_usage(ROOT).free<12*1024**3:r.save(state/'needs_attention.json',dict(reason='Disk below 12 GiB; sync and archive completed units'));return
  code=execute(q,round_name+'.log')
  if code:r.save(state/'needs_attention.json',dict(reason='Grid unit failed',unit=unit['id'],exit_code=code));return
 r.save(state/'complete.json',dict(units=len(units),finished=time.time()))
if __name__=='__main__':main()
