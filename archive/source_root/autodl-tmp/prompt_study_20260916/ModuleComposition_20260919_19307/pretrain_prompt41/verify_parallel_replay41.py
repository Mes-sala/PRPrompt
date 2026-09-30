from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
p=Path(__file__).resolve().parent
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
cm=json.loads((p/'parallel_replay_execution/0/complete.json').read_text());assert cm['count']==6 and cm['replay']
plan=json.loads((p/'parallel_execution_plan.json').read_text());assert plan['shard_runner_sha256']==sha(p/'run_downstream41_shard.py')
a=p/'downstream/ENZYMES/random_s101';b=p/'parallel_replay/ENZYMES/random_s101';cases=[]
for recpath in sorted(b.glob('*/record.json')):
 old=a/recpath.parent.name;new=recpath.parent
 for f in ['sealed.npz']:
  with np.load(old/f) as x,np.load(new/f) as y:
   assert x.files==y.files
   for k in x.files:assert np.array_equal(x[k],y[k]),(old,k)
 r=json.loads(recpath.read_text());r0=json.loads((old/'record.json').read_text())
 for key in set(r)-{'seconds','sealed_sha256','state_sha256'}:assert r[key]==r0[key],(old,key)
 tensors=0
 if recpath.parent.name!='no_prompt':
  x=torch.load(old/'prompt_states.pt',map_location='cpu',weights_only=True);y=torch.load(new/'prompt_states.pt',map_location='cpu',weights_only=True)
  for step in x:
   for module in x[step]:
    if x[step][module] is None:assert y[step][module] is None;continue
    for k in x[step][module]:assert torch.equal(x[step][module][k],y[step][module][k]);tensors+=1
 cases.append({'job':recpath.parent.name,'sealed_sha256':sha(old/'sealed.npz'),'state_tensors_equal':tensors})
assert len(cases)==7
result={'status':'exact_execution_only_replay_passed','cases':cases,'scientific_freeze_sha256':sha(p/'downstream_freeze.json'),'shard_runner_sha256':sha(p/'run_downstream41_shard.py'),'validator_sha256':sha(Path(__file__)),'scope':'Six jobs, all scores/predictions/Prompt state tensors/200-step loss trajectories, plus 20-support no-Prompt baseline; no query target access.','time':time.time()}
(p/'parallel_execution_replay_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':len(cases),'tensors':sum(c['state_tensors_equal'] for c in cases)}))
