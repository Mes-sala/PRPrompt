from pathlib import Path
import json,torch
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());count=0;times={}
for j in p['jobs']:
 if j['direct']:continue
 for seed in p['seeds']:
  for key in ['F00','F01','F10','F11']:
   aa=R/'results'/j['id']/f'{seed}__{key}0';bb=R/'results'/j['id']/f'{seed}__{key}1'
   a=torch.load(aa/'sealed.pt',map_location='cpu',weights_only=True);b=torch.load(bb/'sealed.pt',map_location='cpu',weights_only=True)
   assert a['prompt'].keys()==b['prompt'].keys()
   for k,v in a['prompt'].items():
    if torch.is_tensor(v):assert torch.equal(v,b['prompt'][k]),(j['id'],seed,key,k)
    else:assert v==b['prompt'][k]
   ra=json.loads((aa/'result.json').read_text());rb=json.loads((bb/'result.json').read_text());assert ra['split_sha256']==rb['split_sha256'];assert ra['trace']['fit_key']==rb['trace']['fit_key']==key;assert ra['trace']['training_seconds']==rb['trace']['training_seconds'];assert not ra['trace']['kd_active'];assert ra['trace']['trainable_parameters']==a['prompt']['tokens'].numel()
   times[j['id']+'/'+str(seed)+'/'+key]=ra['trace']['training_seconds'];count+=1
expected=sum(not j['direct'] for j in p['jobs'])*len(p['seeds'])*4;assert count==expected
out=dict(passed=True,shared_training_pairs=count,unique_factorial_fits=len(times),factorial_training_seconds=sum(times.values()),training_seconds_by_fit=times);(R/'pair_audit.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items() if k!='training_seconds_by_fit'})
