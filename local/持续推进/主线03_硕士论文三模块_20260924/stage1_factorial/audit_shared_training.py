from pathlib import Path
import json,torch
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());count=0;maxerr=0.
for j in p['jobs']:
 for seed in p['seeds']:
  d=R/'results'/j['id'];a=d/f'{seed}__T3_normal_gsh_mean';b=d/f'{seed}__T3_normal_gsh_qeps'
  aa=torch.load(a/'sealed.pt',weights_only=True,map_location='cpu');bb=torch.load(b/'sealed.pt',weights_only=True,map_location='cpu')
  for k in ['tokens','tokens2','gate','gate_probability','center','optimizer_gradients']:assert torch.equal(aa['prompt'][k],bb['prompt'][k]),(j['id'],seed,k)
  ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text());assert ra['split_sha256']==rb['split_sha256'];assert ra['trace']['optimizer_diagnostics']==rb['trace']['optimizer_diagnostics']
  trace=ra['trace'];assert len(trace['optimizer_diagnostics'])==100 and aa['prompt']['optimizer_gradients'].shape[0]==100
  assert trace['trainable_parameters']==aa['prompt']['tokens'].numel();assert not trace['kd_active'];assert trace['optimizer82']=='gshampoo'
  for r in trace['optimizer_diagnostics']:
   expected=0. if r['zero_direction'] else .01*r['adam_norm'];err=abs(r['update_norm']-expected);assert err<1e-10;maxerr=max(maxerr,err)
  count+=1
out=dict(passed=True,shared_training_pairs=count,expected_pairs=315,max_step_norm_error=maxerr,steps_per_fit=100,mean_and_qeps_use_identical_prompt=True);assert count==315;(R/'shared_training_audit.json').write_text(json.dumps(out,indent=2));print(out)
