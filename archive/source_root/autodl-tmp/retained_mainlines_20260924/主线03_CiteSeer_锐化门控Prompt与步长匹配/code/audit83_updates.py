from pathlib import Path
import json,sys,numpy as np,torch
r=Path(__file__).resolve().parent;d=r/sys.argv[1];checked=0;maximum=0.;normerr=0.
for f in d.glob('*/result.json'):
 a=json.loads(f.read_text())
 if 'optimizer82' not in a or not a['method'].endswith('_mean'):continue
 p=torch.load(f.parent/'sealed.pt',weights_only=True,map_location='cpu')['prompt'];grads=p['optimizer_gradients'].numpy();mode=a['optimizer82'];lr=a['optimizer_lr'];eps=a['optimizer_epsilon'];theta=np.zeros(grads.shape[1:],dtype=np.float32);L=np.eye(theta.shape[0])*eps;R=np.eye(theta.shape[1])*eps;m=np.zeros_like(theta,dtype=np.float64);v=m.copy()
 for i,gg in enumerate(grads,1):
  g=gg.astype(np.float64);L+=g@g.T;R+=g.T@g;m=.9*m+.1*g;v=.999*v+.001*g*g;adam=(m/(1-.9**i))/(np.sqrt(v/(1-.999**i))+1e-8)
  if mode in ['shampoo','gshampoo']:
   vl,ul=np.linalg.eigh(L);vr,ur=np.linalg.eigh(R);assert vl.min()>0 and vr.min()>0;delta=((ul*vl**(-.25))@ul.T)@g@((ur*vr**(-.25))@ur.T)
  elif mode=='gdiagonal':delta=np.diag(L)[:,None]**(-.25)*g*np.diag(R)[None,:]**(-.25)
  elif mode=='ggradient':delta=g
  elif mode=='gadam':delta=adam
  else:raise ValueError(mode)
  if mode.startswith('g'):
   dn=np.linalg.norm(delta);an=np.linalg.norm(adam);delta=delta*(an/dn) if dn>0 else np.zeros_like(delta);record=a['optimizer_diagnostics'][i-1]
   for k,val in [('direction_norm',dn),('adam_norm',an),('update_norm',lr*np.linalg.norm(delta))]:
    err=abs(val-record[k]);normerr=max(normerr,err);assert np.isclose(val,record[k],atol=1e-8,rtol=1e-7),(f,i,k,err)
   assert record['zero_direction']==(dn==0)
  theta=(theta-lr*delta.astype(np.float32)).astype(np.float32)
 err=float(np.max(np.abs(theta-p['tokens'].numpy())));maximum=max(maximum,err);assert err<3e-5,(f,err);checked+=1
recovery=[]
if 'smoke' in d.name:
 for ck in ['GraphMAE','GraphMAE2']:
  a=torch.load(d/f'CiteSeer__{ck}__127199__head_gad_mean'/'sealed.pt',weights_only=True);b=torch.load(d/f'CiteSeer__{ck}__127199__head_tr1_mean'/'sealed.pt',weights_only=True);err=float((a['scores']-b['scores']).abs().max());assert torch.allclose(a['scores'],b['scores'],atol=5e-6,rtol=1e-5);assert torch.equal(a['prediction'],b['prediction']);recovery.append(dict(checkpoint=ck,max_score_error=err))
out=dict(checked=checked,max_token_error=maximum,max_norm_error=normerr,adam_recovery=recovery,passed=True);(r/(d.name+'_update_audit.json')).write_text(json.dumps(out,indent=2));print(out)
