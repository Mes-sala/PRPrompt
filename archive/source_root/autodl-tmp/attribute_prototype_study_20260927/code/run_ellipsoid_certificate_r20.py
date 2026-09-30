"""R20: covariance ellipsoid score bounds, before reading full-update artifacts."""
import json,time,os,fcntl,math
import numpy as np
import torch
import torch.nn.functional as F
import run_study as r
import expanded_data as data

def arr(x):return x.detach().cpu().numpy()
def certify(lo,hi,pred,norms):
 h=hi.clone();ix=torch.arange(len(pred),device=pred.device);h[ix,pred]=-torch.inf
 return (lo[ix,pred]>h.max(1).values+1e-12)&(norms>1e-12)
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results/ellipsoid_certificate_r20';out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runner=r.sha(__file__);design=r.sha(r.ROOT/'design/R20_方向椭球事前认证_开发验证.md');r.save(out/'process.json',dict(pid=os.getpid(),runner_sha256=runner,started=time.time()));count=0
 for name in data.ALL:
  d,x,adj,pool,query,C,protocol=data.load_data(name);x=x.double();adj=adj.double();z=F.normalize(adj@(adj@x),dim=1)
  geom=json.loads((r.ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text());assert r.thash(x)==geom['input_sha256'] and r.thash(d.edge_index)==geom['graph_sha256']
  folder=out/name;folder.mkdir(exist_ok=True);zz=None;geometry_seconds=0.
  for seed in range(271700,271705):
   rec=folder/f'{seed}.json';pf=folder/f'{seed}.npz'
   if rec.exists():
    old=json.loads(rec.read_text());assert old['runner_sha256']==runner and old['artifact_sha256']==r.sha(pf);count+=1;continue
   src=r.ROOT/'results/cone_certificate_r19'/name/f'{seed}.pre.npz';sj=src.with_suffix('.json');j=json.loads(sj.read_text());assert j['artifact_sha256']==r.sha(src)
   with np.load(src) as p: pre={k:p[k].copy() for k in p.files}
   a=torch.tensor(pre['anchor'],device='cuda');g=torch.tensor(pre['mean'],device='cuda');b=torch.tensor(pre['common'],device='cuda');w=F.normalize(b,dim=1);bn=b.norm(dim=1)
   qi=torch.tensor(pre['query'],device='cuda');si=torch.tensor(pre['support'],device='cuda');zq=z[qi];norms=zq.norm(dim=1);scores=zq@w.T;assert np.allclose(arr(scores),pre['common_scores'],atol=1e-12,rtol=1e-10)
   meta=j['metadata']['spectral'];traj=meta['trajectory'][-1];eta=meta['eta'];eligible=meta['axis_valid'] and eta<float(bn.min());pred=scores.argmax(1)
   lo=torch.tensor(pre['spectral_lower'],device='cuda');hi=torch.tensor(pre['spectral_upper'],device='cuda');elo=-norms[:,None].expand_as(scores).clone();ehi=norms[:,None].expand_as(scores).clone()
   package={k:pre[k] for k in ['anchor','mean','common','support','query','common_scores','node_norms']};package.update(cone_lower=arr(lo),cone_upper=arr(hi),cone_cert=pre['spectral_cert'])
   details=dict(eligible=eligible,eta=eta,Ce=None,gradient_quadratic_min_raw=None,bound_seconds=0.,geometry_precompute_seconds=0.)
   if eligible:
    if zz is None:
     torch.cuda.synchronize();t=time.perf_counter();zz=z.T@z;torch.cuda.synchronize();geometry_seconds=time.perf_counter()-t
    torch.cuda.synchronize();t=time.perf_counter();n=len(z)-len(si);S=(zz-z[si].T@z[si])/n-torch.outer(g,g);S=(S+S.T)/2
    # Covariance action is shared across classes; avoid N x C x D allocation.
    zSz=(zq@S*zq).sum(1);Sw=S@w.T;zSw=zq@Sw;wSw=(w.T*Sw).sum(0)
    quad=(zSz[:,None]-2*scores*zSw+scores.square()*wSw[None,:])/bn.square()[None,:]
    assert float(quad.min())>-1e-9
    # Cancellation-safe outward padding; numerical certificate, not interval arithmetic.
    rounding=1e-12*(1+zSz.abs()[:,None]+2*(scores*zSw).abs()+scores.square()*wSw.abs()[None,:])/bn.square()[None,:]
    qsafe=quad.clamp_min(0)+rounding
    Ce=.5*math.exp(traj['logK'])*math.sqrt(meta['lambda_upper'])*traj['D']/.05
    e=g/g.norm();mc=.5*(a@e)+.5*j['geometry']['alphaZ'];rmin=torch.maximum(bn-eta,mc);assert float(rmin.min())>0
    radius=Ce*qsafe.sqrt()+norms[:,None]*eta**2/rmin.square()[None,:]+1e-12
    elo=torch.maximum(scores-radius,-norms[:,None]);ehi=torch.minimum(scores+radius,norms[:,None]);torch.cuda.synchronize()
    details.update(Ce=Ce,gradient_quadratic_min_raw=float(quad.min()),bound_seconds=time.perf_counter()-t,geometry_precompute_seconds=geometry_seconds)
    package.update(gradient_quadratic=arr(quad),quadratic_padding=arr(rounding),rmin=arr(rmin),ellipsoid_radius=arr(radius),zSz=arr(zSz),zSw=arr(zSw),wSw=arr(wSw))
   jlo=torch.maximum(lo,elo);jhi=torch.minimum(hi,ehi);assert bool((jlo<=jhi+1e-10).all())
   for typ,l,h in [('ellipsoid',elo,ehi),('joint',jlo,jhi)]:
    cert=certify(l,h,pred,norms);package.update({typ+'_lower':arr(l),typ+'_upper':arr(h),typ+'_cert':arr(cert)});details[typ+'_uncertified_pct']=float((~cert).double().mean()*100)
   details['cone_uncertified_pct']=meta['uncertified_pct']
   prepf=folder/f'{seed}.pre.npz';np.savez_compressed(prepf,**package);r.save(folder/f'{seed}.pre.json',dict(dataset=name,seed=seed,details=details,artifact_sha256=r.sha(prepf),source_sha256=r.sha(src),source_json_sha256=r.sha(sj),runner_sha256=runner,design_sha256=design,phase='before loading full update artifacts',finished=time.time()))
   fullsrc=r.ROOT/'results/cone_certificate_r19'/name/f'{seed}.npz'
   with np.load(fullsrc) as f:full_scores=f['full_scores'].copy();full_prototype=f['full_prototype'].copy()
   changed=full_scores.argmax(1)!=pre['common_scores'].argmax(1)
   for typ in ['cone','ellipsoid','joint']:
    assert not np.any(package[typ+'_cert']&changed)
    assert np.all(full_scores>=package[typ+'_lower']-1e-10) and np.all(full_scores<=package[typ+'_upper']+1e-10)
   np.savez_compressed(pf,**package,full_scores=full_scores,full_prototype=full_prototype)
   r.save(rec,dict(dataset=name,seed=seed,details=details,disagreement_pct=float(changed.mean()*100),artifact_sha256=r.sha(pf),pre_artifact_sha256=r.sha(prepf),source_sha256=r.sha(src),full_source_sha256=r.sha(fullsrc),runner_sha256=runner,design_sha256=design,finished=time.time()));count+=1
  print(name,count,[(json.loads((folder/f'{s}.json').read_text())['details']['joint_uncertified_pct']) for s in range(271700,271705)],flush=True)
  del d,x,adj,z,zq,zz;torch.cuda.empty_cache()
 assert count==70;r.save(out/'complete.json',dict(rows=count,datasets=14,runner_sha256=runner,design_sha256=design,finished=time.time()))
if __name__=='__main__':
 with torch.no_grad():main()
