"""Pre-update non-asymptotic cone certificate; development only, no query labels."""
import json,time,os,fcntl,math
import numpy as np
import torch
import torch.nn.functional as F
import run_study as r
import expanded_data as data

def arr(x):return x.detach().cpu().numpy()
def diameter(x):return float(torch.cdist(x,x).max())
def upper_spectrum(S,trace):
 gen=torch.Generator(device='cuda').manual_seed(982719);v=torch.randn(len(S),dtype=S.dtype,device='cuda',generator=gen);v/=v.norm()
 for _ in range(120):
  u=S@v;v=u/u.norm().clamp_min(1e-30)
 u=S@v;lo=max(0.,float(v@u));res=float((u-lo*v).norm());rem=max(0.,trace-lo)
 block=.5*(lo+rem+math.sqrt((lo-rem)**2+4*res**2));frob=float(S.square().sum().sqrt())
 return dict(rayleigh=lo,residual=res,trace=trace,frobenius=frob,upper=min(trace,frob,block)+1e-12)
def budget(D0,DA,DZ,cap,m,lam,beta=.5,tau=.05):
 D=D0;rows=[]
 for t in range(3):
  logK=DZ*D/tau
  B=DZ if lam<=0 or D<=0 else min(DZ,math.exp(min(700,logK+math.log(lam*D/tau))))
  if lam==0 or D==0:B=0.
  nxt=min(cap,((1-beta)*DA+beta*B)/m)+1e-12 if m>0 else 2.
  rows.append(dict(D=D,logK=logK,B=B,next_D=nxt));D=nxt
 return rows
def cone_scores(scores,norms,bnorm,eta):
 good=eta<bnorm
 delta=torch.asin((eta/bnorm).clamp(0,1));s=scores/norms[:,None].clamp_min(1e-30)
 assert float(s.abs().max())<1+1e-10
 theta=torch.acos(s.clamp(-1,1));lower=torch.cos((theta+delta).clamp(max=math.pi))*norms[:,None];upper=torch.cos((theta-delta).clamp(min=0))*norms[:,None]
 lower[:,~good]=-norms[:,None];upper[:,~good]=norms[:,None]
 pred=scores.argmax(1);hi=upper.clone();hi[torch.arange(len(pred),device='cuda'),pred]=-torch.inf
 cert=(lower[torch.arange(len(pred),device='cuda'),pred]>hi.max(1).values+1e-12)&(norms>1e-12)
 return lower,upper,cert
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results/cone_certificate_r19';out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runner=r.sha(__file__);design=r.sha(r.ROOT/'design/R19_事前共同收缩认证_开发验证.md');r.save(out/'process.json',dict(pid=os.getpid(),runner_sha256=runner,started=time.time()));count=0
 for name in data.ALL:
  d,x,adj,pool,query,C,protocol=data.load_data(name);x=x.double();adj=adj.double();z=F.normalize(adj@(adj@x),dim=1)
  geom=json.loads((r.ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text());assert r.thash(x)==geom['input_sha256'] and r.thash(d.edge_index)==geom['graph_sha256']
  torch.cuda.synchronize();start=time.perf_counter();zz=z.T@z;total=z.sum(0);torch.cuda.synchronize();pre_seconds=time.perf_counter()-start
  folder=out/name;folder.mkdir(exist_ok=True)
  for seed in range(271700,271705):
   rec=folder/f'{seed}.json';pf=folder/f'{seed}.npz'
   if rec.exists():
    old=json.loads(rec.read_text());assert old['runner_sha256']==runner and old['artifact_sha256']==r.sha(pf);count+=1;continue
   src=r.ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__graph_anchor'/'prediction.npz'
   with np.load(src) as old:support=old['support'].copy();olda=old['anchor'].copy();oldmu=old['prototype'].copy()
   si=torch.tensor(support,device='cuda');sy=d.y[support].cuda();oh=F.one_hot(sy,C).double();a=oh.T@z[si]/oh.sum(0)[:,None];assert np.allclose(arr(a),olda,atol=1e-12,rtol=1e-10)
   mask=torch.ones(len(z),dtype=torch.bool,device='cuda');mask[si]=False;zu=z[mask];n=len(zu);g=zu.mean(0);e=g/g.norm();S=(zz-z[si].T@z[si])/n-torch.outer(g,g);S=(S+S.T)/2
   trace=float((zu-g).square().sum()/n);spec=upper_spectrum(S,trace);az=float((zu@e).min())-1e-12;aa=float((a@e).min())-1e-12;m=.5*(az+aa)
   radius=float((zu-g).norm(dim=1).max());DZ=min(2.,2*radius)
   if 0<=az<=1:DZ=min(DZ,2*math.sqrt(max(0.,1-az*az)))
   DZ+=1e-12;ra=float((a-(a@e)[:,None]*e).norm(dim=1).max());rz=float((zu-(zu@e)[:,None]*e).norm(dim=1).max());rad=.5*(ra+rz);cap=2*rad/math.sqrt(m*m+rad*rad)+1e-12 if m>0 else 2.
   DA=diameter(a)+1e-12;D0=diameter(F.normalize(a,dim=1))+1e-12;common=.5*a+.5*g;scores=z[query.cuda()]@F.normalize(common,dim=1).T;norms=z[query.cuda()].norm(dim=1);bnorm=common.norm(dim=1)
   package=dict(anchor=arr(a),mean=arr(g),common=arr(common),support=support,query=query.numpy(),common_scores=arr(scores),node_norms=arr(norms));pred=scores.argmax(1);metadata={}
   for typ,lam in [('trace',trace+1e-12),('spectral',spec['upper'])]:
    traj=budget(D0,DA,DZ,cap,m,lam);eta=.5*traj[-1]['B'];lo,hi,cert=cone_scores(scores,norms,bnorm,eta)
    if not (m>0 and aa>0):cert[:]=False;lo[:]=-1;hi[:]=1
    package[typ+'_lower']=arr(lo);package[typ+'_upper']=arr(hi);package[typ+'_cert']=arr(cert);metadata[typ]=dict(lambda_upper=lam,trajectory=traj,eta=eta,uncertified_pct=float((~cert).double().mean()*100),axis_valid=m>0 and aa>0)
   # Seal every pre-update quantity before running full q/u loop.
   prepf=folder/f'{seed}.pre.npz';np.savez_compressed(prepf,**package);r.save(folder/f'{seed}.pre.json',dict(dataset=name,seed=seed,metadata=metadata,spectrum=spec,geometry=dict(alphaZ=az,alphaA=aa,m=m,DZ=DZ,DA=DA,D0=D0,cap=cap),artifact_sha256=r.sha(prepf),runner_sha256=runner,design_sha256=design,source_sha256=r.sha(src),phase='before any full softmax update',finished=time.time()))
   mu=a.clone();actual=[]
   for t in range(3):
    q=(zu@F.normalize(mu,dim=1).T/.05).softmax(-1);mass=q.sum(0);assert float(mass.min())>1e-8;u=q.T@zu/mass[:,None]
    obs=dict(direction_diameter=diameter(F.normalize(mu,dim=1)),soft_diameter=diameter(u),soft_residual=float((u-g).norm(dim=1).max()))
    for typ,b in metadata.items():
     if b['axis_valid']:
      bound=b['trajectory'][t];assert obs['direction_diameter']<=bound['D']+1e-10 and obs['soft_diameter']<=bound['B']+1e-10 and obs['soft_residual']<=bound['B']+1e-10
    actual.append(obs);mu=.5*a+.5*u
   assert np.allclose(arr(mu),oldmu,atol=1e-12,rtol=1e-10)
   full_scores=z[query.cuda()]@F.normalize(mu,dim=1).T;full=full_scores.argmax(1);changed=pred!=full
   for typ,b in metadata.items():
    cert=torch.tensor(package[typ+'_cert'],device='cuda');assert not bool((cert&changed).any())
    lower=torch.tensor(package[typ+'_lower'],device='cuda');upper=torch.tensor(package[typ+'_upper'],device='cuda');assert bool((full_scores>=lower-1e-10).all()) and bool((full_scores<=upper+1e-10).all())
   np.savez_compressed(pf,**package,full_prototype=arr(mu),full_scores=arr(full_scores))
   r.save(rec,dict(dataset=name,seed=seed,metadata=metadata,spectrum=spec,actual=actual,disagreement_pct=float(changed.double().mean()*100),pre_artifact_sha256=r.sha(prepf),artifact_sha256=r.sha(pf),runner_sha256=runner,design_sha256=design,source_sha256=r.sha(src),geometry_precompute_seconds=pre_seconds,scope='No query labels; common prediction and bounds sealed before full iteration',finished=time.time()));count+=1
  print(name,count,[(json.loads((folder/f'{s}.json').read_text())['metadata']['spectral']['uncertified_pct']) for s in range(271700,271705)],flush=True)
  del d,x,adj,z,zu,zz,S;torch.cuda.empty_cache()
 assert count==70;r.save(out/'complete.json',dict(rows=count,datasets=14,runner_sha256=runner,design_sha256=design,finished=time.time()))
if __name__=='__main__':
 with torch.no_grad():main()
