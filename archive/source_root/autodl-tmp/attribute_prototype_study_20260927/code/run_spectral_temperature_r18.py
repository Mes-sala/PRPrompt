"""Single geometry-calibrated temperature intervention, no accuracy selection."""
import os,json,time,fcntl
import numpy as np
import torch
import torch.nn.functional as F
import run_study as r
import expanded_data as data

NAMES=['Computers','Photo','Squirrel','WikiCS','Roman-empire','Amazon-ratings']
def arr(x):return x.detach().cpu().numpy()
def stats(a,m,u,w,beta,J):
 A=a-a.mean(0);U=u-u.mean(0);M=m-m.mean(0);den=A.square().sum();k=(A*U).sum()/den;res=U-k*A
 av=F.normalize(a,dim=1);v=F.normalize(m,dim=1);da=av[:,None]-av[None,:];dv=v[:,None]-v[None,:];d=da.square().sum(-1);mask=torch.triu(torch.ones_like(d,dtype=torch.bool),diagonal=1)&(d>1e-16)
 return dict(v_ratio=float(M.square().sum()/den),soft_axis=float(k),axis=1-beta+beta*float(k),orthogonal_ratio=beta**2*float(res.square().sum()/den),mean_jeffreys=float(J[mask].mean()),max_jeffreys=float(J.max()),angular_min=float(((da*dv).sum(-1)[mask]/d[mask]).min()),min_norm=float(m.norm(dim=1).min()))
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results/spectral_temperature_r18';out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runner=r.sha(__file__);design=r.sha(r.ROOT/'design/R18_谱标定温度因果干预.md');r.save(out/'process.json',dict(pid=os.getpid(),started=time.time(),runner_sha256=runner));count=0
 for name in NAMES:
  d,x,adj,pool,query,C,protocol=data.load_data(name);x=x.double();adj=adj.double();z=F.normalize(adj@(adj@x),dim=1)
  sf=r.ROOT/'results/consensus_spectrum'/f'{name}.json';sr={v['seed']:v for v in json.loads(sf.read_text())['rows']}
  oldgeom=json.loads((r.ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text());assert r.thash(x)==oldgeom['input_sha256'] and r.thash(d.edge_index)==oldgeom['graph_sha256']
  folder=out/name;folder.mkdir(exist_ok=True)
  for seed in range(271700,271720):
   rec=folder/f'{seed}.json';pf=folder/f'{seed}.npz'
   if rec.exists():
    old=json.loads(rec.read_text());assert old['runner_sha256']==runner and old['design_sha256']==design and old['artifact_sha256']==r.sha(pf);count+=4;continue
   oldfile=r.ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__graph_unanchored'/'prediction.npz'
   with np.load(oldfile) as op:support=op['support'].copy();anchor=op['anchor'].copy()
   si=torch.tensor(support,device='cuda');sy=d.y[support].cuda();oh=F.one_hot(sy,C).double();a=oh.T@z[si]/oh.sum(0)[:,None];assert np.allclose(arr(a),anchor,rtol=1e-10,atol=1e-12)
   mask=torch.ones(len(z),dtype=torch.bool,device='cuda');mask[si]=False;zu=z[mask];sp=sr[seed];lo=sp['rho_lower'];hi=sp['rho_upper'];assert hi<1
   calibrated=.05*lo/2;assert calibrated>0;package=dict(support=support,query=query.numpy(),anchor=arr(a));records=[]
   for tid,tau in [('fixed',.05),('spectral',calibrated)]:
    for beta,tag in [(.5,'anchor'),(1.,'unanchored')]:
     m=a.clone();hist=[];minmass=float('inf');zeros=0;err=0.
     for t in range(3):
      w=F.normalize(m,dim=1);logits=zu@(w-w.mean(0)).T/tau;q=logits.softmax(-1);zeros+=int((q==0).sum());mass=q.sum(0);minmass=min(minmass,float(mass.min()));assert minmass>1e-8
      p=q/mass;u=p.T@zu
      if zeros==0:
       lp=p.log();J=((p[:,:,None]-p[:,None,:])*(lp[:,:,None]-lp[:,None,:])).sum(0)
       dot=((w[:,None]-w[None,:])*(u[:,None]-u[None,:])).sum(-1);err=max(err,float((dot-tau*J).abs().max()));assert err<1e-9
      else:
       dot=((w[:,None]-w[None,:])*(u[:,None]-u[None,:])).sum(-1);J=dot/tau
      m=(1-beta)*a+beta*u;hist.append(stats(a,m,u,w,beta,J))
     logits=z[query.cuda()]@F.normalize(m,dim=1).T/.2;prob=logits.softmax(-1);key=tid+'_'+tag;package[key+'_probability']=arr(prob);package[key+'_prototype']=arr(m)
     replay=None
     if tid=='fixed':
      baseline=r.ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__graph_{tag}'/'prediction.npz'
      with np.load(baseline) as old:replay=float(np.max(abs(arr(m)-old['prototype'])));assert np.allclose(arr(m),old['prototype'],rtol=1e-8,atol=1e-11)
     records.append(dict(key=key,beta=beta,tau=tau,rho_lower=lo*.05/tau,rho_upper=hi*.05/tau,trajectory=hist,zero_soft_assignments=zeros,min_mass=minmass,max_kl_identity_error=err,baseline_replay_error=replay));count+=1
   np.savez_compressed(pf,**package)
   r.save(rec,dict(dataset=name,seed=seed,records=records,runner_sha256=runner,design_sha256=design,artifact_sha256=r.sha(pf),spectrum_source_sha256=r.sha(sf),input_sha256=r.thash(x),graph_sha256=r.thash(d.edge_index),scope='probabilities sealed without reading query labels',finished=time.time()))
  r.save(folder/'complete.json',dict(rows=80,runner_sha256=runner,design_sha256=design));print(name,count,flush=True)
  del d,x,adj,z,zu;torch.cuda.empty_cache()
 assert count==480;r.save(out/'complete.json',dict(rows=count,datasets=6,runner_sha256=runner,design_sha256=design,finished=time.time()))
if __name__=='__main__':
 with torch.no_grad():main()
