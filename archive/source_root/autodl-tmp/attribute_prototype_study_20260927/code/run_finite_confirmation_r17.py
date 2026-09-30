"""Frozen finite-iteration mechanism confirmation; no query-label scoring."""
from pathlib import Path
import os,json,time,fcntl
import numpy as np
import torch
import torch.nn.functional as F
import run_study as r
import expanded_data as data
import expanded_grid_cv as cv

def arr(t):return t.detach().cpu().numpy()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results/finite_confirmation_r17';out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runner=r.sha(__file__);design=r.sha(r.ROOT/'design/R17_有限轮机制新抽样确认.md')
 r.save(out/'process.json',dict(pid=os.getpid(),started=time.time(),runner_sha256=runner,design_sha256=design))
 total=0
 for name in data.ALL:
  d,x,adj,pool,query,C,protocol=data.load_data(name);adj=adj.double();conditions={'original':x.double()}
  if float(d.x.min())<0:
   den=(d.x-d.x.min()).sum(1,keepdim=True).clamp_min(1)
   assert torch.equal((d.x-d.x.min())/den,x.cpu())
   conditions['remove_shift_same_denominator']=(d.x/den).cuda().double()
  for mode,inp in conditions.items():
   z=F.normalize(adj@(adj@inp),dim=1);meta=json.loads((r.ROOT/'results/propagation_causal_r15'/f'{name}__{mode}__h2.json').read_text())
   assert r.thash(inp)==meta['input_sha256'] and r.thash(d.edge_index)==meta['graph_sha256']
   # Sparse float64 reductions need not have bitwise identical order. Verify old
   # finite trajectories numerically before using a newly reconstructed geometry.
   old=json.loads((r.ROOT/'results/propagation_causal_r15'/f'{name}__{mode}__h2.json').read_text())['rows'][0]
   oldpf=r.ROOT/'results/attribute_mechanism_r13'/name/'271700__graph_unanchored'/'prediction.npz'
   with np.load(oldpf) as oldp:oldsi=torch.tensor(oldp['support'],device='cuda');olda=oldp['anchor'].copy()
   oldsy=d.y[arr(oldsi)].cuda();oh=F.one_hot(oldsy,C).double();checka=oh.T@z[oldsi]/oh.sum(0)[:,None]
   oldmask=torch.ones(len(z),dtype=torch.bool,device='cuda');oldmask[oldsi]=False;checkz=z[oldmask];replay_error=0.
   if mode=='original':assert np.allclose(arr(checka),olda,atol=1e-12,rtol=1e-10)
   for beta in [.5,1.]:
    cm=checka.clone()
    for t in range(4):
     actual=float((cm-cm.mean(0)).square().sum(1).mean());expected=old['trajectory'][str(beta)][t]['separation'];replay_error=max(replay_error,abs(actual-expected));assert abs(actual-expected)<1e-10
     if t<3:
      cq=(checkz@F.normalize(cm,dim=1).T/.05).softmax(-1);cu=cq.T@checkz/cq.sum(0)[:,None];cm=(1-beta)*checka+beta*cu
   folder=out/f'{name}__{mode}';folder.mkdir(exist_ok=True)
   for seed in range(271800,271820):
    rec=folder/f'{seed}.json';pf=folder/f'{seed}.npz'
    if rec.exists():
     prior=json.loads(rec.read_text());assert prior['runner_sha256']==runner and prior['design_sha256']==design and r.sha(pf)==prior['artifact_sha256'];total+=1;continue
    support,sy,_,_=cv.support_split(d,pool,C,seed,5);assert not torch.isin(support,query).any();si=support.cuda();sy=sy.cuda()
    mask=torch.ones(len(z),dtype=torch.bool,device='cuda');mask[si]=False;zu=z[mask];g=zu.mean(0)
    oh=F.one_hot(sy,C).double();a=oh.T@z[si]/oh.sum(0)[:,None];common=.5*a+.5*g
    bs=z[query.cuda()]@F.normalize(common,dim=1).T
    package=dict(support=support.numpy(),support_labels=arr(sy),query=query.numpy(),anchor=arr(a),common=arr(common),mean=arr(g),common_scores=arr(bs))
    max_kl_error=0.;massmin=float('inf')
    for beta,tag in [(.5,'anchor'),(1.,'unanchored')]:
     m=a.clone();hist=[];js=[]
     for t in range(3):
      w=F.normalize(m,dim=1);q=(zu@w.T/.05).softmax(-1);mass=q.sum(0);massmin=min(massmin,float(mass.min()));assert massmin>1e-8
      p=q/mass[None,:];u=p.T@zu;lp=p.log()
      J=((p[:,:,None]-p[:,None,:])*(lp[:,:,None]-lp[:,None,:])).sum(0)
      dot=((w[:,None,:]-w[None,:,:])*(u[:,None,:]-u[None,:,:])).sum(-1)
      err=float((dot-.05*J).abs().max());max_kl_error=max(max_kl_error,err);assert err<1e-10
      new=(1-beta)*a+beta*u;hist.append(torch.stack([m,u,new]));js.append(J);m=new
     scores=z[query.cuda()]@F.normalize(m,dim=1).T
     package[tag+'_trajectory']=arr(torch.stack(hist));package[tag+'_jeffreys']=arr(torch.stack(js));package[tag+'_prediction']=arr(scores.argmax(1)).astype(np.int16)
    np.savez_compressed(pf,**package)
    r.save(rec,dict(dataset=name,mode=mode,seed=seed,query_nodes=len(query),runner_sha256=runner,design_sha256=design,loader_sha256=r.sha(data.__file__),operator_sha256=r.sha(r.__file__),splitter_sha256=r.sha(cv.__file__),artifact_sha256=r.sha(pf),representation_sha256=r.thash(z),previous_representation_sha256=meta['representation_sha256'],old_trajectory_max_error=replay_error,graph_sha256=meta['graph_sha256'],source_geometry_sha256=r.sha(r.ROOT/'results/propagation_causal_r15'/f'{name}__{mode}__h2.json'),max_kl_identity_error=max_kl_error,min_class_mass=massmin,scope='Query indices and inputs only; no query labels accessed by experiment loop',finished=time.time()))
    total+=1
   r.save(folder/'complete.json',dict(rows=20,runner_sha256=runner,design_sha256=design));print(name,mode,total,flush=True)
   del z,zu
  del d,x,adj,conditions;torch.cuda.empty_cache()
 assert total==340
 r.save(out/'complete.json',dict(rows=total,geometries=17,runner_sha256=runner,design_sha256=design,finished=time.time()))
if __name__=='__main__':
 with torch.no_grad():main()
