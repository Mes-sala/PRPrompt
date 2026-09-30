"""Fixed-attribute causal controls and post-seal mechanism diagnostics; no encoders."""
from pathlib import Path
import argparse,json,time,fcntl,os
import numpy as np
import torch
import torch.nn.functional as F
import run_study as r
import expanded_data as data
import expanded_grid_cv as cv

CONDITIONS=[('raw_no_update',0,False,0.,0,False),('graph_no_update',2,False,0.,0,False),('graph_anchor',2,False,.5,3,False),('graph_unanchored',2,False,1.,3,False),('graph_shuffle',2,False,.5,3,True),('center_no_update',2,True,0.,0,False),('center_anchor',2,True,.5,3,False)]

def trajectory(z,si,sy,C,beta,rounds,shuffle,seed):
    oh=F.one_hot(sy,C).to(z.dtype);a=oh.T@z[si]/oh.sum(0)[:,None];mu=a.clone();mask=torch.ones(len(z),device=z.device,dtype=torch.bool);mask[si]=False;zu=z[mask];g=torch.Generator(device=z.device).manual_seed(seed+33000);perm=torch.randperm(len(zu),generator=g,device=z.device);hist=[]
    for t in range(rounds):
        q=(zu@F.normalize(mu,dim=1).T/.05).softmax(-1)
        if shuffle:q=q[perm]
        mass=q.sum(0);u=q.T@zu/mass.clamp_min(1e-8)[:,None];new=(1-beta)*a+beta*u;hist.append((mu.clone(),u.clone(),new.clone(),mass.clone()));mu=new
    prob=(z@F.normalize(mu,dim=1).T/.2).softmax(-1)
    return prob,a,mu,hist,mask,perm

def run(plan):
    assert r.sha(__file__)==plan['runner_sha256'] and r.sha(data.__file__)==plan['loader_sha256'] and r.sha(r.__file__)==plan['operator_sha256'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    name=plan['dataset'];out=r.ROOT/'results'/plan['round']/name;out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()))
    d,x,adj,pool,query,C,protocol=data.load_data(name);x=x.double();adj=adj.double();h=adj@ (adj@x);delta=(adj-adj.T).coalesce();symmetric=delta._nnz()==0 or float(delta.values().abs().max())<1e-10
    def energy(v):return float((v*(v-adj@v)).sum())
    geom=dict(symmetric=symmetric,energy_input=energy(x),energy_propagated=energy(h),input_sha256=r.thash(x),graph_sha256=r.thash(d.edge_index),scope='before centering or row normalization')
    if symmetric:assert geom['energy_propagated']<=geom['energy_input']+1e-7*max(1,abs(geom['energy_input']))
    r.save(out/'geometry.json',geom);count=0
    for tag,hops,center,beta,rounds,shuffle in CONDITIONS:
        raw=x if hops==0 else h;z=F.normalize(raw-raw.mean(0,keepdim=True) if center else raw,dim=1)
        for seed in plan['seeds']:
            folder=out/(str(seed)+'__'+tag);folder.mkdir(exist_ok=True);record=folder/'result.json'
            if record.exists():
                old=json.loads(record.read_text());assert old['plan_sha256']==r.sha(plan['_path']) and r.sha(folder/'prediction.npz')==old['prediction_sha256'] and r.sha(folder/'diagnostic.npz')==old['diagnostic_sha256'];count+=1;continue
            support,sy,_,_=cv.support_split(d,pool,C,seed,5);si=support.cuda();sy=sy.cuda();assert not torch.isin(support,query).any();r.save(out/'current.json',dict(seed=seed,condition=tag,completed=count,time=time.time()))
            with torch.no_grad():prob,a,mu,hist,mask,perm=trajectory(z,si,sy,C,beta,rounds,shuffle,seed)
            pf=folder/'prediction.npz';np.savez_compressed(pf,probability=prob[query.cuda()].cpu().numpy(),support=support.numpy(),query=query.numpy(),anchor=a.cpu().numpy(),prototype=mu.cpu().numpy());ph=r.sha(pf)
            # Query and non-support labels are accessed only below this sealing boundary.
            qy=d.y[query].cuda();zy=z[query.cuda()];ref=torch.stack([zy[qy==c].mean(0) for c in range(C)]);mean=zy.mean(0);within=float(sum(((zy[qy==c]-ref[c])**2).sum() for c in range(C))/len(zy));between=float(sum((qy==c).sum()*((ref[c]-mean)**2).sum() for c in range(C))/len(zy))
            diag={'reference':ref.cpu().numpy(),'anchor':a.cpu().numpy(),'prototype':mu.cpu().numpy()};purity=[];uy=d.y[mask.cpu()].cuda();zu=z[mask]
            for t,(before,u,after,mass) in enumerate(hist):
                diag.update({f'before_{t}':before.cpu().numpy(),f'pseudo_{t}':u.cpu().numpy(),f'after_{t}':after.cpu().numpy(),f'mass_{t}':mass.cpu().numpy()});q=(zu@F.normalize(before,dim=1).T/.05).softmax(-1)
                if shuffle:q=q[perm]
                purity.append(float(q[torch.arange(len(uy),device=q.device),uy].mean()))
            cosine=zy@F.normalize(mu,dim=1).T;true=cosine[torch.arange(len(qy),device=qy.device),qy].clone();cosine[torch.arange(len(qy),device=qy.device),qy]=-torch.inf;margin=true-cosine.max(1).values;diag['true_class_margin']=margin.cpu().numpy();df=folder/'diagnostic.npz';np.savez_compressed(df,**diag)
            r.save(record,dict(dataset=name,condition=tag,seed=seed,hops=hops,center=center,beta=beta,rounds=rounds,shuffle=shuffle,plan_sha256=r.sha(plan['_path']),prediction_sha256=ph,diagnostic_sha256=r.sha(df),split_sha256=r.thash(torch.cat([support,query])),reference_scope='query empirical class means; post-seal transductive diagnostic',within_scatter=within,between_scatter=between,soft_correct_mass=purity,min_prototype_norm=float(mu.norm(dim=1).min()),mass_clamp_activated=any(bool((row[3]<1e-8).any()) for row in hist),negative_margin_fraction=float((margin<0).double().mean()),**r.measures(d.y[query].numpy(),prob[query.cuda()].cpu().numpy())));count+=1
        del z
    assert count==len(CONDITIONS)*len(plan['seeds']);r.save(out/'complete.json',dict(rows=count,plan_sha256=r.sha(plan['_path']),finished=time.time()));print(name,count,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan;run(p)
