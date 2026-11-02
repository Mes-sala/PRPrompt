"""Locked, dataset-name-free schema rule and masked-statistics scale/hardness risk.

No backbone calls or labels are accepted. Reconstruction risk is a heuristic,
not a confidence interval on classification accuracy.
"""
import math
import torch
from torch.nn import functional as F

FAMILIES=['StatsRoute','ViewMeanStats','RawKernelStats']

@torch.no_grad()
def reconstruction_choice(h,replicates,nodes,active,energy_ratio,shots=(1,3,5)):
    assert h.dtype==torch.float64 and len(replicates)==2
    assert active.shape==(h.shape[1],) and active.dtype==torch.bool
    assert all(x.shape==h.shape and torch.isfinite(x).all() for x in replicates)
    n,d=len(h),(h.shape[1]-3)//2
    groups=[[0,1,2]]+[[3+j,3+d+j] for j in range(d)]
    groups=[[j for j in g if active[j]] for g in groups];groups=[g for g in groups if g]
    assert len(groups)>1 and energy_ratio>=1-1e-6
    risks={k:{f:{mode:torch.zeros(n,dtype=torch.float64) for mode in ['soft','hard']} for f in FAMILIES} for k in shots}
    used=[]
    for held in groups:
        keep=active.clone();keep[held]=False
        if not keep.any():continue
        used.append(held);target=h[:,held];var=target.var(0,unbiased=False).clamp_min(1e-12)
        ref=h.clone();ref[:,~keep]=0
        for version in replicates:
            x=version.clone();x[:,~keep]=0
            stats=F.normalize(x,dim=-1)@F.normalize(ref,dim=-1).T
            view=torch.zeros_like(stats)
            for j in range(d):
                ids=[0,1,2,3+j,3+d+j]
                view+=F.normalize(x[:,ids],dim=-1)@F.normalize(ref[:,ids],dim=-1).T/d
            # Gaussian similarity and negative squared distance have identical neighbour order.
            raw=-torch.cdist(x,ref).square()
            for family,sim in zip(FAMILIES,[stats,view,raw]):
                sim.fill_diagonal_(-float('inf'))
                order=sim.argsort(dim=1,descending=True,stable=True)
                hard=((target[order[:,0]]-target).square()/var).mean(1)
                for shot in shots:
                    k=min(n-1,max(2,math.ceil(math.sqrt(n/shot))))
                    predicted=target[order[:,:k]].mean(1)
                    soft=((predicted-target).square()/var).mean(1)
                    risks[shot][family]['soft']+=soft
                    risks[shot][family]['hard']+=hard
    divisor=len(used)*len(replicates);assert divisor>0
    out={}
    for shot in shots:
        rows=[]
        for family in FAMILIES:
            soft=risks[shot][family]['soft']/divisor;hard=risks[shot][family]['hard']/divisor;delta=hard-soft
            se=float(delta.std(unbiased=True)/math.sqrt(n))
            rows.append(dict(family=family,soft_mse=float(soft.mean()),hard_mse=float(hard.mean()),hard_minus_soft=float(delta.mean()),paired_se=se,hard_heuristic_upper=float(delta.mean())+1.96*se))
        best=min(rows,key=lambda r:(r['soft_mse'],FAMILIES.index(r['family'])))
        target_gain=max(1.,energy_ratio/math.sqrt(shot))
        gain=min([1,3,10],key=lambda g:abs(math.log(g)-math.log(target_gain)))
        base=dict(family=best['family'],train_gain=1)
        out[str(shot)]=dict(family=dict(base,hard=False,inference_gain=1),scale=dict(base,hard=False,inference_gain=gain),full=dict(base,hard=best['hard_heuristic_upper']<0,inference_gain=gain),scores=rows,target_gain=target_gain,neighbours=math.ceil(math.sqrt(n/shot)))
    return dict(shots=out,graphs=n,dimensions=h.shape[1],active_columns=int(active.sum()),heldout_groups=used,energy_ratio=energy_ratio,true_labels_read=False,dataset_name_read=False,encoder_calls=0,proxy='masked-statistics reconstruction; no pseudo classes')

def bandwidth(h):
    dd = torch.cdist(h, h).square()
    v = dd[torch.triu_indices(len(h), len(h), 1).unbind()]
    v = v[v > 1e-12]
    return v.median() if len(v) else h.new_tensor(1.)

def centered(p, si, nodes):
    t = p - p[si].mean(0)
    rms = (t[si].square().mean(1) * nodes[si]).sum().div(nodes[si].sum()).sqrt().clamp_min(1e-6)
    return t / rms


import torch

@torch.no_grad()
def describe(xs):
    x=torch.cat(xs).double();eps=64*torch.finfo(torch.float32).eps
    rms=x.square().mean(0).sqrt();tol=eps*rms.clamp_min(1)
    active=x.std(0,unbiased=False)>tol
    binary=((x.abs()<=tol)|((x-1).abs()<=tol)).all(0)
    allbinary=bool(active.any() and binary[active].all())
    simplex=bool(allbinary and ((x[:,active].sum(1)-1).abs()<=eps).all())
    continuous=active&~binary
    means=torch.stack([a.double().mean(0) for a in xs])
    within=torch.stack([a.double().var(0,unbiased=False) for a in xs]).mean(0).sqrt()
    between=means.std(0,unbiased=False)
    invariant=bool(continuous.any() and (between[continuous]<=eps*within[continuous].clamp_min(1)).all())
    return dict(active_features=int(active.sum()),all_active_binary=allbinary,simplex_rows=simplex,continuous_channels=int(continuous.sum()),constant_graph_mean_for_all_continuous=invariant)

def family_from_schema(schema):
    if schema['all_active_binary'] and schema['simplex_rows']:return 'ViewMeanStats'
    if schema['continuous_channels']>0 and schema['constant_graph_mean_for_all_continuous']:return 'RawKernelStats'
    return 'StatsRoute'

@torch.no_grad()
def choose(h,replicates,nodes,active,energy_ratio,schema,shots=(1,3,5)):
    result=reconstruction_choice(h,replicates,nodes,active,energy_ratio,shots)
    family=family_from_schema(schema)
    for shot in shots:
        cell=result['shots'][str(shot)];risk=next(x for x in cell['scores'] if x['family']==family)
        for mode in ['family','scale','full']:cell[mode]['family']=family
        cell['full']['hard']=risk['hard_heuristic_upper']<0
    result.update(schema=schema,proxy='input distribution structure; reconstruction used only for hardness')
    return result
