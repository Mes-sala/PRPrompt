"""R24: frozen real-graph concentration intervention; no labels in diagnostics.

The NumPy self-check imports no torch. Real runs require a separately frozen
contract, reuse R13 support indices, and seal all limiting predictions first.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT/'design/R24_真实图方向集中尺度_固定干预.md'
OUT = ROOT/'results/concentration_scaling_r24'
CONTRACT = ROOT/'provenance/r24_frozen_contract.json'
DATASETS = ['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon',
            'Cornell','Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings']
SEED = 271700
KS = list(range(13))
BETAS = [.5, 1.]
TAU0 = .05
FLOOR = 100*np.finfo(np.float64).eps


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [finite(v) for v in value]
    return value


def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'w') as f:
        json.dump(finite(value),f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def save_npz(path,package):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'wb') as f:
        np.savez_compressed(f,**package); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def payload():
    deps=['code/expanded_data.py','code/run_study.py',
          'provenance/expanded_datasets.json','provenance/expanded_heterophily_datasets.json',
          'theory/方向集中下的反馈与判别尺度_候选.md',
          'theory/池质量与方向集中_联合极限.md']
    return dict(schema='concentration-scaling-r24-v1',runner_sha256=sha(__file__),
        design_sha256=sha(DESIGN),dependencies={p:sha(ROOT/p) for p in deps},
        datasets=DATASETS,seed=SEED,k=KS,betas=BETAS,rounds=[1,2,3],tau0=TAU0,
        temperature_branches=['fixed','scaled'],axis='ones/sqrt(d)',axis_mass_min=1e-14,
        commutation_epsilon=.5,domain_eps_radius=.1,domain_logit=.1,
        resolution_constant=100,machine_epsilon=np.finfo(np.float64).eps,
        fitting='all domain and resolved nonzero points, minimum 3; no extra epsilon',
        ren=dict(rounds=1,temperature=1.,readout='unnormalized squared Euclidean',
                 replication_policies=['one','inverse_epsilon_squared'],scope='development mechanism intervention, not holdout'))


# Shared pure operators: NumPy checks and torch runtime use these same formulas.
def is_numpy(x): return isinstance(x,(np.ndarray,np.generic))
def sum_(x,axis=None,keep=False):
    if is_numpy(x): return np.sum(x,axis=axis,keepdims=keep)
    return x.sum() if axis is None else x.sum(dim=axis,keepdim=keep)
def mean_(x,axis=0,keep=False):
    return sum_(x,axis,keep)/x.shape[axis]
def sqrt_(x): return np.sqrt(x) if is_numpy(x) else x.sqrt()
def exp_(x): return np.exp(x) if is_numpy(x) else x.exp()
def copy_(x): return x.copy() if is_numpy(x) else x.clone()
def softmax(x):
    top=x.max(axis=1,keepdims=True) if is_numpy(x) else x.max(dim=1,keepdim=True).values
    y=exp_(x-top); return y/sum_(y,1,True)
def logsumexp(x,axis,keep=False):
    if is_numpy(x):
        top=x.max(axis=axis,keepdims=True)
        ans=top+np.log(sum_(np.exp(x-top),axis,True))
        return ans if keep else np.squeeze(ans,axis=axis)
    return x.logsumexp(dim=axis,keepdim=keep)
def responsibility_columns(logits):
    # Same row-softmax then column normalization, without zero class mass.
    logq=logits-logsumexp(logits,1,True)
    logmass=logsumexp(logq,0)
    p=exp_(logq-logmass[None,:])
    if not tensor_finite(p): raise ValueError('Nonfinite column-normalized responsibilities')
    q_unrepresentable_count=sum_(exp_(logq)==0)
    return p,exp_(logmass),logmass,q_unrepresentable_count
def normsq(x): return sum_(x*x,1)
def norm(x): return float(sqrt_(sum_(x*x)))
def arr(x): return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
def tensor_finite(x):
    return bool(np.isfinite(x).all()) if is_numpy(x) else bool(x.isfinite().all())


def node_coordinates(v,eps):
    radial=1/sqrt_(1+eps*eps*normsq(v))
    return radial,radial[:,None]*v


def normalized_coordinates(A,H,eps):
    scale=sqrt_(A*A+eps*eps*normsq(H))
    return A/scale,H/scale[:,None]


def stable_scores(v,radial,A,H,eps):
    """D has exactly the cosine argmax; no subtraction of two values near 1."""
    ah,hh=normalized_coordinates(A,H,eps)
    vn=normsq(v); hn=normsq(hh)
    # (r-ahat)/eps, rationalized using r²+eps²*r²||v||²=1.
    radial_gap=eps*(hn[None,:]-radial[:,None]**2*vn[:,None])/(radial[:,None]+ah[None,:])
    D=2*radial[:,None]*(v@hh.T)-hn[None,:]-radial_gap*radial_gap
    return D


def means_by_class(radial,tangent,onehot,counts):
    return (onehot.T@radial)/counts,(onehot.T@tangent)/counts[:,None]


def update(vu,ru,tu,A,H,anchor_A,anchor_H,eps,tau,beta):
    p,mass,logmass,qzero=responsibility_columns((eps*eps/(2*tau))*stable_scores(vu,ru,A,H,eps))
    ua=p.T@ru; uh=p.T@tu
    return (1-beta)*anchor_A+beta*ua,(1-beta)*anchor_H+beta*uh,ua,uh,mass,logmass,qzero


def euclidean_D(v,h):
    # -||v-h||² after removing the node-common -||v||².
    return 2*v@h.T-normsq(h)[None,:]


def tangent_limit(vu,vq,m,beta):
    h=copy_(m); rows=[]
    for t in range(1,4):
        p,mass,logmass,qzero=responsibility_columns(euclidean_D(vu,h)/(2*TAU0))
        h=(1-beta)*m+beta*(p.T@vu)
        rows.append(dict(H=copy_(h),mass=copy_(mass),logmass=copy_(logmass),
                         q_probability_zero_count=qzero,D=euclidean_D(vq,h)))
    return rows


def leading_terms(vu,vq,m):
    g=mean_(vu); vc=vu-g
    B=m-mean_(m); powers=[]; action=B
    for _ in range(3):
        action=(vc.T@(vc@action.T)).T/len(vu)
        powers.append(copy_(action))
    mt=.5*m+.5*g
    fixed_score=vq@powers[0].T-sum_(mt*powers[0],1)[None,:]
    margins=[(vq-g)@p.T for p in powers]
    return dict(mean=g,B=B,powers=powers,fixed_score=fixed_score,
                unanchored_score=margins,tangent_common_D=euclidean_D(vq,mt))


def radial_loss(v,radial):
    # (1-r)/eps²; valid even when subtraction from 1 is unresolved.
    return normsq(v)*radial*radial/(1+radial)


def stable_euclidean_scores(v,radial,B,H,eps):
    rho=radial_loss(v,radial)
    return 2*radial[:,None]*(v@H.T)-normsq(H)[None,:]-eps*eps*(B[None,:]-rho[:,None])**2


def ren_statistics(vu,ru,tu,anchor_B,anchor_H,eps):
    p,mass,logmass,qzero=responsibility_columns(eps*eps*stable_euclidean_scores(vu,ru,anchor_B,anchor_H,eps))
    rho=radial_loss(vu,ru)
    return p.T@rho,p.T@tu,mass,logmass,qzero


def mass_anchored_coordinates(anchor_B,anchor_H,soft_B,soft_H,logmass,counts,replication):
    logcounts=np.log(counts) if is_numpy(counts) else counts.log()
    logit=logcounts-math.log(replication)-logmass
    gamma=np.exp(-np.logaddexp(0,-logit)) if is_numpy(logit) else logit.sigmoid()
    B=gamma*anchor_B+(1-gamma)*soft_B
    H=gamma[:,None]*anchor_H+(1-gamma)[:,None]*soft_H
    return B,H,gamma


def ren_leading_terms(vq,m,lead,n,counts):
    # Protocol freezes equal K=5; reject silently unequal-mass generalization.
    assert float(counts.min())==float(counts.max())
    kappa=len(m)*float(counts[0])/n
    common=kappa*(m-lead['mean'])
    full=common+2*lead['powers'][0]
    full_contrast=kappa*lead['B']+2*lead['powers'][0]
    common_contrast=kappa*lead['B']
    gamma0=kappa/(1+kappa)
    center=gamma0*m+(1-gamma0)*lead['mean']
    return dict(kappa=kappa,gamma0=gamma0,full_prototype_coefficient=full,
        common_prototype_coefficient=common,
        full_limit_score=2*(vq-lead['mean'])@full_contrast.T,
        common_limit_score=2*(vq-lead['mean'])@common_contrast.T,
        fixed_replication_tangent_center=center,fixed_replication_limit_D=euclidean_D(vq,center))


def centered(x): return x-mean_(x,0,True)
def rowcentered(x): return x-mean_(x,1,True)


def coordinate_norm(A,H,eps):
    return math.sqrt(float(sum_(A*A))+eps*eps*float(sum_(H*H)))


def coordinate_metric(actual_A,actual_H,predicted_H,eps,domain,reference_scale=1.):
    """Prediction is a physical tangent vector, with zero leading radial term."""
    actual=coordinate_norm(actual_A,actual_H,eps)
    predicted=norm(predicted_H)
    error=math.sqrt(float(sum_(actual_A*actual_A))+float(sum_((eps*actual_H-predicted_H)**2)))
    threshold=FLOOR*max(1.,reference_scale,math.sqrt(actual_A.shape[0]))
    resolved=actual>threshold and predicted>threshold
    return dict(amplitude=actual,predicted_amplitude=predicted,error=error,
        relative_error=error/predicted if predicted>0 else None,
        resolution_threshold=threshold,resolved=resolved,
        zero_leading_coefficient=predicted==0,domain=domain,fit_eligible=domain and resolved)


def matrix_metric(actual,predicted,domain,reference_scale=1.):
    actual_norm=norm(actual); predicted_norm=norm(predicted); error=norm(actual-predicted)
    # Frobenius metrics compare with a same-sized matrix of unit cosine scale.
    threshold=FLOOR*max(1.,reference_scale)*math.sqrt(actual.shape[0]*actual.shape[1])
    resolved=actual_norm>threshold and predicted_norm>threshold
    return dict(amplitude=actual_norm,predicted_amplitude=predicted_norm,error=error,
        relative_error=error/predicted_norm if predicted_norm>0 else None,
        resolution_threshold=threshold,resolved=resolved,
        zero_leading_coefficient=predicted_norm==0,domain=domain,fit_eligible=domain and resolved)


def ranking(D,reference,physical_factor):
    a,b=arr(D),arr(reference)
    diff=a.argmax(1)!=b.argmax(1)
    sa=np.sort(a,axis=1); sb=np.sort(b,axis=1)
    gap_a=physical_factor*(sa[:,-1]-sa[:,-2])
    gap_b=physical_factor*(sb[:,-1]-sb[:,-2])
    near=(gap_a<=FLOOR)|(gap_b<=FLOOR)
    return dict(disagreement_count=int(diff.sum()),nodes=len(a),
        disagreement_fraction=float(diff.mean()),unresolved_gap_either_count=int(near.sum()),
        disagreement_outside_unresolved_count=int((diff&~near).sum()),
        min_physical_gap=float(min(gap_a.min(),gap_b.min())))


def slopes(records):
    groups={}
    for row in records:
        if row['temperature']!='fixed': continue
        for t,entry in row['rounds'].items():
            for key,m in entry['metrics'].items():
                if m.get('fit_eligible'):
                    groups.setdefault((row['beta'],int(t),key),[]).append((row['epsilon'],m['amplitude']))
    answer=[]
    for beta in BETAS:
        for t in range(1,4):
            keys=['initial_contrast','feedback','prototype_vs_common','score_vs_common'] if beta==.5 else ['initial_contrast','class_contrast','centered_score']
            for key in keys:
                points=sorted(groups.get((beta,t,key),[]))
                item=dict(beta=beta,round=t,metric=key,points=points,n_points=len(points),slope=None)
                if len(points)>=3:
                    x=np.log([a for a,b in points]); y=np.log([b for a,b in points])
                    coefficient=np.polyfit(x,y,1); item['slope']=float(coefficient[0])
                    residual=y-np.polyval(coefficient,x)
                    item['log_rmse']=float(np.sqrt(np.mean(residual**2)))
                else: item['reason']='Fewer than three predeclared domain/resolved points'
                answer.append(item)
    return answer


def ren_slopes(records):
    answer=[]
    for policy in ['one','inverse_epsilon_squared']:
        keys=['soft_position'] if policy=='one' else ['soft_position','full_vs_mean','common_vs_mean','centered_score','common_centered_score']
        for key in keys:
            points=sorted((r['epsilon'],r['metrics'][key]['amplitude']) for r in records
                          if r['replication_policy']==policy and r['metrics'][key]['fit_eligible'])
            entry=dict(replication_policy=policy,metric=key,points=points,n_points=len(points),slope=None)
            if len(points)>=3:
                x=np.log([a for a,b in points]); y=np.log([b for a,b in points]); fit=np.polyfit(x,y,1)
                entry.update(slope=float(fit[0]),log_rmse=float(np.sqrt(np.mean((y-np.polyval(fit,x))**2))))
            else: entry['reason']='Fewer than three predeclared domain/resolved points'
            answer.append(entry)
    return answer


def selfcheck():
    rng=np.random.default_rng(270924)
    v=rng.normal(size=(73,6))*.3
    support=np.arange(15); sy=np.repeat(np.arange(3),5)
    onehot=np.eye(3)[sy]; counts=onehot.sum(0); vu=v[15:]; vq=v[15:]
    m=onehot.T@v[support]/counts[:,None]; lead=leading_terms(vu,vq,m)
    limits={b:tangent_limit(vu,vq,m,b) for b in BETAS}
    identity_error=0.; update_error=0.; final=[]; unanchored=[]
    for eps in [1.,.25,.0625,.015625]:
        r,tan=node_coordinates(v,eps); As,Hs=means_by_class(r[support],tan[support],onehot,counts)
        direct_z=np.column_stack([r,eps*tan]); direct_anchor=np.column_stack([As,eps*Hs])
        D=stable_scores(v,r,As,Hs,eps)
        canonical=direct_z@(direct_anchor/np.linalg.norm(direct_anchor,axis=1,keepdims=True)).T
        identity_error=max(identity_error,float(np.max(abs(rowcentered(canonical)-.5*eps*eps*rowcentered(D)))))
        for beta in BETAS:
            for branch in ['fixed','scaled']:
                tau=TAU0 if branch=='fixed' else TAU0*eps*eps
                A,H=As.copy(),Hs.copy(); direct=direct_anchor.copy()
                for T in range(1,4):
                    A,H,ua,uh,mass,logmass,qzero=update(vu,r[15:],tan[15:],A,H,As,Hs,eps,tau,beta)
                    q=softmax(direct_z[15:]@(direct/np.linalg.norm(direct,axis=1,keepdims=True)).T/tau)
                    direct=(1-beta)*direct_anchor+beta*(q.T@direct_z[15:])/q.sum(0)[:,None]
                    update_error=max(update_error,float(np.max(abs(direct-np.column_stack([A,eps*H])))))
                    Dout=stable_scores(vq,r[15:],A,H,eps)
                    if eps==.015625 and branch=='fixed' and beta==.5:
                        ga=r[15:].mean(); gh=tan[15:].mean(0)
                        commonA=(1-beta)*As+beta*ga; commonH=(1-beta)*Hs+beta*gh
                        Dc=stable_scores(vq,r[15:],commonA,commonH,eps)
                        gamma=beta if T==1 else beta*(1-beta)
                        pred=gamma*eps**4/TAU0*lead['fixed_score']
                        rel=norm(.5*eps*eps*(Dout-Dc)-pred)/norm(pred)
                        assert rel<.03,(T,rel)
                        final.append(dict(round=T,fixed_score_coefficient_relative_error=rel))
                    if eps==.0625 and branch=='fixed' and beta==1.:
                        pred=eps**(2*T+1)/TAU0**T*lead['powers'][T-1]
                        err=coordinate_metric(centered(A),centered(H),pred,eps,False)['relative_error']
                        scorepred=eps**(2*T+2)/TAU0**T*lead['unanchored_score'][T-1]
                        serr=norm(.5*eps*eps*rowcentered(Dout)-scorepred)/norm(scorepred)
                        assert err<.1 and serr<.1,(T,err,serr)
                        unanchored.append(dict(round=T,prototype_relative_error=err,score_relative_error=serr))
                    if eps==.015625 and branch=='scaled':
                        lim=limits[beta][T-1]
                        assert norm(H-lim['H'])/max(norm(lim['H']),1e-15)<.03
    assert identity_error<2e-14
    assert update_error<2e-10
    # Arbitrary nonnegative graph propagation commutes with the feature map.
    X=rng.uniform(.1,1,size=(11,7)); W=rng.uniform(0,1,size=(11,11)); W/=W.sum(1,keepdims=True)
    e=np.ones(7)/np.sqrt(7); transform=lambda x:(x@e)[:,None]*e+.5*(x-(x@e)[:,None]*e)
    commute=np.max(abs(transform(W@X)-W@transform(X)))
    assert commute<1e-14
    Y=W@X; axis=Y@e; vv=Y/axis[:,None]-e
    rr,tt=node_coordinates(vv,1.); reconstructed=rr[:,None]*e+tt
    assert np.max(abs(reconstructed-Y/np.linalg.norm(Y,axis=1,keepdims=True)))<1e-14
    renlead=ren_leading_terms(vq,m,lead,len(vu),counts); renchecks=[]; renidentity=0.
    for eps in [1.,.25,.0625,.015625]:
        r,tan=node_coordinates(v,eps); rho=radial_loss(v,r)
        Bs=onehot.T@rho[support]/counts; Hs=onehot.T@tan[support]/counts[:,None]
        ub,uh,mass,lmass,qzero=ren_statistics(vu,r[15:],tan[15:],Bs,Hs,eps)
        gb=rho[15:].mean(); gh=tan[15:].mean(0)
        for rep in [1,eps**-2]:
            B,H,gamma=mass_anchored_coordinates(Bs,Hs,ub,uh,lmass,counts,rep)
            D=stable_euclidean_scores(vq,r[15:],B,H,eps)
            zp=np.column_stack([r[15:],eps*tan[15:]])
            mp=np.column_stack([1-eps*eps*B,eps*H])
            canonical=-((zp[:,None,:]-mp[None,:,:])**2).sum(2)
            renidentity=max(renidentity,float(np.max(abs(rowcentered(canonical)-eps*eps*rowcentered(D)))))
            if eps==.015625 and rep>1:
                proto=coordinate_metric(-eps*eps*(B-gb),H-gh,
                    eps**3*renlead['full_prototype_coefficient'],eps,False)
                sp=eps**4*renlead['full_limit_score']
                scoreerr=norm(eps*eps*rowcentered(D)-sp)/norm(sp)
                assert proto['relative_error']<.03 and scoreerr<.03
                renchecks.append(dict(epsilon=eps,prototype_relative_error=proto['relative_error'],score_relative_error=scoreerr))
        if eps==1.:
            rep=4; zp=np.column_stack([r[15:],tan[15:]]); ap=np.column_stack([1-Bs,Hs])
            repeated=np.tile(zp,(rep,1))
            q=softmax(-((repeated[:,None,:]-ap[None,:,:])**2).sum(2))
            explicit=(counts[:,None]*ap+q.T@repeated)/(counts+q.sum(0))[:,None]
            B,H,_=mass_anchored_coordinates(Bs,Hs,ub,uh,lmass,counts,rep)
            assert np.max(abs(explicit-np.column_stack([1-B,H])))<1e-13
    assert renidentity<2e-14
    p,ma,lma,qzero=responsibility_columns(np.array([[-10000.,0.],[-10001.,0.]]))
    assert ma[0]==0 and int(qzero)==2 and np.isfinite(lma).all()
    assert np.max(abs(p[:,0]-np.array([1/(1+np.exp(-1)),1/(1+np.exp(1))])))<1e-12
    report=dict(status='passed',scope='Small NumPy pure operators only; no real graph/GPU',
        score_identity_max_error=identity_error,canonical_update_max_error=update_error,
        commutation_max_error=float(commute),coefficient_checks=final,unanchored_coefficient_checks=unanchored,
        ren_score_identity_max_error=renidentity,ren_joint_coefficient_checks=renchecks,
        extreme_logdomain_check='zero represented mass, finite log mass and correct column mean')
    save_json(ROOT/'reports/concentration_scaling_r24/local_operator_selfcheck.json',report)
    print(json.dumps(report,indent=2))


def tick():
    torch.cuda.synchronize(); return time.perf_counter()


def run(contract_path):
    global torch
    frozen=json.loads(Path(contract_path).read_text())
    assert frozen==payload(),'Contract/source/design mismatch'
    contract_hash=sha(contract_path)
    import torch
    import torch.nn.functional as F
    import torch_geometric
    import expanded_data as data
    import run_study as study
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    assert data.ALL==DATASETS
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=dict(torch=torch.__version__,numpy=np.__version__,pyg=torch_geometric.__version__,
        sources={str(inspect.getsourcefile(f)):sha(inspect.getsourcefile(f)) for f in [NormalizeFeatures,gcn_norm]})
    save_json(OUT/'process.json',dict(pid=os.getpid(),started=time.time(),runtime=runtime,contract_sha256=contract_hash))
    archive=ROOT/'provenance/runner_versions/r24'; archive.mkdir(parents=True,exist_ok=True)
    for p in [Path(__file__),DESIGN]+[ROOT/p for p in frozen['dependencies'] if p.endswith(('.py','.md'))]:
        dest=archive/(sha(p)+'__'+p.name)
        if not dest.exists(): shutil.copy2(p,dest)
        assert sha(dest)==sha(p)
    completed=[]
    with torch.no_grad():
        for name in DATASETS:
            folder=OUT/name; folder.mkdir(exist_ok=True)
            graph_start=tick(); d,x,adj,pool,query,C,protocol=data.load_data(name)
            x=x.double(); adj=adj.double(); Y=adj@(adj@x)
            original=F.normalize(Y,dim=1); e=torch.ones(x.shape[1],device='cuda',dtype=torch.float64)/math.sqrt(x.shape[1])
            axis=Y@e; source=ROOT/'results/attribute_mechanism_r13'/name/f'{SEED}__graph_anchor'/'prediction.npz'
            geometry=json.loads((source.parent.parent/'geometry.json').read_text())
            assert study.thash(x)==geometry['input_sha256'] and study.thash(d.edge_index)==geometry['graph_sha256']
            hashes=dict(input=study.thash(x),raw_input=study.thash(d.x),graph=study.thash(d.edge_index),
                adjacency_indices=study.thash(adj.indices()),adjacency_values=study.thash(adj.values()),
                propagated=study.thash(Y),representation=study.thash(original),query=study.thash(query),
                pool=study.thash(pool),source_sha256=sha(source))
            common_metadata=dict(dataset=name,seed=SEED,protocol=protocol,contract_sha256=contract_hash,
                design_sha256=frozen['design_sha256'],runner_sha256=frozen['runner_sha256'],
                dependencies=frozen['dependencies'],runtime=runtime,input_hashes=hashes)
            invalid=(~axis.isfinite())|(axis<=1e-14)
            if bool(invalid.any()):
                save_npz(folder/'input_family_failure.npz',dict(invalid_indices=arr(torch.where(invalid)[0]),axis_mass=arr(axis)))
                failure=dict(common_metadata,status='inapplicable_input_family',invalid_count=int(invalid.sum()),
                    axis_mass_min=float(axis.min()),reason='At least one node has nonfinite or <=1e-14 axis mass; no nodes removed',
                    artifact_sha256=sha(folder/'input_family_failure.npz'))
                save_json(folder/'done.json',failure); completed.append(failure); continue
            v=Y/axis[:,None]-e
            tangent_error=float((v@e).abs().max())
            R=float(v.norm(dim=1).max())
            assert tangent_error<=1e-10*max(1.,R),'Input tangent error exceeds frozen tolerance'
            # Check exchange with propagation once, after NormalizeFeatures.
            xx=x@e; Xhalf=xx[:,None]*e+.5*(x-xx[:,None]*e)
            left=adj@(adj@Xhalf); right=axis[:,None]*e+.5*(Y-axis[:,None]*e)
            commutation_error=float((left-right).abs().max())
            assert torch.allclose(left,right,atol=1e-12,rtol=1e-10)
            with np.load(source) as old:
                support=old['support'].copy(); old_anchor=old['anchor'].copy()
                assert np.array_equal(query.numpy(),old['query'])
            si=torch.tensor(support,device='cuda'); qi=query.cuda()
            assert not torch.isin(si,qi).any()
            # Only support labels are used. The historical loader determines C/pool.
            sy=d.y[support].cuda(); onehot=F.one_hot(sy,C).double(); counts=onehot.sum(0)
            assert bool((counts==5).all())
            mask=torch.ones(len(v),dtype=torch.bool,device='cuda'); mask[si]=False
            vu=v[mask]; vq=v[qi]
            m=onehot.T@v[si]/counts[:,None]
            geometry_seconds=tick()-graph_start
            start=tick(); lead=leading_terms(vu,vq,m)
            limits={b:tangent_limit(vu,vq,m,b) for b in BETAS}
            ren_lead=ren_leading_terms(vq,m,lead,len(vu),counts)
            prediction_seconds=tick()-start
            prep=dict(support=support,support_labels=arr(sy),support_counts=arr(counts),query=query.numpy(),
                support_v=arr(v[si]),query_v=arr(vq),axis=e.cpu().numpy(),support_m=arr(m),mean_v=arr(lead['mean']),
                class_contrast=arr(lead['B']),fixed_score_coefficient=arr(lead['fixed_score']),
                tangent_common_D=arr(lead['tangent_common_D']))
            for t in range(3):
                prep[f'covariance_power_{t+1}']=arr(lead['powers'][t])
                prep[f'unanchored_score_coefficient_{t+1}']=arr(lead['unanchored_score'][t])
            for beta in BETAS:
                for t,entry in enumerate(limits[beta],1):
                    for key,value in entry.items(): prep[f'limit_b{beta:g}_t{t}_{key}']=arr(value)
            for key,value in ren_lead.items(): prep['ren_'+key]=arr(value)
            pre_npz=folder/f'{SEED}.pre.npz'; pre_json=folder/f'{SEED}.pre.json'
            pre_info=dict(common_metadata,phase='before all actual epsilon trajectories; tangent limit already uses node softmax',
                coordinate_sha256=study.thash(v),support_sha256=study.thash(si),radius=R,
                tangent_error=tangent_error,commutation_epsilon=.5,commutation_max_error=commutation_error,
                geometry_seconds=geometry_seconds,prediction_seconds=prediction_seconds,
                n_nodes=len(v),n_unlabeled=len(vu),classes=C,one_support_draw_per_graph=True)
            if pre_json.exists():
                saved=json.loads(pre_json.read_text())
                for key in ['contract_sha256','input_hashes','runtime','coordinate_sha256','support_sha256']:
                    assert saved[key]==pre_info[key],('Preseal metadata mismatch',key)
                assert saved['artifact_sha256']==sha(pre_npz)
                with np.load(pre_npz) as old:
                    assert set(old.files)==set(prep)
                    for key,value in prep.items(): assert np.array_equal(old[key],value),('Preseal array mismatch',key)
            else:
                save_npz(pre_npz,prep); save_json(pre_json,dict(pre_info,artifact_sha256=sha(pre_npz),sealed_at=time.time()))
            pre_hash=sha(pre_npz); pre_record_hash=sha(pre_json)
            graph_rows=[]; ren_rows=[]
            for k in KS:
                eps=2.**(-k); radial,tangent=node_coordinates(v,eps)
                ru,tu=radial[mask],tangent[mask]; rq=radial[qi]
                As,Hs=means_by_class(radial[si],tangent[si],onehot,counts)
                ga,gh=mean_(ru),mean_(tu)
                domain=eps*R<=.1 and eps*eps*R*R/TAU0<=.1
                init_metric=coordinate_metric(centered(As),centered(Hs),eps*lead['B'],eps,domain)
                # Separate, one-step Ren Euclidean operator. No trajectory shares
                # its assignments with the cosine branches below.
                ren_start=tick()
                rho=radial_loss(v,radial); Bs=(onehot.T@rho[si])/counts; gb=mean_(rho[mask])
                ub,uh,rmass,rlogmass,rqzero=ren_statistics(vu,ru,tu,Bs,Hs,eps)
                ren_assignment_seconds=tick()-ren_start
                for policy in ['one','inverse_epsilon_squared']:
                    rep=1 if policy=='one' else 4**k
                    tag=f'ren__k{k:02d}__{policy}'; artifact=folder/(tag+'.npz'); record=folder/(tag+'.json')
                    if record.exists():
                        saved=json.loads(record.read_text())
                        assert saved['contract_sha256']==contract_hash and saved['pre_artifact_sha256']==pre_hash
                        assert saved['pre_record_sha256']==pre_record_hash and saved['artifact_sha256']==sha(artifact)
                        ren_rows.append(saved); continue
                    begin=tick()
                    B,H,gamma=mass_anchored_coordinates(Bs,Hs,ub,uh,rlogmass,counts,rep)
                    Bc,Hc,gamma_c=mass_anchored_coordinates(Bs,Hs,gb,gh,rlogmass,counts,rep)
                    assert bool((gamma==gamma_c).all())
                    D=stable_euclidean_scores(vq,rq,B,H,eps)
                    Dc=stable_euclidean_scores(vq,rq,Bc,Hc,eps)
                    pack=dict(epsilon=np.array(eps),replication=np.array(rep),anchor_B=arr(Bs),anchor_H=arr(Hs),
                        full_B=arr(B),full_A=arr(1-eps*eps*B),full_H=arr(H),full_D=arr(D),
                        common_B=arr(Bc),common_A=arr(1-eps*eps*Bc),common_H=arr(Hc),common_D=arr(Dc),
                        soft_B=arr(ub),soft_H=arr(uh),mean_B=arr(gb),mean_H=arr(gh),
                        class_mass=arr(rmass),log_class_mass=arr(rlogmass),support_weight=arr(gamma))
                    metrics=dict(soft_position=coordinate_metric(-eps*eps*(ub-gb),uh-gh,
                        2*eps**3*lead['powers'][0],eps,domain))
                    diag=dict(full_vs_true_mass_common=ranking(D,Dc,eps*eps))
                    if policy=='inverse_epsilon_squared':
                        metrics['full_vs_mean']=coordinate_metric(-eps*eps*(B-gb),H-gh,
                            eps**3*ren_lead['full_prototype_coefficient'],eps,domain)
                        metrics['common_vs_mean']=coordinate_metric(-eps*eps*(Bc-gb),Hc-gh,
                            eps**3*ren_lead['common_prototype_coefficient'],eps,domain)
                        full_score=eps*eps*rowcentered(D); common_score=eps*eps*rowcentered(Dc)
                        metrics['centered_score']=matrix_metric(full_score,eps**4*ren_lead['full_limit_score'],domain,4.)
                        metrics['common_centered_score']=matrix_metric(common_score,eps**4*ren_lead['common_limit_score'],domain,4.)
                        metrics['support_weight_scaled']=matrix_metric((gamma/eps**2)[:,None],
                            gamma[:,None]*0+ren_lead['kappa'],False)
                        diag['full_vs_joint_limit']=ranking(D,eps*eps*ren_lead['full_limit_score'],eps*eps)
                        diag['common_vs_joint_limit']=ranking(Dc,eps*eps*ren_lead['common_limit_score'],eps*eps)
                        diag['joint_limit_full_vs_common']=ranking(ren_lead['full_limit_score'],ren_lead['common_limit_score'],eps**4)
                        pack.update(physical_centered_score=arr(full_score),physical_common_centered_score=arr(common_score))
                    else:
                        metrics['tangent_H_to_fixed_replication_limit']=matrix_metric(H,ren_lead['fixed_replication_tangent_center'],False)
                        metrics['centered_D_to_fixed_replication_limit']=matrix_metric(rowcentered(D),
                            rowcentered(ren_lead['fixed_replication_limit_D']),False)
                        diag['full_vs_fixed_replication_limit']=ranking(D,ren_lead['fixed_replication_limit_D'],eps*eps)
                        diag['common_vs_fixed_replication_limit']=ranking(Dc,ren_lead['fixed_replication_limit_D'],eps*eps)
                    if k==0:
                        full=(1-eps*eps*B)[:,None]*e+eps*H
                        canonical=2*original[qi]@full.T-original[qi].square().sum(1)[:,None]-full.square().sum(1)[None,:]
                        pack['canonical_negative_squared_euclidean']=arr(canonical)
                        diag['canonical_euclidean_vs_stable_D']=ranking(canonical,eps*eps*D,1.)
                        diag['canonical_centered_identity_error']=float((rowcentered(canonical)-eps*eps*rowcentered(D)).abs().max())
                    row=dict(common_metadata,operator='Ren fixed-attribute one-step Euclidean inference',
                        experiment_role='development intervention, not holdout',epsilon=eps,k=k,
                        replication_policy=policy,replication=rep,rounds=1,temperature=1.,
                        radius=R,fixed_temperature_local_domain=domain,metrics=metrics,ranking=diag,
                        pre_artifact_sha256=pre_hash,pre_record_sha256=pre_record_hash,
                        assignment_seconds_shared_across_replications=ren_assignment_seconds,
                        branch_seconds=tick()-begin,q_probability_zero_count=int(rqzero),
                        logmass_min=float(rlogmass.min()),mass_underflow_count=int((rmass==0).sum()),status='complete')
                    serial=time.perf_counter(); save_npz(artifact,pack)
                    row['serialization_seconds']=time.perf_counter()-serial; row['artifact_sha256']=sha(artifact)
                    save_json(record,row); ren_rows.append(row)
                for branch in ['fixed','scaled']:
                    tau=TAU0 if branch=='fixed' else TAU0*eps*eps
                    for beta in BETAS:
                        tag=f'k{k:02d}__{branch}__b{beta:g}'
                        artifact=folder/(tag+'.npz'); record=folder/(tag+'.json')
                        if record.exists():
                            saved=json.loads(record.read_text())
                            assert saved['contract_sha256']==contract_hash and saved['pre_artifact_sha256']==pre_hash
                            assert saved['pre_record_sha256']==pre_record_hash and saved['artifact_sha256']==sha(artifact)
                            graph_rows.append(saved); continue
                        save_json(OUT/'current.json',dict(dataset=name,condition=tag,completed_graphs=len(completed),time=time.time()))
                        start=tick(); A,H=copy_(As),copy_(Hs)
                        Ac=(1-beta)*As+beta*ga; Hc=(1-beta)*Hs+beta*gh
                        Dc=stable_scores(vq,rq,Ac,Hc,eps)
                        package=dict(epsilon=np.array(eps),anchor_A=arr(As),anchor_H=arr(Hs),
                            common_A=arr(Ac),common_H=arr(Hc),common_D=arr(Dc),mean_A=arr(ga),mean_H=arr(gh))
                        rounds={}; replay={}
                        for t in range(1,4):
                            update_start=tick()
                            A,H,ua,uh,mass,logmass,qzero=update(vu,ru,tu,A,H,As,Hs,eps,tau,beta)
                            update_seconds=tick()-update_start
                            score_start=tick(); D=stable_scores(vq,rq,A,H,eps)
                            score_seconds=tick()-score_start
                            for key,value in dict(A=A,H=H,soft_A=ua,soft_H=uh,mass=mass,logmass=logmass,D=D).items():
                                package[f't{t}_{key}']=arr(value)
                            metrics={'initial_contrast':init_metric}
                            diag=dict(full_vs_common=ranking(D,Dc,.5*eps*eps))
                            if branch=='fixed':
                                if beta==.5:
                                    factor=1. if t==1 else (1-beta); gamma=beta*factor
                                    feedback_prediction=eps**3/TAU0*factor*lead['powers'][0]
                                    proto_prediction=eps**3/TAU0*gamma*lead['powers'][0]
                                    score_prediction=gamma*eps**4/TAU0*lead['fixed_score']
                                    metrics['feedback']=coordinate_metric(ua-ga,uh-gh,feedback_prediction,eps,domain)
                                    metrics['prototype_vs_common']=coordinate_metric(A-Ac,H-Hc,proto_prediction,eps,domain)
                                    physical_difference=.5*eps*eps*(D-Dc)
                                    metrics['score_vs_common']=matrix_metric(physical_difference,score_prediction,domain)
                                    package[f't{t}_physical_score_difference']=arr(physical_difference)
                                    diag['common_vs_tangent_nearest_center']=ranking(Dc,lead['tangent_common_D'],.5*eps*eps)
                                    diag['full_vs_predicted_corrected_common']=ranking(D,Dc+2*score_prediction/(eps*eps),.5*eps*eps)
                                else:
                                    proto_prediction=eps**(2*t+1)/TAU0**t*lead['powers'][t-1]
                                    score_prediction=eps**(2*t+2)/TAU0**t*lead['unanchored_score'][t-1]
                                    metrics['class_contrast']=coordinate_metric(centered(A),centered(H),proto_prediction,eps,domain)
                                    physical_centered=.5*eps*eps*rowcentered(D)
                                    metrics['centered_score']=matrix_metric(physical_centered,score_prediction,domain)
                                    package[f't{t}_physical_centered_score']=arr(physical_centered)
                                    diag['full_vs_leading_margin_rule']=ranking(D,2*score_prediction/(eps*eps),.5*eps*eps)
                            else:
                                limit=limits[beta][t-1]
                                metrics['tangent_prototype_to_limit']=matrix_metric(H,limit['H'],False)
                                metrics['centered_D_to_limit']=matrix_metric(rowcentered(D),rowcentered(limit['D']),False)
                                diag['full_vs_frozen_tangent_limit']=ranking(D,limit['D'],.5*eps*eps)
                            if k==0:
                                # Explicit original-space comparison; record even near-tie disagreements.
                                full=A[:,None]*e+eps*H
                                canonical=original[qi]@F.normalize(full,dim=1).T
                                package[f't{t}_canonical_cosine']=arr(canonical)
                                diag['canonical_cosine_vs_stable_D']=ranking(canonical, .5*eps*eps*D,1.)
                                diag['canonical_score_identity_centered_max_error']=float((rowcentered(canonical)-.5*eps*eps*rowcentered(D)).abs().max())
                                if beta==.5 and t==3:
                                    # Both temperature branches equal .05 at epsilon=1.
                                    with np.load(source) as old: old_prototype=old['prototype'].copy()
                                    actual_anchor=As[:,None]*e+Hs
                                    replay=dict(anchor_max_error=float(np.max(abs(arr(actual_anchor)-old_anchor))),
                                        prototype_max_error=float(np.max(abs(arr(full)-old_prototype))),
                                        anchor_allclose=bool(np.allclose(arr(actual_anchor),old_anchor,atol=1e-12,rtol=1e-10)),
                                        prototype_allclose=bool(np.allclose(arr(full),old_prototype,atol=1e-10,rtol=1e-8)))
                                    assert replay['anchor_allclose'] and replay['prototype_allclose'],replay
                            rounds[str(t)]=dict(metrics=metrics,ranking=diag,update_seconds=update_seconds,
                                score_seconds=score_seconds,mass_min=float(mass.min()),mass_max=float(mass.max()),
                                logmass_min=float(logmass.min()),mass_underflow_count=int((mass==0).sum()),
                                q_probability_zero_count=int(qzero))
                        calculation_seconds=tick()-start
                        row=dict(common_metadata,epsilon=eps,k=k,temperature=branch,tau=tau,beta=beta,
                            radius=R,epsilon_radius=eps*R,epsilon_squared_radius_squared_over_tau0=eps*eps*R*R/TAU0,
                            fixed_temperature_local_domain=domain,rounds=rounds,r13_replay=replay,
                            pre_artifact_sha256=pre_hash,pre_record_sha256=pre_record_hash,
                            calculation_seconds=calculation_seconds,status='complete')
                        serialize_start=time.perf_counter(); save_npz(artifact,package)
                        row['serialization_seconds']=time.perf_counter()-serialize_start
                        row['artifact_sha256']=sha(artifact); save_json(record,row)
                        graph_rows.append(row)
            fit=slopes(graph_rows); save_json(folder/'slopes.json',dict(contract_sha256=contract_hash,fits=fit))
            save_json(folder/'ren_slopes.json',dict(contract_sha256=contract_hash,fits=ren_slopes(ren_rows)))
            done=dict(common_metadata,status='complete',conditions=len(graph_rows),expected_conditions=len(KS)*2*len(BETAS),
                ren_conditions=len(ren_rows),expected_ren_conditions=len(KS)*2,
                pre_artifact_sha256=pre_hash,pre_record_sha256=pre_record_hash,
                graph_seconds=tick()-graph_start,domain_points=[k for k in KS if 2.**(-k)*R<=.1 and 2.**(-2*k)*R*R/TAU0<=.1],
                slopes_sha256=sha(folder/'slopes.json'),ren_slopes_sha256=sha(folder/'ren_slopes.json'))
            assert done['conditions']==done['expected_conditions']
            assert done['ren_conditions']==done['expected_ren_conditions']
            save_json(folder/'done.json',done); completed.append(done)
            del d,x,adj,Y,original,v,vu,vq
            torch.cuda.empty_cache()
    save_json(OUT/'done.json',dict(status='complete',datasets=completed,contract_sha256=contract_hash,finished=time.time()))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--freeze-only',action='store_true'); parser.add_argument('--contract',type=Path,default=CONTRACT)
    args=parser.parse_args()
    if args.self_check: selfcheck(); return
    if args.freeze_only:
        proposed=payload()
        if args.contract.exists(): assert json.loads(args.contract.read_text())==proposed,'Existing contract differs; do not overwrite'
        else: save_json(args.contract,proposed)
        print(args.contract); return
    if not args.contract.exists(): raise SystemExit('Freeze reviewed design/runner with --freeze-only before any real run')
    run(args.contract)


if __name__=='__main__': main()
