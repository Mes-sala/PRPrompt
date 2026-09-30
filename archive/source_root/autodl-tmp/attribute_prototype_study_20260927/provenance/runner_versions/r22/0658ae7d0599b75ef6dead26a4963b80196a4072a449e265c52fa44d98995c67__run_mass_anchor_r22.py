"""R22: fixed-feature Ren inference operators, mass controls and replication.

No SSH, training, query labels, or accuracy. Freeze first, then execute in ProG.
The small NumPy self-check imports no torch and starts no experiments.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/mass_anchor_r22'
DESIGN = ROOT / 'design/R22_质量锚定与重复节点池_算子验证.md'
DEFAULT_CONTRACT = ROOT / 'provenance/r22_frozen_contract.json'
DATASETS = ['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon',
            'Cornell','Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings']
SEEDS = list(range(271700, 271705))
REPLICATIONS = [1, 2, 4, 8]
PRE_METHODS = ['common_mass0','moment_mass0','moment_mass2','common_equalmass']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [finite_json(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with open(temp, 'w') as file:
        json.dump(finite_json(value), file, ensure_ascii=False, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def save_npz(path, package):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with open(temp, 'wb') as file:
        np.savez_compressed(file, **package)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def contract_payload():
    dependencies = ['code/expanded_data.py', 'code/run_study.py',
                    'provenance/expanded_datasets.json',
                    'provenance/expanded_heterophily_datasets.json']
    return dict(schema='mass-anchor-r22-v1',runner_sha256=sha(__file__),
        design_sha256=sha(DESIGN),dependencies={p:sha(ROOT/p) for p in dependencies},
        datasets=DATASETS,seeds=SEEDS,replications=REPLICATIONS,
        ren=dict(distance='squared_euclidean',temperature=1.,rounds=1,
                 normalize_prototype=False,readout='negative_squared_euclidean'),
        controls=dict(fixed_beta=.5,rounds=[1,3],cosine_tau=.05),
        local_eta_threshold=.1,pre_methods=PRE_METHODS)


def softmax(logits, exp, axis):
    # NumPy 2 arrays also have a .device attribute; test the actual array type.
    maximum = logits.max(axis=axis,keepdims=True) if isinstance(logits,np.ndarray) else logits.max(dim=axis,keepdim=True).values
    shifted = logits-maximum
    weights = exp(shifted)
    total = weights.sum(axis=axis,keepdims=True) if isinstance(weights,np.ndarray) else weights.sum(dim=axis,keepdim=True)
    return weights/total


def euclidean_scores(nodes, prototype):
    return 2*nodes@prototype.T-(nodes*nodes).sum(1)[:,None]-(prototype*prototype).sum(1)[None,:]


def pool_prototype(anchor, soft, mass, counts, replication):
    scaled = replication*mass
    return (counts[:,None]*anchor+scaled[:,None]*soft)/(counts+scaled)[:,None]


def moment_terms(anchor, mean, covariance, size, exp):
    # Exact logits after removing the node-common -||z_i||^2 term.
    v = 2*anchor
    pi = softmax(v@mean-(anchor*anchor).sum(1), exp, axis=0)
    contrast = v-pi@v
    linear = contrast@covariance
    variance = (linear*contrast).sum(1)
    mass0 = size*pi
    mass2 = mass0*(1+.5*(variance-pi@variance))
    return dict(pi=pi,contrast=contrast,soft=mean+linear,mass0=mass0,
                mass2=mass2,projected_variance=variance)


def numpy_selfcheck():
    """Independent explicit duplication and small-dispersion order checks."""
    rng = np.random.default_rng(270922)
    anchor = rng.normal(size=(4,6))*.12
    counts = np.array([5.,5.,5.,5.])
    mean = rng.normal(size=6)*.12
    residual = rng.normal(size=(91,6)); residual -= residual.mean(axis=0)
    rows = []
    for eps in [.1,.03,.01,.003]:
        z = mean+eps*residual
        g = z.mean(0); cov = (z-g).T@(z-g)/len(z)
        approx = moment_terms(anchor,g,cov,len(z),np.exp)
        q = softmax(euclidean_scores(z,anchor),np.exp,1)
        mass=q.sum(0); u=q.T@z/mass[:,None]
        rows.append(dict(epsilon=eps,mass0_error=float(np.linalg.norm(approx['mass0']-mass)),
            mass2_error=float(np.linalg.norm(approx['mass2']-mass)),
            position_error=float(np.linalg.norm(approx['soft']-u))))
    assert rows[-1]['mass2_error'] < rows[-1]['mass0_error']/10
    assert rows[-1]['position_error'] < rows[-2]['position_error']/15
    max_replication_error=0.
    z=mean+.1*residual
    q=softmax(euclidean_scores(z,anchor),np.exp,1); mass=q.sum(0); u=q.T@z/mass[:,None]
    for replication in REPLICATIONS:
        repeated=np.tile(z,(replication,1))
        qr=softmax(euclidean_scores(repeated,anchor),np.exp,1)
        mr=qr.sum(0); ur=qr.T@repeated/mr[:,None]
        full=pool_prototype(anchor,ur,mr,counts,1)
        algebraic=pool_prototype(anchor,u,mass,counts,replication)
        max_replication_error=max(max_replication_error,float(np.max(abs(full-algebraic))))
        a,b=anchor.copy(),anchor.copy()
        for _ in range(3):
            qa=softmax(euclidean_scores(z,a),np.exp,1)
            qb=softmax(euclidean_scores(repeated,b),np.exp,1)
            a=.5*anchor+.5*(qa.T@z/qa.sum(0)[:,None])
            b=.5*anchor+.5*(qb.T@repeated/qb.sum(0)[:,None])
            assert np.max(abs(a-b))<1e-12
    assert max_replication_error<1e-12
    brute=-((z[:,None,:]-anchor[None,:,:])**2).sum(2)
    assert np.max(abs(brute-euclidean_scores(z,anchor)))<1e-12
    rec=dict(status='passed',scope='NumPy operator algebra only; no real-data or GPU run',
        mass_position_orders=rows,max_replication_error=max_replication_error)
    save_json(ROOT/'reports/mass_anchor_r22/local_operator_selfcheck.json',rec)
    print(json.dumps(rec,indent=2))


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def arr(x):
    return x.detach().cpu().numpy()


def exact_statistics(nodes, prototype, kind):
    logits = euclidean_scores(nodes,prototype) if kind=='euclidean' else nodes@F.normalize(prototype,dim=1).T/.05
    q=softmax(logits,torch.exp,1)
    mass=q.sum(0); numerator=q.T@nodes
    if not bool(torch.isfinite(q).all()) or float(mass.min())<=0:
        raise ValueError('Invalid exact soft assignments or class mass')
    return numerator/mass[:,None],mass,numerator,int((q==0).sum())


def invariance_from_statistics(soft,mass,numerator):
    errors={str(rep):float(((rep*numerator)/(rep*mass)[:,None]-soft).abs().max()) for rep in REPLICATIONS}
    assert max(errors.values())<1e-10
    return errors


def score_diagnostics(scores,reference):
    error=scores-reference
    centered=error-error.mean(1,keepdim=True)
    tops=scores.topk(2,dim=1).values
    ref_tops=reference.topk(2,dim=1).values
    changed=scores.argmax(1)!=reference.argmax(1)
    scale=torch.maximum(torch.ones_like(tops[:,0]),scores.abs().max(1).values)
    ref_scale=torch.maximum(torch.ones_like(ref_tops[:,0]),reference.abs().max(1).values)
    near=(tops[:,0]-tops[:,1]<=1e-10*scale)|(ref_tops[:,0]-ref_tops[:,1]<=1e-10*ref_scale)
    return dict(disagreement_pct=float(changed.double().mean()*100),
        centered_score_rmse=float(centered.square().mean().sqrt()),
        max_pairwise_margin_error=float((error.max(1).values-error.min(1).values).max()),
        near_tie_either_pct=float(near.double().mean()*100),
        disagreement_outside_near_ties_pct=float((changed&~near).double().mean()*100))


def explicit_replication_check(nodes,anchor,counts):
    small=nodes[:128]; repeated=small.repeat(8,1)
    u,m,num,_=exact_statistics(small,anchor,'euclidean')
    ur,mr,_,_=exact_statistics(repeated,anchor,'euclidean')
    ren_error=float((pool_prototype(anchor,ur,mr,counts,1)-pool_prototype(anchor,u,m,counts,8)).abs().max())
    result=dict(nodes=len(small),replication=8,ren_max_error=ren_error,
                subset_sha256=r.thash(small),anchor_sha256=r.thash(anchor))
    assert ren_error<1e-10
    for kind in ['euclidean','cosine']:
        a,b=anchor.clone(),anchor.clone(); errors=[]
        for _ in range(3):
            ua,_,_,_=exact_statistics(small,a,kind)
            ub,_,_,_=exact_statistics(repeated,b,kind)
            a=.5*anchor+.5*ua; b=.5*anchor+.5*ub
            errors.append(float((a-b).abs().max()))
        assert max(errors)<1e-10
        result[kind+'_fixed_beta_errors']=errors
    return result


def run(contract_path):
    global torch,F,data,r
    frozen=json.loads(Path(contract_path).read_text())
    assert frozen==contract_payload(),'Source or design changed after freeze'
    contract_hash=sha(contract_path)
    import torch
    import torch.nn.functional as F
    import expanded_data as data
    import run_study as r
    import torch_geometric
    import inspect
    assert data.ALL==DATASETS
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=dict(torch=torch.__version__,numpy=np.__version__,pyg=torch_geometric.__version__)
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    runtime['library_sources']={str(inspect.getsourcefile(obj)):sha(inspect.getsourcefile(obj)) for obj in [NormalizeFeatures,gcn_norm]}
    save_json(OUT/'process.json',dict(pid=os.getpid(),started=time.time(),contract_sha256=contract_hash,runtime=runtime))
    archive=ROOT/'provenance/runner_versions/r22'; archive.mkdir(parents=True,exist_ok=True)
    for source in [Path(__file__),DESIGN]+[ROOT/p for p in frozen['dependencies'] if p.endswith('.py')]:
        target=archive/(sha(source)+'__'+source.name)
        if not target.exists(): shutil.copy2(source,target)
        assert sha(target)==sha(source)
    count=0
    with torch.no_grad():
        for name in DATASETS:
            start=tick()
            d,x,adj,pool,query,C,protocol=data.load_data(name)
            x=x.double(); adj=adj.double(); z=F.normalize(adj@(adj@x),dim=1)
            load_representation_seconds=tick()-start
            old_geometry=json.loads((ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text())
            assert r.thash(x)==old_geometry['input_sha256'] and r.thash(d.edge_index)==old_geometry['graph_sha256']
            source_hashes=dict(raw_input=r.thash(d.x),input=r.thash(x),graph=r.thash(d.edge_index),
                adjacency_indices=r.thash(adj.indices()),adjacency_values=r.thash(adj.values()),
                representation=r.thash(z),query=r.thash(query),pool=r.thash(pool))
            start=tick(); zz=z.T@z; total=z.sum(0); covariance_precompute_seconds=tick()-start
            folder=OUT/name; folder.mkdir(exist_ok=True)
            for seed in SEEDS:
                stem=folder/str(seed); record=stem.with_suffix('.json'); artifact=stem.with_suffix('.npz')
                pre_record=folder/f'{seed}.pre.json'; pre_artifact=folder/f'{seed}.pre.npz'
                source=ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__graph_anchor'/'prediction.npz'
                if record.exists():
                    old=json.loads(record.read_text())
                    assert old['contract_sha256']==contract_hash and old['source_sha256']==sha(source)
                    assert old['input_hashes']==source_hashes and old['runtime']==runtime
                    assert old['artifact_sha256']==sha(artifact) and old['pre_artifact_sha256']==sha(pre_artifact)
                    assert old['pre_record_sha256']==sha(pre_record)
                    count+=1; continue
                save_json(OUT/'current.json',dict(dataset=name,seed=seed,completed=count,time=time.time()))
                draw_start=tick()
                with np.load(source) as old:
                    support=old['support'].copy(); old_anchor=old['anchor'].copy()
                    assert np.array_equal(query.numpy(),old['query'])
                start=tick(); si=torch.tensor(support,device='cuda'); sy=d.y[support].cuda()
                assert not torch.isin(si,query.cuda()).any()
                onehot=F.one_hot(sy,C).double(); counts=onehot.sum(0)
                assert bool((counts==5).all())
                anchor=onehot.T@z[si]/counts[:,None]
                assert np.allclose(arr(anchor),old_anchor,atol=1e-12,rtol=1e-10)
                mask=torch.ones(len(z),dtype=torch.bool,device='cuda'); mask[si]=False
                zu=z[mask]; n=len(zu); qz=z[query.cuda()]
                mean=(total-z[si].sum(0))/n
                covariance=(zz-z[si].T@z[si])/n-torch.outer(mean,mean)
                covariance=(covariance+covariance.T)/2
                radius=float((zu-mean).norm(dim=1).max())
                support_geometry_seconds=tick()-start
                start=tick(); moments=moment_terms(anchor,mean,covariance,n,torch.exp)
                moment_terms_seconds=tick()-start
                mass2_valid=bool(torch.isfinite(moments['mass2']).all() and (moments['mass2']>0).all())
                assert abs(float(moments['mass2'].sum())-n)<1e-8*max(1,n)
                eta=radius*float(moments['contrast'].norm(dim=1).max())
                package=dict(anchor=arr(anchor),support=support,support_labels=arr(sy),support_counts=arr(counts),
                    query=query.numpy(),mean=arr(mean),covariance_diagonal=arr(covariance.diag()),
                    mass0=arr(moments['mass0']),mass2=arr(moments['mass2']),mean_assignment=arr(moments['pi']),
                    moment_soft=arr(moments['soft']),projected_variance=arr(moments['projected_variance']),
                    query_norm_squared=arr(qz.square().sum(1)),query_anchor_dot=arr(qz@anchor.T),
                    query_mean_dot=arr(qz@mean),query_moment_soft_dot=arr(qz@moments['soft'].T))
                predicted={}; predicted_scores={}; branch_seconds={}; score_seconds={}
                invalid={} if mass2_valid else {'moment_mass2':'nonpositive or nonfinite second-order mass; no clipping'}
                for method in PRE_METHODS:
                    if method in invalid: continue
                    soft=moments['soft'] if method.startswith('moment_') else mean.expand_as(anchor)
                    mass=moments['mass2'] if method=='moment_mass2' else (torch.full_like(counts,n/C) if method=='common_equalmass' else moments['mass0'])
                    for rep in REPLICATIONS:
                        key=f'{method}__r{rep}'
                        start=tick(); mu=pool_prototype(anchor,soft,mass,counts,rep); branch_seconds[key]=tick()-start
                        start=tick(); scores=euclidean_scores(qz,mu); score_seconds[key]=tick()-start
                        assert bool(torch.isfinite(mu).all() and torch.isfinite(scores).all())
                        predicted[key]=mu; predicted_scores[key]=scores
                        package[key+'__prototype']=arr(mu); package[key+'__scores']=arr(scores)
                        package[key+'__mass']=arr(rep*mass)
                        package[key+'__support_weight']=arr(counts/(counts+rep*mass))
                start=tick(); initial_scores=euclidean_scores(qz,anchor); initial_score_seconds=tick()-start
                package['initial_euclidean__scores']=arr(initial_scores)
                metadata=dict(dataset=name,seed=seed,protocol=protocol,n_unlabeled=n,classes=C,replications=REPLICATIONS,
                    invalid=invalid,mass2_valid=mass2_valid,node_eta_upper=eta,predeclared_local_domain=eta<=.1,
                    node_radius=radius,covariance_trace=float(covariance.trace()),covariance_sha256=r.thash(covariance),
                    input_hashes=source_hashes,support_sha256=r.thash(si),source_sha256=sha(source),
                    runner_sha256=frozen['runner_sha256'],design_sha256=frozen['design_sha256'],
                    dependencies=frozen['dependencies'],contract_sha256=contract_hash,runtime=runtime,
                    branch_seconds=branch_seconds,score_seconds=score_seconds,moment_terms_seconds=moment_terms_seconds,
                    support_geometry_seconds=support_geometry_seconds,covariance_precompute_seconds=covariance_precompute_seconds,
                    load_representation_seconds=load_representation_seconds,initial_score_seconds=initial_score_seconds)
                # Persist all predictions before any full N-by-C softmax for this draw.
                serialization_start=time.perf_counter(); save_npz(pre_artifact,package)
                save_json(pre_record,dict(metadata,phase='before any full node softmax for this draw',
                    artifact_sha256=sha(pre_artifact),finished=time.time()))
                pre_serialization_seconds=time.perf_counter()-serialization_start
                pre_hash=sha(pre_artifact); pre_record_hash=sha(pre_record)
                start=tick(); full_soft,mass,numerator,zero_count=exact_statistics(zu,anchor,'euclidean')
                full_assignment_seconds=tick()-start
                package.update(full_soft=arr(full_soft),full_mass=arr(mass),full_numerator=arr(numerator),
                               query_full_soft_dot=arr(qz@full_soft.T))
                post={}; post_scores={}; post_branch_seconds={}; post_score_seconds={}
                for method,soft in [('full_ren',full_soft),('common_true_mass',mean.expand_as(anchor))]:
                    for rep in REPLICATIONS:
                        key=f'{method}__r{rep}'
                        start=tick(); mu=pool_prototype(anchor,soft,mass,counts,rep); post_branch_seconds[key]=tick()-start
                        start=tick(); scores=euclidean_scores(qz,mu); post_score_seconds[key]=tick()-start
                        post[key]=mu; post_scores[key]=scores
                        package[key+'__prototype']=arr(mu); package[key+'__scores']=arr(scores)
                        package[key+'__mass']=arr(rep*mass)
                        package[key+'__support_weight']=arr(counts/(counts+rep*mass))
                fixed_records={}; fixed_seconds={}; fixed_score_seconds={}; fixed_comparison_seconds={}
                for kind in ['euclidean','cosine']:
                    start=tick(); mu=anchor.clone(); trajectory=[mu.clone()]; softs=[]; masses=[]; inv=[]
                    for step in range(3):
                        if kind=='euclidean' and step==0:
                            soft,ms,num=full_soft,mass,numerator
                        else: soft,ms,num,_=exact_statistics(zu,mu,kind)
                        inv.append(invariance_from_statistics(soft,ms,num))
                        softs.append(soft.clone()); masses.append(ms.clone())
                        mu=.5*anchor+.5*soft; trajectory.append(mu.clone())
                    fixed_seconds[kind]=tick()-start
                    fixed_records[kind]=dict(replication_error_by_round=inv,
                        first_assignment_reuses_ren=(kind=='euclidean'),
                        timing_excludes_shared_ren_first_assignment=(kind=='euclidean'))
                    package[f'fixed_{kind}__trajectory']=arr(torch.stack(trajectory))
                    package[f'fixed_{kind}__soft']=arr(torch.stack(softs)); package[f'fixed_{kind}__mass']=arr(torch.stack(masses))
                    for step in [1,3]:
                        key=f'fixed_{kind}__step{step}'
                        start=tick()
                        scores=euclidean_scores(qz,trajectory[step]) if kind=='euclidean' else qz@F.normalize(trajectory[step],dim=1).T
                        fixed_score_seconds[key]=tick()-start; package[key+'__scores']=arr(scores)
                        # For cosine only compare argmax: the numerical score
                        # scale/readout differs and is not a matched score error.
                        comparison_start=tick()
                        if kind=='euclidean':
                            fixed_records[kind][f'step{step}_versus_ren_r1']=score_diagnostics(scores,post_scores['full_ren__r1'])
                        else:
                            fixed_records[kind][f'step{step}_ren_prediction_disagreement_pct']=float((scores.argmax(1)!=post_scores['full_ren__r1'].argmax(1)).double().mean()*100)
                        fixed_comparison_seconds[key]=tick()-comparison_start
                    if kind=='cosine':
                        with np.load(source) as old: old_full=old['prototype'].copy()
                        assert np.allclose(arr(mu),old_full,atol=1e-11,rtol=1e-8)
                        fixed_records[kind]['r13_replay_max_error']=float(np.max(abs(arr(mu)-old_full)))
                start=tick(); endpoints={}; intervention={}
                all_mu={**predicted,**post}; all_scores={**predicted_scores,**post_scores}
                for key,mu in all_mu.items():
                    method,rep_text=key.rsplit('__r',1); rep=int(rep_text); ref=post[f'full_ren__r{rep}']
                    endpoints[key]=dict(score_diagnostics(all_scores[key],post_scores[f'full_ren__r{rep}']),
                        prototype_max_error=float((mu-ref).norm(dim=1).max()),
                        prototype_frobenius_error=float((mu-ref).norm()),
                        actual_support_weight=arr(counts/(counts+rep*mass)).tolist(),
                        used_support_weight=package[key+'__support_weight'].tolist())
                    intervention[key]=dict(score_diagnostics(all_scores[key],all_scores[f'{method}__r1']),
                        prototype_change_from_r1=float((mu-all_mu[f'{method}__r1']).norm()))
                diagnostics=dict(soft_residual_from_mean_per_class=arr((full_soft-mean).norm(dim=1)).tolist(),
                    soft_location_error_per_class=arr((moments['soft']-full_soft).norm(dim=1)).tolist(),
                    soft_location_frobenius_error=float((moments['soft']-full_soft).norm()),
                    common_location_frobenius_error=float((mean-full_soft).norm()),
                    mass0_absolute_error=arr(moments['mass0']-mass).tolist(),
                    mass2_absolute_error=arr(moments['mass2']-mass).tolist(),
                    mass0_probability_l1_error=float((moments['mass0']-mass).abs().sum()/n),
                    mass2_probability_l1_error=float((moments['mass2']-mass).abs().sum()/n),
                    exact_zero_assignments=zero_count,
                    copy_statistics_invariance=invariance_from_statistics(full_soft,mass,numerator))
                diagnostic_seconds=tick()-start
                explicit=None; explicit_seconds=0.
                if name=='Cora' and seed==271700:
                    start=tick(); explicit=explicit_replication_check(zu,anchor,counts); explicit_seconds=tick()-start
                assert sha(pre_artifact)==pre_hash and sha(pre_record)==pre_record_hash
                serialization_start=time.perf_counter(); save_npz(artifact,package)
                serialization_seconds=time.perf_counter()-serialization_start
                save_json(record,dict(metadata,endpoint=endpoints,replication_intervention=intervention,
                    diagnostics=diagnostics,fixed_controls=fixed_records,fixed_update_seconds=fixed_seconds,
                    fixed_score_seconds=fixed_score_seconds,fixed_comparison_seconds=fixed_comparison_seconds,
                    full_assignment_seconds=full_assignment_seconds,
                    post_branch_seconds=post_branch_seconds,post_score_seconds=post_score_seconds,
                    diagnostic_seconds=diagnostic_seconds,explicit_replication_check=explicit,
                    explicit_replication_seconds=explicit_seconds,pre_serialization_seconds=pre_serialization_seconds,
                    final_serialization_seconds=serialization_seconds,
                    draw_total_seconds_before_final_record=tick()-draw_start,artifact_sha256=sha(artifact),
                    pre_artifact_sha256=pre_hash,pre_record_sha256=pre_record_hash,
                    scope='Fixed-feature inference operators only; no query labels or accuracy; true-mass common pool is post-hoc.',
                    finished=time.time()))
                count+=1
                print(json.dumps(dict(dataset=name,seed=seed,completed=count,local=eta<=.1,
                    mass2_valid=mass2_valid,ren_rep8_disagreement=intervention['full_ren__r8']['disagreement_pct'])),flush=True)
            save_json(folder/'complete.json',dict(rows=5,contract_sha256=contract_hash,finished=time.time()))
            del d,x,adj,z,zz,zu,covariance,qz
            torch.cuda.empty_cache()
    assert count==70
    save_json(OUT/'complete.json',dict(rows=count,datasets=14,replications=REPLICATIONS,
        contract_sha256=contract_hash,runner_sha256=frozen['runner_sha256'],
        design_sha256=frozen['design_sha256'],finished=time.time()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--contract',type=Path,default=DEFAULT_CONTRACT)
    args=parser.parse_args()
    if args.self_check:
        numpy_selfcheck(); return
    if args.freeze_only:
        payload=contract_payload()
        if args.contract.exists(): assert json.loads(args.contract.read_text())==payload
        else: save_json(args.contract,payload)
        print(json.dumps(dict(contract=str(args.contract),sha256=sha(args.contract)),indent=2)); return
    if not args.contract.exists(): raise FileNotFoundError('Run --freeze-only after review before execution')
    try: run(args.contract)
    except Exception:
        save_json(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
