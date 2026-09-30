"""R25: frozen new-condition confirmation of the R22 Ren inference operator.

Pre/post archives are disjoint; query scores are recoverable from dot products.
Canonical argmax and normalized top-two margins preserve numerical tie audits.
No query labels, training, temperature selection, or nearest-score selection.
"""
from pathlib import Path
import argparse
import fcntl
import inspect
import json
import os
import shutil
import time
import traceback

import numpy as np
import run_mass_anchor_r22 as base

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/mass_anchor_confirmation_r25'
DESIGN=ROOT/'design/R25_Ren算子新条件确认.md'
CONTRACT=ROOT/'provenance/r25_frozen_contract.json'
DATASETS=list(base.DATASETS)
HOPS=[1,2,3]
SEEDS=list(range(271900,271910))
REPS=[1,8]
PRE_METHODS=list(base.PRE_METHODS)


def payload():
    dependencies=['code/run_mass_anchor_r22.py','code/expanded_data.py','code/run_study.py',
                  'provenance/expanded_datasets.json','provenance/expanded_heterophily_datasets.json']
    return dict(schema='mass-anchor-confirmation-r25-v1',runner_sha256=base.sha(__file__),
        design_sha256=base.sha(DESIGN),dependencies={x:base.sha(ROOT/x) for x in dependencies},
        datasets=DATASETS,hops=HOPS,seeds=SEEDS,replications=REPS,shots=5,
        operator=dict(distance='squared_euclidean',temperature=1.,rounds=1,
                      normalize_prototype=False,readout='negative_squared_euclidean'),
        local_eta_threshold=.1,pre_methods=PRE_METHODS,
        near_tie_normalized_gap=1e-10,storage='disjoint_pre_post_with_query_dots_and_canonical_argmax')


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def arr(x):return x.detach().cpu().numpy()


def prediction_summary(scores):
    top=scores.topk(2,dim=1).values
    scale=torch.maximum(torch.ones_like(top[:,0]),scores.abs().max(1).values)
    return scores.argmax(1),(top[:,0]-top[:,1])/scale


def exact_statistics(nodes,anchor):
    q=base.softmax(base.euclidean_scores(nodes,anchor),torch.exp,1)
    mass=q.sum(0)
    if float(mass.min())<=0 or not bool(torch.isfinite(q).all()):
        raise ValueError('Invalid exact node assignments')
    return q.T@nodes/mass[:,None],mass,int((q==0).sum())


def canonical_disagreement(pred,gap,reference,refgap):
    changed=pred!=reference
    near=(gap<=1e-10)|(refgap<=1e-10)
    return dict(disagreement_pct=float(np.mean(changed)*100),
                near_tie_either_pct=float(np.mean(near)*100),
                disagreement_outside_near_ties_pct=float(np.mean(changed&~near)*100))


def support_draw(pool,pool_labels,classes,seed):
    gen=torch.Generator().manual_seed(seed)
    sampled=[]
    for c in range(classes):
        candidates=pool[pool_labels==c]
        assert len(candidates)>=5
        sampled.append(candidates[torch.randperm(len(candidates),generator=gen)[:5]])
    return torch.cat(sampled)


def numpy_selfcheck():
    rng=np.random.default_rng(270925)
    z=rng.normal(size=(73,7));z/=np.linalg.norm(z,axis=1,keepdims=True)
    anchor=rng.normal(size=(4,7))*.15;counts=np.full(4,5.)
    g=z.mean(0);cov=(z-g).T@(z-g)/len(z)
    terms=base.moment_terms(anchor,g,cov,len(z),np.exp)
    mass=base.softmax(base.euclidean_scores(z,anchor),np.exp,1).sum(0)
    errors=[]
    for rep in REPS:
        mu=base.pool_prototype(anchor,terms['soft'],terms['mass2'],counts,rep)
        weight=counts/(counts+rep*terms['mass2'])
        dot=(z@anchor.T)*weight[None,:]+(z@terms['soft'].T)*(1-weight)[None,:]
        reconstructed=2*dot-np.square(z).sum(1)[:,None]-np.square(mu).sum(1)[None,:]
        brute=-np.square(z[:,None,:]-mu[None,:,:]).sum(2)
        errors.append(float(np.max(abs(reconstructed-brute))))
    assert max(errors)<1e-12
    result=dict(status='passed',score_reconstruction_errors=errors,
        exact_mass_sum=float(mass.sum()),scope='Synthetic NumPy algebra only; no new graph conditions executed')
    base.save_json(ROOT/'reports/mass_anchor_confirmation_r25/local_selfcheck.json',result)
    print(json.dumps(result,indent=2))


def run(contract_path):
    global torch,F,data,r
    frozen=json.loads(Path(contract_path).read_text())
    assert frozen==payload(),'Source/design changed after freezing'
    contract_hash=base.sha(contract_path)
    import torch
    import torch.nn.functional as F
    import expanded_data as data
    import run_study as r
    import torch_geometric
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    assert data.ALL==DATASETS
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=dict(torch=torch.__version__,numpy=np.__version__,pyg=torch_geometric.__version__,
        library_sources={str(inspect.getsourcefile(obj)):base.sha(inspect.getsourcefile(obj))
                         for obj in [NormalizeFeatures,gcn_norm]})
    base.save_json(OUT/'process.json',dict(pid=os.getpid(),started=time.time(),
                   contract_sha256=contract_hash,runtime=runtime))
    archive=ROOT/'provenance/runner_versions/r25';archive.mkdir(parents=True,exist_ok=True)
    for source in [Path(__file__),DESIGN]+[ROOT/x for x in frozen['dependencies']]:
        target=archive/(base.sha(source)+'__'+source.name)
        if not target.exists():shutil.copy2(source,target)
        assert base.sha(target)==base.sha(source)
    count=0
    with torch.no_grad():
        for name in DATASETS:
            start=tick();d,x,adj,pool,query,C,protocol=data.load_data(name)
            x=x.double();adj=adj.double();load_seconds=tick()-start
            assert not torch.isin(pool,query).any()
            # Only training-pool labels are used for new support sampling.
            pool_labels=d.y[pool].clone()
            supports={seed:support_draw(pool,pool_labels,C,seed) for seed in SEEDS}
            old_geometry=json.loads((ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text())
            assert r.thash(x)==old_geometry['input_sha256'] and r.thash(d.edge_index)==old_geometry['graph_sha256']
            shared_hashes=dict(raw_input=r.thash(d.x),input=r.thash(x),graph=r.thash(d.edge_index),
                adjacency_indices=r.thash(adj.indices()),adjacency_values=r.thash(adj.values()),
                query=r.thash(query),pool=r.thash(pool),pool_labels=r.thash(pool_labels))
            raw_h=x
            for hops in HOPS:
                start=tick();raw_h=adj@raw_h;z=F.normalize(raw_h,dim=1);propagation_seconds=tick()-start
                assert float(z.norm(dim=1).max())<=1+1e-12
                hashes=dict(shared_hashes,representation=r.thash(z))
                start=tick();zz=z.T@z;total=z.sum(0);covariance_precompute_seconds=tick()-start
                folder=OUT/name/f'h{hops}';folder.mkdir(parents=True,exist_ok=True)
                geometry=dict(dataset=name,hops=hops,nodes=len(z),features=z.shape[1],classes=C,
                    input_hashes=hashes,protocol=protocol,contract_sha256=contract_hash,
                    load_seconds=load_seconds,incremental_propagation_seconds=propagation_seconds,
                    covariance_precompute_seconds=covariance_precompute_seconds,runtime=runtime)
                geometry_file=folder/'geometry.json'
                if geometry_file.exists():
                    previous=json.loads(geometry_file.read_text())
                    assert previous['input_hashes']==hashes and previous['contract_sha256']==contract_hash
                    assert previous['runtime']==runtime
                else:base.save_json(geometry_file,geometry)
                geometry_hash=base.sha(geometry_file)
                for seed in SEEDS:
                    record=folder/f'{seed}.json';pre_record=folder/f'{seed}.pre.json'
                    pre_artifact=folder/f'{seed}.pre.npz';post_artifact=folder/f'{seed}.post.npz'
                    support=supports[seed]
                    if record.exists():
                        previous=json.loads(record.read_text())
                        assert previous['contract_sha256']==contract_hash
                        assert previous['geometry_sha256']==geometry_hash and previous['support_sha256']==r.thash(support)
                        assert previous['runtime']==runtime
                        assert previous['pre_artifact_sha256']==base.sha(pre_artifact)
                        assert previous['pre_record_sha256']==base.sha(pre_record)
                        assert previous['post_artifact_sha256']==base.sha(post_artifact)
                        count+=1;continue
                    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Less than 3 GiB available before new output')
                    base.save_json(OUT/'current.json',dict(dataset=name,hops=hops,seed=seed,completed=count,time=time.time()))
                    draw_start=tick();start=tick()
                    si=support.cuda();sy=d.y[support].cuda()
                    assert not torch.isin(si,query.cuda()).any()
                    onehot=F.one_hot(sy,C).double();counts=onehot.sum(0)
                    assert bool((counts==5).all())
                    anchor=onehot.T@z[si]/counts[:,None]
                    mask=torch.ones(len(z),dtype=torch.bool,device='cuda');mask[si]=False
                    zu=z[mask];qz=z[query.cuda()];n=len(zu)
                    mean=(total-z[si].sum(0))/n
                    covariance=(zz-z[si].T@z[si])/n-torch.outer(mean,mean)
                    covariance=(covariance+covariance.T)/2
                    radius=float((zu-mean).norm(dim=1).max())
                    support_geometry_seconds=tick()-start
                    start=tick();terms=base.moment_terms(anchor,mean,covariance,n,torch.exp);moment_seconds=tick()-start
                    mass2_valid=bool(torch.isfinite(terms['mass2']).all() and (terms['mass2']>0).all())
                    assert abs(float(terms['mass2'].sum())-n)<1e-8*max(1,n)
                    eta=radius*float(terms['contrast'].norm(dim=1).max())
                    invalid={} if mass2_valid else {'moment_mass2':'nonpositive/nonfinite mass; no clipping'}
                    start=tick()
                    pre=dict(anchor=arr(anchor),mean=arr(mean),support=arr(si),support_labels=arr(sy),support_counts=arr(counts),
                        query=query.numpy(),query_norm_squared=arr(qz.square().sum(1)),
                        covariance_diagonal=arr(covariance.diag()),mass0=arr(terms['mass0']),mass2=arr(terms['mass2']),
                        mean_assignment=arr(terms['pi']),moment_soft=arr(terms['soft']),projected_variance=arr(terms['projected_variance']),
                        query_anchor_dot=arr(qz@anchor.T),query_mean_dot=arr(qz@mean),query_moment_soft_dot=arr(qz@terms['soft'].T))
                    query_dot_seconds=tick()-start
                    branch_seconds={};score_seconds={}
                    for method in PRE_METHODS:
                        if method in invalid:continue
                        soft=terms['soft'] if method.startswith('moment_') else mean.expand_as(anchor)
                        mass=terms['mass2'] if method=='moment_mass2' else (
                            torch.full_like(counts,n/C) if method=='common_equalmass' else terms['mass0'])
                        for rep in REPS:
                            key=f'{method}__r{rep}'
                            start=tick();mu=base.pool_prototype(anchor,soft,mass,counts,rep);branch_seconds[key]=tick()-start
                            start=tick();scores=base.euclidean_scores(qz,mu);pred,gap=prediction_summary(scores);score_seconds[key]=tick()-start
                            pre[key+'__prototype']=arr(mu);pre[key+'__prediction']=arr(pred)
                            pre[key+'__normalized_gap']=arr(gap)
                    metadata=dict(dataset=name,hops=hops,seed=seed,classes=C,n_unlabeled=n,protocol=protocol,
                        invalid=invalid,mass2_valid=mass2_valid,node_eta_upper=eta,predeclared_local_domain=eta<=.1,
                        node_radius=radius,covariance_trace=float(covariance.trace()),covariance_sha256=r.thash(covariance),
                        support_sha256=r.thash(si),input_hashes=hashes,geometry_sha256=geometry_hash,
                        contract_sha256=contract_hash,runner_sha256=frozen['runner_sha256'],design_sha256=frozen['design_sha256'],
                        dependencies=frozen['dependencies'],runtime=runtime,
                        support_geometry_seconds=support_geometry_seconds,moment_terms_seconds=moment_seconds,
                        pre_query_dot_seconds=query_dot_seconds,pre_branch_seconds=branch_seconds,pre_score_seconds=score_seconds,
                        load_seconds=load_seconds,incremental_propagation_seconds=propagation_seconds,
                        covariance_precompute_seconds=covariance_precompute_seconds)
                    start=time.perf_counter();base.save_npz(pre_artifact,pre)
                    base.save_json(pre_record,dict(metadata,phase='before any full node softmax for this draw',
                        pre_artifact_sha256=base.sha(pre_artifact),finished=time.time()))
                    pre_serialization_seconds=time.perf_counter()-start
                    pre_hash=base.sha(pre_artifact);pre_record_hash=base.sha(pre_record)
                    # Only now evaluate the full N-by-C assignment.
                    start=tick();u,mass,zeros=exact_statistics(zu,anchor);full_assignment_seconds=tick()-start
                    start=tick();post=dict(full_soft=arr(u),full_mass=arr(mass),query_full_soft_dot=arr(qz@u.T));post_dot_seconds=tick()-start
                    post_branch_seconds={};post_score_seconds={}
                    for method,soft in [('full_ren',u),('common_true_mass',mean.expand_as(anchor))]:
                        for rep in REPS:
                            key=f'{method}__r{rep}'
                            start=tick();mu=base.pool_prototype(anchor,soft,mass,counts,rep);post_branch_seconds[key]=tick()-start
                            start=tick();scores=base.euclidean_scores(qz,mu);pred,gap=prediction_summary(scores);post_score_seconds[key]=tick()-start
                            post[key+'__prototype']=arr(mu);post[key+'__prediction']=arr(pred);post[key+'__normalized_gap']=arr(gap)
                    assert not (set(pre)&set(post)),'Pre/post archives must not duplicate data arrays'
                    start=time.perf_counter();endpoint={};intervention={}
                    arrays={**pre,**post}
                    for method in PRE_METHODS+['full_ren','common_true_mass']:
                        if method in invalid:continue
                        for rep in REPS:
                            key=f'{method}__r{rep}';ref=f'full_ren__r{rep}'
                            endpoint[key]=dict(canonical_disagreement(arrays[key+'__prediction'],arrays[key+'__normalized_gap'],
                                arrays[ref+'__prediction'],arrays[ref+'__normalized_gap']),
                                prototype_max_error=float(np.linalg.norm(arrays[key+'__prototype']-arrays[ref+'__prototype'],axis=1).max()))
                            intervention[key]=canonical_disagreement(arrays[key+'__prediction'],arrays[key+'__normalized_gap'],
                                arrays[f'{method}__r1__prediction'],arrays[f'{method}__r1__normalized_gap'])
                    diagnostics=dict(mass0_probability_l1_error=float((terms['mass0']-mass).abs().sum()/n),
                        mass2_probability_l1_error=float((terms['mass2']-mass).abs().sum()/n),
                        common_location_error=float((mean-u).norm()),moment_location_error=float((terms['soft']-u).norm()),
                        support_weight_r1_mean=float((counts/(counts+mass)).mean()),
                        support_weight_r8_mean=float((counts/(counts+8*mass)).mean()),zero_assignments=zeros)
                    diagnostic_seconds=time.perf_counter()-start
                    assert base.sha(pre_artifact)==pre_hash and base.sha(pre_record)==pre_record_hash
                    start=time.perf_counter();base.save_npz(post_artifact,post);post_serialization_seconds=time.perf_counter()-start
                    base.save_json(record,dict(metadata,endpoint=endpoint,replication_intervention=intervention,
                        diagnostics=diagnostics,full_assignment_seconds=full_assignment_seconds,post_query_dot_seconds=post_dot_seconds,
                        post_branch_seconds=post_branch_seconds,post_score_seconds=post_score_seconds,
                        diagnostic_seconds=diagnostic_seconds,pre_serialization_seconds=pre_serialization_seconds,
                        post_serialization_seconds=post_serialization_seconds,draw_total_seconds=tick()-draw_start,
                        pre_artifact_sha256=pre_hash,pre_record_sha256=pre_record_hash,
                        post_artifact_sha256=base.sha(post_artifact),
                        scope='New support draws and h1/h2/h3 conditions on the same 14 graphs; no query labels or accuracy; disjoint pre/post archives.',
                        finished=time.time()))
                    count+=1
                    print(json.dumps(dict(dataset=name,hops=hops,seed=seed,completed=count,domain=eta<=.1,
                        moment2_r1_disagreement=endpoint.get('moment_mass2__r1',{}).get('disagreement_pct'),
                        common_mass0_r1_disagreement=endpoint['common_mass0__r1']['disagreement_pct'])),flush=True)
                base.save_json(folder/'complete.json',dict(rows=len(SEEDS),contract_sha256=contract_hash,
                                                          geometry_sha256=geometry_hash,finished=time.time()))
                del zz,z,zu,qz,covariance;torch.cuda.empty_cache()
            base.save_json(OUT/name/'complete.json',dict(rows=len(HOPS)*len(SEEDS),contract_sha256=contract_hash,finished=time.time()))
            del d,x,adj,raw_h;torch.cuda.empty_cache()
    assert count==420
    base.save_json(OUT/'complete.json',dict(rows=count,datasets=len(DATASETS),hops=HOPS,seeds=SEEDS,
        contract_sha256=contract_hash,runner_sha256=frozen['runner_sha256'],design_sha256=frozen['design_sha256'],finished=time.time()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true')
    ap.add_argument('--self-check',action='store_true');ap.add_argument('--contract',type=Path,default=CONTRACT)
    args=ap.parse_args()
    if args.self_check:numpy_selfcheck();return
    if args.freeze_only:
        frozen=payload()
        if args.contract.exists():assert json.loads(args.contract.read_text())==frozen
        else:base.save_json(args.contract,frozen)
        print(json.dumps(dict(contract=str(args.contract),sha256=base.sha(args.contract)),indent=2));return
    if not args.contract.exists():raise FileNotFoundError('Review and freeze first')
    try:run(args.contract)
    except Exception:
        base.save_json(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
