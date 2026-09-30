"""Independent R25 audit from compact pre/post arrays. NumPy only.

Canonical argmax is retained, score arrays are reconstructed from query dots.
Position benefits use equal mass approximations; domain and hop strata remain
separate. No query labels, graph replay, threshold changes, or method selection.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'results/mass_anchor_confirmation_r25'
OUT=ROOT/'reports/mass_anchor_confirmation_r25'
CONTRACT=ROOT/'provenance/r25_frozen_contract.json'
DATASETS=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell',
          'Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings']
HOPS=[1,2,3];SEEDS=list(range(271900,271910));REPS=[1,8]
PRE_METHODS=['common_mass0','moment_mass0','moment_mass2','common_equalmass']
ALL_METHODS=PRE_METHODS+['full_ren','common_true_mass']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def close(x,y,key,atol=2e-10,rtol=2e-8):
    assert np.allclose(x,y,atol=atol,rtol=rtol),(key,x,y)
    return float(np.max(np.abs(np.asarray(x)-np.asarray(y))))


def source_check(value,path):
    if path.exists() and sha(path)==value:return
    archived=ROOT/'provenance/runner_versions/r25'/f'{value}__{path.name}'
    assert archived.exists() and sha(archived)==value,('Source mismatch',path,value)


def contract_check():
    c=read(CONTRACT)
    assert c['schema']=='mass-anchor-confirmation-r25-v1'
    assert c['datasets']==DATASETS and c['hops']==HOPS and c['seeds']==SEEDS and c['replications']==REPS
    assert c['shots']==5 and c['local_eta_threshold']==.1 and c['near_tie_normalized_gap']==1e-10
    assert c['operator']==dict(distance='squared_euclidean',temperature=1.,rounds=1,
        normalize_prototype=False,readout='negative_squared_euclidean')
    assert c['pre_methods']==PRE_METHODS
    source_check(c['runner_sha256'],ROOT/'code/run_mass_anchor_confirmation_r25.py')
    source_check(c['design_sha256'],ROOT/'design/R25_Ren算子新条件确认.md')
    for p,h in c['dependencies'].items():source_check(h,ROOT/p)
    return c,sha(CONTRACT)


def pooled(anchor,soft,mass,counts,rep):
    weight=counts/(counts+rep*mass)
    return weight[:,None]*anchor+(1-weight)[:,None]*soft,weight


def scores_from_dots(mu,weight,anchor_dot,soft_dot,norm_squared):
    dot=anchor_dot*weight[None,:]+soft_dot*(1-weight)[None,:]
    return 2*dot-norm_squared[:,None]-np.square(mu).sum(1)[None,:]


def normalized_gap(scores):
    top=np.partition(scores,-2,axis=1)[:,-2:]
    return np.abs(top[:,0]-top[:,1])/np.maximum(1.,np.abs(scores).max(1))


def canonical_metrics(pred,gap,ref,refgap):
    change=pred!=ref;near=(gap<=1e-10)|(refgap<=1e-10)
    return dict(disagreement_pct=float(change.mean()*100),near_tie_either_pct=float(near.mean()*100),
        disagreement_outside_near_ties_pct=float((change&~near).mean()*100))


def csv_write(name,rows):
    path=OUT/name
    if not rows:path.write_text('');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def collect(allow_partial=False):
    contract,contract_hash=contract_check()
    rows=[];paired=[];mass_rows=[];rep_rows=[];cost_rows=[];geometry_rows=[];invalid=[]
    supports={};verified=0;array_count=0;score_arrays=0;rank_changes=0;unsafe_rank_changes=0
    max_proto_error=0.;max_gap_error=0.;max_scalar_error=0.
    for name in DATASETS:
        original_geometry=read(ROOT/'results/attribute_mechanism_r13'/name/'geometry.json')
        for hops in HOPS:
            folder=SRC/name/f'h{hops}';gp=folder/'geometry.json'
            if not gp.exists():
                if allow_partial:continue
                raise FileNotFoundError(gp)
            geom=read(gp);geometry_hash=sha(gp)
            assert geom['contract_sha256']==contract_hash and geom['dataset']==name and geom['hops']==hops
            assert geom['input_hashes']['input']==original_geometry['input_sha256']
            assert geom['input_hashes']['graph']==original_geometry['graph_sha256']
            geometry_rows.append(dict(dataset=name,hops=hops,nodes=geom['nodes'],features=geom['features'],
                classes=geom['classes'],first_geometry_load_seconds=geom['load_seconds'],
                first_geometry_incremental_propagation_seconds=geom['incremental_propagation_seconds'],
                first_geometry_covariance_precompute_seconds=geom['covariance_precompute_seconds']))
            checked_hop=0
            for seed in SEEDS:
                rp=folder/f'{seed}.json'
                if not rp.exists():
                    if allow_partial:continue
                    raise FileNotFoundError(rp)
                rec=read(rp);pr=folder/f'{seed}.pre.json';pre_rec=read(pr)
                pa=folder/f'{seed}.pre.npz';qa=folder/f'{seed}.post.npz'
                assert rec['contract_sha256']==pre_rec['contract_sha256']==contract_hash
                assert rec['runner_sha256']==contract['runner_sha256'] and rec['design_sha256']==contract['design_sha256']
                assert rec['dependencies']==contract['dependencies']
                assert rec['geometry_sha256']==geometry_hash and rec['input_hashes']==geom['input_hashes']
                assert rec['runtime']==geom['runtime']
                assert rec['pre_artifact_sha256']==pre_rec['pre_artifact_sha256']==sha(pa)
                assert rec['post_artifact_sha256']==sha(qa) and rec['pre_record_sha256']==sha(pr)
                assert pre_rec['phase']=='before any full node softmax for this draw'
                for key,value in pre_rec.items():
                    if key not in ['phase','pre_artifact_sha256','finished']:assert rec[key]==value,(name,hops,seed,key)
                with np.load(pa) as pfile,np.load(qa) as qfile:
                    assert not (set(pfile.files)&set(qfile.files)),('Duplicated pre/post keys',name,hops,seed)
                    p={k:pfile[k] for k in pfile.files};q={k:qfile[k] for k in qfile.files}
                assert not any('label' in k and k!='support_labels' for k in list(p)+list(q))
                assert not any(k.endswith('__scores') for k in list(p)+list(q))
                for key,value in {**p,**q}.items():
                    if key!='mass2' or rec['mass2_valid']:assert np.isfinite(value).all(),(name,hops,seed,key)
                    array_count+=1
                anchor=p['anchor'];mean=p['mean'];counts=p['support_counts'];C=len(anchor);n=rec['n_unlabeled']
                assert C==rec['classes']==geom['classes'] and np.all(counts==5)
                assert n==geom['nodes']-len(p['support'])
                assert np.array_equal(np.bincount(p['support_labels'],minlength=C),counts)
                assert np.intersect1d(p['support'],p['query']).size==0
                assert ahash(p['support'])==rec['support_sha256'] and ahash(p['query'])==rec['input_hashes']['query']
                if (name,seed) not in supports:supports[(name,seed)]=(p['support'].copy(),p['support_labels'].copy())
                else:
                    assert np.array_equal(p['support'],supports[(name,seed)][0])
                    assert np.array_equal(p['support_labels'],supports[(name,seed)][1])
                logits=2*anchor@mean-np.square(anchor).sum(1)
                pi=np.exp(logits-logits.max());pi/=pi.sum()
                contrast=2*anchor-pi@(2*anchor)
                action=p['moment_soft']-mean
                variance=(action*contrast).sum(1)
                mass0=n*pi;mass2=mass0*(1+.5*(variance-pi@variance))
                eta=rec['node_radius']*np.linalg.norm(contrast,axis=1).max()
                for value,expected,label in [(pi,p['mean_assignment'],'mean class assignment'),
                    (mass0,p['mass0'],'mass0'),(variance,p['projected_variance'],'variance'),
                    (eta,rec['node_eta_upper'],'eta'),(p['covariance_diagonal'].sum(),rec['covariance_trace'],'trace')]:
                    max_scalar_error=max(max_scalar_error,close(value,expected,label))
                mass_valid=bool(np.isfinite(mass2).all() and np.all(mass2>0))
                assert mass_valid==rec['mass2_valid']
                if mass_valid:close(mass2,p['mass2'],'mass2')
                assert rec['predeclared_local_domain']==(eta<=.1)
                actual_mass=q['full_mass'];u=q['full_soft'];moment=p['moment_soft']
                assert np.all(actual_mass>0) and np.isclose(actual_mass.sum(),n,rtol=1e-10)
                diag=dict(mass0_probability_l1_error=float(np.abs(mass0-actual_mass).sum()/n),
                    mass2_probability_l1_error=float(np.abs(mass2-actual_mass).sum()/n),
                    common_location_error=float(np.linalg.norm(mean-u)),moment_location_error=float(np.linalg.norm(moment-u)),
                    support_weight_r1_mean=float((counts/(counts+actual_mass)).mean()),
                    support_weight_r8_mean=float((counts/(counts+8*actual_mass)).mean()))
                for key,value in diag.items():
                    if np.isfinite(value):max_scalar_error=max(max_scalar_error,close(value,rec['diagnostics'][key],key))
                common_fields=dict(dataset=name,hops=hops,seed=seed,domain=rec['predeclared_local_domain'],eta=eta)
                mass_rows.append(dict(**common_fields,**diag,mass2_valid=mass_valid))
                branch={}
                for method in ALL_METHODS:
                    if method in rec['invalid']:
                        assert method=='moment_mass2' and not mass_valid
                        assert not any(k.startswith(method+'__') for k in p)
                        invalid.append(dict(**common_fields,method=method,reason=rec['invalid'][method]));continue
                    source=q if method in ['full_ren','common_true_mass'] else p
                    masses=actual_mass if method in ['full_ren','common_true_mass'] else (
                        mass2 if method=='moment_mass2' else np.full(C,n/C) if method=='common_equalmass' else mass0)
                    if method.startswith('moment_'):soft=moment;dot=p['query_moment_soft_dot']
                    elif method=='full_ren':soft=u;dot=q['query_full_soft_dot']
                    else:soft=np.broadcast_to(mean,anchor.shape);dot=np.broadcast_to(p['query_mean_dot'][:,None],p['query_anchor_dot'].shape)
                    for rep in REPS:
                        key=f'{method}__r{rep}';proto,weight=pooled(anchor,soft,masses,counts,rep)
                        original=source[key+'__prototype'];pred=source[key+'__prediction'];gap=source[key+'__normalized_gap']
                        max_proto_error=max(max_proto_error,close(proto,original,key+' prototype',atol=2e-12,rtol=2e-10))
                        assert pred.shape==gap.shape==(len(p['query']),) and np.all((pred>=0)&(pred<C))
                        assert np.all(gap>=0)
                        score=scores_from_dots(proto,weight,p['query_anchor_dot'],dot,p['query_norm_squared'])
                        restored_gap=normalized_gap(score)
                        max_gap_error=max(max_gap_error,close(restored_gap,gap,key+' normalized gap',atol=3e-12,rtol=2e-7))
                        changed=score.argmax(1)!=pred
                        outside=(restored_gap>1e-10)&(gap>1e-10)
                        rank_changes+=int(changed.sum());unsafe_rank_changes+=int((changed&outside).sum())
                        assert not np.any(changed&outside),(name,hops,seed,key,'non-tied restored rank mismatch')
                        branch[key]=dict(prototype=original,prediction=pred,gap=gap,score=score)
                        score_arrays+=1
                local_rows={}
                for key,item in branch.items():
                    method,rr=key.rsplit('__r',1);rep=int(rr);ref=branch[f'full_ren__r{rep}']
                    result=canonical_metrics(item['prediction'],item['gap'],ref['prediction'],ref['gap'])
                    result['prototype_max_error']=float(np.linalg.norm(item['prototype']-ref['prototype'],axis=1).max())
                    for field,value in result.items():max_scalar_error=max(max_scalar_error,close(value,rec['endpoint'][key][field],field))
                    delta=item['score']-ref['score']
                    result.update(reconstructed_centered_score_rmse=float(np.sqrt(np.mean((delta-delta.mean(1,keepdims=True))**2))),
                        reconstructed_max_pairwise_margin_error=float(np.ptp(delta,axis=1).max()))
                    row=dict(**common_fields,method=method,replication=rep,phase='posthoc' if method=='common_true_mass' else 'reference' if method=='full_ren' else 'pre',**result)
                    rows.append(row);local_rows[(method,rep)]=row
                    first=branch[f'{method}__r1']
                    intervention=canonical_metrics(item['prediction'],item['gap'],first['prediction'],first['gap'])
                    for field,value in intervention.items():max_scalar_error=max(max_scalar_error,close(value,rec['replication_intervention'][key][field],field))
                    rep_rows.append(dict(**common_fields,method=method,replication=rep,**intervention))
                for rep in REPS:
                    for candidate,reference,label in [('moment_mass0','common_mass0','position_same_mass'),
                        ('moment_mass2','moment_mass0','second_order_mass_same_position'),
                        ('moment_mass2','common_true_mass','prediction_vs_posthoc_position_control'),
                        ('common_mass0','common_equalmass','mean_mass_vs_equal_mass')]:
                        if (candidate,rep) not in local_rows or (reference,rep) not in local_rows:continue
                        a,b=local_rows[(candidate,rep)],local_rows[(reference,rep)]
                        paired.append(dict(**common_fields,replication=rep,contrast=label,candidate=candidate,reference=reference,
                            candidate_disagreement_pct=a['disagreement_pct'],reference_disagreement_pct=b['disagreement_pct'],
                            disagreement_reduction_pp=b['disagreement_pct']-a['disagreement_pct'],
                            outside_tie_reduction_pp=b['disagreement_outside_near_ties_pct']-a['disagreement_outside_near_ties_pct'],
                            prototype_error_reduction=b['prototype_max_error']-a['prototype_max_error']))
                cost_rows.append(dict(**common_fields,
                    support_geometry_seconds=rec['support_geometry_seconds'],moment_terms_seconds=rec['moment_terms_seconds'],
                    pre_query_dot_seconds=rec['pre_query_dot_seconds'],pre_pool_seconds=sum(rec['pre_branch_seconds'].values()),
                    pre_score_seconds=sum(rec['pre_score_seconds'].values()),full_assignment_seconds=rec['full_assignment_seconds'],
                    post_query_dot_seconds=rec['post_query_dot_seconds'],post_pool_seconds=sum(rec['post_branch_seconds'].values()),
                    post_score_seconds=sum(rec['post_score_seconds'].values()),diagnostic_seconds=rec['diagnostic_seconds'],
                    serialization_seconds=rec['pre_serialization_seconds']+rec['post_serialization_seconds'],
                    draw_total_seconds=rec['draw_total_seconds'],
                    rec_load_seconds=rec['load_seconds'],rec_covariance_precompute_seconds=rec['covariance_precompute_seconds'],
                    rec_incremental_propagation_seconds=rec['incremental_propagation_seconds']))
                verified+=1;checked_hop+=1
            if not allow_partial:
                marker=read(folder/'complete.json')
                assert checked_hop==10 and marker['rows']==10 and marker['contract_sha256']==contract_hash
                assert marker['geometry_sha256']==geometry_hash
        if not allow_partial:
            marker=read(SRC/name/'complete.json');assert marker['rows']==30 and marker['contract_sha256']==contract_hash
    marker=read(SRC/'complete.json') if (SRC/'complete.json').exists() else None
    complete=verified==420 and marker is not None
    if marker:
        assert marker['rows']==420 and marker['hops']==HOPS and marker['seeds']==SEEDS
        assert marker['contract_sha256']==contract_hash and marker['runner_sha256']==contract['runner_sha256']
    if not allow_partial:assert complete,'Not all frozen conditions are complete'
    verification=dict(complete=complete,draws=verified,expected_draws=420,pre_post_disjoint=True,
        finite_arrays_checked=array_count,reconstructed_euclidean_score_arrays=score_arrays,
        canonical_rank_reconstruction_changes=rank_changes,changes_outside_near_ties=unsafe_rank_changes,
        max_prototype_pool_reconstruction_error=max_proto_error,max_normalized_gap_reconstruction_error=max_gap_error,
        max_reported_scalar_difference=max_scalar_error,support_sets_checked_across_hops=len(supports),
        invalid=invalid,query_labels_read=False,contract_sha256=contract_hash,
        scope='Frozen sources and geometry/pre/post hashes; disjoint archives; support identity across hops; pool formulas, moment-mass algebra, Euclidean query-dot scores, canonical argmax/gap consistency and reported endpoint/replication statistics. No independent whole-graph covariance/propagation/node-softmax replay or support resampling from unsaved pool arrays. Same 14 graphs; new support draws and h1/h3 geometry conditions, not independent graph generalization.')
    return rows,paired,mass_rows,rep_rows,cost_rows,geometry_rows,verification


def summarize(rows,paired,mass,rep_rows,costs):
    summary=[];domains=[];contrasts=[];mass_summary=[];rep_summary=[];cost_summary=[]
    for name in DATASETS:
        for hops in HOPS:
            for method in ALL_METHODS:
                for rep in REPS:
                    part=[x for x in rows if x['dataset']==name and x['hops']==hops and x['method']==method and x['replication']==rep]
                    if not part:continue
                    summary.append(dict(dataset=name,hops=hops,method=method,replication=rep,draws=len(part),
                        domain_draws=sum(x['domain'] for x in part),
                        disagreement_mean_pct=float(np.mean([x['disagreement_pct'] for x in part])),
                        disagreement_max_pct=max(x['disagreement_pct'] for x in part),
                        outside_tie_mean_pct=float(np.mean([x['disagreement_outside_near_ties_pct'] for x in part])),
                        prototype_error_mean=float(np.mean([x['prototype_max_error'] for x in part]))))
            part=[x for x in mass if x['dataset']==name and x['hops']==hops]
            if part:
                mass_summary.append(dict(dataset=name,hops=hops,draws=len(part),domain_draws=sum(x['domain'] for x in part),
                    eta_mean=float(np.mean([x['eta'] for x in part])),eta_max=max(x['eta'] for x in part),
                    mass0_l1_mean=float(np.mean([x['mass0_probability_l1_error'] for x in part])),
                    mass2_l1_mean=float(np.mean([x['mass2_probability_l1_error'] for x in part])),
                    mass2_improved_draws=sum(x['mass2_probability_l1_error']<x['mass0_probability_l1_error'] for x in part)))
            part=[x for x in costs if x['dataset']==name and x['hops']==hops]
            if part:
                cost_summary.append(dict(dataset=name,hops=hops,draws=len(part),
                    **{k+'_mean':float(np.mean([x[k] for x in part])) for k in part[0]
                       if k not in ['dataset','hops','seed','domain','eta']}))
    for hops in HOPS:
        for dom in [True,False]:
            for method in ALL_METHODS:
                for rep in REPS:
                    part=[x for x in rows if x['hops']==hops and x['domain']==dom and x['method']==method and x['replication']==rep]
                    if not part:continue
                    domains.append(dict(hops=hops,domain='declared' if dom else 'outside',method=method,replication=rep,
                        draws=len(part),datasets=len(set(x['dataset'] for x in part)),
                        disagreement_mean_pct=float(np.mean([x['disagreement_pct'] for x in part])),
                        disagreement_max_pct=max(x['disagreement_pct'] for x in part),
                        outside_tie_mean_pct=float(np.mean([x['disagreement_outside_near_ties_pct'] for x in part]))))
                    changes=[x for x in rep_rows if x['hops']==hops and x['domain']==dom and x['method']==method and x['replication']==rep]
                    rep_summary.append(dict(hops=hops,domain='declared' if dom else 'outside',method=method,replication=rep,
                        draws=len(changes),self_r1_disagreement_mean_pct=float(np.mean([x['disagreement_pct'] for x in changes])),
                        self_r1_disagreement_max_pct=max(x['disagreement_pct'] for x in changes),
                        outside_tie_mean_pct=float(np.mean([x['disagreement_outside_near_ties_pct'] for x in changes]))))
            for rep in REPS:
                for label in sorted(set(x['contrast'] for x in paired)):
                    part=[x for x in paired if x['hops']==hops and x['domain']==dom and x['replication']==rep and x['contrast']==label]
                    if not part:continue
                    delta=[x['disagreement_reduction_pp'] for x in part]
                    contrasts.append(dict(hops=hops,domain='declared' if dom else 'outside',replication=rep,contrast=label,
                        draws=len(part),datasets=len(set(x['dataset'] for x in part)),
                        candidate_error_mean_pct=float(np.mean([x['candidate_disagreement_pct'] for x in part])),
                        reference_error_mean_pct=float(np.mean([x['reference_disagreement_pct'] for x in part])),
                        reduction_mean_pp=float(np.mean(delta)),reduction_min_pp=min(delta),reduction_max_pp=max(delta),
                        improved_draws=sum(x>1e-12 for x in delta),tied_draws=sum(abs(x)<=1e-12 for x in delta),
                        worsened_draws=sum(x<-1e-12 for x in delta),
                        outside_tie_reduction_mean_pp=float(np.mean([x['outside_tie_reduction_pp'] for x in part]))))
    return summary,domains,contrasts,mass_summary,rep_summary,cost_summary


def make_report(summary,domains,contrasts,verification):
    lines=['# R25：Ren型位置矩的新条件确认','',
        f"独立核验{verification['draws']}/420组，完成标记：{verification['complete']}。没有查询标签或Accuracy。",'',
        '本轮保留同14图，在h1/h2/h3和10个新支持抽样上确认；h2是同几何新支持复核，h1/h3是传播条件外推，不能称新图泛化。η≤0.1阈值保持不变，不追认域外成功满足局部假设。','',
        '## 同质量的位置对照','',
        '下面只比较moment_mass0和common_mass0，两者采用相同零阶类质量。正的误差下降表示位置矩更接近完整Ren输出；不是Accuracy提升。','',
        '|h|范围|r|抽样数|共同池平均异判|位置矩平均异判|下降pp|改善/相同/恶化|最差下降pp|',
        '|---:|---|---:|---:|---:|---:|---:|---|---:|']
    for x in contrasts:
        if x['contrast']!='position_same_mass':continue
        lines.append(f"|{x['hops']}|{x['domain']}|{x['replication']}|{x['draws']}|{x['reference_error_mean_pct']:.5f}%|{x['candidate_error_mean_pct']:.5f}%|{x['reduction_mean_pp']:.5f}|{x['improved_draws']}/{x['tied_draws']}/{x['worsened_draws']}|{x['reduction_min_pp']:.5f}|")
    lines+=['','## 预声明域的覆盖与最坏误差','',
        '|h|范围|分支|r|抽样数|图数|平均异判|最坏异判|近并列外平均异判|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for x in domains:
        if x['method'] not in ['moment_mass0','moment_mass2','common_true_mass']:continue
        lines.append(f"|{x['hops']}|{x['domain']}|{x['method']}|{x['replication']}|{x['draws']}|{x['datasets']}|{x['disagreement_mean_pct']:.5f}%|{x['disagreement_max_pct']:.5f}%|{x['outside_tie_mean_pct']:.5f}%|")
    lines+=['','## 全14图的h分层结果','',
        '表内为r=1平均异判率；完整复制影响与r=8结果全部保留在CSV，不按表现选择h。','',
        '|数据集|h|声明域抽样|共同mass0|位置mass0|位置mass2|真实质量共同池|',
        '|---|---:|---:|---:|---:|---:|---:|']
    idx={(x['dataset'],x['hops'],x['method'],x['replication']):x for x in summary}
    for name in DATASETS:
        for hops in HOPS:
            main=idx.get((name,hops,'common_mass0',1))
            if not main:continue
            def cell(method):
                item=idx.get((name,hops,method,1));return f"{item['disagreement_mean_pct']:.4f}%" if item else '弃权/缺失'
            lines.append(f"|{name}|{hops}|{main['domain_draws']}/{main['draws']}|"+'|'.join(cell(m) for m in ['common_mass0','moment_mass0','moment_mass2','common_true_mass'])+'|')
    lines+=['','## 解释边界和验证范围','',
        '- `paired_contrasts.csv`将同质量的位置增量，与同位置的二阶质量增量分开。后者不因质量L1误差改善就自动获得分类价值。',
        '- `replication_summary.csv`保留r8/r1实际变化。复制不增几何信息；真实质量共同池是后验位置对照，不能当作事前预测器。',
        '- 域内外、最坏抽样及近并列全部保留；η域是冻结诊断，不是严格证书。不同h结果不用于挑选最好传播底座。',
        f"- 重建{verification['reconstructed_euclidean_score_arrays']}个欧式score矩阵，与规范argmax有{verification['canonical_rank_reconstruction_changes']}次排名差，近并列范围外{verification['changes_outside_near_ties']}次。主指标按规范argmax复算。",
        f"- 已核对{verification['finite_arrays_checked']}个紧凑数组、pre/post不重叠、所有来源与geometry哈希、{verification['support_sets_checked_across_hops']}套支持集合在h间一致。",
        '- 可以独立核对池化、预测质量、查询内积分数和端点统计；没有全图Σ、全节点分配或原支持池数组的本地重放，不能称完全独立全流程复现。',
        '- `geometry_cost.csv`是首次geometry记录的共享成本；`cost_per_draw.csv`保留实际抽样运行成本。恢复运行时两者可不同，不能按所有抽样重复累加一次矩预计算，也不据此宣传加速。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    rows,paired,mass,reps,costs,geometry,verification=collect(args.allow_partial)
    summary,domains,contrasts,mass_summary,rep_summary,cost_summary=summarize(rows,paired,mass,reps,costs)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,value in [('per_draw.csv',rows),('paired_contrasts.csv',paired),('mass_position_per_draw.csv',mass),
        ('replication_per_draw.csv',reps),('cost_per_draw.csv',costs),('geometry_cost.csv',geometry),
        ('summary.csv',summary),('domain_by_hops.csv',domains),('contrast_by_hops_domain.csv',contrasts),
        ('mass_position_summary.csv',mass_summary),('replication_summary.csv',rep_summary),('cost_summary.csv',cost_summary)]:csv_write(name,value)
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    (OUT/'新条件位置矩确认结论.md').write_text(make_report(summary,domains,contrasts,verification))
    print(json.dumps(dict(draws=verification['draws'],complete=verification['complete'],
        endpoint_rows=len(rows),paired_rows=len(paired),score_arrays=verification['reconstructed_euclidean_score_arrays'],output=str(OUT)),ensure_ascii=False))


if __name__=='__main__':main()
