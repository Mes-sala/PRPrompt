"""R26: pre-only certificate sealing, followed by a separate post audit.

No q, actual N/u, or label arrays are read in either phase. Phase B reads only
sealed prototype, prediction and normalized-gap arrays from R25 post packages.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
DESIGN=ROOT/'design/R26_Ren事前位置贡献证书_离线规则.md'
THEORY=ROOT/'theory/Ren一步_矩近似的事前误差与间隔判据_候选.md'
PURE=ROOT/'theory/check_ren_moment_certificate_candidate.py'
CONTRACT=ROOT/'provenance/r26_frozen_contract.json'
SOURCE=ROOT/'results/mass_anchor_confirmation_r25'
OUT=ROOT/'results/preonly_certificate_r26'
REPORT=ROOT/'reports/preonly_certificate_r26'
DATASETS=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell','Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings']
HOPS=[1,2,3];SEEDS=list(range(271900,271910));METHODS=['moment0','moment2'];REPS=[1,8]
GAP=1e-10;BUDGET=.01;AUDIT_TOL=1e-12
spec=importlib.util.spec_from_file_location('ren_certificate_candidate',PURE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def save_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('');return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def payload():
    # This is provenance, not R25 result data. Phase A never reads its post data.
    r25=ROOT/'provenance/r25_frozen_contract.json'
    return dict(schema='preonly-ren-certificate-r26-v1',runner_sha256=sha(__file__),
        design_sha256=sha(DESIGN),theory_sha256=sha(THEORY),pure_functions_sha256=sha(PURE),
        r25_contract_sha256=sha(r25),datasets=DATASETS,hops=HOPS,seeds=SEEDS,methods=METHODS,
        replications=REPS,eta_threshold=.1,normalized_score_guard=GAP,group_budget=BUDGET,
        radius_audit_tolerance=AUDIT_TOL,arithmetic='float64 numerical validation of real bounds, not interval IEEE',
        pre_data_access='only R25 pre.npz and pre.json',
        post_array_allowlist=['prototype','prediction','normalized_gap'],
        selection='Separate k0/k2; no best-of selection; whole pre phase before any post access')


def check_contract(path):
    value=read(path);assert value==payload(),'R26 frozen source/design/dependency changed'
    return value,sha(path)


def identities():
    for name in DATASETS:
        for hops in HOPS:
            for seed in SEEDS:yield name,hops,seed


def paths(name,hops,seed):
    return SOURCE/name/f'h{hops}'/str(seed),OUT/name/f'h{hops}'/str(seed)


def protected_winner(interval):
    nominal,lo,hi=(interval[k] for k in ('nominal','lower','upper'))
    candidate=np.argmax(nominal,axis=1);other=hi.copy();other[np.arange(len(hi)),candidate]=-np.inf
    margin=lo[np.arange(len(lo)),candidate]-other.max(1)
    scale=np.maximum(1.,np.maximum(np.abs(lo).max(1),np.abs(hi).max(1)))
    valid=interval['valid'].all(1)&np.isfinite(margin)&np.isfinite(scale)
    safe=valid&(margin>GAP*scale)
    normalized=np.zeros(len(margin));normalized[valid]=margin[valid]/scale[valid]
    return np.where(safe,candidate,-1),normalized


def threeway(yf,yc):
    valid=(yf>=0)&(yc>=0)
    state=np.full(len(yf),2,dtype=np.int8);state[valid&(yf==yc)]=0;state[valid&(yf!=yc)]=1
    lower=float(np.mean(state==1));upper=float(np.mean(state!=0))
    action='ignorable' if upper<=BUDGET else ('must_keep' if lower>BUDGET else 'abstain')
    return state,dict(certified_same=int((state==0).sum()),certified_different=int((state==1).sum()),
                      abstain=int((state==2).sum()),queries=len(state),lower=lower,upper=upper,action=action)


def invalid_case(queries,classes,features,reason):
    q,c=queries,classes;lo=np.full((q,c),-np.inf);hi=np.zeros((q,c));winner=np.full(q,-1,dtype=np.int64)
    arrays=dict(full_center=np.zeros((c,features)),common_center=np.zeros((c,features)),
                full_radius=np.full(c,np.inf),common_radius=np.full(c,np.inf),
                full_lower=lo,full_upper=hi,common_lower=lo.copy(),common_upper=hi.copy(),
                full_winner=winner,common_winner=winner.copy(),
                full_normalized_cert_margin=np.zeros(q),common_normalized_cert_margin=np.zeros(q))
    state,summary=threeway(winner,winner);arrays['state']=state
    return arrays,dict(status='abstain',reason=reason,**summary)


def make_pre_case(a,rec,method,rep):
    Q=len(a['query']);C=len(a['anchor']);D=a['anchor'].shape[1]
    try:
        if any(not np.isfinite(value).all() for value in a.values()):raise ValueError('Nonfinite pre array')
        if np.any(a['covariance_diagonal']<0):raise ValueError('Negative covariance diagonal')
        if np.any(a['query_norm_squared']<0):raise ValueError('Negative query squared norm')
        n=rec['n_unlabeled'];pi=a['mean_assignment'];nu=a['projected_variance']
        if not np.allclose(a['mass0'],n*pi,atol=1e-12,rtol=1e-10):raise ValueError('mass0 formula mismatch')
        expected2=n*pi*(1+.5*(nu-pi@nu))
        if not np.allclose(a['mass2'],expected2,atol=1e-12,rtol=1e-10):raise ValueError('mass2 formula mismatch')
        geom=core.geometry_certificate(a['anchor'],a['mean'],a['support_counts'],pi,a['moment_soft'],nu,
            rec['node_radius'],rec['node_eta_upper'],n,rep,method)
        original='moment_mass0' if method=='moment0' else 'moment_mass2'
        if not np.allclose(geom['full_center'],a[f'{original}__r{rep}__prototype'],atol=1e-12,rtol=1e-10):
            raise ValueError('Frozen R25 prototype mismatch')
        cert=core.query_certificate(geom,a['query_norm_squared'],a['query_anchor_dot'],
             a['query_mean_dot'],a['query_moment_soft_dot'])
        yf,mf=protected_winner(cert['full']);yc,mc=protected_winner(cert['common']);state,summary=threeway(yf,yc)
        result={key:geom[key] for key in ('full_center','common_center','full_radius','common_radius','B','J','delta','mass_ratio_lower','mhat')}
        result.update(full_winner=yf,common_winner=yc,state=state,
                      full_normalized_cert_margin=mf,common_normalized_cert_margin=mc)
        for part in ('full','common'):
            for end in ('lower','upper'):result[f'{part}_{end}']=cert[part][end]
        invalid_distance=int((~cert['full']['valid'].all(1)|~cert['common']['valid'].all(1)).sum())
        return result,dict(status='bounded',reason='',invalid_query_distances=invalid_distance,**summary)
    except (ValueError,KeyError,FloatingPointError) as exc:
        result,summary=invalid_case(Q,C,D,str(exc));summary['invalid_query_distances']=0
        return result,summary


def seal_pre(contract_path):
    frozen,contract_hash=check_contract(contract_path)
    expected=list(identities());missing=[]
    for name,hops,seed in expected:
        src,_=paths(name,hops,seed)
        for suffix in ('.pre.json','.pre.npz'):
            if not src.with_suffix(suffix).is_file():missing.append(str(src.with_suffix(suffix)))
    if missing:raise FileNotFoundError(f'Need all 420 pre pairs before certification; missing {len(missing)} files')
    # No R25 post path is inspected, opened or hashed anywhere in this phase.
    started=time.time();units=[];all_rows=[];allowed=['anchor','mean','support_counts','mean_assignment',
        'projected_variance','moment_soft','mass0','mass2','covariance_diagonal','query','query_norm_squared',
        'query_anchor_dot','query_mean_dot','query_moment_soft_dot']
    for name,hops,seed in expected:
        src,dst=paths(name,hops,seed);source_record=src.with_suffix('.pre.json');source_artifact=src.with_suffix('.pre.npz')
        rec=read(source_record);assert (rec['dataset'],rec['hops'],rec['seed'])==(name,hops,seed)
        assert rec['contract_sha256']==frozen['r25_contract_sha256']
        source_npz_hash=sha(source_artifact);source_json_hash=sha(source_record)
        assert source_npz_hash==rec['pre_artifact_sha256']
        output_npz=dst.with_suffix('.precert.npz');output_json=dst.with_suffix('.precert.json')
        if output_json.exists():
            old=read(output_json)
            assert old['r26_contract_sha256']==contract_hash and old['source_pre_sha256']==source_npz_hash
            assert old['source_pre_record_sha256']==source_json_hash and old['artifact_sha256']==sha(output_npz)
            unit=old
        else:
            with np.load(source_artifact) as package:
                arrays={key:package[key].copy() for key in allowed}
                for method in ('moment_mass0','moment_mass2'):
                    for rep in REPS:
                        key=f'{method}__r{rep}__prototype'
                        if key in package:arrays[key]=package[key].copy()
            pack={'query':arrays['query']};rows=[];begin=time.perf_counter()
            for method in METHODS:
                for rep in REPS:
                    result,summary=make_pre_case(arrays,rec,method,rep);key=f'{method}__r{rep}'
                    for field,value in result.items():pack[key+'__'+field]=value
                    rows.append(dict(dataset=name,hops=hops,seed=seed,method=method,replication=rep,
                        eta=rec['node_eta_upper'],declared_domain=rec['node_eta_upper']<=.1,**summary))
            calculation_seconds=time.perf_counter()-begin;start=time.perf_counter();save_npz(output_npz,pack)
            unit=dict(dataset=name,hops=hops,seed=seed,r26_contract_sha256=contract_hash,
                source_pre_sha256=source_npz_hash,source_pre_record_sha256=source_json_hash,
                artifact_sha256=sha(output_npz),phase='pre only; no R25 post data accessed',rows=rows,
                calculation_seconds=calculation_seconds,serialization_seconds=time.perf_counter()-start,finished=time.time())
            save_json(output_json,unit)
        all_rows.extend(unit['rows'])
        units.append(dict(dataset=name,hops=hops,seed=seed,source_pre_sha256=source_npz_hash,
             source_pre_record_sha256=source_json_hash,artifact_sha256=sha(output_npz),record_sha256=sha(output_json)))
    assert len(units)==420 and len(all_rows)==1680
    seal=dict(status='all_pre_sealed',r26_contract_sha256=contract_hash,units=units,unit_count=420,
              branch_count=1680,pre_data_only=True,actual_q_N_u_or_labels_accessed=False,
              started=started,sealed_at=time.time())
    seal_path=OUT/'pre_phase_seal.json'
    if seal_path.exists():
        old=read(seal_path);assert old['units']==units and old['r26_contract_sha256']==contract_hash
    else:save_json(seal_path,seal)
    write_csv(REPORT/'pre_only_branches.csv',all_rows)
    print(json.dumps(dict(status='all_pre_sealed',units=420,branches=1680,seal_sha256=sha(seal_path)),indent=2))


def verify_post(contract_path):
    frozen,contract_hash=check_contract(contract_path);seal_path=OUT/'pre_phase_seal.json'
    seal=read(seal_path);assert seal['status']=='all_pre_sealed' and seal['unit_count']==420
    assert seal['branch_count']==1680 and seal['r26_contract_sha256']==contract_hash and seal['pre_data_only']
    assert len(seal['units'])==420
    assert {(u['dataset'],u['hops'],u['seed']) for u in seal['units']}==set(identities())
    # Verify every pre certificate and source BEFORE the first post-data access.
    for unit in seal['units']:
        src,dst=paths(unit['dataset'],unit['hops'],unit['seed'])
        for path,key in [(src.with_suffix('.pre.npz'),'source_pre_sha256'),(src.with_suffix('.pre.json'),'source_pre_record_sha256'),
                         (dst.with_suffix('.precert.npz'),'artifact_sha256'),(dst.with_suffix('.precert.json'),'record_sha256')]:
            assert sha(path)==unit[key],('Changed pre material',path)
    started=time.time();rows=[];source_post_hashes=[]
    counters=dict(branches=0,wrong_full_winner=0,wrong_common_winner=0,wrong_threeway=0,
        certified_full_near_tie=0,certified_common_near_tie=0,
        full_ball_violations_beyond_audit_tolerance=0,common_ball_violations_beyond_audit_tolerance=0,
        max_full_ball_raw_violation=0.,max_common_ball_raw_violation=0.,interval_count_violations=0)
    for unit in seal['units']:
        name,hops,seed=unit['dataset'],unit['hops'],unit['seed'];src,dst=paths(name,hops,seed)
        meta=read(dst.with_suffix('.precert.json'))
        assert meta['finished']<=seal['sealed_at']
        with np.load(dst.with_suffix('.precert.npz')) as pack:p={k:pack[k].copy() for k in pack.files}
        # Phase B starts only after the all-pre validation above.
        rec=read(src.with_suffix('.json'));postpath=src.with_suffix('.post.npz')
        assert rec['contract_sha256']==frozen['r25_contract_sha256']
        assert rec['pre_artifact_sha256']==unit['source_pre_sha256']
        assert rec['pre_record_sha256']==unit['source_pre_record_sha256']
        posthash=sha(postpath);assert posthash==rec['post_artifact_sha256']
        source_post_hashes.append(dict(dataset=name,hops=hops,seed=seed,post_sha256=posthash,record_sha256=sha(src.with_suffix('.json'))))
        with np.load(postpath) as post:
            actual={}
            for rep in REPS:
                for kind in ('full_ren','common_true_mass'):
                    for field in ('prototype','prediction','normalized_gap'):
                        key=f'{kind}__r{rep}__{field}';actual[key]=post[key].copy()
            # No access to full_soft, full_mass, query_full_soft_dot, or labels.
        for branch in meta['rows']:
            method,rep=branch['method'],branch['replication'];prefix=f'{method}__r{rep}__'
            yf=p[prefix+'full_winner'];yc=p[prefix+'common_winner'];state=p[prefix+'state']
            pyf=actual[f'full_ren__r{rep}__prediction'];pyc=actual[f'common_true_mass__r{rep}__prediction']
            actual_diff=pyf!=pyc;wrongf=int(((yf>=0)&(yf!=pyf)).sum());wrongc=int(((yc>=0)&(yc!=pyc)).sum())
            wrong=int(((state==0)&actual_diff).sum()+((state==1)&~actual_diff).sum())
            raw={};violations={};near={}
            for kind,source_kind,winner in [('full','full_ren',yf),('common','common_true_mass',yc)]:
                proto=actual[f'{source_kind}__r{rep}__prototype'];center=p[prefix+kind+'_center'];radius=p[prefix+kind+'_radius']
                error=np.linalg.norm(proto-center,axis=1)-radius
                raw[kind]=max(0.,float(np.max(error)));violations[kind]=int((error>AUDIT_TOL).sum())
                near[kind]=int(((winner>=0)&(actual[f'{source_kind}__r{rep}__normalized_gap']<=GAP)).sum())
                counters[kind+'_ball_violations_beyond_audit_tolerance']+=violations[kind]
                counters['max_'+kind+'_ball_raw_violation']=max(counters['max_'+kind+'_ball_raw_violation'],raw[kind])
                counters['certified_'+kind+'_near_tie']+=near[kind]
            value=float(actual_diff.mean());out_of_count=not (branch['lower']<=value<=branch['upper'])
            counters['interval_count_violations']+=int(out_of_count)
            counters['wrong_full_winner']+=wrongf;counters['wrong_common_winner']+=wrongc;counters['wrong_threeway']+=wrong;counters['branches']+=1
            rows.append(dict(**branch,actual_disagreement=value,wrong_full_winner=wrongf,wrong_common_winner=wrongc,
                wrong_threeway=wrong,full_ball_raw_violation=raw['full'],common_ball_raw_violation=raw['common'],
                full_ball_violations=violations['full'],common_ball_violations=violations['common'],
                certified_full_near_tie=near['full'],certified_common_near_tie=near['common']))
    bad=sum(counters[k] for k in ('wrong_full_winner','wrong_common_winner','wrong_threeway','certified_full_near_tie',
        'certified_common_near_tie','full_ball_violations_beyond_audit_tolerance','common_ball_violations_beyond_audit_tolerance','interval_count_violations'))
    summary=[]
    for name in DATASETS:
        for hops in HOPS:
            for method in METHODS:
                for rep in REPS:
                    group=[r for r in rows if (r['dataset'],r['hops'],r['method'],r['replication'])==(name,hops,method,rep)]
                    summary.append(dict(dataset=name,hops=hops,method=method,replication=rep,draws=len(group),
                        domain_draws=sum(r['declared_domain'] for r in group),ignorable=sum(r['action']=='ignorable' for r in group),
                        must_keep=sum(r['action']=='must_keep' for r in group),abstain=sum(r['action']=='abstain' for r in group),
                        lower_mean_pct=100*float(np.mean([r['lower'] for r in group])),
                        upper_mean_pct=100*float(np.mean([r['upper'] for r in group])),
                        actual_mean_pct=100*float(np.mean([r['actual_disagreement'] for r in group]))))
    keep=sum(r['action']=='must_keep' for r in rows);ignore=sum(r['action']=='ignorable' for r in rows)
    decision='go_nonempty_preservation_certificate' if not bad and keep else 'stop_archive_no_rule_tuning'
    verification=dict(status='complete',**counters,audit_passed=not bad,decision=decision,must_keep_branches=keep,
        ignorable_branches=ignore,abstain_branches=len(rows)-keep-ignore,
        pre_phase_seal_sha256=sha(seal_path),r26_contract_sha256=contract_hash,
        actual_q_N_u_or_label_arrays_accessed=False,prototype_and_canonical_predictions_only=True,
        post_audit_seconds=time.time()-started,source_post_hashes=source_post_hashes,
        limitations=['Ordinary float64, not outward-rounded interval arithmetic',
         'No local reconstruction of full q, covariance or actual scores; post prototype balls and canonical predictions are audited',
         'Same graphs with new support/depth conditions, not independent new graphs',
         'No query labels or Accuracy; keep means necessary to preserve this operator output'])
    write_csv(REPORT/'branches.csv',rows);write_csv(REPORT/'summary.csv',summary);save_json(REPORT/'verification.json',verification)
    lines=['# R26：事前位置贡献证书的离线核验','',f'420组/1680分支完成；审计通过：{not bool(bad)}。决定：`{decision}`。','',
        f'整组1%预算下：必须保留{keep}分支，可忽略{ignore}分支，弃权{len(rows)-keep-ignore}分支。两种k分别报告，没有择优选择。','',
        '这里的“必须保留”只表示删除类别专属位置不能保持原Ren预测，不表示更高Accuracy。',
        '所有420组区间和三分规则在读取任何post数据前封存；核验阶段也未读取真实N/u/q或标签数组。',
        '普通float64只核验解析界，不是严格区间算术。完整分数未重建；实际原型落球和规范赢家分别核验。','',
        '|数据集|h|矩规则|r|域内|可忽略|必须保留|弃权|平均下界%|平均上界%|实际异判%|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f'|{r["dataset"]}|{r["hops"]}|{r["method"]}|{r["replication"]}|{r["domain_draws"]}|{r["ignorable"]}|{r["must_keep"]}|{r["abstain"]}|{r["lower_mean_pct"]:.6f}|{r["upper_mean_pct"]:.6f}|{r["actual_mean_pct"]:.6f}|')
    (REPORT/'Ren事前位置贡献证书_核验报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:verification[k] for k in ('status','branches','audit_passed','decision','must_keep_branches','ignorable_branches','abstain_branches')},indent=2))


def selfcheck():
    core.selfcheck()
    near=dict(nominal=np.array([[0.,-1e-11],[0.,-1e-8]]),lower=np.array([[0.,-1e-11],[0.,-1e-8]]),
              upper=np.array([[0.,-1e-11],[0.,-1e-8]]),valid=np.ones((2,2),bool))
    winner,_=protected_winner(near);assert np.array_equal(winner,[-1,0])
    state,summary=threeway(np.array([0,1,-1]),np.array([0,0,1]));assert np.array_equal(state,[0,1,2])
    assert summary['lower']==1/3 and summary['upper']==2/3 and summary['action']=='must_keep'
    invalid,record=invalid_case(3,2,4,'synthetic invalid');assert record['action']=='abstain'
    result=dict(status='passed',scope='Only synthetic NumPy and policy tests; no R25 data accessed',
        runner_sha256=sha(__file__),pure_functions_sha256=sha(PURE),normalized_guard=GAP)
    save_json(REPORT/'local_selfcheck.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contract',type=Path,default=CONTRACT)
    mode=parser.add_mutually_exclusive_group(required=True)
    for option in ('freeze-only','pre-only','verify-only','self-check'):mode.add_argument('--'+option,action='store_true')
    args=parser.parse_args()
    if args.self_check:selfcheck();return
    if args.freeze_only:
        value=payload()
        if args.contract.exists():assert read(args.contract)==value,'Existing R26 contract differs'
        else:save_json(args.contract,value)
        print(args.contract);return
    if not args.contract.exists():raise FileNotFoundError('Review and freeze R26 before processing real pre data')
    if args.pre_only:seal_pre(args.contract)
    elif args.verify_only:verify_post(args.contract)


if __name__=='__main__':main()
