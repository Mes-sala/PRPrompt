"""Independent local audit of R24 sealed predictions and saved trajectories.

Never imports the runner, torch, graph loaders, or query labels. Recomputes
readouts, endpoint errors, rankings and all-point fits with NumPy. It cannot
recreate the entire unlabeled pool from the exported query/support coordinates;
that limitation is explicit in its report. No server access or experiment launch.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
DATASETS=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell',
          'Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings']
TAU=.05
RESOLUTION=100*np.finfo(np.float64).eps


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text())
def load_npz(path):
    with np.load(path,allow_pickle=False) as f: return {k:f[k] for k in f.files}
def norm(x): return float(np.linalg.norm(x))
def cc(x): return x-x.mean(axis=0,keepdims=True)
def rc(x): return x-x.mean(axis=1,keepdims=True)


def json_value(x):
    if isinstance(x,np.generic): return json_value(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict): return {str(k):json_value(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_value(v) for v in x]
    return x


def write_json(path,x):
    Path(path).write_text(json.dumps(json_value(x),ensure_ascii=False,indent=2,allow_nan=False))


class Audit:
    def __init__(self): self.items=[]
    def check(self,name,ok,**data):
        self.items.append(dict(check=name,passed=bool(ok),**json_value(data)))
    def close(self,name,a,b,rtol=1e-8,atol=1e-12):
        a,b=np.asarray(a),np.asarray(b)
        ok=a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False)
        error=float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None
        self.check(name,ok,max_absolute_error=error,rtol=rtol,atol=atol)
    @property
    def failures(self): return [x for x in self.items if not x['passed']]


def cosine_D(v,A,H,eps):
    # Independently derive distance to normalized prototype in radial/tangent
    # coordinates. The radial distance is rationalized, not subtracted near 1.
    vn=np.einsum('ij,ij->i',v,v)
    r=1/np.sqrt(1+eps*eps*vn)
    s=np.sqrt(A*A+eps*eps*np.einsum('ij,ij->i',H,H))
    ah=A/s; hh=H/s[:,None]; hn=np.einsum('ij,ij->i',hh,hh)
    gap=eps*(hn[None,:]-(r*r*vn)[:,None])/(r[:,None]+ah[None,:])
    return 2*(r[:,None]*(v@hh.T))-hn[None,:]-gap*gap


def euclidean_D(v,B,H,eps):
    vn=np.einsum('ij,ij->i',v,v); r=1/np.sqrt(1+eps*eps*vn)
    rho=vn*r*r/(1+r)
    return 2*r[:,None]*(v@H.T)-(H*H).sum(1)[None,:]-eps*eps*(B[None,:]-rho[:,None])**2


def prototype_error(radial,tangent,leading,eps,domain):
    actual=np.concatenate([radial[:,None],eps*tangent],axis=1)
    prediction=np.concatenate([np.zeros((len(radial),1)),leading],axis=1)
    return error_summary(actual,prediction,domain,RESOLUTION*max(1.,math.sqrt(len(radial))))


def matrix_error(actual,prediction,domain,reference_scale=1.):
    return error_summary(actual,prediction,domain,
                         RESOLUTION*max(1.,reference_scale)*math.sqrt(actual.size))


def error_summary(actual,prediction,domain,threshold):
    a,p=norm(actual),norm(prediction); error=norm(actual-prediction)
    resolved=a>threshold and p>threshold
    status='zero_leading_coefficient' if p==0 else ('resolved' if resolved else 'unresolved')
    dot=float(np.sum(actual*prediction))
    return dict(amplitude=a,predicted_amplitude=p,error=error,
        relative_error=error/p if p else None,direction_cosine=dot/(a*p) if a*p else None,
        coefficient_gain=dot/(p*p) if p else None,resolution_threshold=threshold,
        resolved=resolved,zero_leading_coefficient=p==0,domain=bool(domain),
        fit_eligible=bool(domain and resolved),numerical_status=status)


def ranking(a,b,factor):
    pa=a.argmax(1); pb=b.argmax(1); changed=pa!=pb
    # Two largest values by partition; classification uses saved score arrays.
    ta=np.partition(a,-2,axis=1)[:,-2:]; tb=np.partition(b,-2,axis=1)[:,-2:]
    ga=factor*(ta.max(1)-ta.min(1)); gb=factor*(tb.max(1)-tb.min(1))
    unresolved=(ga<=RESOLUTION)|(gb<=RESOLUTION)
    return dict(disagreement_count=int(changed.sum()),nodes=len(a),
        disagreement_fraction=float(changed.mean()),unresolved_gap_either_count=int(unresolved.sum()),
        unresolved_gap_either_fraction=float(unresolved.mean()),
        disagreement_outside_unresolved_count=int((changed&~unresolved).sum()),
        min_physical_gap=float(min(ga.min(),gb.min())))


def check_metric(audit,prefix,actual,saved):
    # Independent mean/norm reductions may differ close to cancellation. This
    # small audit allowance is one tenth of the frozen resolution threshold;
    # it never affects either independently calculated eligibility boolean.
    rounding_allowance=.1*actual['resolution_threshold']
    for key in ['amplitude','predicted_amplitude','error','relative_error','resolution_threshold']:
        if actual[key] is None or saved[key] is None:
            audit.check(prefix+'/'+key,actual[key] is None and saved[key] is None)
        else:
            # Norm reduction implementations can differ; this tolerance audits
            # computation only and NEVER changes domain or eligibility thresholds.
            allowance=rounding_allowance
            if key=='relative_error': allowance/=max(actual['predicted_amplitude'],1e-300)
            if key=='resolution_threshold': allowance=0.
            audit.close(prefix+'/'+key,actual[key],saved[key],rtol=2e-7,atol=allowance)
    for key in ['resolved','zero_leading_coefficient','domain','fit_eligible']:
        audit.check(prefix+'/'+key,actual[key]==saved[key],independent=actual[key],runner=saved[key])


def check_ranking(audit,prefix,actual,saved):
    for key in ['disagreement_count','nodes','unresolved_gap_either_count','disagreement_outside_unresolved_count']:
        audit.check(prefix+'/'+key,actual[key]==saved[key],independent=actual[key],runner=saved[key])


def precheck(p,meta,audit,prefix):
    # These formulas do not use any trajectory output. The covariance actions
    # themselves remain presealed inputs because full unlabeled v is not exported.
    v=p['query_v']; m=p['support_m']; g=p['mean_v']; B=m-m.mean(0)
    labels=p['support_labels']; classes=len(m)
    recomputed=np.stack([p['support_v'][labels==c].mean(0) for c in range(classes)])
    audit.close(prefix+'/support means',recomputed,m)
    audit.close(prefix+'/class contrast',B,p['class_contrast'])
    audit.check(prefix+'/support-query disjoint',not np.intersect1d(p['support'],p['query']).size)
    audit.check(prefix+'/five supports per class',bool(np.all(p['support_counts']==5)))
    observed_radius=max(np.linalg.norm(v,axis=1).max(),np.linalg.norm(p['support_v'],axis=1).max())
    audit.check(prefix+'/exported coordinate radius bounded by full radius',observed_radius<=meta['radius']*(1+1e-12))
    sigmaB=p['covariance_power_1']; mt=.5*m+.5*g
    fixed=v@sigmaB.T-np.einsum('ij,ij->i',mt,sigmaB)[None,:]
    audit.close(prefix+'/fixed score coefficient',fixed,p['fixed_score_coefficient'])
    for t in range(1,4):
        act=p[f'covariance_power_{t}']
        audit.close(prefix+f'/unanchored coefficient {t}',(v-g)@act.T,p[f'unanchored_score_coefficient_{t}'])
        centered_error=norm(act.mean(0))/max(1.,norm(act))
        audit.check(prefix+f'/centered covariance action {t}',centered_error<=1e-10,relative_centering_error=centered_error)
    kappa=classes*5/meta['n_unlabeled']; gamma0=kappa/(1+kappa)
    audit.close(prefix+'/Ren kappa',kappa,p['ren_kappa'])
    audit.close(prefix+'/Ren full vector coefficient',kappa*(m-g)+2*sigmaB,p['ren_full_prototype_coefficient'])
    audit.close(prefix+'/Ren common vector coefficient',kappa*(m-g),p['ren_common_prototype_coefficient'])
    audit.close(prefix+'/Ren full score coefficient',2*(v-g)@(kappa*B+2*sigmaB).T,p['ren_full_limit_score'])
    audit.close(prefix+'/Ren common score coefficient',2*kappa*(v-g)@B.T,p['ren_common_limit_score'])
    audit.close(prefix+'/Ren fixed-replication center',gamma0*m+(1-gamma0)*g,p['ren_fixed_replication_tangent_center'])
    for beta in [.5,1.]:
        for t in range(1,4):
            h=p[f'limit_b{beta:g}_t{t}_H']
            expected=2*v@h.T-(h*h).sum(1)[None,:]
            audit.close(prefix+f'/scaled limit readout b{beta:g} t{t}',expected,p[f'limit_b{beta:g}_t{t}_D'])
    return dict(independent_recomputed='Support means, coefficient readouts, score identities, errors, rankings, fits',
        not_independently_recomputed='Full-unlabeled covariance actions, tangent-limit assignments, full graph radius and propagation; only hash-linked runtime records available',
        limit_disagreement=ranking(p['ren_full_limit_score'],p['ren_common_limit_score'],1.))


def analyze_cosine(package,row,p,audit):
    eps=row['epsilon']; beta=row['beta']; temperature=row['temperature']; k=row['k']
    domain=eps*row['radius']<=.1 and eps*eps*row['radius']**2/TAU<=.1
    audit.check('domain',domain==row['fixed_temperature_local_domain'])
    audit.check('epsilon matches frozen grid',eps==2.**(-k) and k in range(13))
    audit.check('temperature matches branch',row['tau']==(TAU if temperature=='fixed' else TAU*eps*eps))
    audit.close('common readout',cosine_D(p['query_v'],package['common_A'],package['common_H'],eps),package['common_D'])
    As,Hs=package['anchor_A'],package['anchor_H']; Dc=package['common_D']; rows=[]
    initial=prototype_error(cc(As),cc(Hs),eps*p['class_contrast'],eps,domain)
    for t in range(1,4):
        A,H,D=package[f't{t}_A'],package[f't{t}_H'],package[f't{t}_D']
        recomputed=cosine_D(p['query_v'],A,H,eps)
        audit.close(f't{t}/readout',recomputed,D)
        # Canonical rankings stay tied to saved D, while cross-machine sensitive
        # near-tie readout recomputation is reported separately, not overwritten.
        saved_vs_recomputed=ranking(D,recomputed,.5*eps*eps)
        metrics={'initial_contrast':initial}; ranks={'full_vs_common':ranking(D,Dc,.5*eps*eps)}
        if temperature=='fixed' and beta==.5:
            factor=1. if t==1 else .5; gamma=.5*factor
            metrics['feedback']=prototype_error(package[f't{t}_soft_A']-package['mean_A'],
                package[f't{t}_soft_H']-package['mean_H'],factor*eps**3/TAU*p['covariance_power_1'],eps,domain)
            metrics['prototype_vs_common']=prototype_error(A-package['common_A'],H-package['common_H'],
                gamma*eps**3/TAU*p['covariance_power_1'],eps,domain)
            actual=.5*eps*eps*(D-Dc); prediction=gamma*eps**4/TAU*p['fixed_score_coefficient']
            metrics['score_vs_common']=matrix_error(actual,prediction,domain)
            audit.close(f't{t}/physical difference',actual,package[f't{t}_physical_score_difference'])
            ranks['common_vs_tangent_nearest_center']=ranking(Dc,p['tangent_common_D'],.5*eps*eps)
            ranks['full_vs_predicted_corrected_common']=ranking(D,Dc+2*prediction/(eps*eps),.5*eps*eps)
        elif temperature=='fixed':
            metrics['class_contrast']=prototype_error(cc(A),cc(H),eps**(2*t+1)/TAU**t*p[f'covariance_power_{t}'],eps,domain)
            prediction=eps**(2*t+2)/TAU**t*p[f'unanchored_score_coefficient_{t}']
            actual=.5*eps*eps*rc(D)
            metrics['centered_score']=matrix_error(actual,prediction,domain)
            audit.close(f't{t}/physical centered score',actual,package[f't{t}_physical_centered_score'])
            ranks['full_vs_leading_margin_rule']=ranking(D,2*prediction/(eps*eps),.5*eps*eps)
        else:
            lh=p[f'limit_b{beta:g}_t{t}_H']; ld=p[f'limit_b{beta:g}_t{t}_D']
            metrics['tangent_prototype_to_limit']=matrix_error(H,lh,False)
            metrics['centered_D_to_limit']=matrix_error(rc(D),rc(ld),False)
            ranks['full_vs_frozen_tangent_limit']=ranking(D,ld,.5*eps*eps)
        if k==0:
            canonical=package[f't{t}_canonical_cosine']
            ranks['canonical_cosine_vs_stable_D']=ranking(canonical,.5*D,1.)
        for key,m in metrics.items(): check_metric(audit,f't{t}/metric/{key}',m,row['rounds'][str(t)]['metrics'][key])
        for key,r in ranks.items(): check_ranking(audit,f't{t}/ranking/{key}',r,row['rounds'][str(t)]['ranking'][key])
        rows.append(dict(operator='cosine',dataset=row['dataset'],k=k,epsilon=eps,beta=beta,
            temperature=temperature,round=t,domain=domain,metrics=metrics,ranking=ranks,
            readout_recomputed_vs_saved=saved_vs_recomputed,
            q_probability_zero_count=row['rounds'][str(t)].get('q_probability_zero_count'),
            mass_underflow_count=int(np.sum(package[f't{t}_mass']==0))))
    return rows


def analyze_ren(package,row,p,audit):
    eps=row['epsilon']; k=row['k']; policy=row['replication_policy']
    domain=eps*row['radius']<=.1 and eps*eps*row['radius']**2/TAU<=.1
    audit.check('domain',domain==row['fixed_temperature_local_domain'])
    audit.check('replication policy',row['replication']==(1 if policy=='one' else 4**k))
    v=p['query_v']; H=package['full_H']; Hc=package['common_H']; B=package['full_B']; Bc=package['common_B']
    D,Dc=package['full_D'],package['common_D']; gb,gh=package['mean_B'],package['mean_H']
    audit.close('Ren full readout',euclidean_D(v,B,H,eps),D)
    audit.close('Ren common readout',euclidean_D(v,Bc,Hc,eps),Dc)
    logit=np.log(p['support_counts'])-math.log(row['replication'])-package['log_class_mass']
    gamma=np.exp(-np.logaddexp(0,-logit))
    audit.close('Ren support weight',gamma,package['support_weight'])
    audit.close('Ren mixed H',gamma[:,None]*package['anchor_H']+(1-gamma)[:,None]*package['soft_H'],H)
    audit.close('Ren common mixed H',gamma[:,None]*package['anchor_H']+(1-gamma)[:,None]*gh,Hc)
    metrics={'soft_position':prototype_error(-eps*eps*(package['soft_B']-gb),package['soft_H']-gh,
        2*eps**3*p['covariance_power_1'],eps,domain)}
    ranks={'full_vs_true_mass_common':ranking(D,Dc,eps*eps)}
    if policy=='inverse_epsilon_squared':
        metrics['full_vs_mean']=prototype_error(-eps*eps*(B-gb),H-gh,eps**3*p['ren_full_prototype_coefficient'],eps,domain)
        metrics['common_vs_mean']=prototype_error(-eps*eps*(Bc-gb),Hc-gh,eps**3*p['ren_common_prototype_coefficient'],eps,domain)
        actual=eps*eps*rc(D); common=eps*eps*rc(Dc)
        metrics['centered_score']=matrix_error(actual,eps**4*p['ren_full_limit_score'],domain,4.)
        metrics['common_centered_score']=matrix_error(common,eps**4*p['ren_common_limit_score'],domain,4.)
        metrics['support_weight_scaled']=matrix_error((gamma/eps**2)[:,None],np.full((len(gamma),1),float(p['ren_kappa'])),False)
        audit.close('Ren centered physical score',actual,package['physical_centered_score'])
        audit.close('Ren common centered physical score',common,package['physical_common_centered_score'])
        ranks['full_vs_joint_limit']=ranking(D,eps*eps*p['ren_full_limit_score'],eps*eps)
        ranks['common_vs_joint_limit']=ranking(Dc,eps*eps*p['ren_common_limit_score'],eps*eps)
        ranks['joint_limit_full_vs_common']=ranking(p['ren_full_limit_score'],p['ren_common_limit_score'],eps**4)
    else:
        metrics['tangent_H_to_fixed_replication_limit']=matrix_error(H,p['ren_fixed_replication_tangent_center'],False)
        metrics['centered_D_to_fixed_replication_limit']=matrix_error(rc(D),rc(p['ren_fixed_replication_limit_D']),False)
        ranks['full_vs_fixed_replication_limit']=ranking(D,p['ren_fixed_replication_limit_D'],eps*eps)
        ranks['common_vs_fixed_replication_limit']=ranking(Dc,p['ren_fixed_replication_limit_D'],eps*eps)
    if k==0:
        ranks['canonical_euclidean_vs_stable_D']=ranking(package['canonical_negative_squared_euclidean'],eps*eps*D,1.)
    for key,m in metrics.items(): check_metric(audit,'metric/'+key,m,row['metrics'][key])
    for key,r in ranks.items(): check_ranking(audit,'ranking/'+key,r,row['ranking'][key])
    return dict(operator='ren',dataset=row['dataset'],k=k,epsilon=eps,replication_policy=policy,
        round=1,domain=domain,metrics=metrics,ranking=ranks,
        q_probability_zero_count=row.get('q_probability_zero_count'),
        mass_underflow_count=int(np.sum(package['class_mass']==0)))


def expected_order(row,key):
    if row['operator']=='ren':
        return 4 if key in ['centered_score','common_centered_score'] else (3 if key in ['soft_position','full_vs_mean','common_vs_mean'] else None)
    if row['temperature']!='fixed': return None
    if key=='initial_contrast': return 1
    if key in ['feedback','prototype_vs_common']: return 3
    if key=='score_vs_common': return 4
    if key=='class_contrast': return 2*row['round']+1
    if key=='centered_score': return 2*row['round']+2
    return None


def summarize_groups(rows):
    groups={}
    for row in rows:
        family=(row.get('temperature'),row.get('beta')) if row['operator']=='cosine' else (row['replication_policy'],)
        for key,metric in row['metrics'].items():
            identity=(row['dataset'],row['operator'],*family,row['round'],key)
            groups.setdefault(identity,[]).append((row,metric))
    output=[]
    for key,values in groups.items():
        values.sort(key=lambda z:z[0]['k'])
        row0=values[0][0]; valid=[(r,m) for r,m in values if m['fit_eligible']]
        point=[dict(k=r['k'],epsilon=r['epsilon'],**m) for r,m in values]
        result=dict(dataset=row0['dataset'],operator=row0['operator'],temperature=row0.get('temperature'),
            beta=row0.get('beta'),replication_policy=row0.get('replication_policy'),round=row0['round'],
            metric=key[-1],expected_order=expected_order(row0,key[-1]),points=point,
            domain_points=sum(m['domain'] for r,m in values),
            unresolved_points=sum(not m['resolved'] for r,m in values),
            domain_unresolved_points=sum(m['domain'] and not m['resolved'] for r,m in values),
            eligible_k=[r['k'] for r,m in valid],slope=None,log_rmse=None)
        if result['expected_order'] is not None and len(valid)>=3:
            x=np.log([r['epsilon'] for r,m in valid]); y=np.log([m['amplitude'] for r,m in valid])
            fit=np.polyfit(x,y,1)
            result.update(slope=float(fit[0]),log_rmse=float(np.sqrt(np.mean((y-np.polyval(fit,x))**2))),
                          slope_minus_expected=float(fit[0]-result['expected_order']))
        else:
            result['slope_reason']='not a predeclared power-law test' if result['expected_order'] is None else 'fewer than 3 domain/resolved points'
        if valid:
            result.update(first_eligible_relative_error=valid[0][1]['relative_error'],
                last_eligible_relative_error=valid[-1][1]['relative_error'],
                first_eligible_k=valid[0][0]['k'],last_eligible_k=valid[-1][0]['k'],
                error_increase_steps=sum(valid[i][1]['relative_error']>valid[i-1][1]['relative_error'] for i in range(1,len(valid))))
        output.append(result)
    return output


def fmt(x,digits=5): return '—' if x is None else f'{x:.{digits}g}'
def find_group(groups,name,operator,metric,**selectors):
    return next((g for g in groups if g['dataset']==name and g['operator']==operator and g['metric']==metric
                 and all(g.get(k)==v for k,v in selectors.items())),None)


def write_report(out,summary,groups,rows):
    lines=['# R24 真实图尺度干预：独立复算', '',
        f"状态：{summary['status']}。全部 14 图均列入；本轮每图仅一份支持 seed 271700，是开发性机制干预。",
        '', '**不读查询标签，不计算准确率。** 排名一致、稳定坐标仍有可分辨的 argmax 和理论系数吻合，均不能转写成正确率改善。',
        '', f"哈希与数值核验失败条目：{summary['audit_failure_count']}。核验失败时只报告诊断，不认定机制结果已经验收。",
        '', '## 覆盖与数值域', '', '|图|状态|余弦条件/52|Ren 条件/26|域内 k|', '|---|---|---:|---:|---|']
    for g in summary['graphs']:
        lines.append(f"|{g['dataset']}|{g['status']}|{g.get('cosine_conditions',0)}|{g.get('ren_conditions',0)}|{','.join(map(str,g.get('domain_k',[]))) or '无'}|")
    lines+=['','`k` 越大，集中程度越强。域固定为 εR≤0.1 且 ε²R²/0.05≤0.1；阈值为原协议的 100×float64 机器精度。域外和 unresolved 点均保留在原始表中，不进入幂律拟合。',
        '', '## 固定温度：半锚定三轮的四阶分数差', '',
        '|图|可用 k（全部）|拟合阶数 / 预测 4|首/末可用系数相对误差|域内 unresolved 点|', '|---|---|---:|---|---:|']
    for name in DATASETS:
        g=find_group(groups,name,'cosine','score_vs_common',temperature='fixed',beta=.5,round=3)
        lines.append(f"|{name}|{','.join(map(str,g['eligible_k'])) if g else '—'}|{fmt(g.get('slope')) if g else '—'}|{fmt(g.get('first_eligible_relative_error'))+'/'+fmt(g.get('last_eligible_relative_error')) if g else '—'}|{g['domain_unresolved_points'] if g else '—'}|")
    lines+=['','相对误差比较完整向量/分数与预封存主项，不能仅用拟合阶数接近 4 宣称系数正确。上表首末点由全部可用点的固定顺序决定，未挑选尾段。',
        '', '## 无锚定：三轮七阶原型对比与八阶分数', '',
        '|图|原型阶数（可用点）|分数阶数（可用点）|末可用原型/分数系数相对误差|', '|---|---:|---:|---|']
    for name in DATASETS:
        a=find_group(groups,name,'cosine','class_contrast',temperature='fixed',beta=1.,round=3)
        b=find_group(groups,name,'cosine','centered_score',temperature='fixed',beta=1.,round=3)
        cell=lambda g:f"{fmt(g.get('slope'))} ({len(g['eligible_k'])})" if g else '—'
        lines.append(f"|{name}|{cell(a)}|{cell(b)}|{fmt(a.get('last_eligible_relative_error')) if a else '—'}/{fmt(b.get('last_eligible_relative_error')) if b else '—'}|")
    lines+=['','若三轮高阶量在进入局部域前已低于原数值阈值，本轮不能实证检验该阶数；不能降低阈值或用其他尺度段替代。极小但符号有结构的真实数 margin，与 float64 的原生余弦可分辨能力分开。',
        '', '## Ren 复制—集中联合极限', '',
        '|图|四阶分数拟合（可用点）|末可用系数相对误差|预封存极限 full/common 分歧|k=12 实际 full/common 分歧|k=12 任一排名 gap unresolved|', '|---|---:|---:|---:|---:|---:|']
    for name in DATASETS:
        g=find_group(groups,name,'ren','centered_score',replication_policy='inverse_epsilon_squared')
        r=next((r for r in rows if r['dataset']==name and r['operator']=='ren' and r['replication_policy']=='inverse_epsilon_squared' and r['k']==12),None)
        coverage=next(c for c in summary['graphs'] if c['dataset']==name)
        frac=coverage.get('ren_limit_disagreement_fraction')
        ranks=r['ranking']['full_vs_true_mass_common'] if r else {}
        lines.append(f"|{name}|{fmt(g.get('slope'))+' ('+str(len(g['eligible_k']))+')' if g else '—'}|{fmt(g.get('last_eligible_relative_error')) if g else '—'}|{fmt(100*frac) if frac is not None else '—'}%|{fmt(100*ranks['disagreement_fraction']) if ranks else '—'}%|{fmt(100*ranks['unresolved_gap_either_fraction']) if ranks else '—'}%|")
    lines+=['','联合极限改变了取极限路径：位置反馈变小的同时，支持锚定权重也按 ε² 变小。因此非零极限分歧若成立，说明绝对位移不能独立判断决策影响；它不意味着复制增加信息或准确率。实际质量共同池是后验解释对照。',
        '', '## 温度缩放分支', '',
        '|图|β=.5 三轮 k=12 极限分数相对误差|β=1 三轮 k=12 极限分数相对误差|', '|---|---:|---:|']
    for name in DATASETS:
        vals=[]
        for beta in [.5,1.]:
            r=next((r for r in rows if r['dataset']==name and r['operator']=='cosine' and r['temperature']=='scaled' and r['beta']==beta and r['round']==3 and r['k']==12),None)
            vals.append(fmt(r['metrics']['centered_D_to_limit']['relative_error']) if r else '—')
        lines.append(f"|{name}|{vals[0]}|{vals[1]}|")
    lines+=['','这里只报告冻结的终点 k=12，同时全部 13 点已保留。切向极限事前计算过节点 softmax；它是跨条件预测，不是免节点计算的认证。',
        '', '## 复算范围与限制', '',
        '- 独立核验文件哈希、支持均值及预封存分数公式；从 A/H/B 与 query_v 重算稳定读出，并从保存的规范分数独立统计误差、排名、原阈值及全部可用点斜率。',
        '- 输出未保存整个未标注池 v，因此本脚本不重建协方差作用、切向极限的责任分配、全图半径和传播交换。它们作为带哈希的运行证据，不伪称独立完整重跑。',
        '- `all_points.csv` 保留全部指标和域外/unresolved 状态；`groups.json` 保留每组完整曲线、全可用点、系数方向与增益；`rankings.json` 单列近等分数与重算读出敏感性。',
        '- 数值审计容差只用于比较两个实现的数值结果，不改变实验域、100×机器精度分辨阈值或拟合入选条件。没有人为增加“顶刊通过”阈值。',
        '', '本轮能支撑的结论取决于上面的有效覆盖和误差曲线；不存在用少数漂亮图代替全范围验收的默认结论。']
    (out/'真实图尺度干预_独立复算.md').write_text('\n'.join(lines)+'\n')


def main_analysis(results,out,contract_path,allow_partial):
    contract=read_json(contract_path); contract_hash=sha(contract_path)
    assert contract['datasets']==DATASETS and contract['k']==list(range(13))
    assert contract['seed']==271700 and contract['tau0']==TAU and contract['resolution_constant']==100
    out.mkdir(parents=True,exist_ok=True)
    audit=Audit(); all_rows=[]; graphs=[]; files={str(contract_path):contract_hash}; saved_fits=[]
    global_done=results/'done.json'
    if global_done.exists():
        final=read_json(global_done); files[str(global_done)]=sha(global_done)
        audit.check('global completion contract',final['contract_sha256']==contract_hash)
        audit.check('global completion status',final['status']=='complete')
    for name in DATASETS:
        folder=results/name; donepath=folder/'done.json'
        if not donepath.exists():
            graphs.append(dict(dataset=name,status='pending_or_incomplete',cosine_conditions=0,ren_conditions=0)); continue
        done=read_json(donepath); files[str(donepath)]=sha(donepath)
        audit.check(name+'/contract link',done['contract_sha256']==contract_hash)
        if done['status']=='inapplicable_input_family':
            file=folder/'input_family_failure.npz'; audit.check(name+'/failure artifact hash',sha(file)==done['artifact_sha256'])
            files[str(file)]=sha(file); package=load_npz(file)
            indices=np.flatnonzero((~np.isfinite(package['axis_mass']))|(package['axis_mass']<=1e-14))
            audit.check(name+'/input-family exclusion',np.array_equal(indices,package['invalid_indices']) and len(indices)>0)
            audit.check(name+'/invalid node count',len(indices)==done['invalid_count'])
            graphs.append(dict(dataset=name,status='inapplicable_input_family',invalid_nodes=len(indices),cosine_conditions=0,ren_conditions=0)); continue
        prepath=folder/'271700.pre.npz'; metapath=folder/'271700.pre.json'
        meta=read_json(metapath); p=load_npz(prepath)
        for path,expected in [(prepath,done['pre_artifact_sha256']),(metapath,done['pre_record_sha256'])]:
            files[str(path)]=sha(path); audit.check(name+'/preseal hash '+path.name,files[str(path)]==expected)
        audit.check(name+'/preseal artifact link',meta['artifact_sha256']==files[str(prepath)])
        audit.check(name+'/preseal contract',meta['contract_sha256']==contract_hash)
        preinfo=precheck(p,meta,audit,name)
        for filename,hashkey,operator in [('slopes.json','slopes_sha256','cosine'),('ren_slopes.json','ren_slopes_sha256','ren')]:
            path=folder/filename
            files[str(path)]=sha(path); audit.check(name+'/'+filename+'/hash',files[str(path)]==done[hashkey])
            for fit in read_json(path)['fits']: saved_fits.append(dict(dataset=name,operator=operator,**fit))
        coverage=dict(dataset=name,status='complete',cosine_conditions=0,ren_conditions=0,
            radius=meta['radius'],domain_k=[k for k in range(13) if 2.**(-k)*meta['radius']<=.1 and 2.**(-2*k)*meta['radius']**2/TAU<=.1],
            ren_limit_disagreement_fraction=preinfo['limit_disagreement']['disagreement_fraction'],precheck_scope=preinfo)
        expected=[(f'k{k:02d}__{branch}__b{beta:g}',False) for k in range(13) for branch in ['fixed','scaled'] for beta in [.5,1.]]
        expected += [(f'ren__k{k:02d}__{policy}',True) for k in range(13) for policy in ['one','inverse_epsilon_squared']]
        for tag,isren in expected:
            record=folder/(tag+'.json'); artifact=folder/(tag+'.npz')
            if not record.exists() or not artifact.exists():
                audit.check(name+'/'+tag+'/exists',False); coverage['status']='incomplete_artifacts'; continue
            row=read_json(record); files[str(record)]=sha(record); files[str(artifact)]=sha(artifact)
            audit.check(name+'/'+tag+'/hash',files[str(artifact)]==row['artifact_sha256'])
            audit.check(name+'/'+tag+'/preseal link',row['pre_artifact_sha256']==files[str(prepath)] and row['pre_record_sha256']==files[str(metapath)])
            audit.check(name+'/'+tag+'/contract',row['contract_sha256']==contract_hash)
            package=load_npz(artifact); local=Audit()
            if isren:
                all_rows.append(analyze_ren(package,row,p,local)); coverage['ren_conditions']+=1
            else:
                all_rows.extend(analyze_cosine(package,row,p,local)); coverage['cosine_conditions']+=1
            for item in local.items: item['check']=name+'/'+tag+'/'+item['check']
            audit.items.extend(local.items)
        graphs.append(coverage)
    missing=[g for g in graphs if g['status'] not in ['complete','inapplicable_input_family']]
    groups=summarize_groups(all_rows)
    for saved in saved_fits:
        selectors=dict(round=saved.get('round',1))
        if saved['operator']=='cosine': selectors.update(beta=saved['beta'],temperature='fixed')
        else: selectors.update(replication_policy=saved['replication_policy'])
        group=find_group(groups,saved['dataset'],saved['operator'],saved['metric'],**selectors)
        prefix=saved['dataset']+'/fit/'+saved['metric']+'/'+str(selectors)
        audit.check(prefix+'/exists',group is not None)
        if group is None: continue
        points=sorted((p['epsilon'],p['amplitude']) for p in group['points'] if p['fit_eligible'])
        audit.check(prefix+'/number of eligible points',len(points)==saved['n_points'])
        audit.check(prefix+'/exact epsilon selection',[a for a,b in points]==[p[0] for p in saved['points']])
        if group['slope'] is None or saved['slope'] is None:
            audit.check(prefix+'/null slope',group['slope'] is None and saved['slope'] is None)
        else: audit.close(prefix+'/slope',group['slope'],saved['slope'],rtol=1e-6,atol=1e-8)
    status='partial' if missing or not global_done.exists() else ('audit_failed' if audit.failures else 'complete_audited')
    summary=dict(status=status,graphs=graphs,audit_failure_count=len(audit.failures),contract_sha256=contract_hash,
        analyzer_sha256=sha(__file__),created_at=time.time(),partial_allowed=allow_partial,
        scope='One old support draw per graph; development intervention; no query labels or accuracy',
        total_cosine_conditions=sum(g.get('cosine_conditions',0) for g in graphs),
        total_ren_conditions=sum(g.get('ren_conditions',0) for g in graphs))
    write_json(out/'summary.json',summary); write_json(out/'audit.json',dict(checks=audit.items,failures=audit.failures))
    write_json(out/'source_manifest.json',files); write_json(out/'groups.json',groups)
    write_json(out/'rankings.json',[{k:v for k,v in row.items() if k!='metrics'} for row in all_rows])
    write_json(out/'all_rows.json',all_rows)
    flat=[]
    for row in all_rows:
        for key,metric in row['metrics'].items():
            base={k:row.get(k) for k in ['dataset','operator','temperature','beta','replication_policy','round','k','epsilon']}
            flat.append(dict(base,metric=key,expected_order=expected_order(row,key),**metric))
    if flat:
        columns=list(dict.fromkeys(k for row in flat for k in row))
        with open(out/'all_points.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(flat)
    write_report(out,summary,groups,all_rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if audit.failures: return 2
    if (missing or not global_done.exists()) and not allow_partial: return 3
    return 0


def selfcheck():
    rng=np.random.default_rng(2709241); v=rng.normal(size=(19,4)); H=rng.normal(size=(3,4))*.1
    A=np.array([.9,.95,.97]); eps=.3
    r=1/np.sqrt(1+eps*eps*(v*v).sum(1)); z=np.column_stack([r,eps*r[:,None]*v])
    proto=np.column_stack([A,eps*H]); direct=z@(proto/np.linalg.norm(proto,axis=1,keepdims=True)).T
    assert np.max(abs(rc(direct)-.5*eps*eps*rc(cosine_D(v,A,H,eps))))<1e-14
    B=(1-A)/(eps*eps); ed=-((z[:,None,:]-proto[None,:,:])**2).sum(2)
    assert np.max(abs(rc(ed)-eps*eps*rc(euclidean_D(v,B,H,eps))))<1e-14
    rows=[]
    for k in range(5,11):
        e=2.**(-k); prediction=np.ones((3,4))*e**3
        metric=prototype_error(np.zeros(3),prediction/e,prediction,e,True)
        rows.append(dict(dataset='synthetic',operator='cosine',temperature='fixed',beta=.5,round=1,k=k,
                         epsilon=e,metrics={'feedback':metric}))
    group=summarize_groups(rows)[0]
    assert len(group['eligible_k'])==6 and abs(group['slope']-3)<1e-12
    small=matrix_error(np.ones((5,3))*1e-30,np.ones((5,3))*1e-30,True)
    assert not small['fit_eligible'] and small['numerical_status']=='unresolved'
    tied=np.array([[1.,1.],[1.,0.]]); other=np.array([[1.,1.],[0.,1.]])
    rank=ranking(tied,other,1.)
    assert rank['disagreement_count']==1 and rank['unresolved_gap_either_count']==1
    print(json.dumps(dict(status='passed',scope='Local NumPy analysis self-check only',fitted_order=group['slope']),indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results/concentration_scaling_r24')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/concentration_scaling_r24/independent')
    parser.add_argument('--contract',type=Path,default=ROOT/'provenance/r24_frozen_contract.json')
    parser.add_argument('--allow-partial',action='store_true'); parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check: selfcheck(); return
    raise SystemExit(main_analysis(args.results,args.report_dir,args.contract,args.allow_partial))


if __name__=='__main__': main()
