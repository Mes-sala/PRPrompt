"""R21: frozen, label-free development validation of finite-round predictors.

Run from the existing ProG environment. No query labels enter this runner.
Predictions and every applicability diagnostic are sealed before full updates.
Timing separates covariance construction, prediction, full update and audit.
"""
import fcntl
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import expanded_data as data
import run_study as r

TAU, BETA, ROUNDS = .05, .5, 3
SEEDS = list(range(271700, 271705))
METHODS = ['common_mean', 'g_linear_one', 'g_linear_three',
           'baseline_linear_one', 'baseline_linear_three',
           'node_moment_one', 'node_moment_three']
OUT = r.ROOT / 'results/moment_prediction_r21'
DESIGN = r.ROOT / 'design/R21_有限轮矩预测_开发验证.md'


def arr(x):
    return x.detach().cpu().numpy()


def tick():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def directions(x):
    norms = x.norm(dim=1)
    if not bool(torch.isfinite(x).all()) or float(norms.min()) <= 1e-14:
        raise ValueError('Prototype direction undefined or numerically unresolved')
    return x / norms[:, None]


def response(d, baseline, covariance):
    norm = baseline.norm()
    if float(norm) <= 1e-14:
        raise ValueError('Common baseline is zero or numerically unresolved')
    axis = baseline / norm
    projected = d - (d @ axis)[:, None] * axis
    return projected @ covariance / (TAU * norm)


def safe_exp(x):
    return math.exp(x) if x < 700 else float('inf')


def prototype_linear_bound(mu, linear, g, covariance):
    """One-step bound F(mu) versus g+linear, no node-softmax access."""
    w = directions(mu)
    dw = w-w.mean(0)
    eta = float(dw.norm(dim=1).max()) / TAU
    t = dw @ covariance / TAU
    normalization = (t-linear).norm(dim=1)
    trivial = 1+(g+linear).norm(dim=1)
    if eta < 100:
        exp2 = math.exp(2*eta)
        remainder = 2*exp2*eta*eta
        bound = normalization + exp2*(math.sqrt(max(0., float(covariance.trace())))*remainder
                        + t.norm(dim=1)*(eta+remainder))
        bound = torch.minimum(bound, trivial)
    else:
        bound = trivial
    return bound+1e-12, dict(prototype_eta_upper=eta,
                            normalization_linearization_max=float(normalization.max()))


def node_moment_update(mu, g, covariance, node_radius):
    """Only a C-way mean-logit softmax; never an N-by-C node softmax."""
    w = directions(mu)
    pi = (w @ g / TAU).softmax(0)
    v = w-pi@w
    correction = v @ covariance / TAU
    soft = g+correction
    eta = node_radius*float(v.norm(dim=1).max())/TAU
    trivial = 1+soft.norm(dim=1)
    if eta < 100:
        bound = 2*math.exp(4*eta)*eta*eta*(
            math.sqrt(max(0., float(covariance.trace())))+correction.norm(dim=1))
        bound = torch.minimum(bound, trivial)
    else:
        bound = trivial
    mass_variance = ((v@covariance)*v).sum(1)/TAU**2
    mass_second = pi*(1+.5*(mass_variance-pi@mass_variance))
    return soft, bound+1e-12, dict(node_eta_upper=eta,
            mass0_min=float(pi.min()), mass2_min=float(mass_second.min()),
            mass2_sum=float(mass_second.sum()),
            mass2_positive=bool((mass_second>0).all()))


def moment_trajectory(anchor, g, covariance, radius, kind):
    """Prediction only; input is moments, anchors and one cached radius."""
    b0 = anchor.mean(0)
    bstar = (1-BETA)*b0+BETA*g
    A = anchor-b0
    mu = anchor.clone()
    traj, softs, bounds, records = [mu.clone()], [], [], []
    tube_valid, tube_error = True, 0.
    for step in range(ROUNDS):
        if kind == 'node':
            soft, bound, info = node_moment_update(mu, g, covariance, radius)
            condition = info['node_eta_upper'] <= .1
        else:
            baseline = g if kind == 'g' else (b0 if step == 0 else bstar)
            d = mu-mu.mean(0)
            linear = response(d, baseline, covariance)
            soft = g+linear
            bound, info = prototype_linear_bound(mu, linear, g, covariance)
            info['relative_radius'] = float(d.norm(dim=1).max()/baseline.norm())
            info['baseline_norm'] = float(baseline.norm())
            info['anchor_mean_distance_to_g_relative'] = float((b0-g).norm()/g.norm().clamp_min(1e-30))
            condition = (info['prototype_eta_upper'] <= .1 and info['relative_radius'] <= .1)
            if kind == 'g':
                # The fixed-g expansion additionally needs a small mean shift.
                condition = condition and info['anchor_mean_distance_to_g_relative'] <= .1
        newmu = (1-BETA)*anchor+BETA*soft
        if tube_valid:
            m = float(mu.norm(dim=1).min())-tube_error
            if m <= 0:
                tube_valid, tube_error = False, float('inf')
            else:
                tube_error = BETA*(float(bound.max())+4*tube_error/(TAU*m))
                trivial = float((newmu.norm(dim=1)+(1-BETA)*anchor.norm(dim=1)+BETA).max())
                tube_error = min(tube_error, trivial)
        records.append(dict(round=step+1, **info, local_remainder_upper=float(bound.max()),
                            tube_valid=tube_valid,
                            prototype_error_upper=tube_error if tube_valid else None,
                            predeclared_small_variation=condition,
                            min_predicted_norm=float(newmu.norm(dim=1).min())))
        softs.append(soft.clone()); bounds.append(bound.clone())
        mu = newmu; traj.append(mu.clone())
    return dict(trajectory=torch.stack(traj), soft=torch.stack(softs),
                bound=torch.stack(bounds), records=records)


def exact_soft(zu, mu):
    w = directions(mu)
    # Shared-class subtraction leaves softmax unchanged in real arithmetic.
    q = (zu @ (w-w.mean(0)).T/TAU).softmax(1)
    mass = q.sum(0)
    if float(mass.min()) <= 1e-8:
        raise ValueError('Soft class mass outside the original unclamped contract')
    return q.T@zu/mass[:, None], mass, int((q == 0).sum())


def endpoint_metrics(predicted, full, scores, full_scores, common):
    w, wf = directions(predicted), directions(full)
    c = len(w)
    pair_mask = torch.triu(torch.ones((c,c),dtype=torch.bool,device=w.device),diagonal=1)
    dp = w[:,None]-w[None,:]
    df = wf[:,None]-wf[None,:]
    pn, fn = dp.norm(dim=2), df.norm(dim=2)
    usable = pair_mask & (pn>1e-12) & (fn>1e-12)
    cos = ((dp*df).sum(2)/(pn*fn).clamp_min(1e-30)).clamp(-1,1)
    angles = torch.acos(cos[usable])*180/math.pi
    centered_pred = predicted-predicted.mean(0)
    centered_full = full-full.mean(0)
    contrast_scale = float(centered_full.norm())
    residual_scale = float((full-common).norm())
    score_error = scores-full_scores
    centered_error = score_error-score_error.mean(1,keepdim=True)
    return dict(prototype_max_error=float((predicted-full).norm(dim=1).max()),
                prototype_frobenius_error=float((predicted-full).norm()),
                relative_class_contrast_error=float((centered_pred-centered_full).norm())/contrast_scale
                    if contrast_scale>1e-14 else None,
                error_relative_to_full_minus_common=float((predicted-full).norm())/residual_scale
                    if residual_scale>1e-14 else None,
                unit_direction_max_error=float((w-wf).norm(dim=1).max()),
                max_boundary_rotation_degrees=float(angles.max()) if angles.numel() else None,
                mean_boundary_rotation_degrees=float(angles.mean()) if angles.numel() else None,
                boundary_degenerate_pairs=int((pair_mask&~usable).sum()),
                score_centered_rmse=float(centered_error.square().mean().sqrt()),
                max_pairwise_margin_error=float((score_error.max(1).values-score_error.min(1).values).max()),
                prediction_disagreement_pct=float((scores.argmax(1)!=full_scores.argmax(1)).double().mean()*100))


def json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k:json_finite(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    return value


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True,exist_ok=True)
    lock = open(OUT/'run.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner, design = r.sha(__file__), r.sha(DESIGN)
    deps = {p.name:r.sha(p) for p in [r.ROOT/'code/run_study.py',r.ROOT/'code/expanded_data.py']}
    r.save(OUT/'process.json',dict(pid=os.getpid(),started=time.time(),runner_sha256=runner,
                                  design_sha256=design,dependencies=deps))
    count = 0
    for name in data.ALL:
        dataset_start=tick()
        d,x,adj,pool,query,C,protocol = data.load_data(name)
        x,adj=x.double(),adj.double()
        z=F.normalize(adj@(adj@x),dim=1)
        load_representation_seconds=tick()-dataset_start
        geom=json.loads((r.ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text())
        assert r.thash(x)==geom['input_sha256'] and r.thash(d.edge_index)==geom['graph_sha256']
        assert float(z.norm(dim=1).max())<=1+1e-12
        start=tick(); zz=z.T@z; total=z.sum(0); covariance_precompute_seconds=tick()-start
        folder=OUT/name; folder.mkdir(exist_ok=True)
        for seed in SEEDS:
            rec=folder/f'{seed}.json'; artifact=folder/f'{seed}.npz'
            pre_rec=folder/f'{seed}.pre.json'; pre_artifact=folder/f'{seed}.pre.npz'
            if rec.exists():
                old=json.loads(rec.read_text())
                assert old['runner_sha256']==runner and old['design_sha256']==design
                assert old['artifact_sha256']==r.sha(artifact) and old['pre_artifact_sha256']==r.sha(pre_artifact)
                assert old['pre_record_sha256']==r.sha(pre_rec)
                count+=1; continue
            r.save(OUT/'current.json',dict(dataset=name,seed=seed,completed=count,time=time.time()))
            source=r.ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__graph_anchor'/'prediction.npz'
            # Do not load the old target prototype before prediction sealing.
            with np.load(source) as old:
                support=old['support'].copy(); old_anchor=old['anchor'].copy()
            start=tick()
            si=torch.tensor(support,device='cuda')
            assert not torch.isin(si,query.cuda()).any()
            sy=d.y[support].cuda()  # The sole label access in this runner.
            onehot=F.one_hot(sy,C).double()
            anchor=onehot.T@z[si]/onehot.sum(0)[:,None]
            assert np.allclose(arr(anchor),old_anchor,atol=1e-12,rtol=1e-10)
            mask=torch.ones(len(z),dtype=torch.bool,device='cuda'); mask[si]=False
            zu=z[mask]; n=len(zu)
            g=(total-z[si].sum(0))/n
            covariance=(zz-z[si].T@z[si])/n-torch.outer(g,g)
            covariance=(covariance+covariance.T)/2
            radius=float((zu-g).norm(dim=1).max())
            support_geometry_seconds=tick()-start
            qz=z[query.cuda()]
            common=(1-BETA)*anchor+BETA*g
            package=dict(anchor=arr(anchor),mean=arr(g),support=support,query=query.numpy(),
                         covariance_diagonal=arr(covariance.diag()),query_norm=arr(qz.norm(dim=1)))
            predicted={'common_mean':common}
            metadata={'common_mean':dict(predeclared_small_variation=None,records=[])}
            prediction_seconds={}; results={}; invalid={}
            start=tick(); directions(common); prediction_seconds['common_mean']=tick()-start
            for kind,prefix in [('g','g_linear'),('baseline','baseline_linear'),('node','node_moment')]:
                start=tick()
                try:
                    result=moment_trajectory(anchor,g,covariance,radius,kind)
                    seconds=tick()-start
                    results[kind]=result
                    package[prefix+'__trajectory']=arr(result['trajectory'])
                    package[prefix+'__soft_approx']=arr(result['soft'])
                    package[prefix+'__local_remainder_upper']=arr(result['bound'])
                    for steps,suffix in [(1,'one'),(3,'three')]:
                        key=prefix+'_'+suffix
                        predicted[key]=result['trajectory'][steps]
                        metadata[key]=dict(predeclared_small_variation=all(s['predeclared_small_variation'] for s in result['records'][:steps]),
                                           records=result['records'][:steps])
                    # One-step shares the prefix; separately timed below rather
                    # than dividing the three-step cost or calling it equal.
                    prediction_seconds[prefix+'_three']=seconds
                    start=tick()
                    if kind=='node':
                        soft,_,_=node_moment_update(anchor,g,covariance,radius)
                    else:
                        b=g if kind=='g' else anchor.mean(0)
                        linear=response(anchor-anchor.mean(0),b,covariance)
                        soft=g+linear
                        prototype_linear_bound(anchor,linear,g,covariance)
                    first=(1-BETA)*anchor+BETA*soft; directions(first)
                    prediction_seconds[prefix+'_one']=tick()-start
                except ValueError as exc:
                    invalid[prefix]=str(exc)
            score_seconds={}
            predicted_scores={}
            for key,mu in predicted.items():
                start=tick(); scores=qz@directions(mu).T; score_seconds[key]=tick()-start
                predicted_scores[key]=scores
                package[key+'__prototype']=arr(mu); package[key+'__scores']=arr(scores)
            # All eligibility quantities and all approximate endpoints are now
            # persisted. No full soft assignment has run for this seed.
            np.savez_compressed(pre_artifact,**package)
            r.save(pre_rec,json_finite(dict(dataset=name,seed=seed,phase='before full node softmax',
                tau=TAU,beta=BETA,rounds=ROUNDS,metadata=metadata,invalid=invalid,
                node_radius=radius,covariance_trace=float(covariance.trace()),
                covariance_sha256=r.thash(covariance),representation_sha256=r.thash(z),
                support_sha256=r.thash(si),source_sha256=r.sha(source),
                runner_sha256=runner,design_sha256=design,dependencies=deps,
                pre_artifact_sha256=r.sha(pre_artifact),prediction_seconds=prediction_seconds,
                score_seconds=score_seconds,support_geometry_seconds=support_geometry_seconds,
                covariance_precompute_seconds=covariance_precompute_seconds,
                load_representation_seconds=load_representation_seconds,finished=time.time())))
            sealed_pre_hash=r.sha(pre_artifact); sealed_record_hash=r.sha(pre_rec)
            start=tick(); mu=anchor.clone(); full_traj=[mu.clone()]; full_records=[]
            for step in range(ROUNDS):
                soft,mass,zeros=exact_soft(zu,mu)
                mu=(1-BETA)*anchor+BETA*soft; full_traj.append(mu.clone())
                full_records.append(dict(round=step+1,min_mass=float(mass.min()),zero_assignments=zeros))
            full_update_seconds=tick()-start
            with np.load(source) as old: old_full=old['prototype'].copy()
            replay_error=float(np.max(abs(arr(mu)-old_full)))
            assert np.allclose(arr(mu),old_full,atol=1e-11,rtol=1e-8)
            start=tick(); full_scores=qz@directions(mu).T; full_score_seconds=tick()-start
            package['full__trajectory']=arr(torch.stack(full_traj)); package['full__scores']=arr(full_scores)
            endpoint={key:endpoint_metrics(m,mu,predicted_scores[key],full_scores,common)
                      for key,m in predicted.items()}
            first_scores=qz@directions(full_traj[1]).T
            endpoint['full_one']=endpoint_metrics(full_traj[1],mu,first_scores,full_scores,common)
            package['full_one__scores']=arr(first_scores)
            # Post-seal audit at predicted inputs. These extra softmax calls are
            # only for validating bounds; never fed back into predictions.
            start=tick(); audits={}
            for kind,result in results.items():
                entries=[]; exact_at_pred=[]
                for step in range(ROUNDS):
                    inp=result['trajectory'][step]
                    exact,mass,zeros=exact_soft(zu,inp)
                    residual=(exact-result['soft'][step]).norm(dim=1)
                    upper=result['bound'][step]
                    assert bool((residual<=upper+1e-9).all()), (name,seed,kind,step)
                    current_full=full_traj[step+1]
                    actual_error=float((result['trajectory'][step+1]-current_full).norm(dim=1).max())
                    stage=result['records'][step]
                    if stage['tube_valid']:
                        assert actual_error<=stage['prototype_error_upper']+1e-9
                    w=directions(inp); pi=(w@g/TAU).softmax(0); v=w-pi@w
                    exact_node_eta=float(((zu-g)@v.T/TAU).abs().max())
                    entry=dict(round=step+1,actual_local_residual=float(residual.max()),
                               stated_local_upper=float(upper.max()),actual_trajectory_error=actual_error,
                               exact_node_eta_postseal=exact_node_eta,
                               exact_mass_min=float(mass.min()),zero_assignments=zeros)
                    entries.append(entry); exact_at_pred.append(exact)
                audits[kind]=entries
                package[kind+'__exact_soft_at_predicted_input']=arr(torch.stack(exact_at_pred))
            audit_seconds=tick()-start
            assert r.sha(pre_artifact)==sealed_pre_hash and r.sha(pre_rec)==sealed_record_hash
            np.savez_compressed(artifact,**package)
            r.save(rec,json_finite(dict(dataset=name,seed=seed,protocol=protocol,
                endpoint=endpoint,metadata=metadata,invalid=invalid,postseal_audits=audits,
                full_records=full_records,baseline_replay_max_error=replay_error,
                prediction_seconds=prediction_seconds,score_seconds=score_seconds,
                full_update_seconds=full_update_seconds,full_score_seconds=full_score_seconds,
                bound_audit_seconds=audit_seconds,support_geometry_seconds=support_geometry_seconds,
                covariance_precompute_seconds=covariance_precompute_seconds,
                load_representation_seconds=load_representation_seconds,
                runner_sha256=runner,design_sha256=design,dependencies=deps,
                artifact_sha256=r.sha(artifact),pre_artifact_sha256=sealed_pre_hash,
                pre_record_sha256=sealed_record_hash,source_sha256=r.sha(source),
                scope='No query labels or accuracy. Approximate predictions and eligibility sealed before full node softmax.',
                finished=time.time())))
            count+=1
            print(json.dumps(dict(dataset=name,seed=seed,completed=count,
                node_disagreement=endpoint.get('node_moment_three',{}).get('prediction_disagreement_pct'),
                node_eligible=metadata.get('node_moment_three',{}).get('predeclared_small_variation'))),flush=True)
        r.save(folder/'complete.json',dict(rows=len(SEEDS),runner_sha256=runner,design_sha256=design))
        del d,x,adj,z,zu,zz,covariance,qz
        torch.cuda.empty_cache()
    assert count==70
    r.save(OUT/'complete.json',dict(rows=count,datasets=len(data.ALL),methods=METHODS,
            tau=TAU,beta=BETA,rounds=ROUNDS,runner_sha256=runner,design_sha256=design,
            dependencies=deps,finished=time.time()))


if __name__=='__main__':
    try:
        with torch.no_grad():
            main()
    except Exception:
        r.save(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
