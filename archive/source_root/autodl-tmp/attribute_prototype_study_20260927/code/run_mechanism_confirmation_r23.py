"""R23 frozen confirmation: 14 graphs x 3 depths x 10 new support draws.

Local construction only; execute explicitly in the existing ProG environment.
All certificates and approximate endpoints are sealed before full node softmax.
"""
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import expanded_data as data
import expanded_grid_cv as cv
import run_study as r
import run_cone_certificate_r19 as cone
import run_ellipsoid_certificate_r20 as ellipsoid
import run_moment_prediction_r21 as moment


TAU, BETA, ROUNDS = .05, .5, 3
DEPTHS, SEEDS = (1, 2, 3), tuple(range(271900, 271910))
DATASETS = ('Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo', 'Actor',
            'Chameleon', 'Cornell', 'Squirrel', 'WikiCS', 'CS', 'Physics',
            'Roman-empire', 'Amazon-ratings')
OUT = r.ROOT / 'results/mechanism_confirmation_r23'
DESIGN = r.ROOT / 'design/R23_新传播深度与新支持抽样_机制确认.md'
DEPENDENCIES = r.ROOT / 'provenance/R23_frozen_dependencies.json'


def arr(x):
    return x.detach().cpu().numpy()


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def certificate_inputs(zu, anchor, g, covariance, qz, scores, common):
    """Frozen R19/R20 formulas, using moments and inputs only."""
    start = tick()
    trace = float((zu - g).square().sum() / len(zu))
    spec = cone.upper_spectrum(covariance, trace)
    e = g / g.norm()
    az = float((zu @ e).min()) - 1e-12
    aa = float((anchor @ e).min()) - 1e-12
    m = .5 * (az + aa)
    node_radius = float((zu - g).norm(dim=1).max())
    dz = min(2., 2 * node_radius)
    if 0 <= az <= 1:
        dz = min(dz, 2 * math.sqrt(max(0., 1 - az * az)))
    dz += 1e-12
    ra = float((anchor - (anchor @ e)[:, None] * e).norm(dim=1).max())
    rz = float((zu - (zu @ e)[:, None] * e).norm(dim=1).max())
    rad = .5 * (ra + rz)
    cap = 2 * rad / math.sqrt(m * m + rad * rad) + 1e-12 if m > 0 else 2.
    da = cone.diameter(anchor) + 1e-12
    d0 = cone.diameter(moment.directions(anchor)) + 1e-12
    norms = qz.norm(dim=1)
    bnorm = common.norm(dim=1)
    package, records = {}, {}
    axis_valid = m > 0 and aa > 0
    for typ, lam in [('trace', trace + 1e-12), ('spectral', spec['upper'])]:
        trajectory = cone.budget(d0, da, dz, cap, m, lam)
        eta = .5 * trajectory[2]['B']
        lower, upper, cert = cone.cone_scores(scores, norms, bnorm, eta)
        if not axis_valid:
            cert[:] = False
            lower[:] = -1
            upper[:] = 1
        package.update({typ + '__lower': arr(lower), typ + '__upper': arr(upper),
                        typ + '__cert': arr(cert)})
        records[typ] = dict(lambda_upper=lam, trajectory=trajectory, eta=eta,
                            axis_valid=axis_valid,
                            uncertified_pct=float((~cert).double().mean() * 100))
    cone_seconds = tick() - start
    start = tick()
    spectral = records['spectral']
    last, eta = spectral['trajectory'][2], spectral['eta']
    eligible = axis_valid and eta < float(bnorm.min())
    elo = -norms[:, None].expand_as(scores).clone()
    ehi = norms[:, None].expand_as(scores).clone()
    details = dict(eligible=eligible, Ce=None, gradient_quadratic_min_raw=None)
    if eligible:
        w = moment.directions(common)
        zSz = (qz @ covariance * qz).sum(1)
        Sw = covariance @ w.T
        zSw = qz @ Sw
        wSw = (w.T * Sw).sum(0)
        quad = (zSz[:, None] - 2 * scores * zSw + scores.square() * wSw[None, :]) / bnorm.square()[None, :]
        assert float(quad.min()) > -1e-9
        padding = 1e-12 * (1 + zSz.abs()[:, None] + 2 * (scores * zSw).abs()
                          + scores.square() * wSw.abs()[None, :]) / bnorm.square()[None, :]
        ce = .5 * math.exp(last['logK']) * math.sqrt(spectral['lambda_upper']) * last['D'] / .05
        mc = .5 * (anchor @ e) + .5 * az
        rmin = torch.maximum(bnorm - eta, mc)
        assert float(rmin.min()) > 0
        radius = ce * (quad.clamp_min(0) + padding).sqrt() + norms[:, None] * eta ** 2 / rmin.square()[None, :] + 1e-12
        elo = torch.maximum(scores - radius, -norms[:, None])
        ehi = torch.minimum(scores + radius, norms[:, None])
        details.update(Ce=ce, gradient_quadratic_min_raw=float(quad.min()))
        package.update(gradient_quadratic=arr(quad), quadratic_padding=arr(padding),
                       rmin=arr(rmin), ellipsoid_radius=arr(radius),
                       zSz=arr(zSz), zSw=arr(zSw), wSw=arr(wSw))
    lower = torch.tensor(package['spectral__lower'], device=qz.device)
    upper = torch.tensor(package['spectral__upper'], device=qz.device)
    jlo, jhi = torch.maximum(lower, elo), torch.minimum(upper, ehi)
    assert bool((jlo <= jhi + 1e-10).all())
    pred = scores.argmax(1)
    for typ, lo, hi in [('ellipsoid', elo, ehi), ('joint', jlo, jhi)]:
        cert = ellipsoid.certify(lo, hi, pred, norms)
        package.update({typ + '__lower': arr(lo), typ + '__upper': arr(hi),
                        typ + '__cert': arr(cert)})
        records[typ] = dict(uncertified_pct=float((~cert).double().mean() * 100))
    details['seconds'] = tick() - start
    geometry = dict(alphaZ=az, alphaA=aa, m=m, DZ=dz, DA=da, D0=d0,
                    cap=cap, node_radius=node_radius, covariance_trace=trace)
    return package, records, details, geometry, spec, cone_seconds


def main():
    assert tuple(data.ALL) == DATASETS
    assert (moment.TAU, moment.BETA, moment.ROUNDS) == (TAU, BETA, ROUNDS)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = json.loads(DEPENDENCIES.read_text())
    deps = {name: r.sha(r.ROOT / 'code' / name) for name in frozen['files']}
    assert deps == frozen['files'], 'Frozen project dependency changed; do not silently refreeze.'
    runner, design, dependency_manifest = r.sha(__file__), r.sha(DESIGN), r.sha(DEPENDENCIES)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / 'run.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, device=torch.cuda.get_device_name(),
                       cuda=torch.version.cuda,
                       packages={name: importlib.metadata.version(name) for name in
                                 ('torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn')})
    r.save(OUT / 'process.json', dict(pid=os.getpid(), started=time.time(), runner_sha256=runner,
                                    design_sha256=design, dependencies=deps,
                                    dependency_manifest_sha256=dependency_manifest, environment=environment))
    completed = 0
    for name in DATASETS:
        start = tick()
        d, x, adj, pool, query, classes, protocol = data.load_data(name)
        x, adj = x.double(), adj.double()
        load_seconds = tick() - start
        contract_path = r.ROOT / 'results/attribute_mechanism_r13' / name / 'geometry.json'
        contract = json.loads(contract_path.read_text())
        assert r.thash(x) == contract['input_sha256']
        assert r.thash(d.edge_index) == contract['graph_sha256']
        input_contract = dict(input_sha256=r.thash(x), graph_sha256=r.thash(d.edge_index),
                              adjacency_indices_sha256=r.thash(adj.indices()),
                              adjacency_values_sha256=r.thash(adj.values()),
                              pool_sha256=r.thash(pool), query_sha256=r.thash(query),
                              source_contract_sha256=r.sha(contract_path))
        supports = {}
        for seed in SEEDS:
            support, sy, _, _ = cv.support_split(d, pool, classes, seed, 5)
            assert not torch.isin(support, query).any()
            assert len(support) == 5 * classes and len(torch.unique(support)) == len(support)
            supports[seed] = (support, sy)
        dataset_folder = OUT / name
        dataset_folder.mkdir(exist_ok=True)
        r.save(dataset_folder / 'support_manifest.json', dict(
            seeds={str(seed): dict(support=s.tolist(), support_sha256=r.thash(s)) for seed, (s, _) in supports.items()},
            input_contract=input_contract, dependencies=deps,
            runner_sha256=runner, design_sha256=design))
        propagated = x
        for hops in DEPTHS:
            start = tick()
            propagated = adj @ propagated
            z = F.normalize(propagated, dim=1)
            representation_seconds = tick() - start
            assert float(z.norm(dim=1).max()) <= 1 + 1e-12
            representation_hash = r.thash(z)
            start = tick()
            zz = z.T @ z
            covariance_precompute_seconds = tick() - start
            folder = dataset_folder / f'h{hops}'
            folder.mkdir(exist_ok=True)
            condition_hashes = dict(**input_contract, representation_sha256=representation_hash)
            r.save(folder / 'geometry.json', dict(dataset=name, hops=hops, hashes=condition_hashes,
                load_seconds=load_seconds, representation_increment_seconds=representation_seconds,
                covariance_precompute_seconds=covariance_precompute_seconds,
                runner_sha256=runner, design_sha256=design, dependencies=deps))
            for seed in SEEDS:
                rec = folder / f'{seed}.json'
                artifact = folder / f'{seed}.npz'
                pre_rec = folder / f'{seed}.pre.json'
                pre_artifact = folder / f'{seed}.pre.npz'
                if rec.exists():
                    old = json.loads(rec.read_text())
                    assert old['runner_sha256'] == runner and old['design_sha256'] == design
                    assert old['dependencies'] == deps and old['dependency_manifest_sha256'] == dependency_manifest
                    assert old['artifact_sha256'] == r.sha(artifact)
                    assert old['pre_artifact_sha256'] == r.sha(pre_artifact)
                    assert old['pre_record_sha256'] == r.sha(pre_rec)
                    for key, value in condition_hashes.items():
                        assert old['hashes'][key] == value
                    assert old['hashes']['support_sha256'] == r.thash(supports[seed][0])
                    assert old['hashes']['support_labels_sha256'] == r.thash(supports[seed][1])
                    completed += 1
                    continue
                r.save(OUT / 'current.json', dict(dataset=name, hops=hops, seed=seed,
                                               completed=completed, time=time.time()))
                start = tick()
                support, sy = supports[seed]
                si, sy = support.cuda(), sy.cuda()
                onehot = F.one_hot(sy, classes).double()
                anchor = onehot.T @ z[si] / onehot.sum(0)[:, None]
                moment.directions(anchor)
                mask = torch.ones(len(z), dtype=torch.bool, device=z.device)
                mask[si] = False
                zu = z[mask]
                g = zu.mean(0)
                assert float(g.norm()) > 1e-14
                covariance = (zz - z[si].T @ z[si]) / len(zu) - torch.outer(g, g)
                covariance = (covariance + covariance.T) / 2
                node_radius = float((zu - g).norm(dim=1).max())
                support_geometry_seconds = tick() - start
                qz = z[query.cuda()]
                start = tick()
                common = (1 - BETA) * anchor + BETA * g
                moment.directions(common)
                common_prediction_seconds = tick() - start
                start = tick()
                common_scores = qz @ moment.directions(common).T
                common_score_seconds = tick() - start
                package = dict(anchor=arr(anchor), mean=arr(g), common=arr(common),
                               support=arr(support), query=arr(query), node_norms=arr(qz.norm(dim=1)),
                               covariance_diagonal=arr(covariance.diag()))
                cert_package, certificates, ellipsoid_record, geometry, spectrum, cone_seconds = certificate_inputs(
                    zu, anchor, g, covariance, qz, common_scores, common)
                package.update(cert_package)
                predicted = {'common_mean': common}
                predicted_scores = {'common_mean': common_scores}
                prediction_seconds = {'common_mean': common_prediction_seconds}
                score_seconds = {'common_mean': common_score_seconds}
                prediction_metadata = {'common_mean': dict(predeclared_small_variation=None, records=[])}
                moment_results, invalid = {}, {}
                for kind, prefix in [('baseline', 'baseline_linear'), ('node', 'node_moment')]:
                    start = tick()
                    try:
                        result = moment.moment_trajectory(anchor, g, covariance, node_radius, kind)
                        mu_pred = result['trajectory'][3]
                        moment.directions(mu_pred)
                        method = prefix + '_three'
                        predicted[method] = mu_pred
                        moment_results[kind] = result
                        prediction_seconds[method] = tick() - start
                        prediction_metadata[method] = dict(
                            predeclared_small_variation=all(row['predeclared_small_variation'] for row in result['records']),
                            records=result['records'])
                        package[prefix + '__trajectory'] = arr(result['trajectory'])
                        package[prefix + '__soft_approx'] = arr(result['soft'])
                        package[prefix + '__local_remainder_upper'] = arr(result['bound'])
                        start = tick()
                        predicted_scores[method] = qz @ moment.directions(mu_pred).T
                        score_seconds[method] = tick() - start
                    except ValueError as exc:
                        invalid[prefix] = str(exc)
                for method, mu_pred in predicted.items():
                    package[method + '__prototype'] = arr(mu_pred)
                    package[method + '__scores'] = arr(predicted_scores[method])
                hashes = dict(**condition_hashes, support_sha256=r.thash(support),
                              support_labels_sha256=r.thash(sy),
                              covariance_sha256=r.thash(covariance))
                timings = dict(dataset_load_seconds=load_seconds,
                               representation_increment_seconds=representation_seconds,
                               covariance_precompute_seconds=covariance_precompute_seconds,
                               support_geometry_seconds=support_geometry_seconds,
                               cone_budget_seconds=cone_seconds,
                               ellipsoid_bound_seconds=ellipsoid_record['seconds'],
                               prediction_seconds=prediction_seconds, score_seconds=score_seconds)
                start = time.perf_counter()
                np.savez_compressed(pre_artifact, **package)
                pre_save_seconds = time.perf_counter() - start
                r.save(pre_rec, moment.json_finite(dict(
                    dataset=name, hops=hops, seed=seed, phase='before any full node softmax',
                    tau=TAU, beta=BETA, rounds=ROUNDS, protocol=protocol,
                    certificates=certificates, ellipsoid=ellipsoid_record,
                    geometry=geometry, spectrum=spectrum,
                    prediction_metadata=prediction_metadata, invalid=invalid,
                    timings=timings, pre_payload_save_seconds=pre_save_seconds,
                    hashes=hashes, runner_sha256=runner, design_sha256=design,
                    dependencies=deps, dependency_manifest_sha256=dependency_manifest,
                    artifact_sha256=r.sha(pre_artifact), finished=time.time())))
                pre_hash, pre_record_hash = r.sha(pre_artifact), r.sha(pre_rec)

                # All prediction rules/outputs above are already sealed.
                start = tick()
                mu, full_traj, full_soft, full_records = anchor.clone(), [anchor.clone()], [], []
                full_one_seconds = None
                full_update_seconds = 0.
                for step in range(ROUNDS):
                    update_start = tick()
                    soft, mass, zeros = moment.exact_soft(zu, mu)
                    next_mu = (1 - BETA) * anchor + BETA * soft
                    update_seconds = tick() - update_start
                    full_update_seconds += update_seconds
                    obs = dict(round=step + 1, direction_diameter=cone.diameter(moment.directions(mu)),
                               soft_diameter=cone.diameter(soft),
                               soft_residual=float((soft - g).norm(dim=1).max()),
                               min_mass=float(mass.min()), zero_assignments=zeros,
                               update_seconds=update_seconds)
                    for typ in ('trace', 'spectral'):
                        if certificates[typ]['axis_valid']:
                            bound = certificates[typ]['trajectory'][step]
                            assert obs['direction_diameter'] <= bound['D'] + 1e-10
                            assert obs['soft_diameter'] <= bound['B'] + 1e-10
                            assert obs['soft_residual'] <= bound['B'] + 1e-10
                    mu = next_mu
                    full_soft.append(soft.clone())
                    full_traj.append(mu.clone())
                    full_records.append(obs)
                    if step == 0:
                        full_one_seconds = update_seconds
                full_loop_with_diagnostics_seconds = tick() - start
                start = tick()
                full_scores = qz @ moment.directions(mu).T
                full_score_seconds = tick() - start
                start = tick()
                first_scores = qz @ moment.directions(full_traj[1]).T
                full_one_score_seconds = tick() - start
                package.update(full__trajectory=arr(torch.stack(full_traj)),
                               full__soft=arr(torch.stack(full_soft)),
                               full__prototype=arr(mu), full__scores=arr(full_scores),
                               full_one__prototype=arr(full_traj[1]), full_one__scores=arr(first_scores))
                endpoint = {method: moment.endpoint_metrics(mu_pred, mu, predicted_scores[method], full_scores, common)
                            for method, mu_pred in predicted.items()}
                endpoint['full_one'] = moment.endpoint_metrics(full_traj[1], mu, first_scores, full_scores, common)
                changed = common_scores.argmax(1) != full_scores.argmax(1)
                for typ in ('trace', 'spectral', 'ellipsoid', 'joint'):
                    lower = torch.tensor(package[typ + '__lower'], device=z.device)
                    upper = torch.tensor(package[typ + '__upper'], device=z.device)
                    cert = torch.tensor(package[typ + '__cert'], device=z.device)
                    assert not bool((cert & changed).any())
                    assert bool((full_scores >= lower - 1e-10).all())
                    assert bool((full_scores <= upper + 1e-10).all())
                start = tick()
                audits = {}
                for kind, result in moment_results.items():
                    audit_rows, exact_rows = [], []
                    for step in range(ROUNDS):
                        exact, mass, zeros = moment.exact_soft(zu, result['trajectory'][step])
                        residual = (exact - result['soft'][step]).norm(dim=1)
                        bound = result['bound'][step]
                        assert bool((residual <= bound + 1e-9).all())
                        actual_error = float((result['trajectory'][step + 1] - full_traj[step + 1]).norm(dim=1).max())
                        stage = result['records'][step]
                        if stage['tube_valid']:
                            assert actual_error <= stage['prototype_error_upper'] + 1e-9
                        w = moment.directions(result['trajectory'][step])
                        pi = (w @ g / TAU).softmax(0)
                        v = w - pi @ w
                        exact_eta = float(((zu - g) @ v.T / TAU).abs().max())
                        audit_rows.append(dict(round=step + 1, actual_local_residual=float(residual.max()),
                                               stated_local_upper=float(bound.max()),
                                               actual_trajectory_error=actual_error,
                                               exact_node_eta_postseal=exact_eta,
                                               min_mass=float(mass.min()), zero_assignments=zeros))
                        exact_rows.append(exact)
                    audits[kind] = audit_rows
                    package[kind + '__exact_soft_at_predicted_input'] = arr(torch.stack(exact_rows))
                audit_seconds = tick() - start
                common_disagreement = endpoint['common_mean']['prediction_disagreement_pct']
                node = endpoint.get('node_moment_three')
                node_domain = prediction_metadata.get('node_moment_three', {}).get('predeclared_small_variation', False)
                nontrivial = bool(node is not None and node_domain and common_disagreement > 1.
                                  and node['prediction_disagreement_pct'] <= 1.)
                assert r.sha(pre_artifact) == pre_hash and r.sha(pre_rec) == pre_record_hash
                start = time.perf_counter()
                np.savez_compressed(artifact, **package)
                final_save_seconds = time.perf_counter() - start
                r.save(rec, moment.json_finite(dict(
                    dataset=name, hops=hops, seed=seed, protocol=protocol,
                    endpoint=endpoint, certificates=certificates, ellipsoid=ellipsoid_record,
                    geometry=geometry, spectrum=spectrum,
                    prediction_metadata=prediction_metadata, invalid=invalid,
                    full_records=full_records, postseal_audits=audits,
                    common_disagreement_pct=common_disagreement,
                    nontrivial_node_threshold_crossing=nontrivial,
                    timings=dict(**timings, full_update_seconds=full_update_seconds,
                                 full_loop_with_diagnostics_seconds=full_loop_with_diagnostics_seconds,
                                 full_one_prefix_seconds=full_one_seconds,
                                 full_score_seconds=full_score_seconds,
                                 full_one_score_seconds=full_one_score_seconds,
                                 postseal_audit_seconds=audit_seconds,
                                 pre_payload_save_seconds=pre_save_seconds,
                                 final_payload_save_seconds=final_save_seconds),
                    hashes=hashes, runner_sha256=runner, design_sha256=design,
                    dependencies=deps, dependency_manifest_sha256=dependency_manifest,
                    artifact_sha256=r.sha(artifact), pre_artifact_sha256=pre_hash,
                    pre_record_sha256=pre_record_hash,
                    scope='No evaluation labels or accuracy; all approximate outputs/eligibility sealed before full updates.',
                    finished=time.time())))
                completed += 1
                print(json.dumps(dict(dataset=name, hops=hops, seed=seed, completed=completed,
                                      joint_uncertified_pct=certificates['joint']['uncertified_pct'],
                                      node_domain=node_domain,
                                      node_disagreement=None if node is None else node['prediction_disagreement_pct'])), flush=True)
            r.save(folder / 'complete.json', dict(rows=len(SEEDS), runner_sha256=runner,
                design_sha256=design, dependencies=deps, representation_sha256=representation_hash))
            # Covariance/unit tensors may never be constructed when a whole
            # condition is resumed from verified completed records.
            del z, zz
            torch.cuda.empty_cache()
        del d, x, adj, propagated
        torch.cuda.empty_cache()
    assert completed == 420
    r.save(OUT / 'complete.json', dict(rows=completed, datasets=14, depths=list(DEPTHS), seeds=list(SEEDS),
        tau=TAU, beta=BETA, rounds=ROUNDS, runner_sha256=runner, design_sha256=design,
        dependencies=deps, dependency_manifest_sha256=dependency_manifest,
        finished=time.time()))


if __name__ == '__main__':
    try:
        with torch.no_grad():
            main()
    except Exception:
        r.save(OUT / 'failure.json', dict(time=time.time(), traceback=traceback.format_exc()))
        raise
