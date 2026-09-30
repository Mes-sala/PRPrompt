"""Pure NumPy candidate certificate and synthetic self-check; no graph or labels.

The certificate functions accept only pre-softmax geometry and query dot products.
Exact q/u/N appear exclusively in the self-check reference after certificates.
All bounds are real-arithmetic statements. Float64 checks are not interval proofs.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

DOMAIN_ETA = .1


def softmax(x, axis=-1):
    v = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return v / np.sum(v, axis=axis, keepdims=True)


def geometry_certificate(anchor, mean, counts, pi, moment_soft,
                         projected_variance, radius, eta, pool_size,
                         replication, method):
    """Pre-only prototype centers/radii; separate results for moment0/moment2.

    R25 sources: anchor, mean, support_counts, mean_assignment, moment_soft,
    projected_variance; metadata node_radius/node_eta_upper/n_unlabeled.
    No actual assignment, class count, or exact soft prototype is accepted.
    """
    a, g, K, p, soft, nu = [np.asarray(x, dtype=np.float64) for x in
                            (anchor, mean, counts, pi, moment_soft, projected_variance)]
    if method not in ('moment0', 'moment2'):
        raise ValueError('Only frozen moment0/moment2 are supported')
    if not all(np.isfinite(x).all() for x in (a, g, K, p, soft, nu)):
        raise ValueError('Nonfinite pre geometry')
    if not (np.all(K > 0) and np.all(p > 0) and np.all(nu >= 0)):
        raise ValueError('Nonpositive count/pi or negative projected variance')
    if not np.isclose(p.sum(), 1., atol=1e-12, rtol=1e-12):
        raise ValueError('pi must sum to one')
    if not (math.isfinite(radius) and radius >= 0 and math.isfinite(eta)
            and 0 <= eta <= DOMAIN_ETA and pool_size > 0 and replication > 0):
        raise ValueError('Outside fixed eta domain or invalid scalar geometry')
    d = 2 * a - p @ (2 * a)
    eta_reconstructed = radius * np.linalg.norm(d, axis=1).max()
    if not np.isclose(eta, eta_reconstructed, atol=1e-12, rtol=1e-10):
        raise ValueError('eta is inconsistent with radius and anchor contrast')
    h = soft - g
    nubar = p @ nu
    B = np.exp(eta) * eta * (nu / 6 + (2 / 3 + eta / 2) * nubar)
    J = radius * (.5 * (nu + nubar) + B)
    delta0 = .5 * np.abs(nu - nubar) + B
    lower = np.maximum(np.exp(-eta * eta / 2), 1 - delta0)
    mhat = np.ones_like(p) if method == 'moment0' else 1 + .5 * (nu - nubar)
    delta = delta0 if method == 'moment0' else B
    if np.any(mhat <= 0):
        raise ValueError('Nonpositive approximate relative class count')
    T = replication * pool_size * p
    weight = K / (K + T * mhat)
    full_center = weight[:, None] * a + (1 - weight)[:, None] * soft
    common_center = weight[:, None] * a + (1 - weight)[:, None] * g
    factor = T / (K + T * lower)
    full_radius = factor * (J + np.abs(1 - mhat) * np.linalg.norm(h, axis=1)
                            + delta * np.linalg.norm(full_center - g, axis=1))
    common_radius = factor * delta * np.linalg.norm(common_center - g, axis=1)
    return dict(method=method, weight=weight, full_center=full_center,
                common_center=common_center, full_radius=full_radius,
                common_radius=common_radius, B=B, J=J, delta=delta,
                delta0=delta0, mass_ratio_lower=lower, mhat=mhat,
                h=h, nubar=nubar, eta=eta, radius=radius)


def score_intervals(query_norm_squared, query_center_dot, center, radius):
    """Exact distance-to-ball intervals in real arithmetic.

    A negative squared distance causes per-query/class invalidation rather than
    silent clipping. This guards input inconsistency; it is not rigorous outward
    rounding of all preceding NumPy arithmetic.
    """
    distance2 = (np.asarray(query_norm_squared)[:, None]
                 - 2 * np.asarray(query_center_dot)
                 + np.sum(center * center, axis=1)[None, :])
    valid = np.isfinite(distance2) & (distance2 >= 0)
    distance = np.sqrt(np.where(valid, distance2, np.nan))
    lo = -(distance + radius[None, :]) ** 2
    hi = -np.maximum(0., distance - radius[None, :]) ** 2
    nominal = -distance2
    return dict(lower=lo, upper=hi, nominal=nominal, valid=valid)


def certified_winner(interval):
    score, lower, upper = (interval[k] for k in ('nominal', 'lower', 'upper'))
    candidate = np.argmax(score, axis=1)
    other_upper = upper.copy()
    other_upper[np.arange(len(score)), candidate] = -np.inf
    certified = (interval['valid'].all(1)
                 & (lower[np.arange(len(score)), candidate] > other_upper.max(1)))
    return np.where(certified, candidate, -1)


def query_certificate(geom, query_norm_squared, query_anchor_dot,
                      query_mean_dot, query_moment_soft_dot):
    """The dot products are precisely those already stored in each R25 pre NPZ."""
    weight = geom['weight'][None, :]
    full_dot = weight * query_anchor_dot + (1 - weight) * query_moment_soft_dot
    common_dot = weight * query_anchor_dot + (1 - weight) * query_mean_dot[:, None]
    full = score_intervals(query_norm_squared, full_dot,
                           geom['full_center'], geom['full_radius'])
    common = score_intervals(query_norm_squared, common_dot,
                             geom['common_center'], geom['common_radius'])
    yf, yc = certified_winner(full), certified_winner(common)
    both = (yf >= 0) & (yc >= 0)
    same, different = both & (yf == yc), both & (yf != yc)
    unknown = ~both
    return dict(full=full, common=common, full_winner=yf, common_winner=yc,
                certified_same=same, certified_different=different, abstain=unknown,
                disagreement_lower=float(different.mean()),
                disagreement_upper=float((different | unknown).mean()))


def normalized(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def selfcheck():
    rng = np.random.default_rng(270926)
    families = []
    for draw in range(100):
        C, dim, n = int(rng.integers(2,6)), int(rng.integers(2,12)), int(rng.integers(20,91))
        eps = float(10 ** rng.uniform(-2., -.55))
        tangent = rng.normal(size=(n + 5*C, dim - 1))
        z = normalized(np.column_stack([np.ones(len(tangent)), eps*tangent]))
        a = z[:5*C].reshape(C,5,dim).mean(1)
        families.append((f'random{draw}', z[5*C:], a))
    # Fixed counterexample geometry at several concentrations. Physical repeated
    # pools keep the synthetic check within R25's replication choices {1, 8}.
    v = np.array([[6,0],[-6,0],[0,1],[0,-1],[1,2],[-1,-2],[1,-2],[-1,2]], float)
    m = np.array([[.3,.3],[-.3,-.3]])
    for eps, copies in [(1/16,32),(1/32,128),(1/64,512)]:
        pool = normalized(np.column_stack([np.ones(8),eps*v]))
        a = normalized(np.column_stack([np.ones(2),eps*m]))
        families.append((f'counterexample_eps{eps}',np.tile(pool,(copies,1)),a))
    totals = dict(families=0, domain_families=0, certificate_cases=0,
                  certified_same=0, certified_different=0, abstain=0,
                  wrong_certified_full=0, wrong_certified_common=0,
                  wrong_same_or_different=0, max_full_radius_violation=0.,
                  max_common_radius_violation=0., max_score_interval_violation=0.,
                  max_relative_mass_bound_violation=0., max_numerator_bound_violation=0.,
                  max_prototype_identity_residual=0.)
    counterexamples=[]
    for name,z,a in families:
        totals['families'] += 1
        n,C=len(z),len(a);g=z.mean(0);x=z-g;S=x.T@x/n
        p=softmax(2*a@g-np.sum(a*a,1));d=2*a-p@(2*a)
        h=d@S;nu=np.sum(h*d,1);R=float(np.linalg.norm(x,axis=1).max())
        eta=R*float(np.linalg.norm(d,axis=1).max())
        if eta>DOMAIN_ETA:continue
        totals['domain_families']+=1
        K=np.full(C,5.);soft=g+h
        # Seal the Python values of all candidate predictions before exact q.
        predictions=[]
        for rep in (1,8):
            for method in ('moment0','moment2'):
                geom=geometry_certificate(a,g,K,p,soft,nu,R,eta,n,rep,method)
                cert=query_certificate(geom,np.sum(z*z,1),z@a.T,z@g,z@soft.T)
                predictions.append((rep,method,geom,cert))
        # Independent reference starts here. No reference value was an input.
        logits=-np.sum((z[:,None,:]-a[None,:,:])**2,axis=2)
        q=softmax(logits);N=q.sum(0);u=q.T@z/N[:,None]
        mratio=N/(n*p);t=x@d.T;f=q/p[None,:]
        enum=x.T@(f-1-t)/n;enum=enum.T
        for rep,method,geom,cert in predictions:
            M=(K[:,None]*a+rep*N[:,None]*u)/(K+rep*N)[:,None]
            common=(K[:,None]*a+rep*N[:,None]*g)/(K+rep*N)[:,None]
            score=-np.sum((z[:,None,:]-M[None,:,:])**2,axis=2)
            cscore=-np.sum((z[:,None,:]-common[None,:,:])**2,axis=2)
            for key,value in (
                ('max_full_radius_violation',np.max(np.linalg.norm(M-geom['full_center'],axis=1)-geom['full_radius'])),
                ('max_common_radius_violation',np.max(np.linalg.norm(common-geom['common_center'],axis=1)-geom['common_radius'])),
                ('max_relative_mass_bound_violation',np.max(np.abs(mratio-geom['mhat'])-geom['delta'])),
                ('max_numerator_bound_violation',np.max(np.linalg.norm(enum,axis=1)-geom['J']))):
                totals[key]=max(totals[key],float(value))
                assert value<2e-12,(name,rep,method,key,value)
            T=rep*n*p;center=geom['full_center']
            identity=T[:,None]/(K+T*mratio)[:,None]*(enum+(1-geom['mhat'])[:,None]*h
                     +(geom['mhat']-mratio)[:,None]*(center-g))
            identity_error=float(np.max(abs(M-center-identity)))
            totals['max_prototype_identity_residual']=max(totals['max_prototype_identity_residual'],identity_error)
            assert identity_error<2e-12
            for target,actual in [('full',score),('common',cscore)]:
                intervals=cert[target]
                violation=max(float(np.max(intervals['lower']-actual)),float(np.max(actual-intervals['upper'])),0.)
                totals['max_score_interval_violation']=max(totals['max_score_interval_violation'],violation)
                assert violation<2e-12,(name,target,violation)
                winner=cert[target+'_winner'];wrong=int(((winner>=0)&(winner!=actual.argmax(1))).sum())
                totals['wrong_certified_'+target]+=wrong;assert wrong==0
            actual_diff=score.argmax(1)!=cscore.argmax(1)
            wrong=int((cert['certified_same']&actual_diff).sum()+(cert['certified_different']&~actual_diff).sum())
            totals['wrong_same_or_different']+=wrong;assert wrong==0
            for key in ('certified_same','certified_different','abstain'):
                totals[key]+=int(cert[key].sum())
            totals['certificate_cases']+=1
            if name.startswith('counterexample'):
                counterexamples.append(dict(name=name,replication=rep,method=method,eta=eta,
                    actual_disagreement=float(actual_diff.mean()),lower=cert['disagreement_lower'],
                    upper=cert['disagreement_upper']))
    report=dict(status='passed',scope='Synthetic pure NumPy verification only; no real graph or query labels',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),domain_eta=DOMAIN_ETA,
        totals=totals,counterexample_cases=counterexamples,
        caveat='Float64 tolerance audit of real-arithmetic bounds, not interval arithmetic; no efficacy claim')
    out=Path(__file__).with_name('Ren一步_矩近似事前界_纯NumPy核验.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    selfcheck()
