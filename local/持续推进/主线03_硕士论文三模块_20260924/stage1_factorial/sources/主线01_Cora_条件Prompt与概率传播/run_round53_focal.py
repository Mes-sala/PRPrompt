"""Frozen-backbone 5-shot prompt search; query labels accessed after sealing predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.conv.gcn_conv import gcn_norm

sys.path.insert(0, '/root/ProG')
from prompt_graph.model import GCN
from prompt_graph.utils import act


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def thash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def ridge_teacher(r, support, sy, classes):
    rn = F.normalize(r, dim=1)
    xs = rn[support].double()
    gram = xs @ xs.T
    y = F.one_hot(sy, classes).double()
    eye = torch.eye(len(support), dtype=torch.float64, device=r.device)
    losses, candidates = {}, {}
    for lam in [0.01, 0.1, 1.0, 10.0]:
        inverse = torch.linalg.solve(gram + lam * eye, eye)
        alpha = inverse @ y
        loo = y - alpha / inverse.diag()[:, None]
        losses[str(lam)] = F.cross_entropy(loo / 0.2, sy).item()
        candidates[lam] = alpha
    selected = min(candidates, key=lambda lam: losses[str(lam)])
    # Only the small support Gram solve uses double precision.
    score = ((rn @ rn[support].T) @ candidates[selected].float()) / 0.2
    return score.detach(), {'ridge_lambda': selected, 'ridge_loo_ce': losses}


def memory_prompt(x, r, support, sy, classes, seed, method):
    if method == 'input05mean':
        return 0.5 * x + 0.5 * x.mean(0, keepdim=True), {}
    labels = sy
    if method == 'input05perm':
        gen = torch.Generator(device=x.device).manual_seed(seed + 100000)
        labels = sy[torch.randperm(len(sy), generator=gen, device=x.device)]
    oh = F.one_hot(labels, classes).float()
    keys = oh.T @ r[support] / oh.sum(0)[:, None]
    values = oh.T @ x[support] / oh.sum(0)[:, None]
    q = (F.normalize(r, dim=1) @ F.normalize(keys, dim=1).T / 0.2).softmax(-1)
    alpha = {'input025': 0.25, 'input075': 0.75}.get(method, 0.5)
    xp = x + alpha * (q @ values - x)
    return xp, {'keys': keys.cpu(), 'values': values.cpu(), 'alpha': alpha, 'group_labels': labels.cpu()}


def make_filters(x, edges):
    src, dst = edges[:, edges[0] != edges[1]]
    n = len(x)
    xn = F.normalize(x, dim=1)
    similarities = torch.cat([(xn[src[i:i+8192]] * xn[dst[i:i+8192]]).sum(1) for i in range(0, len(src), 8192)])
    out = {}
    for name, weight in [('plain', torch.ones_like(similarities)), ('bilateral', (5 * similarities).exp())]:
        degree = torch.zeros(n, device=x.device).index_add_(0, dst, weight)
        isolated = (degree == 0).nonzero().flatten()
        rows = torch.cat([dst, isolated]); cols = torch.cat([src, isolated])
        weights = torch.cat([weight / degree[dst], torch.ones(len(isolated), device=x.device)])
        op = torch.sparse_coo_tensor(torch.stack([rows, cols]), weights, (n, n)).coalesce()
        assert (torch.sparse.mm(op, torch.ones(n, 1, device=x.device)) - 1).abs().max() < 2e-6
        out[name] = op
    return out


def smooth_probs(q, op):
    for _ in range(2):
        q = 0.5 * q + 0.5 * torch.sparse.mm(op, q)
    q = q / q.sum(1, keepdim=True)
    assert torch.isfinite(q).all() and q.min() >= 0
    return q


@torch.no_grad()
def rectified_teacher(r, support, sy, classes, method, seed):
    torch.manual_seed(seed)
    rn = F.normalize(r, dim=1)
    oh = F.one_hot(sy, classes).float()
    original = (oh.T @ rn[support]) / oh.sum(0)[:, None]
    centers = original.clone()
    mask = torch.ones(len(r), dtype=torch.bool, device=r.device)
    mask[support] = False
    ru = rn[mask]
    beta = 0.5 if 'ref50' in method else (0.0 if 'ref0' in method else 0.25)
    for _ in range(3):
        assignment = (ru @ F.normalize(centers, dim=1).T / 0.05).softmax(-1)
        if method.endswith('shuffle'):
            assignment = assignment[torch.randperm(len(assignment), device=r.device)]
        pseudo = (assignment.T @ ru) / assignment.sum(0).clamp_min(1e-8)[:, None]
        centers = (1 - beta) * original + beta * pseudo
    scores = rn @ F.normalize(centers, dim=1).T / 0.2
    assert torch.isfinite(scores).all() and not mask[support].any()
    return scores, {'rect_beta': beta, 'rect_iterations': 3, 'rect_unlabeled_count': int(mask.sum()), 'rect_support_overlap': int(mask[support].sum()), 'rect_center_change': (centers-original).norm().item()}


def token_prompt(u, r, adj, model, edges, support, sy, classes, seed, method, steps, filters=None):
    torch.manual_seed(seed)
    if 'ref' in method:
        teacher_scores, ridge_info = rectified_teacher(r, support, sy, classes, method, seed)
        teacher = teacher_scores.softmax(-1).detach()
    elif 'ridge' in method:
        teacher_scores, ridge_info = ridge_teacher(r, support, sy, classes)
        teacher = teacher_scores.softmax(-1).detach()
    else:
        teacher = logits(r, support, sy, classes).softmax(-1).detach()
        ridge_info = {}
    if "plain" in method:
        teacher = smooth_probs(teacher, filters["plain"]).detach()
        ridge_info["teacher_filter"] = "plain"
    gate = teacher - teacher.mean(0, keepdim=True)
    if method.endswith('perm'):
        gate = gate[torch.randperm(len(gate), device=gate.device)]
    gate = torch.sparse.mm(adj, gate).detach()
    scale = u.std(0).clamp_min(1e-6).detach()
    tokens = torch.nn.Parameter(torch.zeros(classes, u.shape[1], device=u.device))
    tokens2 = torch.nn.Parameter(torch.zeros_like(tokens))
    params = [tokens, tokens2] if method.endswith('deep') else [tokens]
    hbase = act(u).detach()
    zbase = model.conv_layers[1](hbase, edges).detach()
    hscale = hbase.std(0).clamp_min(1e-6).detach()
    zscale = zbase.std(0).clamp_min(1e-6).detach()
    amp = 10.0 if '10' in method else (3.0 if '3' in method else 1.0)
    def forward():
        if method.endswith('late'):
            return model.conv_layers[1](hbase + amp * (gate @ tokens.tanh()) * hscale, edges)
        if method.endswith('output'):
            return zbase + amp * (gate @ tokens.tanh()) * zscale
        if method.endswith('deep'):
            hp = act(u + (amp / 2**0.5) * (gate @ tokens.tanh()) * scale)
            hp = hp + (amp / 2**0.5) * (gate @ tokens2.tanh()) * hscale
            return model.conv_layers[1](hp, edges)
        hp = act(u + amp * (gate @ tokens.tanh()) * scale)
        return model.conv_layers[1](hp, edges)
    with torch.no_grad():
        noop = (F.normalize(forward(), dim=1) - F.normalize(zbase, dim=1)).abs().max().item()
        assert noop < 2e-5
    if method.startswith('token_random'):
        with torch.no_grad():
            tokens.normal_(0, 0.3)
        return forward().detach(), {'tokens': tokens.detach().cpu(), 'amplitude': amp}, {}
    opt = torch.optim.Adam(params, lr=0.01)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        ce = F.cross_entropy(logits(z, support, sy, classes, loo=True), sy)
        kd = F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        loss = ce + (kd if 'kd' in method else 0) + 0.001 * sum(p.square().mean() for p in params)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        if step in (0, steps-1):
            hist.append({'step': step, 'loss': loss.item(), 'ce': ce.item(), 'kd': kd.item(), 'gradient_norm': tokens.grad.norm().item()})
    return forward().detach(), {'tokens': tokens.detach().cpu(), 'tokens2': tokens2.detach().cpu(), 'amplitude': amp}, {'history': hist, 'prompt_noop_norm_error': noop, 'trainable_parameters': sum(p.numel() for p in params), **ridge_info}


def logits(z, support, y, classes, loo=False):
    z = F.normalize(z, dim=1)
    zs = z[support]
    oh = F.one_hot(y, classes).float()
    sums = oh.T @ zs
    counts = oh.sum(0)
    proto = F.normalize(sums / counts[:, None], dim=1)
    if not loo:
        return z @ proto.T / 0.2
    per = (sums[None] - oh[:, :, None] * zs[:, None])
    per = per / (counts[None, :, None] - oh[:, :, None])
    return (zs[:, None] * F.normalize(per, dim=2)).sum(-1) / 0.2


def train(h, z0, model, edges, support, sy, classes, method, seed, steps):
    if method == 'frozen':
        return z0, {}, {}
    torch.manual_seed(seed)
    a = torch.nn.Parameter(torch.zeros(h.shape[1], device=h.device))
    b = torch.nn.Parameter(torch.zeros_like(a))
    params = [a] if method == 'graphprompt' else [a, b]
    opt = torch.optim.Adam(params, lr=0.01)
    scale = h.std(0).clamp_min(1e-6)
    teacher = logits(z0, support, sy, classes).softmax(-1).detach()
    def forward():
        if method == 'graphprompt':
            return z0 * (1 + 0.5 * a.tanh())
        hp = h * (1 + 0.5 * a.tanh()) + 0.5 * scale * b.tanh()
        return model.conv_layers[1](hp, edges)
    history = []
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        ce = F.cross_entropy(logits(z, support, sy, classes, loo=True), sy)
        loss = ce + 0.001 * sum(p.square().mean() for p in params)
        if 'kl' in method:
            loss = loss + F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        if 'mix' in method:
            zn = F.normalize(z[support], dim=1)
            perm = torch.randperm(len(sy), device=z.device)
            mix = F.normalize((zn + zn[perm]) / 2, dim=1)
            oh = F.one_hot(sy, classes).float()
            proto = F.normalize(oh.T @ zn / oh.sum(0)[:, None], dim=1)
            ml = mix @ proto.T / 0.2
            loss = loss + 0.25 * (F.cross_entropy(ml, sy) + F.cross_entropy(ml, sy[perm]))
        loss.backward()
        assert all(p.grad is None for p in model.parameters())
        assert torch.isfinite(loss)
        opt.step()
        if step in (0, steps - 1):
            history.append({'step': step, 'loss': loss.item(), 'support_loo_ce': ce.item()})
    return forward().detach(), {'a': a.detach().cpu(), 'b': b.detach().cpu()}, {'history': history}


@torch.no_grad()
def knn_operator(z,seed,random=False):
    zn=F.normalize(z,dim=1);n=len(z);ids=[];ws=[];k=min(10,n-1)
    for start in range(0,n,1024):
        end=min(start+1024,n);sim=zn[start:end]@zn.T
        sim[torch.arange(end-start,device=z.device),torch.arange(start,end,device=z.device)]=-torch.inf
        vals,ix=sim.topk(k,dim=1);ids.append(ix);ws.append((vals/.2).softmax(-1))
    ids=torch.cat(ids);weights=torch.cat(ws)
    if random:
        gen=torch.Generator(device=z.device).manual_seed(seed+410000)
        # One shared random permutation of node identities preserves each row's distinct neighbors.
        # Remove self destinations and replace them with the first available non-self node.
        perm=torch.randperm(n,device=z.device,generator=gen);ids=perm[ids]
        for row in (ids==torch.arange(n,device=z.device)[:,None]).any(1).nonzero().flatten().tolist():
            used=set(ids[row].tolist());replacement=next(j for j in range(n) if j!=row and j not in used)
            ids[row,ids[row]==row]=replacement
    assert not (ids==torch.arange(n,device=z.device)[:,None]).any()
    assert torch.allclose(weights.sum(1),torch.ones(n,device=z.device),atol=1e-6)
    row=torch.arange(n,device=z.device)[:,None].expand_as(ids)
    op=torch.sparse_coo_tensor(torch.stack([row.flatten(),ids.flatten()]),weights.flatten(),(n,n)).coalesce()
    return op,ids,weights

@torch.no_grad()
def propagate(p0,op,si,sy,alpha,clamp,steps=20):
    p=p0.clone();target=F.one_hot(sy,p.shape[1]).to(p.dtype)
    if clamp:p[si]=target
    last=0.
    for _ in range(steps):
        new=(1-alpha)*p0+alpha*torch.sparse.mm(op,p)
        if clamp:new[si]=target
        last=(new-p).abs().max().item();p=new
    assert torch.isfinite(p).all() and p.min()>=0
    assert (p.sum(1)-1).abs().max()<2e-5
    if clamp:assert torch.equal(p[si],target)
    return p,last

@torch.no_grad()
def qe_readout(z,si,sy,classes,kind,seed,original,cache,path):
    zn=F.normalize(z,dim=1)
    if kind=='qeorig':
        expanded=F.normalize(zn+torch.sparse.mm(original,zn),dim=1)
        return logits(expanded,si,sy,classes),dict(readout=kind,expanded_embedding_sha256=thash(expanded))
    mode,kstr=kind.split('-');k=int(kstr)
    if 'qe50' not in cache:
        ix=[];vals=[]
        for start in range(0,len(z),1024):
            end=min(start+1024,len(z));sim=zn[start:end]@zn.T
            sim[torch.arange(end-start,device=z.device),torch.arange(start,end,device=z.device)]=-torch.inf
            v,j=sim.topk(50,dim=1);ix.append(j);vals.append(v)
        cache['qe50']=(torch.cat(ix),torch.cat(vals))
    allix,allv=cache['qe50'];ix=allix[:,:k].clone();weights=allv[:,:k].clamp_min(0).pow(3)
    if mode in ['qeuniform','qeuniformrandom','qefixed','qeprob','qepu']:weights=torch.ones_like(weights)
    if mode=='qeps':weights=(allv[:,:k]/.2).softmax(-1)
    if mode in ['qerandom','qeuniformrandom']:
        gen=torch.Generator(device=z.device).manual_seed(seed+460000)
        perm=torch.randperm(len(z),device=z.device,generator=gen);ix=perm[ix]
        for row in (ix==torch.arange(len(z),device=z.device)[:,None]).any(1).nonzero().flatten().tolist():
            used=set(ix[row].tolist());replacement=next(j for j in range(len(z)) if j!=row and j not in used)
            ix[row,ix[row]==row]=replacement
    else:assert mode in ['qe','qeuniform','qefixed','qeprob','qepu','qeps']
    expanded=F.normalize(zn+(zn[ix]*weights[:,:,None]).sum(1),dim=1)
    assert torch.isfinite(expanded).all()
    readscore=logits(expanded,si,sy,classes)
    if mode=='qefixed':
        oh=F.one_hot(sy,classes).float()
        proto=F.normalize(oh.T@zn[si]/oh.sum(0)[:,None],dim=1)
        readscore=expanded@proto.T/.2
    if mode=='qeprob':
        p0=logits(z,si,sy,classes).softmax(-1)
        readscore=((p0+p0[ix].sum(1))/(k+1)).clamp_min(1e-12).log()
    if mode in ['qepu','qeps']:
        rows=torch.arange(len(z),device=z.device)[:,None].expand_as(ix)
        normweights=weights/weights.sum(1,keepdim=True)
        op=torch.sparse_coo_tensor(torch.stack([rows.flatten(),ix.flatten()]),normweights.flatten(),(len(z),len(z))).coalesce()
        pred,_=propagate(logits(z,si,sy,classes).softmax(-1),op,si,sy,.9,True)
        readscore=pred.clamp_min(1e-12).log()
    file=path.parent/(path.name+'_'+kind+'.pt');file.parent.mkdir(exist_ok=True)
    torch.save(dict(indices=ix.cpu().int(),weights=weights.cpu(),self_weight=1.,embedding_sha256=thash(z),
        expanded_embedding_sha256=thash(expanded),variant=kind),file)
    return readscore,dict(readout=kind,expanded_embedding_sha256=thash(expanded),
        neighbor_file=str(file.relative_to(file.parent.parent)),neighbor_sha256=sha(file),qe_k=k,qe_power=3 if mode in ['qe','qerandom'] else (None if mode=='qeps' else 0),
        neighbor_weight_sum_mean=weights.sum(1).mean().item(),support_prototypes_recomputed=mode not in ['qefixed','qeprob','qepu','qeps'],
        aggregation_space='probability' if mode in ['qeprob','qepu','qeps'] else 'embedding',
        operation_steps=20 if mode in ['qepu','qeps'] else 1)

@torch.no_grad()
def diffusion_readout(z,si,sy,classes,kind,seed,original,cache,path):
    score=logits(z,si,sy,classes)
    if kind=='mean':return score,{'readout':kind}
    if kind.startswith('qe'):return qe_readout(z,si,sy,classes,kind,seed,original,cache,path)
    graph,astr,clampstr=kind.split('-');alpha=float(astr);clamp=clampstr=='clamp'
    meta={'readout':kind,'alpha':alpha,'support_clamp':clamp}
    if graph=='orig':op=original
    else:
        if graph not in cache:
            op,ids,weights=knn_operator(z,seed,random=graph=='random')
            file=path.parent/(path.name+'_'+graph+'.pt');file.parent.mkdir(exist_ok=True)
            torch.save({'indices':ids.cpu().to(torch.int32),'weights':weights.cpu(),'embedding_sha256':thash(z),'seed':seed},file)
            cache[graph]=(op,file)
        op,file=cache[graph]
        meta.update(neighbor_file=str(file.relative_to(file.parent.parent)),neighbor_sha256=sha(file))
    p,last=propagate(score.softmax(-1),op,si,sy,alpha,clamp)
    return p.clamp_min(1e-12).log(),dict(meta,last_step_residual=last,propagation_steps=20)


def vat_prompt(u, r, adj, model, edges, support, sy, classes, seed, method, steps, filters=None, vatmode="none", epsilon=0.):
    torch.manual_seed(seed)
    if 'ref' in method:
        teacher_scores, ridge_info = rectified_teacher(r, support, sy, classes, method, seed)
        teacher = teacher_scores.softmax(-1).detach()
    elif 'ridge' in method:
        teacher_scores, ridge_info = ridge_teacher(r, support, sy, classes)
        teacher = teacher_scores.softmax(-1).detach()
    else:
        teacher = logits(r, support, sy, classes).softmax(-1).detach()
        ridge_info = {}
    if "plain" in method:
        teacher = smooth_probs(teacher, filters["plain"]).detach()
        ridge_info["teacher_filter"] = "plain"
    gate = teacher - teacher.mean(0, keepdim=True)
    if method.endswith('perm'):
        gate = gate[torch.randperm(len(gate), device=gate.device)]
    gate = torch.sparse.mm(adj, gate).detach()
    scale = u.std(0).clamp_min(1e-6).detach()
    tokens = torch.nn.Parameter(torch.zeros(classes, u.shape[1], device=u.device))
    tokens2 = torch.nn.Parameter(torch.zeros_like(tokens))
    params = [tokens, tokens2] if method.endswith('deep') else [tokens]
    hbase = act(u).detach()
    zbase = model.conv_layers[1](hbase, edges).detach()
    hscale = hbase.std(0).clamp_min(1e-6).detach()
    zscale = zbase.std(0).clamp_min(1e-6).detach()
    amp = 10.0 if '10' in method else (3.0 if '3' in method else 1.0)
    perturb_scale=hscale if method.endswith('late') else scale
    def forward(delta=None):
        noise=0 if delta is None else delta*perturb_scale
        if method.endswith('late'):
            hp=hbase+amp*(gate@tokens.tanh())*hscale+noise
        else:
            hp=act(u+amp*(gate@tokens.tanh())*scale+noise)
        return model.conv_layers[1](hp,edges)
    with torch.no_grad():
        noop = (F.normalize(forward(), dim=1) - F.normalize(zbase, dim=1)).abs().max().item()
        assert noop < 2e-5
    if method.startswith('token_random'):
        with torch.no_grad():
            tokens.normal_(0, 0.3)
        return forward().detach(), {'tokens': tokens.detach().cpu(), 'amplitude': amp}, {}
    opt = torch.optim.Adam(params, lr=0.01)
    hist = []
    noisegen=torch.Generator(device=u.device).manual_seed(seed+390000)
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        ce = F.cross_entropy(logits(z, support, sy, classes, loo=True), sy)
        kd = F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        loss = ce + (kd if 'kd' in method else 0) + 0.001 * sum(p.square().mean() for p in params)
        consistency=z.new_zeros(());direction_norm=z.new_zeros(())
        if vatmode!='none':
            with torch.no_grad():
                zn=F.normalize(z.detach(),dim=1);oh=F.one_hot(sy,classes).float()
                prototypes=F.normalize(oh.T@zn[support]/oh.sum(0)[:,None],dim=1)
                current=(zn@prototypes.T/.2).softmax(-1).detach()
            def dist(delta):
                pred=F.normalize(forward(delta),dim=1)@prototypes.T/.2
                return F.kl_div(pred.log_softmax(-1),current,reduction='batchmean')
            direction=torch.randn(u.shape,device=u.device,generator=noisegen)
            direction=F.normalize(direction,dim=1)*u.shape[1]**.5
            if vatmode=='vat':
                probe=(.01*direction).detach().requires_grad_(True)
                g=torch.autograd.grad(dist(probe),probe)[0]
                assert torch.isfinite(g).all()
                direction=(F.normalize(g,dim=1)*u.shape[1]**.5).detach()
            delta=(epsilon*direction).detach();direction_norm=delta.square().mean().sqrt()
            consistency=dist(delta);loss=loss+consistency
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        if step in (0, steps-1):
            hist.append({'step': step, 'loss': loss.item(), 'ce': ce.item(), 'kd': kd.item(), 'gradient_norm': tokens.grad.norm().item(), 'consistency': consistency.item(), 'perturb_rms': direction_norm.item()})
    return forward().detach(), {'tokens': tokens.detach().cpu(), 'tokens2': tokens2.detach().cpu(), 'amplitude': amp}, {'vat_mode':vatmode,'epsilon':epsilon,'history': hist, 'prompt_noop_norm_error': noop, 'trainable_parameters': sum(p.numel() for p in params), **ridge_info}


def focal_loss(score,y,gamma,mode='focal',gen=None):
    ce=F.cross_entropy(score,y,reduction='none');pt=(-ce).exp();one=(-torch.expm1(-ce)).clamp_min(0)
    coeff=torch.ones_like(pt) if gamma==0 else one.pow(gamma)+gamma*pt*one.pow(gamma-1)*ce
    if mode=='focal':loss=(one.pow(gamma)*ce).mean()
    elif mode=='scale':loss=coeff.detach().mean()*ce.mean()
    elif mode=='perm':
        ix=torch.randperm(len(y),device=score.device,generator=gen)
        loss=(coeff.detach()[ix]*ce).mean()
    else:raise ValueError(mode)
    return loss,coeff.detach()

def focal_prompt(u, r, adj, model, edges, support, sy, classes, seed, method, steps, filters=None, gamma=0., mode="focal"):
    torch.manual_seed(seed)
    if 'ref' in method:
        teacher_scores, ridge_info = rectified_teacher(r, support, sy, classes, method, seed)
        teacher = teacher_scores.softmax(-1).detach()
    elif 'ridge' in method:
        teacher_scores, ridge_info = ridge_teacher(r, support, sy, classes)
        teacher = teacher_scores.softmax(-1).detach()
    else:
        teacher = logits(r, support, sy, classes).softmax(-1).detach()
        ridge_info = {}
    if "plain" in method:
        teacher = smooth_probs(teacher, filters["plain"]).detach()
        ridge_info["teacher_filter"] = "plain"
    gate = teacher - teacher.mean(0, keepdim=True)
    if method.endswith('perm'):
        gate = gate[torch.randperm(len(gate), device=gate.device)]
    gate = torch.sparse.mm(adj, gate).detach()
    scale = u.std(0).clamp_min(1e-6).detach()
    tokens = torch.nn.Parameter(torch.zeros(classes, u.shape[1], device=u.device))
    tokens2 = torch.nn.Parameter(torch.zeros_like(tokens))
    params = [tokens, tokens2] if method.endswith('deep') else [tokens]
    hbase = act(u).detach()
    zbase = model.conv_layers[1](hbase, edges).detach()
    hscale = hbase.std(0).clamp_min(1e-6).detach()
    zscale = zbase.std(0).clamp_min(1e-6).detach()
    amp = 10.0 if '10' in method else (3.0 if '3' in method else 1.0)
    def forward():
        if method.endswith('late'):
            return model.conv_layers[1](hbase + amp * (gate @ tokens.tanh()) * hscale, edges)
        if method.endswith('output'):
            return zbase + amp * (gate @ tokens.tanh()) * zscale
        if method.endswith('deep'):
            hp = act(u + (amp / 2**0.5) * (gate @ tokens.tanh()) * scale)
            hp = hp + (amp / 2**0.5) * (gate @ tokens2.tanh()) * hscale
            return model.conv_layers[1](hp, edges)
        hp = act(u + amp * (gate @ tokens.tanh()) * scale)
        return model.conv_layers[1](hp, edges)
    with torch.no_grad():
        noop = (F.normalize(forward(), dim=1) - F.normalize(zbase, dim=1)).abs().max().item()
        assert noop < 2e-5
    if method.startswith('token_random'):
        with torch.no_grad():
            tokens.normal_(0, 0.3)
        return forward().detach(), {'tokens': tokens.detach().cpu(), 'amplitude': amp}, {}
    gen=torch.Generator(device=u.device).manual_seed(seed+530000)
    scales=[]
    opt = torch.optim.Adam(params, lr=0.01)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        loo=logits(z,support,sy,classes,loo=True)
        ce=F.cross_entropy(loo,sy)
        supervised,coeff=focal_loss(loo,sy,gamma,mode,gen)
        pt=loo.softmax(-1).gather(1,sy[:,None]).flatten()
        scales.append(dict(mean=coeff.mean().item(),minimum=coeff.min().item(),maximum=coeff.max().item()))
        kd = F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        loss = supervised + (kd if 'kd' in method else 0) + 0.001 * sum(p.square().mean() for p in params)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        if step in (0, steps-1):
            hist.append({'step': step, 'loss': loss.item(), 'ce': ce.item(), 'kd': kd.item(), 'gradient_norm': tokens.grad.norm().item(),'supervised':supervised.item(),'mean_pt':pt.mean().item()})
    return forward().detach(), {'tokens': tokens.detach().cpu(), 'tokens2': tokens2.detach().cpu(), 'amplitude': amp}, {'focal_gamma':gamma,'focal_mode':mode,'coefficient_history':scales,'history': hist, 'prompt_noop_norm_error': noop, 'trainable_parameters': sum(p.numel() for p in params), **ridge_info}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer', 'PubMed'])
    ap.add_argument('--checkpoints', nargs='+', default=['GraphMAE', 'GraphMAE2'])
    ap.add_argument('--methods', nargs='+', default=['frozen', 'graphprompt', 'rawprop2', 'rect_teacher_ref25', 'rect_teacher_ref50', 'rect_teacher_ref25shuffle', 'token_kd3', 'token_kd3_ref25', 'token_kd3_ref50', 'token_kd3_ref25shuffle', 'token_ridge_kd10plainlate'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(30100, 30105)))
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--phase', choices=['dev', 'confirm'], default='dev')
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = sha(__file__)
    config = vars(args) | {'source_sha256': source_hash, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    config_path = out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Refusing resume with changed config/source'
    else:
        save_json(config_path, config)
    for name in args.datasets:
        data = Planetoid('/root/ProG/data/Planetoid', name, transform=NormalizeFeatures())[0]
        classes = int(data.y.max()) + 1
        pool = data.train_mask.nonzero().flatten()
        query_all = data.test_mask.nonzero().flatten()
        gen = torch.Generator().manual_seed(20260923)
        query_all = query_all[torch.randperm(len(query_all), generator=gen)]
        query = query_all[:500] if args.phase == 'dev' else query_all[500:]
        assert not set(pool.tolist()) & set(query_all.tolist())
        graph_hash = hashlib.sha256((thash(data.x) + thash(data.edge_index)).encode()).hexdigest()
        x, edges = data.x.cuda(), data.edge_index.cuda()
        ei, ew = gcn_norm(edges, num_nodes=len(x), add_self_loops=True)
        adj = torch.sparse_coo_tensor(ei.flip(0), ew, (len(x), len(x))).coalesce()
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
        filters = make_filters(x, edges)
        for ck in args.checkpoints:
            ckpath = Path(f'/root/ProG/Experiment/pre_trained_model/{name}/{ck}.GCN.128hidden_dim.pth')
            if ck.startswith('Random_s'):
                ckpath = out / 'random_backbones' / f'{name}__{ck}.pth'
                ckpath.parent.mkdir(exist_ok=True)
                if not ckpath.exists():
                    init_seed = int(ck.split('_s')[1])
                    torch.manual_seed(init_seed)
                    random_model = GCN(x.shape[1], 128, 128, num_layer=2)
                    torch.save(random_model.state_dict(), ckpath)
                    save_json(Path(str(ckpath) + '.json'), {'variant': 'frozen_random', 'pretrained': False, 'init_seed': init_seed, 'checkpoint_sha256': sha(ckpath)})
            state = torch.load(ckpath, map_location='cpu', weights_only=True)
            model = GCN(x.shape[1], 128, 128, num_layer=2).cuda().eval()
            model.load_state_dict(state, strict=True)
            model.requires_grad_(False)
            before = {k: thash(v) for k, v in model.state_dict().items()}
            with torch.no_grad():
                u = model.conv_layers[0](x, edges).detach()
                h = act(u).detach()
                z0 = model.conv_layers[1](h, edges).detach()
                reference = model(x, edges)
                err = (z0 - reference).abs().max().item()
                # GPU scatter reductions are not bitwise deterministic in float32.
                assert torch.allclose(z0, reference, atol=5e-5, rtol=1e-5), err
                assert (F.normalize(z0, dim=1) - F.normalize(reference, dim=1)).abs().max() < 1e-5
            sidecar = json.loads(Path(str(ckpath) + '.json').read_text()) if Path(str(ckpath) + '.json').exists() else {'source_parity': 'legacy_unverified'}
            cksha = sha(ckpath)
            if 'checkpoint_sha256' in sidecar:
                assert cksha == sidecar['checkpoint_sha256']
            for seed in args.seeds:
                gen = torch.Generator().manual_seed(seed)
                support = torch.cat([p[torch.randperm(len(p), generator=gen)[:5]] for c in range(classes) for p in [pool[data.y[pool] == c]]])
                sy = data.y[support].cuda()
                si = support.cuda()
                assert torch.all(torch.bincount(sy, minlength=classes) == 5)
                split_hash = hashlib.sha256((thash(support) + thash(query)).encode()).hexdigest()
                embedding_cache={}
                neighbor_cache={}
                for method in args.methods:
                    folder = out / f'{name}__{ck}__{seed}__{method}'
                    folder.mkdir(exist_ok=True)
                    if (folder / 'result.json').exists():
                        continue
                    started = time.time()
                    score_override = None
                    if method.startswith('head_'):
                        _,source,kind=method.split('_')
                        if source not in embedding_cache:
                            if source=='base':embedding_cache[source]=(z0,{}, {})
                            elif source.startswith(('focal','scale','perm')):
                                assert name=='Cora'
                                mode=source[:-1];gamma=int(source[-1])
                                embedding_cache[source]=focal_prompt(u,r,adj,model,edges,si,sy,classes,seed,'token_ridge_kd10plainlate',args.steps,filters,gamma,mode)
                            else:
                                root_method='token_ridge_kd10plainlate' if name=='Cora' else 'token_kd3_ref50'
                                embedding_cache[source]=(vat_prompt(u,r,adj,model,edges,si,sy,classes,seed,root_method,args.steps,filters,'vat',.3) if name=='PubMed' and ck=='GraphMAE' else token_prompt(u,r,adj,model,edges,si,sy,classes,seed,root_method,args.steps,filters))
                        z,prompt,oldtrace=embedding_cache[source]
                        score_override,headtrace=diffusion_readout(z,si,sy,classes,kind,seed,filters['plain'],neighbor_cache.setdefault(source,{}),out/'neighbors'/f'{name}__{ck}__{seed}__{source}')
                        trace=dict(oldtrace,**headtrace,embedding_sha256=thash(z),training_readout='cosine_unchanged')
                    elif method=='vat_reference':
                        root_method='token_ridge_kd10plainlate' if name=='Cora' else 'token_kd3_ref50'
                        z,prompt,trace=vat_prompt(u,r,adj,model,edges,si,sy,classes,seed,root_method,args.steps,filters,'vat',.3)
                    elif method.startswith('rect_teacher'):
                        score_override, trace = rectified_teacher(r, si, sy, classes, method, seed)
                        z, prompt = r, {}
                    elif method.startswith('ridge_teacher'):
                        score_override, trace = ridge_teacher(r, si, sy, classes)
                        if method.endswith("plain"):
                            score_override = smooth_probs(score_override.softmax(-1), filters["plain"]).clamp_min(1e-12).log()
                            trace["teacher_filter"] = "plain"
                        z, prompt = r, {}
                    elif method.startswith('token'):
                        z, prompt, trace = token_prompt(u, r, adj, model, edges, si, sy, classes, seed, method, args.steps, filters)
                    elif method in ('raw', 'rawprop2'):
                        z, prompt, trace = (x if method == 'raw' else r), {}, {}
                    elif method.startswith('input') or method == 'direct05':
                        with torch.no_grad():
                            xp, prompt = memory_prompt(x, r, si, sy, classes, seed, method)
                            z = xp if method == 'direct05' else model(xp, edges)
                        trace = {'input_relative_change': ((xp-x).norm()/x.norm()).item()}
                    else:
                        z, prompt, trace = train(h, z0, model, edges, si, sy, classes, method, seed, args.steps)
                    with torch.no_grad():
                        scores = (score_override if score_override is not None else logits(z, si, sy, classes))[query.cuda()].cpu()
                    pred = scores.argmax(-1)
                    # Persist predictions, indices and prompt before reading query labels.
                    torch.save({'query': query, 'support': support, 'scores': scores, 'prediction': pred, 'prompt': prompt}, folder / 'sealed.pt')
                    predsha = sha(folder / 'sealed.pt')
                    assert before == {k: thash(v) for k, v in model.state_dict().items()}
                    accuracy = (pred == data.y[query]).float().mean().item()
                    result = dict(dataset=name, checkpoint=ck, seed=seed, method=method, phase=args.phase, accuracy=accuracy,
                                  seconds=time.time()-started, checkpoint_sha256=cksha, source_sha256=source_hash,
                                  graph_sha256=graph_hash, split_sha256=split_hash, prediction_sha256=predsha,
                                  query_size=len(query), support_size=len(support), frozen_verified=True, noop_error=err,
                                  provenance=sidecar, **trace)
                    save_json(folder / 'result.json', result)
                    print(json.dumps({k: result[k] for k in ['dataset', 'checkpoint', 'seed', 'method', 'accuracy', 'seconds']}), flush=True)
    save_json(out / 'complete.json', {'completed_at': time.time(), 'results': len(list(out.glob('*/result.json')))})


if __name__ == '__main__':
    main()
