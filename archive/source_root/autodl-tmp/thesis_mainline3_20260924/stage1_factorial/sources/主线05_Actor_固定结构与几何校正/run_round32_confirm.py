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
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
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


def selfloop_forward(model, x, edges, lambdas, only_self=False):
    n = len(x)
    base = edges[:, edges[0] != edges[1]]
    if only_self:
        base = base[:, :0]
    loops = torch.arange(n, device=x.device).repeat(2, 1)
    prompted_edges = torch.cat([base, loops], dim=1)
    z = x
    for i in range(2):
        weights = torch.cat([torch.ones(base.shape[1], device=x.device), lambdas[i].expand(n)])
        z = model.conv_layers[i](z, prompted_edges, edge_weight=weights)
        if i == 0:
            z = act(z)
    return z


def structural_prompt(model, x, edges, raw, si, sy, classes, method, steps):
    if method in ['self1','self10','self100','selfonly']:
        value = {'self1':1.,'self10':10.,'self100':100.,'selfonly':1.}[method]
        with torch.no_grad():
            z = selfloop_forward(model, x, edges, x.new_full((2,), value), method=='selfonly')
        return z, {'lambda': [value,value]}, {'structural_trainable_parameters':0}
    loglambda = torch.nn.Parameter(x.new_zeros(2))
    opt = torch.optim.Adam([loglambda],lr=.05)
    teacher,_ = ridge_teacher(raw,si,sy,classes)
    teacher = teacher.softmax(-1).detach()
    hist=[]
    for step in range(steps):
        opt.zero_grad()
        z = selfloop_forward(model,x,edges,loglambda.exp())
        ce=F.cross_entropy(logits(z,si,sy,classes,loo=True),sy)
        kd=F.kl_div(logits(z,si,sy,classes).log_softmax(-1),teacher,reduction='batchmean')
        loss=ce+.001*loglambda.square().mean()+(kd if method.endswith('kd') else 0)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(loglambda.grad).all()
        assert all(p.grad is None for p in model.parameters())
        if step in (0,steps-1):hist.append({'step':step,'ce':ce.item(),'kd':kd.item(),'gradient':loglambda.grad.detach().cpu().tolist(),'lambda':loglambda.exp().detach().cpu().tolist()})
        opt.step()
        with torch.no_grad():loglambda.clamp_(-4.605170186,4.605170186)
    return selfloop_forward(model,x,edges,loglambda.exp()).detach(), {'lambda':loglambda.exp().detach().cpu().tolist()}, {'structural_trainable_parameters':2,'history':hist}


def combined_prompt(model, x, edges, raw, si, sy, classes, seed, method, steps):
    torch.manual_seed(seed)
    root, variant = method.split('_')
    only = root == 'mixonly'
    lam = 1.0 if root in ['mix1','mixonly'] else 100.0
    n = len(x)
    base = edges[:, edges[0] != edges[1]]
    if only:
        base = base[:, :0]
    pe = torch.cat([base, torch.arange(n,device=x.device).repeat(2,1)],1)
    ew = torch.cat([x.new_ones(base.shape[1]),x.new_full((n,),lam)])
    with torch.no_grad():
        h = act(model.conv_layers[0](x,pe,edge_weight=ew)).detach()
        zb = model.conv_layers[1](h,pe,edge_weight=ew).detach()
        ref = selfloop_forward(model,x,edges,x.new_full((2,),lam),only)
        assert torch.allclose(zb,ref,atol=5e-5,rtol=1e-5)
    meta = {'self_weight':lam,'self_only':only,'token_amplitude':3.0,'prompt_position':'after_first_activation_before_second_conv'}
    if variant == 'base':
        return zb, {}, meta
    if variant == 'gp':
        z,params,trace = train(h,zb,model,edges,si,sy,classes,'graphprompt',seed,steps)
        return z,params,dict(meta,**trace)
    ts,info = ridge_teacher(raw,si,sy,classes)
    q=ts.softmax(-1).detach();gate=q-q.mean(0,keepdim=True)
    scale=h.std(0).clamp_min(1e-6)
    token=torch.nn.Parameter(x.new_zeros((classes,h.shape[1])))
    def forward():return model.conv_layers[1](h+3*(gate@token.tanh())*scale,pe,edge_weight=ew)
    with torch.no_grad():
        noop=(F.normalize(forward(),dim=1)-F.normalize(zb,dim=1)).abs().max().item()
        assert noop<2e-5,noop
    meta.update(info);meta['combination_noop_error']=noop
    if variant=='random':
        with torch.no_grad():token.normal_(0,.3)
    else:
        opt=torch.optim.Adam([token],lr=.01);hist=[]
        for step in range(steps):
            opt.zero_grad();z=forward()
            ce=F.cross_entropy(logits(z,si,sy,classes,loo=True),sy)
            kd=F.kl_div(logits(z,si,sy,classes).log_softmax(-1),q,reduction='batchmean')
            loss=ce+.001*token.square().mean()+(kd if variant=='kd' else 0)
            loss.backward();assert torch.isfinite(loss) and torch.isfinite(token.grad).all()
            assert all(p.grad is None for p in model.parameters())
            opt.step()
            if step in (0,steps-1):hist.append({'step':step,'ce':ce.item(),'kd':kd.item(),'gradient_norm':token.grad.norm().item()})
        meta['history']=hist
    return forward().detach(),{'tokens':token.detach().cpu()},meta


@torch.no_grad()
def geometry_stats(z, kind):
    zd=z.double();mu=zd.mean(0,keepdim=True);c=(zd-mu).T@(zd-mu)/(len(z)-1)
    eye=torch.eye(z.shape[1],device=z.device,dtype=torch.float64)
    tau=c.trace()/z.shape[1]
    floor=(tau.abs()*1e-12).clamp_min(1e-30)
    if kind=='original':mu=torch.zeros_like(mu);w=eye
    elif kind=='center':w=eye
    elif kind=='diag10':w=torch.diag((.9*c.diag()+.1*tau).clamp_min(floor).rsqrt())
    else:
        alpha={'full0':0.,'full10':.1,'full50':.5,'full100':1.}[kind]
        vals,vec=torch.linalg.eigh((1-alpha)*c+alpha*tau*eye)
        w=(vec*vals.clamp_min(floor).rsqrt()[None])@vec.T
    assert torch.isfinite(w).all()
    return mu.to(z.dtype),w.to(z.dtype),{'cov_trace':c.trace().item(),'cov_eigen_min':torch.linalg.eigvalsh(c)[0].item(),'cov_eigen_max':torch.linalg.eigvalsh(c)[-1].item()}


def geometry_prompt(model,x,edges,raw,si,sy,classes,seed,method,steps,stats_path):
    torch.manual_seed(seed)
    _,kind,variant=method.split('_')
    with torch.no_grad():
        h=act(model.conv_layers[0].lin(x)+model.conv_layers[0].bias).detach()
        zb=(model.conv_layers[1].lin(h)+model.conv_layers[1].bias).detach()
        reference=selfloop_forward(model,x,edges,x.new_ones(2),True)
        assert torch.allclose(zb,reference,atol=5e-5,rtol=1e-5)
        if stats_path.exists():
            saved=torch.load(stats_path,map_location=x.device,weights_only=True);mu,w=saved['mu'],saved['w'];meta=saved['meta']
        else:
            mu,w,meta=geometry_stats(zb,kind)
            stats_path.parent.mkdir(exist_ok=True)
            torch.save({'mu':mu.cpu(),'w':w.cpu(),'meta':meta},stats_path)
        gz=(zb-mu)@w
    trace=dict(meta,geometry=kind,geometry_file=str(stats_path.relative_to(stats_path.parent.parent)),geometry_sha256=sha(stats_path),statistics_labels_used=False)
    if variant=='base':return gz,{},trace
    if variant=='gp':
        z,params,t=train(h,gz,model,edges,si,sy,classes,'graphprompt',seed,steps)
        return z,params,dict(trace,**t)
    teacher,info=ridge_teacher(raw,si,sy,classes);q=teacher.softmax(-1).detach();gate=q-q.mean(0,keepdim=True)
    token=torch.nn.Parameter(x.new_zeros((classes,h.shape[1])));scale=h.std(0).clamp_min(1e-6)
    def forward():
        z=model.conv_layers[1].lin(h+3*(gate@token.tanh())*scale)+model.conv_layers[1].bias
        return (z-mu)@w
    with torch.no_grad():
        noop=(F.normalize(forward(),dim=1)-F.normalize(gz,dim=1)).abs().max().item();assert noop<2e-5,noop
    if variant=='random':
        with torch.no_grad():token.normal_(0,.3)
        return forward().detach(),{'tokens':token.detach().cpu()},dict(trace,**info,geometry_noop_error=noop,random_token=True)
    opt=torch.optim.Adam([token],lr=.01);hist=[]
    for step in range(steps):
        opt.zero_grad();z=forward();ce=F.cross_entropy(logits(z,si,sy,classes,loo=True),sy)
        loss=ce+.001*token.square().mean();loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(token.grad).all()
        assert all(p.grad is None for p in model.parameters());opt.step()
        if step in (0,steps-1):hist.append({'step':step,'ce':ce.item(),'gradient_norm':token.grad.norm().item()})
    return forward().detach(),{'tokens':token.detach().cpu()},dict(trace,**info,history=hist,geometry_noop_error=noop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--datasets', nargs='+', default=['Cornell', 'Actor', 'Chameleon'])
    ap.add_argument('--checkpoints', nargs='+', default=['GraphCL', 'GraphMAE'])
    ap.add_argument('--methods', nargs='+', default=['frozen', 'graphprompt', 'raw', 'rawridge', 'geom_original_base', 'geom_original_gp', 'geom_original_ce', 'geom_center_base', 'geom_center_gp', 'geom_center_ce', 'geom_diag10_base', 'geom_diag10_gp', 'geom_diag10_ce', 'geom_full10_base', 'geom_full10_gp', 'geom_full10_ce', 'geom_full50_base', 'geom_full50_gp', 'geom_full50_ce'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(49100,49105)))
    ap.add_argument('--fold', type=int, default=0)
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
        asset = Path('/root/autodl-tmp/accuracy_prompt_20260923/assets5090')
        if name == 'Cornell':
            data = WebKB(str(asset / 'data/Cornell'), 'Cornell')[0]
        elif name == 'Actor':
            data = Actor(str(asset / 'data/Actor'))[0]
        elif name == 'Chameleon':
            data = WikipediaNetwork(str(asset / 'data/WikipediaNetwork'), 'chameleon', geom_gcn_preprocess=True)[0]
        else:
            raise ValueError(name)
        classes = int(data.y.max()) + 1
        pool = data.train_mask[:, args.fold].nonzero().flatten()
        query = (data.val_mask if args.phase == 'dev' else data.test_mask)[:, args.fold].nonzero().flatten()
        assert not set(pool.tolist()) & set(query.tolist())
        assert not set(data.val_mask[:, args.fold].nonzero().flatten().tolist()) & set(data.test_mask[:, args.fold].nonzero().flatten().tolist())
        graph_hash = hashlib.sha256((thash(data.x) + thash(data.edge_index)).encode()).hexdigest()
        x, edges = data.x.cuda(), data.edge_index.cuda()
        ei, ew = gcn_norm(edges, num_nodes=len(x), add_self_loops=True)
        adj = torch.sparse_coo_tensor(ei.flip(0), ew, (len(x), len(x))).coalesce()
        xn = NormalizeFeatures()(data.clone()).x.cuda()
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, xn)).detach()
        filters = make_filters(x, edges)
        for ck in args.checkpoints:
            ckpath = asset / f'Experiment/pre_trained_model/{name}/{ck}.GCN.128hidden_dim.pth'
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
                for method in args.methods:
                    folder = out / f'{name}__{ck}__{seed}__{method}'
                    folder.mkdir(exist_ok=True)
                    if (folder / 'result.json').exists():
                        continue
                    started = time.time()
                    score_override = None
                    if method.startswith('geom_'):
                        stats_path=out/'geometry_stats'/f'{name}__{ck}__{method.split(chr(95))[1]}.pt'
                        z,prompt,trace=geometry_prompt(model,x,edges,xn,si,sy,classes,seed,method,args.steps,stats_path)
                    elif method.startswith('mix'):
                        z, prompt, trace = combined_prompt(model,x,edges,xn,si,sy,classes,seed,method,args.steps)
                    elif method.startswith('self'):
                        z, prompt, trace = structural_prompt(model, x, edges, xn, si, sy, classes, method, args.steps)
                    elif method == 'rawridge':
                        score_override, trace = ridge_teacher(xn, si, sy, classes)
                        z, prompt = xn, {}
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
                        z, prompt, trace = (xn if method == 'raw' else r), {}, {}
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
                                  query_size=len(query), support_size=len(support), fold=args.fold, backbone_input="raw", teacher_input="NormalizeFeatures", directed_edges=not data.is_undirected(), frozen_verified=True, noop_error=err,
                                  provenance=sidecar, **trace)
                    save_json(folder / 'result.json', result)
                    print(json.dumps({k: result[k] for k in ['dataset', 'checkpoint', 'seed', 'method', 'accuracy', 'seconds']}), flush=True)
    save_json(out / 'complete.json', {'completed_at': time.time(), 'results': len(list(out.glob('*/result.json')))})


if __name__ == '__main__':
    main()
