"""Support-only factorial operators; leave the historical and v2 trainers unchanged."""
import copy
import time
import torch
from torch import nn
from torch.nn import functional as F
from prprompt.model import normalized_adjacency, attribute_features, prototypes, prototype_route, PrototypeRoutedPrompt
from prprompt.train import seed_all, state_hash, tensor_hash, fit_predict
from prprompt.protocol import support_folds
from experiments.v1_f100.historical import load
from experiments.boundary_20261002.filters import construct_filters_once

ROUTES = ['Soft', 'Hard', 'HardMatched', 'Shared', 'None']
HEADS = ['Linear', 'Prototype', 'CenteredPrototype', 'Ridge']
BASELINES = ['FullFineTune', 'GPF', 'GPFPlus', 'GraphPromptDiag']

def methods(shots):
    return [r + '__' + h for r in ROUTES for h in HEADS] + BASELINES + [
        'F100OneShotAdapt' if shots == 1 else 'F100Historical']

def fixed(method):
    return method in ['None__Prototype', 'None__CenteredPrototype', 'None__Ridge']

def readout(z, head_kind, support, sy, classes, head=None):
    if head_kind == 'CenteredPrototype':
        z = z - z.mean(0, keepdim=True)
    zn = F.normalize(z, dim=1)
    if head_kind == 'Linear':
        return head(zn)
    if head_kind in ['Prototype', 'CenteredPrototype']:
        oh = F.one_hot(sy, classes).to(zn.dtype)
        centers = F.normalize((oh.T @ zn[support]) / oh.sum(0)[:, None], dim=1)
        return zn @ centers.T / .2
    if head_kind == 'Ridge':
        zs = zn[support].double()
        eye = torch.eye(len(support), device=z.device, dtype=torch.float64)
        targets = F.one_hot(sy, classes).double()
        dual = torch.linalg.solve(zs @ zs.T + eye, targets)
        return (zn.double() @ zs.T @ dual).to(z.dtype)
    raise ValueError(head_kind)

@torch.no_grad()
def routing(kind, x, edges, hidden, support, sy, classes):
    adj = normalized_adjacency(edges, len(x), x.dtype)
    attrs = attribute_features(x, adj)
    soft, p = prototype_route(attrs, adj, support, sy, classes, .1)
    if kind == 'Soft':
        route = soft
    elif kind in ['Hard', 'HardMatched']:
        hard = F.one_hot(p.argmax(1), classes).to(x.dtype)
        route = torch.sparse.mm(adj, hard - hard.mean(0, keepdim=True))
        if kind == 'HardMatched':
            route = route * (soft.norm() / route.norm().clamp_min(1e-12))
    elif kind == 'Shared':
        route = torch.full((len(x), classes), 1/classes, device=x.device, dtype=x.dtype)
    else:
        raise ValueError(kind)
    return route, {'probability_entropy': float(-(p*p.clamp_min(1e-12).log()).sum(1).mean()),
                   'assignment_sha256': tensor_hash(p), 'route_sha256': tensor_hash(route),
                   'route_frobenius_norm': float(route.norm()), 'soft_route_frobenius_norm': float(soft.norm()),
                   'hard_soft_agreement': 1., 'route_rank': int(torch.linalg.matrix_rank(route.double()).item())}

def replay_baseline_embedding(method, state, backbone, x, edges):
    model = copy.deepcopy(backbone).to(x.device).eval()
    model.load_state_dict(state['model'])
    if method in ['GPF', 'GPFPlus']:
        p = state['prompt']; basis = p['basis'].to(x.device)
        addition = basis if method == 'GPF' else F.linear(x, p['router.weight'].to(x.device), p['router.bias'].to(x.device)).softmax(-1) @ basis
        return model(x + addition, edges)
    z = model(x, edges)
    if method == 'GraphPromptDiag':
        z = z * state['prompt']['weight'].to(x.device)
    return z

def fit(method, backbone, x, edges, support, sy, classes, config, seed=0):
    """No query labels or global labels argument exists. Returns canonical scores/embedding."""
    start = time.perf_counter()
    seed_all(seed)
    if method in BASELINES:
        initial_scores, state, audit = fit_predict(method, backbone, x, edges, support, sy, classes, config, seed)
        with torch.no_grad():
            z = replay_baseline_embedding(method, state, backbone, x, edges)
            if method == 'GraphPromptDiag':
                scores = readout(z, 'Prototype', support, sy, classes)
            else:
                scores = F.linear(F.normalize(z, dim=1), state['head']['weight'].to(x.device), state['head']['bias'].to(x.device))
        audit['final_forward_score_drift'] = float((scores-initial_scores).abs().max())
        return scores, z, state, audit

    model = copy.deepcopy(backbone).to(x.device).eval().requires_grad_(False)
    before = state_hash(model)
    if method in ['F100Historical', 'F100OneShotAdapt']:
        counts = torch.bincount(sy, minlength=classes)
        assert bool((counts == (1 if method == 'F100OneShotAdapt' else counts[0])).all())
        assert method != 'F100Historical' or int(counts.min()) >= 2
        old = load()
        if method == 'F100OneShotAdapt':
            original = old['logits']
            old['logits'] = lambda z, s, y, c, loo=False: original(z, s, y, c, loo=False)[s] if loo else original(z, s, y, c, loo=False)
        adj = normalized_adjacency(edges, len(x), x.dtype)
        with torch.no_grad():
            u = model.conv_layers[0](x, edges).detach()
            r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
            filters, filter_audit = construct_filters_once(x, edges, old['make_filters'])
        z, prompt, trace = old['optimizer_prompt'](u, r, adj, model, edges, support, sy, classes, seed,
            'token_ridge_kd10plain', config['steps'], filters, 'none', .1, 'raw', 'adam', .01)
        with torch.no_grad():
            scores = old['logits'](z-z.mean(0, keepdim=True), support, sy, classes)
        gradients = prompt['optimizer_gradients']
        assert torch.isfinite(gradients).all()
        state = {'prompt': prompt, 'trace': trace, 'historical_one_shot_loss_adapted': method == 'F100OneShotAdapt'}
        audit = {'method': method, 'trainable_parameters': classes*128, 'prompt_parameters': classes*128,
                 'head_parameters': 0, 'backbone_parameters': 0, 'query_labels_received': False,
                 'backbone_before': before, 'backbone_after': state_hash(model), 'backbone_frozen': True,
                 'prompt_gradient_max': float(gradients.norm(dim=(1, 2)).max()),
                 'loss': 'inclusive_prototype_ce' if method == 'F100OneShotAdapt' else 'leave_one_out_prototype_ce',
                 'adam_actual_epsilon': 1e-8, 'fit_seconds': time.perf_counter()-start, **filter_audit}
        return scores, z, state, audit

    kind, head_kind = method.split('__')
    assert kind in ROUTES and head_kind in HEADS
    with torch.no_grad():
        hidden = model.conv_layers[0](x, edges).detach()
        z0 = model(x, edges).detach()
    info = {}
    prompt = None
    if kind != 'None':
        route, info = routing(kind, x, edges, hidden, support, sy, classes)
        prompt = PrototypeRoutedPrompt(route, hidden, 10.).to(x.device)
    seed_all(seed + 1_000_003)
    head = nn.Linear(z0.shape[1], classes).to(x.device) if head_kind == 'Linear' else None
    params = (list(prompt.parameters()) if prompt else []) + (list(head.parameters()) if head else [])
    def embedding():
        return model.conv_layers[1](F.leaky_relu(prompt(hidden)), edges) if prompt else z0
    if prompt:
        with torch.no_grad():
            noop = embedding()
            assert torch.allclose(noop, z0, atol=5e-5, rtol=1e-5)
            info['zero_prompt_max_error'] = float((noop-z0).abs().max())
    history = []
    maximum_gradient = 0.
    if params:
        optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
        for step in range(config['steps']):
            optimizer.zero_grad(set_to_none=True)
            z = embedding()
            if head_kind == 'Ridge':
                # Same support rows of the full dual predictor, without computing unused query rows during training.
                zs = F.normalize(z, dim=1)[support].double()
                kernel = zs @ zs.T
                dual = torch.linalg.solve(kernel+torch.eye(len(support), device=z.device, dtype=torch.float64), F.one_hot(sy, classes).double())
                support_scores = (kernel @ dual).to(z.dtype)
            else:
                support_scores = readout(z, head_kind, support, sy, classes, head)[support]
            loss = F.cross_entropy(support_scores, sy)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None for p in model.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            grad = float(prompt.tokens.grad.norm()) if prompt else 0.
            maximum_gradient = max(maximum_gradient, grad)
            optimizer.step()
            history.append({'step': step, 'support_loss': float(loss.detach()), 'prompt_gradient_norm': grad})
    with torch.no_grad():
        z = embedding().detach()
        scores = readout(z, head_kind, support, sy, classes, head).detach()
    after = state_hash(model)
    assert before == after
    state = {'route_kind': kind, 'head_kind': head_kind, 'history': history}
    if prompt:
        state['prompt'] = {k: v.detach().cpu() for k, v in prompt.state_dict().items()}
    if head:
        state['head'] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    audit = {'method': method, 'backbone_frozen': True, 'backbone_before': before, 'backbone_after': after,
             'query_labels_received': False, 'trainable_parameters': sum(p.numel() for p in params),
             'prompt_parameters': sum(p.numel() for p in prompt.parameters()) if prompt else 0,
             'head_parameters': sum(p.numel() for p in head.parameters()) if head else 0,
             'backbone_parameters': 0, 'prompt_gradient_max': maximum_gradient,
             'prototype_training_self_inclusive': head_kind in ['Prototype', 'CenteredPrototype'],
             'fit_seconds': time.perf_counter()-start, **info}
    return scores, z, state, audit

def select(method, backbone, x, edges, support, sy, classes, config, seed):
    base = dict(config['fit'])
    if fixed(method) or method.startswith('F100'):
        return base, {'selection': 'parameter_free' if fixed(method) else 'fixed_historical_recipe',
                      'trials': [], 'query_labels_used': False}
    folds = support_folds(support, sy, classes)
    if not folds or config.get('smoke', False):
        return base, {'selection': 'smoke_fixed' if config.get('smoke') else 'one_shot_predeclared',
                      'trials': [], 'query_labels_used': False}
    trials = []
    for lr in config['selection']['learning_rates']:
        for wd in config['selection']['weight_decays']:
            candidate = dict(base, lr=lr, weight_decay=wd)
            losses = []
            for i, (s, y, v, vy) in enumerate(folds):
                scores, _, _, _ = fit(method, backbone, x, edges, s, y, classes, candidate, seed+i)
                losses.append(float(F.cross_entropy(scores[v], vy)))
            trials.append({'config': candidate, 'fold_ce': losses, 'mean_ce': sum(losses)/len(losses)})
    best = min(trials, key=lambda t: t['mean_ce'])
    return best['config'], {'selection': 'support_only_stratified_leave_one_per_class',
                            'trials': trials, 'query_labels_used': False}

def saved_readout(state, z, support, sy, classes, device):
    method = state['method']
    if method.startswith('F100'):
        return load()['logits'](z-z.mean(0, keepdim=True), support, sy, classes)
    if method == 'GraphPromptDiag':
        return readout(z, 'Prototype', support, sy, classes)
    head_kind = 'Linear' if method in BASELINES else method.split('__')[1]
    head = None
    if head_kind == 'Linear':
        head = nn.Linear(z.shape[1], classes).to(device)
        head.load_state_dict(state['head'])
    return readout(z, head_kind, support, sy, classes, head)
