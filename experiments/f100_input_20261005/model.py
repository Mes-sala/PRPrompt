"""Only the returned gate changes; legacy AST bodies and active training recipe stay fixed."""
import copy
import json
import torch
from torch.nn import functional as F
from prprompt.model import normalized_adjacency
from prprompt.train import state_hash
from experiments.prog_split_20261002.numeric_diagnostics import load
from experiments.boundary_20261002.filters import construct_filters_once
from experiments.boundary_20261002.order_guard import install_order_guard, finalize_order_record

CONTROLS = ['Hard', 'HardMatched', 'Shared']

def transform_gate(soft, info, adjacency, kind):
    classes = info['probability'].shape[1]
    if kind == 'Soft':
        return soft, False
    if kind == 'Shared':
        return torch.full_like(soft, 1/classes), False
    hard = F.one_hot(info['probability'].argmax(1), classes).to(soft.dtype)
    gate = torch.sparse.mm(adjacency, hard-hard.mean(0, keepdim=True))
    degenerate = bool(gate.norm() == 0)
    if kind == 'HardMatched': gate = gate * (soft.norm()/gate.norm().clamp_min(1e-12))
    return gate, degenerate

def fit_input(kind, backbone, x, edges, s, y, classes, seed, steps):
    model = copy.deepcopy(backbone).eval().requires_grad_(False)
    before = state_hash(model); old = load(); shots = int(torch.bincount(y, minlength=classes).min())
    order_record = install_order_guard(old)
    install_input_forward(old)
    if shots == 1:
        original = old['logits']
        old['logits'] = lambda z, si, sy, c, loo=False: original(z, si, sy, c)[si] if loo else original(z, si, sy, c)
    original_gate = old['temperature_gate']; info_record = {}
    def gate_function(scores, adjacency, temperature, mode):
        soft, info = original_gate(scores, adjacency, temperature, mode)
        gate, degenerate = transform_gate(soft, info, adjacency, kind)
        info_record.update(kind=kind, hard_route_degenerate=degenerate, soft_norm=float(soft.norm()), actual_norm=float(gate.norm()),
                           norm_match_feasible=not degenerate, probability_role='common_soft_teacher')
        # Preserve common teacher probability so the legacy teacher-agreement check remains meaningful.
        return gate, dict(info, ratio=gate.norm()/info['base'].norm().clamp_min(1e-12))
    old['temperature_gate'] = gate_function
    adj = normalized_adjacency(edges, len(x), x.dtype)
    with torch.no_grad():
        u = x.detach(); r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
        filters, filter_audit = construct_filters_once(x, edges, old['make_filters'])
    z, prompt, trace = old['optimizer_prompt'](u, r, adj, model, edges, s, y, classes, seed,
        'token_ridge_kd10plain', steps, filters, 'none', .1, 'raw', 'adam', .01)
    with torch.no_grad(): scores = old['logits'](z-z.mean(0, keepdim=True), s, y, classes)
    assert state_hash(model) == before and all(p.grad is None for p in model.parameters())
    assert torch.isfinite(prompt['optimizer_gradients']).all()
    order_audit = finalize_order_record(order_record, prompt)
    return scores, z, {'prompt': prompt, 'trace': trace, 'control': info_record}, {
        'backbone_frozen': True, 'backbone_before': before, 'backbone_after': before, 'query_labels_received': False,
        'prompt_gradient_max': float(prompt['optimizer_gradients'].abs().max()), 'adam_actual_epsilon': 1e-8,
        'injection_layer': 'input', 'prompt_parameters': classes*x.shape[1], 'one_shot_loss_adapted': shots == 1, 'prefit_order_guard': order_audit, **filter_audit, **info_record}

def install_input_forward(ns):
    """Patch only injection forward and base activation; retain historical training body."""
    import ast
    from experiments.v1_f100.historical import REFERENCE
    tree=ast.parse((REFERENCE/'run_round83_graft.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='optimizer_prompt')
    count=0
    for i,n in enumerate(fn.body):
        if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='hbase':
            fn.body[i]=ast.parse('hbase = act(model.conv_layers[0](u, edges)).detach()').body[0];count+=1
        if isinstance(n,ast.FunctionDef) and n.name=='forward':
            fn.body[i]=ast.parse('def forward():\n    prompted = u + amp * (gate @ tokens.tanh()) * scale\n    return model.conv_layers[1](act(model.conv_layers[0](prompted, edges)), edges)').body[0];count+=1
        if isinstance(n,ast.Assert) and 'gate_info' in ast.unparse(n.test):
            fn.body[i]=ast.parse("_checked_prefit_order(teacher, gate_info['probability'])").body[0];count+=1
    assert count==3
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(REFERENCE/'run_round83_graft.py'),'exec'),ns)
