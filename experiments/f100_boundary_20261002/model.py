"""Only the returned gate changes; legacy AST bodies and active training recipe stay fixed."""
import copy
import json
import torch
from torch.nn import functional as F
from prprompt.model import normalized_adjacency
from prprompt.train import state_hash
from experiments.v1_f100.historical import load

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

def fit_control(kind, backbone, x, edges, s, y, classes, seed, steps):
    model = copy.deepcopy(backbone).eval().requires_grad_(False)
    before = state_hash(model); old = load(); shots = int(torch.bincount(y, minlength=classes).min())
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
        u = model.conv_layers[0](x, edges).detach(); r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
        filter_failures = []
        for attempt in range(3):
            try:
                filters = old['make_filters'](x, edges)
                break
            except AssertionError as error:
                # Same construction and assertion, before optimizer or score access; no RNG is consumed here.
                trace = error.__traceback__
                frames = []
                while trace is not None:
                    frames.append(dict(function=trace.tb_frame.f_code.co_name, line=trace.tb_lineno))
                    trace = trace.tb_next
                assert frames[-1]['function'] == 'make_filters'
                filter_failures.append(dict(attempt=attempt, frames=frames, original_assertion_failed=True))
                print(json.dumps(dict(event='pre_fit_filter_assert_failure', attempt=attempt, frames=frames)), flush=True)
                if attempt == 2: raise
    z, prompt, trace = old['optimizer_prompt'](u, r, adj, model, edges, s, y, classes, seed,
        'token_ridge_kd10plain', steps, filters, 'none', .1, 'raw', 'adam', .01)
    with torch.no_grad(): scores = old['logits'](z-z.mean(0, keepdim=True), s, y, classes)
    assert state_hash(model) == before and all(p.grad is None for p in model.parameters())
    assert torch.isfinite(prompt['optimizer_gradients']).all()
    return scores, z, {'prompt': prompt, 'trace': trace, 'control': info_record}, {
        'backbone_frozen': True, 'backbone_before': before, 'backbone_after': before, 'query_labels_received': False,
        'prompt_gradient_max': float(prompt['optimizer_gradients'].abs().max()), 'adam_actual_epsilon': 1e-8,
        'one_shot_loss_adapted': shots == 1, 'pre_fit_filter_assert_failures': filter_failures,
        'pre_fit_filter_attempts': len(filter_failures)+1, **info_record}

def test(device):
    from prprompt.model import TwoLayerGCN
    from experiments.boundary_20261002.model import fit
    import json
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(831); x = torch.rand(80, 19, device=device); ids = torch.arange(80, device=device)
    edges = torch.stack([torch.cat([ids, ids]), torch.cat([(ids+1)%80, (ids+7)%80])])
    model = TwoLayerGCN(19).to(device).eval().requires_grad_(False)
    errors = []
    for shots in [1, 3, 5]:
        s = torch.arange(3*shots, device=device); y = torch.arange(3, device=device).repeat_interleave(shots)
        parent = 'F100OneShotAdapt' if shots == 1 else 'F100Historical'
        a, za, _, _ = fit(parent, model, x, edges, s, y, 3, {'steps': 3}, 0)
        b, zb, _, _ = fit_control('Soft', model, x, edges, s, y, 3, 0, 3)
        e = float((a-b).abs().max()); assert e < (1e-12 if device == 'cpu' else 1e-5); errors.append(e)
        for kind in CONTROLS:
            scores, z, state, audit = fit_control(kind, model, x, edges, s, y, 3, 0, 3)
            assert torch.isfinite(scores).all()
            if kind == 'HardMatched' and not audit['hard_route_degenerate']:
                assert abs(audit['soft_norm']-audit['actual_norm']) < 1e-5
    # A hard argmax collapse is a valid zero gate, not a fictitious norm-matched control.
    eye = torch.eye(8, device=device).to_sparse(); scores = torch.stack([torch.ones(8,device=device)*2, torch.arange(8,device=device)/8],1)
    soft, info = load()['temperature_gate'](scores, eye, .1, 'raw')
    gate, degenerate = transform_gate(soft, info, eye, 'HardMatched')
    assert degenerate and torch.count_nonzero(gate) == 0 and soft.norm() > 0
    print(json.dumps({'device':device, 'legacy_soft_score_errors':errors, 'three_controls_all_shots_finite':True, 'degenerate_hard_case_tested':True}))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); test(p.parse_args().device)
