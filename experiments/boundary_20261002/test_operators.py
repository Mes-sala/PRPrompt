"""Meaningful parity, routing-scale and support-only head tests before any formal fit."""
import argparse
import json
import torch
from prprompt.model import TwoLayerGCN
from prprompt.train import fit_predict, state_hash
from experiments.boundary_20261002.model import fit, routing, methods, saved_readout

def test(device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(991)
    x = torch.rand(80, 19, device=device)
    x = x / x.sum(1, keepdim=True)
    ids = torch.arange(80, device=device)
    edges = torch.stack([torch.cat([ids, ids]), torch.cat([(ids+1)%80, (ids+7)%80])])
    model = TwoLayerGCN(19).to(device).eval().requires_grad_(False)
    s = torch.arange(15, device=device)
    y = torch.arange(3, device=device).repeat_interleave(5)
    config = {'steps': 5, 'lr': .01, 'weight_decay': .0001, 'temperature': .1, 'amplitude': 10., 'gpf_tokens': 20}
    before = state_hash(model)
    parity = []; parameter_errors = []
    for seed in [0, 1, 2]:
        original, old_state, _ = fit_predict('PRPrompt', model, x, edges, s, y, 3, config, seed)
        ours, z, state, audit = fit('Soft__Linear', model, x, edges, s, y, 3, config, seed)
        error = float((original-ours).abs().max())
        assert error < (1e-7 if device == 'cpu' else 1e-5), error
        weight_error = float((old_state['head']['weight']-state['head']['weight']).abs().max())
        token_error = float((old_state['prompt']['tokens']-state['prompt']['tokens']).abs().max())
        assert max(weight_error, token_error) < (1e-12 if device == 'cpu' else 1e-5)
        parameter_errors.append([weight_error, token_error])
        state['method'] = 'Soft__Linear'
        assert torch.equal(ours, saved_readout(state, z, s, y, 3, device))
        parity.append(error)
    with torch.no_grad():
        hidden = model.conv_layers[0](x, edges)
        soft, _ = routing('Soft', x, edges, hidden, s, y, 3)
        matched, _ = routing('HardMatched', x, edges, hidden, s, y, 3)
        assert torch.allclose(soft.norm(), matched.norm(), atol=1e-6)
        shared, _ = routing('Shared', x, edges, hidden, s, y, 3)
        assert torch.equal(shared[0], shared[-1]) and float(shared.norm()) > 0
    checked = 0
    for shots in [1, 3, 5]:
        si = torch.cat([torch.arange(c*5, c*5+shots, device=device) for c in range(3)])
        sy = torch.arange(3, device=device).repeat_interleave(shots)
        for method in methods(shots):
            scores, z, state, audit = fit(method, model, x, edges, si, sy, 3, dict(config, steps=2), 0)
            assert scores.shape == (80, 3) and torch.isfinite(scores).all()
            state['method'] = method
            rebuilt = saved_readout(state, z, si, sy, 3, device)
            assert torch.equal(scores, rebuilt), (method, float((scores-rebuilt).abs().max()))
            if audit['backbone_frozen']:
                assert audit['backbone_before'] == audit['backbone_after']
            checked += 1
    assert state_hash(model) == before and all(p.grad is None for p in model.parameters())
    print(json.dumps({'device': device, 'soft_linear_v2_score_errors': parity, 'parameter_max_errors': parameter_errors,
                      'method_shot_cells_tested': checked, 'saved_embedding_readout_exact': True,
                      'hard_scale_match': True, 'one_shot_F100_loss_adaptation_tested': True}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu')
    test(parser.parse_args().device)
