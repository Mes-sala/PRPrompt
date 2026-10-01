"""Exact-tie fault cases, rejection, RNG preservation and legacy fit parity."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from experiments.boundary_20261002.order_guard import check_order, save_order_diagnostic, verify_order_diagnostic
from experiments.boundary_20261002.model import fit, readout
from experiments.boundary_20261002.filters import construct_filters_once
from experiments.v1_f100.historical import load
from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.protocol import sha_file
from tools.audit_boundary_delivery import check_order_guard


def test(device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261002)
    base = torch.tensor([[.5, .5], [.7, .3]], device=device)
    temperature = torch.tensor([[.49, .51], [.8, .2]], device=device)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
    audit = check_order(base, temperature)
    assert not audit['original_argmax_assertion_passed'] and audit['accepted_exact_tie_rows'] == 1
    assert audit['genuine_order_reversal_rows'] == 0
    rejected = 0
    for bad in [torch.tensor([[.8, .2], [.2, .8]], device=device),
                torch.tensor([[float('nan'), .5], [.8, .2]], device=device)]:
        try:
            check_order(base, bad)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Invalid order silently accepted')
    assert rejected == 2 and torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None: assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        state = {'prompt':{'prefit_base_teacher':base.cpu(), 'gate_probability':temperature.cpu()}}
        stored = {'prefit_order_guard':audit}
        save_order_diagnostic(folder, state, stored)
        assert verify_order_diagnostic(folder, state, stored) == audit
        policy = {'policy':audit['policy']}
        assert check_order_guard(folder, {'audit':stored}, policy) == audit
        changed = base.cpu().numpy().copy(); changed[0, 0] = .9
        np.savez_compressed(folder/'prefit_order.npz', base_teacher=changed, gate_probability=temperature.cpu().numpy())
        stored['prefit_order_sha256'] = sha_file(folder/'prefit_order.npz')
        for checker in [lambda:verify_order_diagnostic(folder, state, stored),
                        lambda:check_order_guard(folder, {'audit':stored}, policy)]:
            try: checker()
            except AssertionError: pass
            else: raise AssertionError('Changed saved teacher silently accepted')
    x = torch.rand(80, 19, device=device)
    ids = torch.arange(80, device=device)
    edges = torch.stack([torch.cat([ids, ids]), torch.cat([(ids+1)%80, (ids+7)%80])])
    model = TwoLayerGCN(19).to(device).eval().requires_grad_(False)
    adj = normalized_adjacency(edges, len(x), x.dtype)
    with torch.no_grad():
        u = model.conv_layers[0](x, edges).detach()
        r = torch.sparse.mm(adj, torch.sparse.mm(adj, x)).detach()
    comparisons = []
    for shots in [1, 3, 5]:
        support = torch.arange(3*shots, device=device)
        labels = torch.arange(3, device=device).repeat_interleave(shots)
        old = load()
        if shots == 1:
            original = old['logits']
            old['logits'] = lambda z, s, y, c, loo=False: original(z, s, y, c)[s] if loo else original(z, s, y, c)
        filters, _ = construct_filters_once(x, edges, old['make_filters'])
        z, p, _ = old['optimizer_prompt'](u, r, adj, model, edges, support, labels, 3, 0,
            'token_ridge_kd10plain', 100, filters, 'none', .1, 'raw', 'adam', .01)
        original_scores = old['logits'](z-z.mean(0, keepdim=True), support, labels, 3)
        after_old_rng = torch.get_rng_state().clone()
        after_old_cuda = torch.cuda.get_rng_state(device).clone() if cuda_rng is not None else None
        method = 'F100OneShotAdapt' if shots == 1 else 'F100Historical'
        scores, new_z, state, new_audit = fit(method, model, x, edges, support, labels, 3, {'steps':100}, 0)
        assert new_audit['prefit_order_guard']['original_argmax_assertion_passed']
        errors = {'scores':float((scores-original_scores).abs().max()),
                  'embedding':float((new_z-z).abs().max()), 'tokens':float((state['prompt']['tokens']-p['tokens']).abs().max())}
        if device == 'cpu': assert max(errors.values()) == 0
        else:
            assert torch.all((new_z-z).abs() < 2e-6+1e-6*z.abs())
            assert errors['scores'] < 1e-5 and errors['tokens'] < 1e-5
        assert torch.equal(after_old_rng, torch.get_rng_state())
        if after_old_cuda is not None: assert torch.equal(after_old_cuda, torch.cuda.get_rng_state(device))
        comparisons.append(dict(shots=shots, errors=errors, rng_exact=True))
    print(json.dumps(dict(device=device, exact_tie_accepted=True, original_failure_flag_retained=True,
        untied_reversal_and_nan_rejected=True, guard_rng_unchanged=True,
        saved_arrays_torch_and_independent_numpy_verified=True, changed_saved_teacher_rejected=True,
        historical_100_step_parity=comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu')
    test(parser.parse_args().device)
