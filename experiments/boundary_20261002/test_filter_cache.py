"""Fault injection, immutable-input rejection and legacy F100 parity for R2."""
import argparse
import json
import torch
from prprompt.model import TwoLayerGCN
from experiments.v1_f100.historical import load
from experiments.boundary_20261002.filters import construct_filters_once, _FILTER_CACHE
from experiments.boundary_20261002 import model as boundary_model


def test(device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(831)
    x = torch.rand(80, 19, device=device)
    ids = torch.arange(80, device=device)
    edges = torch.stack([torch.cat([ids, ids]), torch.cat([(ids+1)%80, (ids+7)%80])])
    original = load()['make_filters']
    calls = 0
    def make_filters(a, e):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise AssertionError('Injected historical pre-fit failure')
        return original(a, e)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
    filters, audit = construct_filters_once(x, edges, make_filters)
    assert calls == 3 and audit['pre_fit_filter_attempts'] == 3
    assert len(audit['pre_fit_filter_assert_failures']) == 2
    cached, second = construct_filters_once(x, edges, make_filters)
    assert cached is filters and calls == 3 and second['filters_acquired_from_cache']
    assert second['pre_fit_filter_attempts'] == 0 and second['pre_fit_filter_assert_failures'] == []
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    with torch.no_grad():
        x.add_(0)
    try:
        construct_filters_once(x, edges, make_filters)
    except AssertionError as error:
        assert 'mutated' in str(error)
    else:
        raise AssertionError('Mutated graph incorrectly used cached filters')
    _FILTER_CACHE.clear()
    failures = 0
    def make_filters(a, e):
        nonlocal failures
        failures += 1
        raise AssertionError('Always fails')
    try:
        construct_filters_once(x, edges, make_filters)
    except AssertionError as error:
        assert str(error) == 'Always fails' and failures == 3
    else:
        raise AssertionError('Unbounded filter retries')
    def unknown(a, e):
        raise AssertionError('Different failure')
    try:
        construct_filters_once(x, edges, unknown)
    except AssertionError as error:
        assert 'Unidentified' in str(error)
    else:
        raise AssertionError('Unidentified failure was retried')
    _FILTER_CACHE.clear()
    backbone = TwoLayerGCN(19).to(device).eval().requires_grad_(False)
    errors, parameter_errors, cache_hits = [], [], []
    cache_function = boundary_model.construct_filters_once
    for shots in [1, 3, 5]:
        s = torch.arange(3*shots, device=device)
        y = torch.arange(3, device=device).repeat_interleave(shots)
        method = 'F100OneShotAdapt' if shots == 1 else 'F100Historical'
        boundary_model.construct_filters_once = lambda a, e, f: (f(a, e), {})
        old_scores, old_z, old_state, _ = boundary_model.fit(method, backbone, x, edges, s, y, 3, {'steps': 3}, 0)
        boundary_model.construct_filters_once = cache_function
        scores, z, state, audit = boundary_model.fit(method, backbone, x, edges, s, y, 3, {'steps': 3}, 0)
        cache_hits.append(audit['filters_acquired_from_cache'])
        assert audit['filters_acquired_from_cache'] == (shots != 1), 'Fresh historical AST load bypassed filter cache'
        error = max(float((old_scores-scores).abs().max()), float((old_z-z).abs().max()))
        parameter_error = float((old_state['prompt']['tokens']-state['prompt']['tokens']).abs().max())
        assert error < (1e-12 if device == 'cpu' else 1e-5)
        assert parameter_error < (1e-12 if device == 'cpu' else 1e-5)
        errors.append(error)
        parameter_errors.append(parameter_error)
    print(json.dumps(dict(device=device, bounded_retries_and_stop=True, cached_attempts_not_double_counted=True,
        immutable_input_guard=True, rng_unchanged=True, unidentified_assertion_rejected=True,
        legacy_f100_score_embedding_errors=errors, legacy_f100_token_errors=parameter_errors,
        cache_reused_across_fresh_AST_loads=cache_hits)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    test(parser.parse_args().device)
