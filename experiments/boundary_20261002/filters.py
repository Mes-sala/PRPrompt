"""Bounded pre-fit construction of immutable historical graph filters.

Retain the unmodified historical assertion and arithmetic. Cache a successful
construction only for the exact same immutable input tensor objects in one worker.
"""
import json
import hashlib
import marshal
import torch

_FILTER_CACHE = {}


def construct_filters_once(x, edges, constructor):
    constructor_hash = hashlib.sha256(marshal.dumps(constructor.__code__)).hexdigest()
    key = (id(x), id(edges), constructor_hash)
    versions = (x._version, edges._version)
    if key in _FILTER_CACHE:
        entry = _FILTER_CACHE[key]
        assert entry['x'] is x and entry['edges'] is edges
        assert entry['versions'] == versions, 'Cached graph inputs were mutated'
        return entry['filters'], dict(filters_acquired_from_cache=True,
            pre_fit_filter_attempts=0, pre_fit_filter_assert_failures=[],
            filter_cache_origin_attempts=entry['attempts'],
            filter_cache_origin_failures=entry['failures'])
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None
    failures = []
    for attempt in range(1, 4):
        try:
            filters = constructor(x, edges)
        except AssertionError as error:
            frames = []
            trace = error.__traceback__
            while trace is not None:
                frames.append(dict(function=trace.tb_frame.f_code.co_name, line=trace.tb_lineno))
                trace = trace.tb_next
            assert frames[-1]['function'] == 'make_filters', 'Unidentified assertion cannot be retried'
            failures.append(dict(attempt=attempt, frames=frames, original_assertion_failed=True))
            print(json.dumps(dict(event='pre_fit_filter_assert_failure', attempt=attempt, frames=frames)), flush=True)
            assert torch.equal(cpu_rng, torch.get_rng_state()), 'Filter construction consumed CPU RNG'
            if cuda_rng is not None:
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state(x.device)), 'Filter construction consumed CUDA RNG'
            if attempt == 3:
                raise
            continue
        assert torch.equal(cpu_rng, torch.get_rng_state()), 'Filter construction consumed CPU RNG'
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(x.device)), 'Filter construction consumed CUDA RNG'
        _FILTER_CACHE[key] = dict(x=x, edges=edges, versions=versions, filters=filters,
                                 attempts=attempt, failures=failures)
        return filters, dict(filters_acquired_from_cache=False,
            pre_fit_filter_attempts=attempt, pre_fit_filter_assert_failures=failures,
            filter_cache_origin_attempts=attempt, filter_cache_origin_failures=failures)
    raise AssertionError('Unreachable filter construction state')
