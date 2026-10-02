"""Audit cross-platform float64 reductions without changing sealed results."""
import math
import sys


def compare_records(actual, expected):
    # All compared metrics/effects are on the unit scale. This is a numerical
    # allowance of 64 binary64 epsilons, not a statistical acceptance margin.
    bound = 64 * sys.float_info.epsilon
    differences = []
    assert len(actual) == len(expected), 'record count mismatch'
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert a.keys() == b.keys(), ('schema mismatch', i)
        for key in a:
            x, y = a[key], b[key]
            if isinstance(x, float) and isinstance(y, float):
                assert math.isfinite(x) and math.isfinite(y), (i, key)
                error = abs(x - y)
                assert error <= bound, ('numeric mismatch', i, key, x, y, bound)
                if x != y:
                    differences.append(dict(record=i, field=key, local=x, server=y, absolute_error=error))
            else:
                assert type(x) is type(y) and x == y, ('metadata mismatch', i, key)
    return dict(exact_equal=not differences, absolute_bound=bound,
                maximum_absolute_error=max((d['absolute_error'] for d in differences), default=0.),
                differences=differences)
