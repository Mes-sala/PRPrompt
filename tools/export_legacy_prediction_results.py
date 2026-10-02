"""Export historical sealed predictions without their embedded Prompt parameters.

Run with ProG Python on the server. Original PT files are only read; caches and
model parameters remain remotely. Every exported array is reloaded and compared.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

RESULT_KEYS = {'query', 'support', 'scores', 'prediction', 'probabilities', 'logits'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def export(plan, output):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(plan['root'])
    records = []
    for item in plan['files']:
        source = Path(item['remote_path'])
        assert source.is_file() and source.stat().st_size == item['size'], str(source)
        assert sha(source) == item['expected_sha256'], str(source)
        relative = Path(item['local_path']).relative_to(root)
        record = dict(local_source=item['local_path'], remote_source=str(source),
                      source_sha256=item['expected_sha256'], bytes=item['size'], exported=None)
        if source.name == 'sealed.pt':
            data = torch.load(source, map_location='cpu', weights_only=True)
            assert isinstance(data, dict) and {'query', 'support', 'scores', 'prediction'} <= set(data)
            arrays = {k: data[k].detach().cpu().numpy() for k in RESULT_KEYS & set(data)}
            target = output / relative.with_suffix('.predictions.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            with np.load(target, allow_pickle=False) as recovered:
                assert set(recovered.files) == set(arrays)
                for key, value in arrays.items():
                    assert recovered[key].dtype == value.dtype
                    assert np.array_equal(recovered[key], value, equal_nan=True)
            record.update(exported=str(target.relative_to(output)), exported_sha256=sha(target),
                arrays={k: dict(dtype=str(v.dtype), shape=list(v.shape)) for k, v in arrays.items()},
                omitted_keys=sorted(set(data) - RESULT_KEYS), exact_array_replay_passed=True)
        records.append(record)
        if len(records) % 500 == 0:
            print(json.dumps(dict(legacy_sources_verified=len(records))), flush=True)
    receipt = dict(schema='legacy-predictions-without-model-parameters-v1',
        remote_originals_unchanged=True, all_source_sha256_verified=True,
        exact_array_replay_passed=True, files=records, verified_at=time.time())
    (output / 'export_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    receipt = export(json.loads(args.plan.read_text()), args.output)
    print(json.dumps(dict(verified=len(receipt['files']),
                          exported=sum(bool(r['exported']) for r in receipt['files']))))
