"""Fixed recipe, disjoint support shards, own seals before reading reference scores."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sample_support, seal_predictions, evaluate_sealed, sha_file, save_json
from prprompt.train import tensor_hash, state_hash
from experiments.boundary_20261002.model import saved_readout
from experiments.boundary_20261002.state_io import save_state
from experiments.f100_boundary_20261002.prepare import read_plan, digest
from experiments.f100_boundary_20261002.model import fit_control
from experiments.boundary_20261002.order_guard import save_order_diagnostic

PAIR_FIELDS = ['checkpoint_sha256', 'graph_sha256', 'support_sha256', 'query_sha256', 'support_labels_sha256']


def parent_methods(shots):
    return ['F100OneShotAdapt' if shots == 1 else 'F100Historical', 'None__CenteredPrototype']


def copy_references(plan, workspace, row):
    refs = {}
    dirname = 'smoke' if row['smoke'] else 'results'
    for method in parent_methods(row['shots']):
        relative = Path(row['checkpoint_id']) / str(row['shots']) / str(row['split_seed']) / str(row['train_seed']) / method
        src = Path(plan['parent_workspace']) / dirname / relative
        ref = json.loads((src / 'metrics.json').read_text())
        assert ref['plan_sha256'] == plan['parent_plan_sha256'] and ref['smoke'] == row['smoke']
        assert ref['method'] == method
        assert all(ref[k] == row[k] for k in ['checkpoint_id', 'shots', 'split_seed', 'train_seed', 'split_policy_sha256', 'evaluation_labels_sha256'])
        assert ref['selected_config']['steps'] == row['steps']
        assert all(ref[k] == row[k] for k in PAIR_FIELDS)
        dest = workspace / ('smoke_references' if row['smoke'] else 'references') / relative
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        names = ['metrics.json', 'predictions.npz', 'seal.json', 'selection.json']
        if 'prefit_order_guard' in ref['audit']:
            names.append('prefit_order.npz')
        for name in names:
            if name == 'predictions.npz': assert sha_file(src / name) == ref['predictions_sha256']
            if name == 'selection.json': assert sha_file(src / name) == ref['selection_sha256']
            expected = sha_file(src / name)
            if (dest / name).exists():
                assert sha_file(dest / name) == expected
            else:
                tmp = dest / (name + '.tmp')
                shutil.copy2(src / name, tmp)
                tmp.replace(dest / name)
            hashes[name] = expected
        refs[method] = dict(relative=str(dest.relative_to(workspace)), files_sha256=hashes)
    return refs


def run(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(a.plan)
    job = plan['jobs'][a.job_index]
    spec = plan['datasets'][job['dataset']]
    assert sha_file(spec['snapshot']) == spec['snapshot_sha256']
    assert sha_file(job['checkpoint']) == job['checkpoint_sha256']
    data = torch.load(spec['snapshot'], weights_only=True, map_location='cpu')
    x, edges = data['x'].to(a.device), data['edges'].to(a.device)
    labels, pool, query = data['labels'], data['pool'], data['query']
    classes = spec['classes']
    model = TwoLayerGCN(x.shape[1]).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(job['checkpoint'], weights_only=True, map_location='cpu'))
    before = state_hash(model)
    root = a.plan.parent / ('smoke' if a.smoke else 'results') / job['id'] / str(a.shots)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f'running_shard_{a.shard}.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        splits = plan['split_seeds'][:1] if a.smoke else plan['split_seeds'][a.shard::3]
        seeds = plan['train_seeds'][:1] if a.smoke else plan['train_seeds']
        count = 0
        for split in splits:
            support = sample_support(pool, labels[pool], classes, a.shots, split)
            sy = labels[support]
            assert not torch.isin(support, query).any()
            assert (torch.bincount(sy, minlength=classes) == a.shots).all()
            for seed in seeds:
                for control in plan['controls']:
                    method = 'F100_' + control
                    folder = root / str(split) / str(seed) / method
                    folder.mkdir(parents=True, exist_ok=True)
                    meta = dict(plan_sha256=plan['plan_sha256'], parent_plan_sha256=plan['parent_plan_sha256'],
                        checkpoint_id=job['id'], checkpoint_sha256=job['checkpoint_sha256'], dataset=job['dataset'],
                        family=job['family'], graph_sha256=spec['graph_sha256'], support_labels_sha256=tensor_hash(sy),
                        split_policy_sha256=digest(dict(pool=spec['pool_sha256'], query=spec['query_sha256'])),
                        shots=a.shots, split_seed=split, train_seed=seed, method=method, control=control, smoke=a.smoke,
                        steps=plan['smoke_steps'] if a.smoke else plan['steps'], pretrain_seed='unknown_original')
                    signature = digest(meta)
                    if (folder / 'metrics.json').exists():
                        r = json.loads((folder / 'metrics.json').read_text())
                        assert r['signature'] == signature
                        for name, key in [('state.pt.gz', 'state_sha256'), ('predictions.npz', 'predictions_sha256'), ('selection.json', 'selection_sha256')]:
                            assert sha_file(folder / name) == r[key]
                        assert copy_references(plan, a.plan.parent, r) == r['references']
                        count += 1
                        continue
                    start = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(a.device)
                    s, y = support.to(a.device), sy.to(a.device)
                    scores, z, state, audit = fit_control(control, model, x, edges, s, y, classes, seed, meta['steps'])
                    assert state_hash(model) == before and not audit['query_labels_received']
                    state.update(method=method, raw_embedding=z.cpu(), support=support, support_labels=sy, query=query)
                    with torch.no_grad(): assert torch.equal(scores, saved_readout(state, z, s, y, classes, a.device))
                    save_state(state, folder / 'state.pt.gz')
                    save_order_diagnostic(folder, state, audit)
                    selection = dict(selection='fixed_historical_recipe', trials=[], query_labels_used=False,
                        lr=.01, weight_decay=0., adam_epsilon=1e-8, prompt_penalty=.001, temperature=.1,
                        one_shot_loss_adapted=a.shots == 1)
                    save_json(folder / 'selection.json', selection)
                    seal = seal_predictions(folder, scores, s, query.to(a.device), dict(meta, signature=signature))
                    metrics = evaluate_sealed(folder, labels[query])
                    torch.cuda.synchronize(a.device)
                    row = dict(meta, signature=signature, predictions_sha256=seal['predictions_sha256'],
                        support_sha256=seal['support_sha256'], query_sha256=seal['query_sha256'], metrics=metrics,
                        state_sha256=sha_file(folder / 'state.pt.gz'), selection_sha256=sha_file(folder / 'selection.json'),
                        embedding_sha256=tensor_hash(z.cpu()), evaluation_labels_sha256=tensor_hash(labels[query]), audit=audit,
                        total_seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(a.device),
                        peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(a.device))
                    # Reference scores are read only after this unit's predictions have been sealed and evaluated.
                    row['references'] = copy_references(plan, a.plan.parent, row)
                    save_json(folder / 'metrics.json', row)
                    count += 1
                    print(json.dumps(dict(job=job['id'], shots=a.shots, control=control, complete=count)), flush=True)
                    del z, state, scores
        save_json(root / f'completion_shard_{a.shard}.json', dict(units=count, smoke=a.smoke, plan_sha256=plan['plan_sha256']))
    finally:
        lock.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--job-index', type=int, required=True)
    p.add_argument('--shots', type=int, choices=[1, 3, 5], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
