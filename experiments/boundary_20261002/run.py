from experiments.boundary_20261002.state_io import save_state, load_state
"""Resumable disjoint support shards; no global labels enter a fitting API."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sample_support, seal_predictions, evaluate_sealed, sha_file, save_json
from prprompt.train import tensor_hash, state_hash
from experiments.boundary_20261002.prepare import read_plan, digest
from experiments.boundary_20261002.model import select, fit, saved_readout
from experiments.boundary_20261002.order_guard import save_order_diagnostic

def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(args.plan); job = plan['jobs'][args.job_index]; shots = args.shots
    data_spec = plan['datasets'][job['dataset']]
    assert sha_file(job['checkpoint']) == job['checkpoint_sha256']
    assert sha_file(data_spec['snapshot']) == data_spec['snapshot_sha256']
    data = torch.load(data_spec['snapshot'], map_location='cpu', weights_only=True)
    labels, pool, query = (data[k] for k in ['labels', 'pool', 'query'])
    x, edges = data['x'].to(args.device), data['edges'].to(args.device)
    classes = data_spec['classes']
    model = TwoLayerGCN(x.shape[1]).to(args.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(job['checkpoint'], map_location='cpu', weights_only=True))
    before = state_hash(model)
    root = args.plan.parent/('smoke' if args.smoke else 'results')/job['id']/str(shots)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/f'running_shard_{args.shard}.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        splits = plan['split_seeds'][:1] if args.smoke else plan['split_seeds'][args.shard::3]
        seeds = plan['train_seeds'][:1] if args.smoke else plan['train_seeds']
        config = json.loads(json.dumps(plan['config']))
        if args.smoke: config['fit']['steps'] = 2; config['smoke'] = True
        count = 0
        for split in splits:
            support = sample_support(pool, labels[pool], classes, shots, split)
            sy = labels[support]; s, y = support.to(args.device), sy.to(args.device)
            assert not torch.isin(support, query).any()
            assert bool((torch.bincount(sy, minlength=classes) == shots).all())
            for seed in seeds:
                for method in plan['methods_by_shot'][str(shots)]:
                    folder = root/str(split)/str(seed)/method; folder.mkdir(parents=True, exist_ok=True)
                    meta = dict(plan_sha256=plan['plan_sha256'], dataset=job['dataset'], family=job['family'],
                        checkpoint_id=job['id'], checkpoint_sha256=job['checkpoint_sha256'], graph_sha256=data_spec['graph_sha256'],
                        split_policy_sha256=digest({'pool': data_spec['pool_sha256'], 'query': data_spec['query_sha256']}),
                        shots=shots, split_seed=split, train_seed=seed, method=method, smoke=args.smoke,
                        support_labels_sha256=tensor_hash(sy), pretrain_seed='unknown_original')
                    signature = digest(meta)
                    if (folder/'metrics.json').exists():
                        r = json.loads((folder/'metrics.json').read_text()); assert r['signature'] == signature
                        for fn, k in [('predictions.npz', 'predictions_sha256'), ('state.pt.gz', 'state_sha256'), ('selection.json', 'selection_sha256')]:
                            assert sha_file(folder/fn) == r[k]
                        count += 1; continue
                    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats(args.device)
                    chosen, selection = select(method, model, x, edges, s, y, classes, config, seed)
                    scores, z, state, audit = fit(method, model, x, edges, s, y, classes, chosen, seed)
                    assert state_hash(model) == before and all(p.grad is None for p in model.parameters())
                    assert not audit['query_labels_received'] and not selection['query_labels_used']
                    state.update(method=method, raw_embedding=z.cpu(), support=support, support_labels=sy, query=query)
                    with torch.no_grad():
                        rebuilt = saved_readout(state, z, s, y, classes, args.device)
                        assert torch.equal(scores, rebuilt), method
                    save_state(state, folder/'state.pt.gz'); save_json(folder/'selection.json', selection)
                    save_order_diagnostic(folder, state, audit)
                    seal = seal_predictions(folder, scores, support, query.to(args.device), dict(meta, signature=signature))
                    metrics = evaluate_sealed(folder, labels[query])
                    torch.cuda.synchronize(args.device)
                    row = dict(meta, signature=signature, predictions_sha256=seal['predictions_sha256'],
                        support_sha256=seal['support_sha256'], query_sha256=seal['query_sha256'],
                        state_sha256=sha_file(folder/'state.pt.gz'), selection_sha256=sha_file(folder/'selection.json'),
                        embedding_sha256=tensor_hash(z.cpu()), evaluation_labels_sha256=tensor_hash(labels[query]),
                        selected_config=chosen, metrics=metrics, audit=audit, total_seconds=time.perf_counter()-start,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(args.device),
                        peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(args.device),
                        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=args.device))
                    save_json(folder/'metrics.json', row); count += 1
                    print(json.dumps({'job': job['id'], 'shots': shots, 'complete': count, 'method': method,
                                      'seconds': row['total_seconds']}), flush=True)
                    del z, state, scores
        save_json(root/f'completion_shard_{args.shard}.json', dict(units=count, smoke=args.smoke, plan_sha256=plan['plan_sha256']))
    finally:
        lock.unlink()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job-index', type=int, required=True); parser.add_argument('--shots', type=int, choices=[1, 3, 5], required=True)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--shard', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--smoke', action='store_true'); run(parser.parse_args())
