from experiments.boundary_20261002.state_io import save_state, load_state
"""Classifier changes on sealed, detached final representations; never retrain prompts."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from prprompt.protocol import support_folds, sha_file, save_json, seal_predictions, evaluate_sealed
from prprompt.train import seed_all, tensor_hash
from experiments.boundary_20261002.model import readout
from experiments.boundary_20261002.prepare import read_plan, digest

def fit_linear(z, support, sy, classes, cfg, seed):
    seed_all(seed+1_000_003)
    head = nn.Linear(z.shape[1], classes).to(z.device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    zn = F.normalize(z, dim=1)
    for step in range(cfg['steps']):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(zn[support]), sy)
        assert torch.isfinite(loss); loss.backward(); opt.step()
    return head

def probe(z, head_kind, s, y, classes, config, seed):
    assert not z.requires_grad
    head = None; selection = {'selection': 'parameter_free', 'trials': [], 'query_labels_used': False}
    if head_kind == 'Linear':
        chosen = dict(config['fit']); folds = support_folds(s, y, classes)
        trials = []
        if folds and not config.get('smoke'):
            for lr in config['selection']['learning_rates']:
                for wd in config['selection']['weight_decays']:
                    cfg = dict(chosen, lr=lr, weight_decay=wd); losses = []
                    for i, (tr, ty, va, vy) in enumerate(folds):
                        h = fit_linear(z, tr, ty, classes, cfg, seed+i)
                        losses.append(float(F.cross_entropy(h(F.normalize(z[va], dim=1)), vy)))
                    trials.append({'config': cfg, 'fold_ce': losses, 'mean_ce': sum(losses)/len(losses)})
            chosen = min(trials, key=lambda t: t['mean_ce'])['config']
        head = fit_linear(z, s, y, classes, chosen, seed)
        selection = {'selection': 'support_only_cv' if trials else 'one_shot_predeclared_or_smoke',
                     'trials': trials, 'config': chosen, 'query_labels_used': False}
    with torch.no_grad():
        scores = readout(z, head_kind, s, y, classes, head)
    state = {'head_kind': head_kind}
    if head: state['head'] = {k: v.cpu() for k, v in head.state_dict().items()}
    return scores, state, selection

def run(args):
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    plan = read_plan(args.plan); job = plan['jobs'][args.job_index]; shots = args.shots
    data = torch.load(plan['datasets'][job['dataset']]['snapshot'], map_location='cpu', weights_only=True)
    labels = data['labels']; classes = plan['datasets'][job['dataset']]['classes']
    parent_root = args.plan.parent/('smoke' if args.smoke else 'results')/job['id']/str(shots)
    output = args.plan.parent/('smoke_probes' if args.smoke else 'probes')/job['id']/str(shots)
    cfg = json.loads(json.dumps(plan['config']))
    if args.smoke: cfg['fit']['steps'] = 2; cfg['smoke'] = True
    splits = plan['split_seeds'][:1] if args.smoke else plan['split_seeds']
    seeds = plan['train_seeds'][:1] if args.smoke else plan['train_seeds']
    count = 0
    for split in splits:
        for seed in seeds:
            for parent_name in plan['readout_probe_sources']:
                parent_method = ('F100OneShotAdapt' if shots == 1 else 'F100Historical') if parent_name == 'F100' else parent_name
                parent = parent_root/str(split)/str(seed)/parent_method
                row = json.loads((parent/'metrics.json').read_text())
                assert sha_file(parent/'state.pt.gz') == row['state_sha256']
                state = load_state(parent/'state.pt.gz')
                z = state['raw_embedding'].to(args.device).detach()
                assert tensor_hash(z) == row['embedding_sha256']
                s, y, q = (state[k].to(args.device) for k in ['support', 'support_labels', 'query'])
                for head_kind in plan['probe_heads']:
                    folder = output/str(split)/str(seed)/(parent_name+'__'+head_kind)
                    folder.mkdir(parents=True, exist_ok=True)
                    meta = {k: row[k] for k in ['plan_sha256', 'dataset', 'family', 'checkpoint_id', 'checkpoint_sha256',
                        'graph_sha256', 'split_policy_sha256', 'shots', 'split_seed', 'train_seed', 'support_labels_sha256', 'smoke']}
                    meta.update(method=parent_name+'__'+head_kind, parent_method=parent_method,
                        parent_state_sha256=row['state_sha256'], embedding_sha256=row['embedding_sha256'],
                        parent_metrics_sha256=sha_file(parent/'metrics.json'), fixed_embedding=True)
                    signature = digest(meta)
                    if (folder/'metrics.json').exists():
                        saved = json.loads((folder/'metrics.json').read_text()); assert saved['signature'] == signature
                        for fn, key in [('state.pt.gz', 'state_sha256'), ('selection.json', 'selection_sha256'), ('predictions.npz', 'predictions_sha256')]:
                            assert sha_file(folder/fn) == saved[key]
                        count += 1; continue
                    start = time.perf_counter()
                    scores, probe_state, selection = probe(z, head_kind, s, y, classes, cfg, seed)
                    assert tensor_hash(z) == row['embedding_sha256'] and not z.requires_grad
                    probe_state.update(parent_relative=str(parent.relative_to(args.plan.parent)), support=state['support'],
                                       support_labels=state['support_labels'], query=state['query'])
                    save_state(probe_state, folder/'state.pt.gz'); save_json(folder/'selection.json', selection)
                    seal = seal_predictions(folder, scores, s, q, dict(meta, signature=signature))
                    metrics = evaluate_sealed(folder, labels[state['query']])
                    saved = dict(meta, signature=signature, metrics=metrics,
                        predictions_sha256=seal['predictions_sha256'], support_sha256=seal['support_sha256'], query_sha256=seal['query_sha256'],
                        state_sha256=sha_file(folder/'state.pt.gz'), selection_sha256=sha_file(folder/'selection.json'),
                        total_seconds=time.perf_counter()-start, query_labels_received=False, embedding_updated=False,
                        evaluation_labels_sha256=tensor_hash(labels[state['query']]))
                    save_json(folder/'metrics.json', saved); count += 1
                del state, z
    save_json(output/'completion.json', dict(units=count, smoke=args.smoke, plan_sha256=plan['plan_sha256']))
    print(json.dumps({'job': job['id'], 'shots': shots, 'probe_units': count}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--job-index', type=int, required=True)
    p.add_argument('--shots', type=int, required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
