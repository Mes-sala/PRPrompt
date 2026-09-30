#!/usr/bin/env python
"""Explicit, versioned node few-shot cell. No random hyperparameter search.

python scripts/run_protocol_cell.py --config configs/protocol_v4_cora.json
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROTOCOL = 'node-transductive-kshot-fixed-v4'
# Methods enter only after their predictions use the common scoring boundary.
CAPABILITIES = {
    'RSP': ('full_graph', 'node', 'cosine_support_prototype', 'research_candidate'),
    'FrozenNodePrototype': ('full_graph', 'node', 'support_prototype', 'baseline'),
    'FrozenIdentityPrototype': ('identity_graph', 'node', 'support_prototype', 'baseline'),
    'GraphTOP-Frozen': ('ego', 'center', 'linear', 'baseline'),
    'None': ('full_graph', 'node', 'linear', 'full_finetune'),
    'Frozen': ('full_graph', 'node', 'linear', 'baseline'),
    'FrozenGraph': ('induced', 'mean', 'linear', 'baseline'),
    'FrozenGraphPrototype': ('induced', 'mean', 'support_prototype', 'baseline'),
    'GPF': ('induced', 'mean', 'linear', 'prog_adapted'),
    'GPF-plus': ('induced', 'mean', 'linear', 'prog_adapted'),
    'Gprompt': ('induced', 'mean', 'support_prototype', 'prog_adapted'),
    'All-in-one': ('induced', 'mean', 'linear', 'budget_adapted'),
    'SUPT-soft': ('induced', 'mean', 'linear', 'prog_adapted'),
    'SUPT-hard': ('induced', 'mean', 'linear', 'prog_adapted'),
    'GraphTOP': ('ego', 'center', 'linear', 'prog_adapted'),
    'GPPT': ('full_graph', 'node', 'task_token', 'prog_adapted'),
    'EdgePrompt': ('full_graph', 'node', 'linear', 'simplified_unverified'),
    'EdgePromptplus': ('full_graph', 'node', 'linear', 'simplified_unverified'),
    'SelfPro': ('identity_graph', 'node', 'support_prototype', 'prog_adapted_unverified'),
    'PSP': ('features_only', 'node', 'learned_prototype', 'simplified_unverified'),
}
REQUIRED = {'dataset', 'prompt', 'checkpoint', 'gnn', 'hidden_dim', 'layers', 'shot',
            'folds', 'support_passes', 'learning_rate', 'weight_decay', 'split_seed',
            'train_seed', 'device', 'output_root', 'split_root'}
OPTIONAL = {'prompt_options', 'answer_passes', 'prompt_passes', 'prompt_lr',
            'allow_unverified', 'subgraph_min', 'subgraph_max', 'inference_seed'}


def validate(config):
    missing, unknown = REQUIRED - config.keys(), config.keys() - REQUIRED - OPTIONAL
    if missing or unknown:
        raise ValueError(f'Missing fields: {sorted(missing)}; unknown fields: {sorted(unknown)}')
    if config['prompt'] not in CAPABILITIES:
        raise NotImplementedError(f"{config['prompt']} has no audited NodeTask adapter; skipped")
    if config['gnn'] != 'GCN':
        raise NotImplementedError('v4 currently validates GCN only; other backbones require adapter tests')
    if 'unverified' in CAPABILITIES[config['prompt']][3] and not config.get('allow_unverified', False):
        raise NotImplementedError('Unverified adapted method requires allow_unverified=true for diagnostics')
    for name in ('shot', 'folds', 'hidden_dim', 'layers'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['prompt'] == 'RSP':
        from prompt_graph.prompt.RSP import RSPOptions
        opts = RSPOptions.parse(config.get('prompt_options'))
        if type(config['support_passes']) is not int or config['support_passes'] != 0:
            raise ValueError('RSP requires support_passes=0')
        if config['shot'] < opts.inner_folds or config['shot'] % opts.inner_folds:
            raise ValueError('RSP shot must be >= inner_folds and divisible by it')
    elif type(config['support_passes']) is not int or config['support_passes'] < 1:
        raise ValueError('support_passes must be a positive integer')
    for name in ('learning_rate', 'weight_decay'):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f'Invalid {name}')
    if config['learning_rate'] == 0 and config['prompt'] != 'RSP':
        raise ValueError('learning_rate must be positive')
    if config['prompt'] == 'RSP' and (config['learning_rate'] != 0 or config['weight_decay'] != 0):
        raise ValueError('RSP has no optimizer: learning_rate and weight_decay must be 0')
    if config['prompt'] == 'All-in-one':
        for name in ('answer_passes', 'prompt_passes', 'prompt_lr'):
            if name not in config or config[name] <= 0:
                raise ValueError(f'All-in-one requires positive {name}')
        if config['support_passes'] % (config['answer_passes'] + config['prompt_passes']):
            raise ValueError('All-in-one budget must contain complete alternating cycles')


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run(config, data=None):
    """Inject data in integration tests; production uses the recorded dataset loader."""
    validate(config)
    import torch

    from prompt_graph.data import load4node
    from prompt_graph.data.induced_graph import induced_graphs
    from prompt_graph.evaluation.node_scores import PREDICTION_SINK
    from prompt_graph.tasker.node_task import NodeTask
    from prompt_graph.utils import seed_everything

    seed_everything(config['train_seed'])
    if data is None:
        data, input_dim, output_dim = load4node(config['dataset'])
    else:
        data = data.clone()
        input_dim, output_dim = data.x.size(1), int(data.y.max()) + 1
    # Strict protocol currently requires explicit official pools; no hidden random fallback.
    if config['prompt'] == 'RSP' and getattr(data, 'edge_weight', None) is not None:
        raise ValueError('RSP MVP requires unweighted original data')
    if getattr(data, 'train_mask', None) is None or getattr(data, 'test_mask', None) is None:
        raise ValueError('v4 requires explicit train/test pools; prepare a versioned split first')
    train_pool = data.train_mask[:, 0] if data.train_mask.ndim == 2 else data.train_mask
    test_pool = data.test_mask[:, 0] if data.test_mask.ndim == 2 else data.test_mask
    if (train_pool & test_pool).any() or not train_pool.any() or not test_pool.any():
        raise ValueError('Train/test pools must be disjoint and nonempty')
    graph_digest = hashlib.sha256()
    for name in ('x', 'edge_index', 'y', 'train_mask', 'test_mask'):
        tensor = getattr(data, name).detach().cpu().contiguous()
        graph_digest.update(str((name, tensor.dtype, tuple(tensor.shape))).encode())
        graph_digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    dataset_hash = graph_digest.hexdigest()
    checkpoint = config['checkpoint']
    checkpoint_hash = hash_file(checkpoint) if checkpoint != 'None' else 'random'
    provenance_path = Path(str(checkpoint) + '.json')
    provenance = json.loads(provenance_path.read_text()) if checkpoint != 'None' and provenance_path.exists() else None
    if provenance is not None and provenance.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Checkpoint provenance hash mismatch')
    source = {str(p.relative_to(ROOT)): hash_file(p) for base in ('prompt_graph', 'scripts')
              for p in sorted((ROOT / base).rglob('*.py'))}
    import torch_geometric
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       pyg=torch_geometric.__version__, threads=torch.get_num_threads(),
                       platform=platform.platform())
    identity = dict(protocol=PROTOCOL, config=config, environment=environment, dataset_hash=dataset_hash,
                    checkpoint_hash=checkpoint_hash, checkpoint_metadata=provenance, source_hash=canonical_hash(source))
    run_id = canonical_hash(identity)
    out = Path(config['output_root']).expanduser().resolve() / run_id
    out.mkdir(parents=True, exist_ok=True)
    # Lock the result identity, not a global device. A scheduler owns device queues.
    with open(out / '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / 'result.json').exists():
            cached = json.loads((out / 'result.json').read_text())
            for relative, digest in cached['split_hashes'].items():
                if hash_file(Path(cached['split_directory']) / relative) != digest:
                    raise RuntimeError('Saved split modified; refusing cached result')
            if hash_file(out / 'predictions.json') != cached['predictions_hash']:
                raise RuntimeError('Saved predictions modified; refusing cached result')
            return cached
        write_json(out / 'config.json', config)
        write_json(out / 'identity.json', identity | {'run_id': run_id, 'source_files': source})
        try:
            split = Path(config['split_root']).expanduser().resolve() / canonical_hash({
                'dataset_hash': dataset_hash, 'shot': config['shot'], 'seed': config['split_seed'],
                'policy': 'official-pools-v4'})
            split.mkdir(parents=True, exist_ok=True)
            mode, readout, head, variant = CAPABILITIES[config['prompt']]
            graphs = None
            if mode == 'induced':
                graphs = induced_graphs(data, config.get('subgraph_min', 100), config.get('subgraph_max', 300))
            prompt_options = dict(config.get('prompt_options', {}))
            if config['prompt'] == 'GraphTOP-Frozen':
                prompt_options['disable_prompt'] = True
            task_prompt = 'GraphTOP' if config['prompt'] == 'GraphTOP-Frozen' else config['prompt']
            protocol = {k: config[k] for k in ('support_passes', 'train_seed', 'split_seed')}
            protocol['split_root'] = str(split)
            protocol['inference_seed'] = config.get('inference_seed', config['train_seed'] + 1000000)
            for name in ('answer_passes', 'prompt_passes', 'prompt_lr'):
                if name in config:
                    protocol[name] = config[name]
            with open(split / '.lock', 'w') as split_lock:
                fcntl.flock(split_lock, fcntl.LOCK_EX)
                task = NodeTask(data=data, input_dim=input_dim, output_dim=output_dim,
                    dataset_name=config['dataset'], prompt_type=task_prompt,
                    pre_train_model_path=checkpoint, gnn_type=config['gnn'],
                    hid_dim=config['hidden_dim'], num_layer=config['layers'],
                    epochs=config['support_passes'], task_num=config['folds'], shot_num=config['shot'],
                    lr=config['learning_rate'], wd=config['weight_decay'], device=config['device'],
                    batch_size=config['shot'] * output_dim, graphs_list=graphs,
                    prompt_options=prompt_options, protocol_options=protocol)
            task.data = task.data.to(task.device)
            predictions = []
            token = PREDICTION_SINK.set(predictions)
            try:
                values = task.run()
            finally:
                PREDICTION_SINK.reset(token)
            if len(predictions) != config['folds'] or len(task.run_diagnostics) != config['folds']:
                raise RuntimeError('Every fold must emit one scored prediction and audit record')
            split_hashes = {}
            for fold, pred in enumerate(predictions, 1):
                folder = split / str(fold)
                for p in sorted(folder.iterdir()):
                    if p.is_file():
                        split_hashes[f'{fold}/{p.name}'] = hash_file(p)
                idx = torch.load(folder / 'test_idx.pt', weights_only=True).view(-1)
                # Legacy induced loader enumerates graph-cache center order.
                if mode == 'induced' and config['prompt'] not in {'SUPT-soft', 'SUPT-hard'}:
                    idx = idx.sort().values
                if data.y.cpu()[idx].view(-1).tolist() != pred['labels']:
                    raise RuntimeError('Prediction/query ordering mismatch')
                pred['query_indices'] = idx.tolist()
                pred['fold'] = fold
            current_source = {str(p.relative_to(ROOT)): hash_file(p) for base in ('prompt_graph', 'scripts')
                              for p in sorted((ROOT / base).rglob('*.py'))}
            if current_source != source:
                raise RuntimeError('Source changed during run; refusing mixed-version result')
            try:
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            except subprocess.CalledProcessError:
                git_commit = None
            write_json(out / 'predictions.json', predictions)
            result = dict(status='complete', git_commit=git_commit, environment=environment,
                split_directory=str(split), predictions_hash=hash_file(out / 'predictions.json'), protocol=PROTOCOL, run_id=run_id,
                dataset=config['dataset'], prompt=config['prompt'], variant=variant,
                input_mode=mode, readout=readout, head=head, label_policy='support_only',
                validation_labels=0,
                selection=('support_inner_cv' if config['prompt'] == 'RSP' and
                           config.get('prompt_options', {}).get('fixed_index') is None else
                           'fixed_index' if config['prompt'] == 'RSP' else 'fixed_last'), config=config,
                checkpoint_hash=checkpoint_hash, split_hashes=split_hashes,
                metric_order=['accuracy', 'macro_f1', 'macro_auroc_ovr', 'macro_auprc_ovr'],
                aggregate=list(values), folds=task.run_diagnostics,
                checkpoint_provenance='random' if checkpoint == 'None' else ('metadata_recorded_adapted' if provenance else 'legacy_unverified'),
                output_directory=str(out))
            write_json(out / 'result.json', result)
            (out / 'failure.json').unlink(missing_ok=True)
            return safe_json(result)
        except Exception as exc:
            write_json(out / 'failure.json', {'status': 'failed', 'error': str(exc),
                                            'traceback': traceback.format_exc()})
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    try:
        result = run(config)
    except NotImplementedError as exc:
        print(json.dumps({'status': 'skipped', 'reason': str(exc)}))
        return 2
    print(json.dumps({'status': result['status'], 'output_directory': result['output_directory']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
