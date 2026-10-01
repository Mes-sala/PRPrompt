"""Seal input copies and the complete experiment before formal fitting."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import torch
from prprompt.protocol import load_dataset, sha_file, save_json
from prprompt.train import tensor_hash
from experiments.boundary_20261002.model import methods, HEADS

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'Actor', 'Chameleon']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read_plan(path):
    p = json.loads(Path(path).read_text())
    assert digest({k: v for k, v in p.items() if k != 'plan_sha256'}) == p['plan_sha256']
    for name, expected in p['source'].items():
        assert sha_file(ROOT/name) == expected, name
    return p

def prepare(workspace):
    assert not (workspace/'plan.json').exists(), 'Never replace a frozen plan'
    old_base = Path('/data1/shq/GP/PRPrompt_F100_uncorrected_20261001')
    multi_base = Path('/data1/shq/GP/PRPrompt_F100_multidata_20261001')
    old = json.loads((old_base/'plan.json').read_text())
    multi = json.loads((multi_base/'plan.json').read_text())
    jobs, datasets = [], {}
    for name in DATASETS:
        parent_plan = old if name in ['Cora', 'CiteSeer'] else multi
        prior = next(j for j in parent_plan['jobs'] if j['dataset'] == name and j['family'] == 'Edgepred_Gprompt')
        if name in ['Cora', 'CiteSeer']:
            x, edges, labels, pool, query, classes = load_dataset(name, '/home/shq/ProG/data')
            data = dict(x=x, edges=edges, labels=labels, pool=pool, query=query)
            rowfile = next((old_base/'results'/prior['id']).glob('5/*/*/F100/metrics.json'))
            previous = json.loads(rowfile.read_text())
            graph_hash = digest({'x': tensor_hash(x), 'edges': tensor_hash(edges)})
            assert graph_hash == previous['graph_sha256']
            split_policy = 'official_masks_fold0'
        else:
            spec = multi['datasets'][name]
            assert sha_file(spec['snapshot']) == spec['snapshot_sha256']
            data = torch.load(spec['snapshot'], map_location='cpu', weights_only=True)
            x, edges, labels, pool, query = (data[k] for k in ['x', 'edges', 'labels', 'pool', 'query'])
            classes = spec['classes']; split_policy = spec['split_policy']
            graph_hash = digest({'x': tensor_hash(x), 'edges': tensor_hash(edges)})
            assert graph_hash == spec['graph_sha256']
        assert not torch.isin(pool, query).any()
        assert torch.bincount(labels[pool], minlength=classes).min() >= 5
        snapshot = workspace/'dataset_snapshots'/(name+'.pt')
        snapshot.parent.mkdir(parents=True, exist_ok=True); torch.save(data, snapshot)
        target = workspace/'checkpoints'/(prior['id']+'.pth'); target.parent.mkdir(exist_ok=True)
        assert sha_file(prior['checkpoint']) == prior['checkpoint_sha256']
        shutil.copy2(prior['checkpoint'], target)
        jobs.append(dict(id=prior['id'], dataset=name, family='Edgepred_Gprompt', checkpoint=str(target),
            checkpoint_sha256=prior['checkpoint_sha256'], input_dim=x.shape[1], hidden_dim=128, output_dim=128,
            pretrain_seed='unknown_original', provenance_status=prior.get('provenance_status', 'ProG-adapted_not_fully_verified')))
        datasets[name] = dict(snapshot=str(snapshot), snapshot_sha256=sha_file(snapshot), classes=classes,
            nodes=len(x), input_dim=x.shape[1], graph_sha256=graph_hash, pool_sha256=tensor_hash(pool),
            query_sha256=tensor_hash(query), labels_sha256=tensor_hash(labels), split_policy=split_policy)
    source = {}
    for dirname in ['prprompt', 'experiments/v1_f100', 'experiments/boundary_20261002']:
        for path in sorted((ROOT/dirname).rglob('*')):
            if path.is_file() and path.suffix in ['.py', '.md']:
                source[str(path.relative_to(ROOT))] = sha_file(path)
    source['tools/metric_utils.py'] = sha_file(ROOT/'tools/metric_utils.py')
    plan = dict(schema='prprompt-routing-head-shot-boundary-v1', jobs=jobs, datasets=datasets, source=source,
        shots=[1, 3, 5], methods_by_shot={str(k): methods(k) for k in [1, 3, 5]},
        split_seeds=list(range(30093000, 30093020)), train_seeds=[0, 1, 2], split_shards=3,
        config=dict(fit=dict(steps=100, lr=.01, weight_decay=.0001, temperature=.1, amplitude=10., gpf_tokens=20),
                    selection=dict(learning_rates=[.001, .01], weight_decays=[0., .0001])),
        readout_probe_sources=['Soft__Linear', 'Soft__CenteredPrototype', 'F100'], probe_heads=HEADS,
        expected_units=31500, expected_probe_units=15120, bootstrap_seed=20261002,
        filter_construction_retry=dict(max_attempts=3, only_before_fit=True,
            original_assertion_unchanged=True, cache='exact immutable input tensor objects per worker'),
        audit_rules=dict(metric_atol=1e-12, gate_atol=2e-6, forward_atol=2e-6, forward_rtol=1e-6,
                         serialized_probability_exact=True, full_fresh_forward_is_diagnostic=True),
        environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda),
        boundary='Original Edgepred_Gprompt only; full graph unlabeled visibility, support labels only; no query selection; one-shot F100 explicitly adapted.')
    plan['plan_sha256'] = digest(plan); save_json(workspace/'plan.json', plan)
    print(json.dumps({'expected': plan['expected_units'], 'probes': plan['expected_probe_units'], 'plan_sha256': plan['plan_sha256']}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('workspace', type=Path)
    prepare(parser.parse_args().workspace)
