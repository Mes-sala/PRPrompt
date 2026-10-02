"""Freeze original inputs and generate exact ProG-algorithm task splits once."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from prprompt.protocol import save_json, sha_file
from experiments.prog_split_20261002.splits import split

METHODS = ['FrozenPrototype', 'FullFineTune', 'PRPrompt', 'NoRouting']


def prepare(workspace):
    assert not (workspace/'plan.json').exists()
    parent = Path('/data1/shq/GP/PRPrompt_boundary_20261002_R3')
    old = json.loads((parent/'plan.json').read_text())
    jobs = [dict(j, snapshot=old['datasets'][j['dataset']]['snapshot'],
                 classes=old['datasets'][j['dataset']]['classes']) for j in old['jobs']]
    arxiv = Path('/data1/shq/GP/PRPrompt_ogbn_arxiv_20261002')
    ck = json.loads((arxiv/'checkpoint_provenance.json').read_text())
    jobs.append(dict(id='ogbn-arxiv', dataset='ogbn-arxiv', classes=40,
                     checkpoint=str(arxiv/'input/pretrained.pth'), checkpoint_sha256=ck['checkpoint_sha256'],
                     snapshot=str(arxiv/'input/dataset.pt'), pretrain_seed=42,
                     family='Edgepred_Gprompt_fullbatch_no_projection'))
    tasks = []
    for j in jobs:
        assert sha_file(j['checkpoint']) == j['checkpoint_sha256']
        d = torch.load(j['snapshot'], map_location='cpu', weights_only=True)
        j.update(nodes=len(d['labels']), input_dim=d['x'].shape[1])
        out = workspace/'splits'/j['dataset'];out.mkdir(parents=True)
        np.savez_compressed(out/'labels.npz', labels=d['labels'].numpy())
        for seed in old['split_seeds']:
            for shots in [1,3,5]:
                support,pool,query=split(d['labels'],j['classes'],shots,seed)
                counts=torch.bincount(d['labels'][support],minlength=j['classes'])
                np.savez_compressed(out/f'{shots}_{seed}.npz',support=support.numpy(),pool=pool.numpy(),query=query.numpy())
                tasks.append(dict(dataset=j['dataset'],shots=shots,split_seed=seed,
                    support_size=len(support),query_size=len(query),support_class_counts=counts.tolist(),
                    support_query_overlap=int(torch.isin(support,query).sum()),
                    nominal_shot_complete=bool((counts==shots).all())))
    plan=dict(schema='original-ProG-two-independent-permutations-v1',jobs=jobs,methods=METHODS,
              shots=[1,3,5],split_seeds=old['split_seeds'],train_seeds=[0,1,2],shards=5,
              config=old['config'],bootstrap_seed=20261002,expected_units=5760,
              protocol='Exact ProG algorithm; overlap retained at explicit user request; not a held-out-only evaluation',
              upstream_commit='087aab45e25c336936d90d37540f140527567ebe',
              environment=dict(python=sys.version,torch=str(torch.__version__)))
    save_json(workspace/'split_inventory.json',tasks)
    save_json(workspace/'plan.json',plan)
    print(json.dumps(dict(tasks=len(tasks),shortages=[t for t in tasks if not t['nominal_shot_complete']],expected_units=5760)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);prepare(p.parse_args().workspace)
