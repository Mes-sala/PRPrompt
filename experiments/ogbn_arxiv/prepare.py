"""Read official CSV data directly; avoid mutating the shared OGB environment."""
import argparse
import json
import zipfile
import numpy as np
import pandas as pd
import torch
from prprompt.protocol import sha_file, save_json
from prprompt.train import tensor_hash
from .common import digest, source_hashes, METHODS, setup_torch

def prepare(w):
    setup_torch()
    assert not (w/'plan.json').exists(), 'Never overwrite a frozen experiment plan'
    archive = w/'input/arxiv.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        def read(suffix, dtype):
            names = [n for n in z.namelist() if n.endswith(suffix)]
            assert len(names) == 1, (suffix, names)
            with z.open(names[0]) as stream:
                return pd.read_csv(stream, header=None, compression='gzip').to_numpy(dtype=dtype)
        x = torch.from_numpy(read('/raw/node-feat.csv.gz', np.float32))
        edges = torch.from_numpy(read('/raw/edge.csv.gz', np.int64).T.copy())
        labels = torch.from_numpy(read('/raw/node-label.csv.gz', np.int64).reshape(-1))
        pool = torch.from_numpy(read('/split/time/train.csv.gz', np.int64).reshape(-1))
        valid = torch.from_numpy(read('/split/time/valid.csv.gz', np.int64).reshape(-1))
        query = torch.from_numpy(read('/split/time/test.csv.gz', np.int64).reshape(-1))
    assert x.shape == (169343, 128) and edges.shape == (2, 1166243)
    assert len(pool) == 90941 and len(valid) == 29799 and len(query) == 48603
    assert torch.isfinite(x).all() and len(labels.unique()) == 40
    assert not torch.isin(pool, query).any() and not torch.isin(valid, query).any()
    assert torch.bincount(labels[pool], minlength=40).min() >= 5
    assert torch.bincount(labels[query], minlength=40).min() > 0
    snapshot = w/'input/dataset.pt'
    torch.save(dict(x=x, edges=edges, labels=labels, pool=pool, query=query, valid=valid), snapshot)
    counts = dict(train=torch.bincount(labels[pool], minlength=40).tolist(),
                  test=torch.bincount(labels[query], minlength=40).tolist())
    plan = dict(schema='PRPrompt-ogbn-arxiv-20261002-v1', dataset='ogbn-arxiv',
        methods=METHODS, shots=[1,3,5], split_seeds=list(range(30093000,30093020)),
        train_seeds=[0,1,2], split_shards=2, expected_units=1260,
        config=dict(fit=dict(steps=100,lr=.01,weight_decay=.0001,temperature=.1,amplitude=10.,gpf_tokens=20),
                    selection=dict(learning_rates=[.001,.01],weight_decays=[0.,.0001])),
        pretrain=dict(seed=42,steps=1000,lr=.001,weight_decay=.00005,temperature=.2,
                      loss_edge_chunk=32768,checkpoint_policy='last_fixed_step',
                      family='Edgepred_Gprompt_fullbatch_no_projection',uses_node_labels=False),
        bootstrap_seed=20261002, snapshot_sha256=sha_file(snapshot), archive_sha256=sha_file(archive),
        graph_sha256=digest(dict(x=tensor_hash(x),edges=tensor_hash(edges))),
        pool_sha256=tensor_hash(pool),query_sha256=tensor_hash(query),labels_sha256=tensor_hash(labels),
        official_split_sizes=dict(train=len(pool),valid=len(valid),test=len(query)),class_counts=counts,
        source=source_hashes(),audit_rules=dict(saved_readout_atol=2e-6,metric_atol=1e-12,
                 fresh_forward_is_diagnostic=True,probability_replay_exact=True),
        source_urls=['https://ogb.stanford.edu/docs/nodeprop/',
                     'https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'])
    plan['plan_sha256']=digest(plan)
    save_json(w/'plan.json',plan)
    print(json.dumps(dict(plan_sha256=plan['plan_sha256'],units=1260,classes=40)),flush=True)

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);prepare(p.parse_args().workspace)
