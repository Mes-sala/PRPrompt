"""Unchanged fit functions, original ProG splits, compact results-first recording."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from prprompt.model import TwoLayerGCN
from prprompt.protocol import save_json
from experiments.boundary_20261002.model import fit, select, saved_readout
from experiments.f100_boundary_20261002.model import fit_control
from experiments.boundary_20261002.state_io import save_state
from tools.metric_utils import numpy_metrics
from experiments.prog_split_20261002.numeric_diagnostics import install, ERRORS


def run(a):
    install()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    w=a.workspace;p=json.loads((w/'plan.json').read_text());j=p['jobs'][a.job]
    d=torch.load(j['snapshot'],map_location='cpu',weights_only=True)
    x,edges=d['x'].to(a.device),d['edges'].to(a.device);labels=d['labels'];classes=j['classes']
    model=TwoLayerGCN(x.shape[1]).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(j['checkpoint'],map_location='cpu',weights_only=True))
    root=w/('smoke' if a.smoke else 'results')/j['dataset']/str(a.shots)
    root.mkdir(parents=True,exist_ok=True);lock=root/f'shard_{a.shard}.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        splits=p['split_seeds'][:1] if a.smoke else p['split_seeds'][a.shard::p['shards']]
        seeds=[0] if a.smoke else p['train_seeds']
        cfg=copy.deepcopy(p['config'])
        if a.smoke:cfg['fit']['steps']=2;cfg['smoke']=True
        for split in splits:
            with np.load(w/'splits'/j['dataset']/f'{a.shots}_{split}.npz') as sfile:
                support=torch.from_numpy(sfile['support'].copy());query=torch.from_numpy(sfile['query'].copy())
            s=support.to(a.device);y=labels[support].to(a.device)
            counts=torch.bincount(y,minlength=classes);assert counts.min()>0
            for seed in seeds:
                for method in p['methods']:
                    folder=root/str(split)/str(seed)/method;folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'metrics.json').exists():continue
                    start=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device)
                    if method in ['PRPrompt','NoRouting']:
                        chosen=dict(steps=cfg['fit']['steps'],lr=.01,weight_decay=0.)
                        selection=dict(selection='fixed_historical_recipe',query_labels_used=False,trials=[])
                        # Existing common control implementation supports unequal class counts.
                        scores,z,state,audit=fit_control('Soft' if method=='PRPrompt' else 'Shared',model,x,edges,s,y,classes,seed,chosen['steps'])
                        state['method']='F100OneShotAdapt' if counts.min()==1 else 'F100Historical'
                    else:
                        name='None__CenteredPrototype' if method=='FrozenPrototype' else method
                        chosen,selection=select(name,model,x,edges,s,y,classes,cfg,seed)
                        scores,z,state,audit=fit(name,model,x,edges,s,y,classes,chosen,seed)
                        state['method']=name
                    assert not audit['query_labels_received'] and not selection['query_labels_used']
                    # Save predictions before scoring; no extra repeated file-hash scans.
                    probabilities=scores[query.to(a.device)].detach().cpu().double().softmax(-1).numpy()
                    assert np.isfinite(probabilities).all()
                    np.savez_compressed(folder/'predictions.npz',probabilities=probabilities,support=support.numpy(),query=query.numpy())
                    metrics=numpy_metrics(labels[query].numpy(),probabilities)
                    with torch.no_grad():
                        rebuilt=saved_readout(state,z,s,y,classes,a.device)
                        error=float((scores-rebuilt).abs().max());assert error<2e-6
                    state.update(raw_embedding=z.cpu(),support=support,support_labels=y.cpu(),query=query)
                    save_state(state,folder/'state.pt.gz');save_json(folder/'selection.json',selection)
                    torch.cuda.synchronize(a.device)
                    row=dict(dataset=j['dataset'],checkpoint_id=j['id'],checkpoint_sha256=j['checkpoint_sha256'],
                        family=j['family'],shots=a.shots,split_seed=split,train_seed=seed,method=method,smoke=a.smoke,
                        support_class_counts=counts.cpu().tolist(),support_query_overlap=int(torch.isin(support,query).sum()),
                        nominal_shot_complete=bool((counts==a.shots).all()),
                        protocol=p['schema'],metrics=metrics,selected_config=chosen,audit=audit,
                        engineering_revision='filter_row_sum_diagnostic_only',filter_row_sum_errors=list(ERRORS),
                        saved_readout_max_error=error,total_seconds=time.perf_counter()-start,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device))
                    save_json(folder/'metrics.json',row)
                    print(json.dumps({k:row[k] for k in ['dataset','shots','split_seed','train_seed','method','total_seconds']}),flush=True)
                    del scores,z,state,rebuilt,probabilities
    finally:lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);p.add_argument('--job',type=int,required=True)
    p.add_argument('--shots',type=int,required=True);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');run(p.parse_args())
