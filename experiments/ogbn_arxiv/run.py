"""Resumable fixed-condition workers; state is retained only on the server."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from prprompt.model import TwoLayerGCN
from prprompt.protocol import sample_support, seal_predictions, evaluate_sealed, save_json, sha_file
from prprompt.train import state_hash, tensor_hash
from experiments.boundary_20261002.model import fit, select, saved_readout, replay_baseline_embedding, readout
from experiments.f100_boundary_20261002.model import fit_control
from experiments.v1_f100.historical import load
from .common import setup_torch, load_plan, digest

def implementation(method,shots):
    if method=='Frozen':return 'None__CenteredPrototype'
    if method=='PRPrompt':return 'F100OneShotAdapt' if shots==1 else 'F100Historical'
    return method

def run(a):
    setup_torch();w=a.workspace;plan=load_plan(w)
    assert sha_file(w/'input/dataset.pt')==plan['snapshot_sha256']
    ck=json.loads((w/'checkpoint_provenance.json').read_text())
    assert sha_file(w/'input/pretrained.pth')==ck['checkpoint_sha256']
    d=torch.load(w/'input/dataset.pt',map_location='cpu',weights_only=True)
    x=d['x'].to(a.device);edges=d['edges'].to(a.device);labels=d['labels'];pool=d['pool'];query=d['query']
    model=TwoLayerGCN(128).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(w/'input/pretrained.pth',map_location='cpu',weights_only=True))
    before=state_hash(model);classes=40
    root=w/('smoke' if a.smoke else 'results');root.mkdir(exist_ok=True)
    lock=root/f'worker_{a.shard}.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        splits=plan['split_seeds'][:1] if a.smoke else plan['split_seeds'][a.shard::plan['split_shards']]
        seeds=plan['train_seeds'][:1] if a.smoke else plan['train_seeds']
        cfg=copy.deepcopy(plan['config'])
        if a.smoke:cfg['fit']['steps']=2;cfg['smoke']=True
        for split in splits:
            for shots in plan['shots']:
                support=sample_support(pool,labels[pool],classes,shots,split);sy=labels[support]
                assert not torch.isin(support,query).any()
                s=support.to(a.device);y=sy.to(a.device)
                for seed in seeds:
                    for method in plan['methods']:
                        folder=root/str(shots)/str(split)/str(seed)/method;folder.mkdir(parents=True,exist_ok=True)
                        meta=dict(plan_sha256=plan['plan_sha256'],dataset='ogbn-arxiv',shots=shots,
                            split_seed=split,train_seed=seed,method=method,smoke=a.smoke,
                            checkpoint_sha256=ck['checkpoint_sha256'],graph_sha256=plan['graph_sha256'],
                            support_labels_sha256=tensor_hash(sy))
                        signature=digest(meta)
                        if (folder/'metrics.json').exists():
                            row=json.loads((folder/'metrics.json').read_text());assert row['signature']==signature
                            for name in ['state.pt','predictions.npz','selection.json','unit_audit.json']:
                                assert sha_file(folder/name)==row['files'][name],name
                            continue
                        start=time.time();torch.cuda.reset_peak_memory_stats(a.device)
                        name=implementation(method,shots)
                        if method=='SharedPrompt':
                            chosen=dict(steps=cfg['fit']['steps'],lr=.01,weight_decay=0.)
                            selection=dict(selection='fixed_same_F100_recipe',query_labels_used=False,trials=[])
                            scores,z,state,audit=fit_control('Shared',model,x,edges,s,y,classes,seed,chosen['steps'])
                            state['method']='F100OneShotAdapt' if shots==1 else 'F100Historical'
                        else:
                            chosen,selection=select(name,model,x,edges,s,y,classes,cfg,seed)
                            scores,z,state,audit=fit(name,model,x,edges,s,y,classes,chosen,seed)
                            state['method']=name
                        assert before==state_hash(model) and all(p.grad is None for p in model.parameters())
                        assert not audit['query_labels_received'] and not selection['query_labels_used']
                        state.update(raw_embedding=z.cpu(),support=support,support_labels=sy,query=query,
                                     sealed_query_logits=scores[query.to(a.device)].cpu())
                        with torch.no_grad():
                            rebuilt=saved_readout(state,z,s,y,classes,a.device)
                            readout_error=float((rebuilt-scores).abs().max())
                            changes=int((rebuilt.argmax(1)!=scores.argmax(1)).sum())
                            assert readout_error<plan['audit_rules']['saved_readout_atol'] and changes==0
                            if method in ['PRPrompt','SharedPrompt']:
                                u=model.conv_layers[0](x,edges);scale=u.std(0).clamp_min(1e-6)
                                p=state['prompt'];g=p['gate'].to(a.device);t=p['tokens'].to(a.device)
                                fresh_z=model.conv_layers[1](F.leaky_relu(u+10*(g@t.tanh())*scale),edges)
                                fresh=load()['logits'](fresh_z-fresh_z.mean(0,keepdim=True),s,y,classes)
                            elif method=='Frozen':
                                fresh_z=model(x,edges);fresh=readout(fresh_z,'CenteredPrototype',s,y,classes)
                            else:
                                fresh_z=replay_baseline_embedding(method,state,model,x,edges)
                                if method=='GraphPromptDiag':fresh=readout(fresh_z,'Prototype',s,y,classes)
                                else:fresh=F.linear(F.normalize(fresh_z,dim=1),state['head']['weight'].to(a.device),state['head']['bias'].to(a.device))
                            diagnostic=dict(logit_max_error=float((fresh-scores).abs().max()),
                                all_node_class_changes=int((fresh.argmax(1)!=scores.argmax(1)).sum()),
                                query_class_changes=int((fresh[query].argmax(1)!=scores[query].argmax(1)).sum()),
                                role='diagnostic_not_replacement_for_sealed_predictions')
                            del fresh,fresh_z,rebuilt
                        torch.save(state,folder/'state.pt');save_json(folder/'selection.json',selection)
                        seal=seal_predictions(folder,scores,support,query.to(a.device),dict(meta,signature=signature))
                        with np.load(folder/'predictions.npz') as sealed:
                            expected=state['sealed_query_logits'].double().softmax(-1).numpy()
                            assert np.array_equal(sealed['probabilities'],expected)
                        unit_audit=dict(backbone_input_hash=before,saved_readout_max_error=readout_error,
                            saved_readout_class_changes=changes,serialized_probability_exact=True,
                            fresh_forward=diagnostic,training=audit,state_retained_on_server=True)
                        save_json(folder/'unit_audit.json',unit_audit)
                        # First access of test labels for scoring follows immutable prediction/state sealing.
                        metrics=evaluate_sealed(folder,labels[query]);torch.cuda.synchronize(a.device)
                        row=dict(meta,signature=signature,metrics=metrics,
                            support_sha256=seal['support_sha256'],query_sha256=seal['query_sha256'],
                            files={n:sha_file(folder/n) for n in ['state.pt','predictions.npz','selection.json','unit_audit.json']},
                            elapsed_seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(a.device),
                            peak_reserved=torch.cuda.max_memory_reserved(a.device))
                        save_json(folder/'metrics.json',row)
                        print(json.dumps(dict(shots=shots,split=split,seed=seed,method=method,seconds=row['elapsed_seconds'],peak=row['peak_reserved'])),flush=True)
                        del state,scores,z
        save_json(root/f'completion_{a.shard}.json',dict(plan_sha256=plan['plan_sha256'],complete=True))
    finally:
        lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--shard',type=int,choices=[0,1],default=0);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
