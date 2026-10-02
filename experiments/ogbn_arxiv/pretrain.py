"""Unlabeled fixed-budget full-edge contrastive pretraining, chunked exact gradient."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch_geometric.utils import structured_negative_sampling
from prprompt.model import TwoLayerGCN
from prprompt.train import seed_all, tensor_hash
from prprompt.protocol import sha_file, save_json
from .common import load_plan, setup_torch

def pretrain(w,device):
    setup_torch();plan=load_plan(w);cfg=plan['pretrain'];seed_all(cfg['seed'])
    data=torch.load(w/'input/dataset.pt',map_location='cpu',weights_only=True)
    x=data['x'].to(device);edges=data['edges'].to(device);n=len(x)
    # Only x and edges are used; no node labels or label-derived optimizer choices.
    del data
    eye=torch.arange(n,device=device);positive=torch.cat([edges,torch.stack([eye,eye])],1)
    triple=structured_negative_sampling(positive.cpu(),num_nodes=n)
    triples=torch.stack(triple).to(device)
    torch.save(triples.cpu(),w/'input/pretrain_triples.pt')
    adjacency=torch.sparse_coo_tensor(positive,torch.ones(positive.shape[1],device=device),(n,n)).coalesce()
    model=TwoLayerGCN(128).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    log=w/'logs/pretraining_steps.jsonl';start=time.time()
    for step in range(cfg['steps']):
        optimizer.zero_grad(set_to_none=True)
        representation=torch.sparse.mm(adjacency,model(x,edges))
        detached=representation.detach().requires_grad_(True);total=0.
        for offset in range(0,triples.shape[1],cfg['loss_edge_chunk']):
            v,a,b=triples[:,offset:offset+cfg['loss_edge_chunk']]
            p=F.cosine_similarity(detached[v],detached[a],dim=-1)/cfg['temperature']
            q=F.cosine_similarity(detached[v],detached[b],dim=-1)/cfg['temperature']
            loss=F.softplus(q-p).sum()/triples.shape[1]
            assert torch.isfinite(loss)
            loss.backward();total+=float(loss.detach())
        representation.backward(detached.grad)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        with log.open('a') as f:f.write(json.dumps(dict(step=step+1,loss=total,elapsed=time.time()-start))+'\n')
        if (step+1)%10==0:print(json.dumps(dict(pretrain_step=step+1,loss=total)),flush=True)
    target=w/'input/pretrained.pth';torch.save(model.cpu().state_dict(),target)
    save_json(w/'checkpoint_provenance.json',dict(plan_sha256=plan['plan_sha256'],
        checkpoint_sha256=sha_file(target),triples_sha256=sha_file(w/'input/pretrain_triples.pt'),
        config=cfg,node_labels_used=False,elapsed_seconds=time.time()-start,
        graph_sha256=plan['graph_sha256'],architecture='TwoLayerGCN_128_128_LeakyReLU_0.01'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();pretrain(a.workspace,a.device)
