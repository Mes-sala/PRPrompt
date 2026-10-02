"""Meaningful small-graph check of chunked pretraining gradients."""
import argparse
import copy
import json
import torch
from torch.nn import functional as F
from prprompt.model import TwoLayerGCN
from .common import setup_torch

def check(device):
    setup_torch();torch.manual_seed(41)
    x=torch.randn(19,7,device=device,dtype=torch.float64)
    ids=torch.arange(19,device=device);edges=torch.stack([ids,(ids+1)%19])
    a=TwoLayerGCN(7,8,8).to(device).double();b=copy.deepcopy(a)
    triples=torch.stack([ids,(ids+1)%19,(ids+5)%19])
    def objective(z,t):
        v,p,n=t
        return F.softplus((F.cosine_similarity(z[v],z[n])-F.cosine_similarity(z[v],z[p]))/.2).sum()/19
    objective(a(x,edges),triples).backward()
    z=b(x,edges);leaf=z.detach().requires_grad_(True)
    for start in range(0,19,4):objective(leaf,triples[:,start:start+4]).backward()
    z.backward(leaf.grad)
    errors=[float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(a.parameters(),b.parameters())]
    assert max(errors)<1e-12,errors
    print(json.dumps(dict(device=device,chunked_gradient_max_error=max(errors))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');check(p.parse_args().device)
