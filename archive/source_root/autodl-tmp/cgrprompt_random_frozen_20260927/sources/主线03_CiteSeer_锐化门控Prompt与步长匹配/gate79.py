import torch
@torch.no_grad()
def temperature_gate(scores,adj,temperature,mode):
    assert temperature>0
    basep=scores.softmax(-1)
    probability=(scores*(.2/temperature)).softmax(-1)
    base=torch.sparse.mm(adj,basep-basep.mean(0,keepdim=True))
    raw=torch.sparse.mm(adj,probability-probability.mean(0,keepdim=True))
    assert base.norm()>0 and raw.norm()>0
    ratio=raw.norm()/base.norm()
    if mode=='raw':gate=raw
    elif mode=='matched':gate=raw/ratio
    elif mode=='amplitude':gate=base*ratio
    else:raise ValueError(mode)
    return gate,dict(probability=probability,ratio=ratio,base=base,raw=raw)
