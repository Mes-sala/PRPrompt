"""Frozen-encoder Prompt adapters and original four-route readout."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_mean_pool
from prompt_graph.model import GCN
from prompt_graph.prompt import GcnLayers
from prompt_graph.prompt.GPF import GPF_plus
from prompt_graph.prompt.GPrompt import Gprompt
from prompt_graph.utils import process
from repair31_common import admission,episode,episodes,fingerprint,simplex
from run_transfer5 import full_key
from stable_random_graph import stats
from run_graph1 import featurek,normk
from ridge24_stable import fit_family,bordered
from train_masked41 import digest,write

METHODS=['GPF-plus','Gprompt','GPF-plus+Gprompt']


class MultiAdapter(torch.nn.Module):
    """Preserve the native GcnLayers forward, collecting actual layer outputs."""
    def __init__(self,dim):
        super().__init__();self.core=GcnLayers(dim,128,2,.3);self.cache={}

    def forward(self,x,edge_index,prompt_type=None):
        key=(x.shape[0],digest_bytes(edge_index.detach().cpu().numpy().tobytes()),str(x.dtype),str(x.device))
        if key not in self.cache:
            edge=edge_index.detach().cpu().numpy();n=x.shape[0]
            a=sp.csr_matrix((np.ones(edge.shape[1]),(edge[0],edge[1])),shape=(n,n))
            a=(process.normalize_adj(a+sp.eye(n))+sp.eye(n)).tocoo()
            self.cache[key]=torch.sparse_coo_tensor(torch.tensor(np.stack([a.row,a.col]),dtype=torch.long),torch.tensor(a.data,dtype=x.dtype),a.shape).coalesce().to(x.device)
        hs=[];handles=[layer.register_forward_hook(lambda module,args,out:hs.append(out)) for layer in self.core.convs]
        try:out=self.core(x,self.cache[key],True,False).squeeze(0)
        finally:
            for handle in handles:handle.remove()
        assert len(hs)==2
        return [x,*hs] if prompt_type=='DAGPrompT' else out


def digest_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def graphs41(ds):
    data=TUDataset(str(BASE/'graph_assets/raw_datasets'),ds,use_node_attr=True)
    graphs=[Data(x=g.x.clone(),edge_index=g.edge_index.clone()) for g in data]
    assert [full_key(g) for g in graphs]==admission()['datasets'][ds]['keys']
    return graphs,data.num_features


def model41(dim,path,kind='standard'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if kind=='multig':
        m=MultiAdapter(dim);m.core.load_state_dict(state,strict=True)
    else:
        m=GCN(dim,128,128,num_layer=2);m.load_state_dict(state,strict=True)
    return m.eval().requires_grad_(False)


def prompts41(dim,nc,method):
    torch.manual_seed(4242)
    f=GPF_plus(dim,20) if 'GPF-plus' in method else None
    c=Gprompt(128) if 'Gprompt' in method else None
    torch.manual_seed(42);head=torch.nn.Linear(128,nc)
    return f,c,head


def node41(m,b,f=None,c=None):
    hs=m(b.x if f is None else f.add(b.x),b.edge_index,prompt_type='DAGPrompT')[1:]
    if c is not None:hs=[hs[0],c(hs[1])]
    return hs


def train_step41(m,b,y,f,c,head,opt):
    assert not m.training and all(not p.requires_grad for p in m.parameters())
    opt.zero_grad();z=global_mean_pool(node41(m,b,f,c)[-1],b.batch)
    if c is None:logits=head(z)
    else:
        centers=torch.stack([z[y==i].mean(0) for i in range(head.out_features)])
        logits=F.cosine_similarity(z[:,None,:],centers[None,:,:],dim=-1)/.1
    loss=F.cross_entropy(logits,y);assert torch.isfinite(loss)
    loss.backward();assert all(p.grad is None for p in m.parameters());opt.step()
    return float(loss.detach())


@torch.no_grad()
def pooled41(m,graphs,f=None,c=None):
    m=m.cpu().double().eval()
    if isinstance(m,MultiAdapter):m.cache.clear()
    if f is not None:f=f.cpu().double().eval()
    if c is not None:c=c.cpu().double().eval()
    blocks=[[] for _ in range(6)]
    for b in DataLoader(graphs,batch_size=64,shuffle=False):
        assert b.y is None;b.x=b.x.double()
        for dest,v in zip(blocks,[v for h in node41(m,b,f,c) for v in stats(h,b.batch)]):dest.append(v)
    return [torch.cat(b) for b in blocks]


def kernels41(blocks):
    ks=[featurek(b) for b in blocks]
    return {'H':ks[3].numpy(),'M':normk(sum(ks)/6).numpy()}


def static41(ds,ep):
    with np.load(BASE/'matrix29_features'/(ds+'_columns.npz')) as d:
        ix=np.searchsorted(d['support_union'],ep['support']);assert np.array_equal(d['support_union'][ix],ep['support'])
        return {k:(d[key][ep['support']][:,ix],d[key][ep['query']][:,ix]) for k,key in [('W','WL3'),('A','attributes_rbf')]}


def ridge41(K,G,Y):
    fitted=fit_family(K,G,Y,True)['cross_scores'];reference=bordered(K,G,Y,True)[3]
    a=simplex(fitted).mean(0);b=simplex(reference).mean(0)
    assert np.allclose(a,b,atol=1e-5,rtol=1e-5),'ridge algebra discrepancy'
    return a,b


def readout41(ks,static,ep):
    s,q=ep['support'],ep['query'];Y=np.eye(int(ep['support_labels'].max())+1)[ep['support_labels']]
    all_k=dict(static);all_k.update({k:(v[np.ix_(s,s)],v[np.ix_(q,s)]) for k,v in ks.items()})
    scores={};refs={}
    for k,(K,G) in all_k.items():scores[k],refs[k]=ridge41(K,G,Y)
    out={};reference={}
    for name,parts in [('four','WAHM'),('HM','HM'),('WA','WA'),('H','H'),('M','M')]:
        out[name]=sum(scores[k] for k in parts)/len(parts)
        reference[name]=sum(refs[k] for k in parts)/len(parts)
        assert np.array_equal(np.round(out[name],10).argmax(-1),np.round(reference[name],10).argmax(-1)),name+' ridge class mismatch requires frozen numerical recovery'
    return out,max(float(np.max(abs(out[k]-reference[k]))) for k in out)


def checkpoint_specs41(ds):
    old=BASE/'repair31_training'/ds
    specs=[(n,'standard',old/n/'encoder.pth') for n in ['random_s101','DGI_both_s101','GraphCL_full_s101','SimGRACE_full_s101','GraphMAE_full_s101']]
    specs += [(m+'_s101','standard',ROOT/'masked_training'/ds/(m+'_s101')/'encoder.pth') for m in ['GraphMAE2','S2GAE','Bandana']]
    specs += [(m+'_'+v+'_s101','standard',ROOT/'edge_training'/ds/(m+'_'+v+'_s101')/'encoder.pth') for m in ['Edgepred_GPPT','Edgepred_Gprompt'] for v in ['native','full']]
    p=ROOT/'multig_training'/ds/'GraphMultiGprompt_repaired_s101'
    specs += [('GraphMultiGprompt_random_s101','multig',p/'random_encoder.pth'),('GraphMultiGprompt_repaired_s101','multig',p/'encoder.pth')]
    return specs
