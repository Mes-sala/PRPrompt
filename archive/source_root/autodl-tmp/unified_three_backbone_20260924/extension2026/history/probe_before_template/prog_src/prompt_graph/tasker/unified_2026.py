"""MTG and DAPrompt prompt-stage adapters for the fixed frozen-checkpoint protocol."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data,Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import coalesce,to_undirected
from prompt_graph.utils import act
from prompt_graph.tasker.unified_comparison import seed_all,graph_cache,batch_from

class MTGPrompt(nn.Module):
    def __init__(self,dims,m_num=10):
        super().__init__();self.MTG_vectors=nn.ParameterList([nn.Parameter(torch.empty(m_num,d)) for d in dims]);self.proj_vectors=nn.ModuleList([nn.Linear(d,m_num) for d in dims]);self.reset_parameters()
    def reset_parameters(self):
        for p in self.MTG_vectors:nn.init.kaiming_normal_(p,mode='fan_in',nonlinearity='leaky_relu',a=.01)
        for p in self.proj_vectors:p.reset_parameters()
    def embed(self,model,x,e,batch):
        for i,conv in enumerate(model.conv_layers):
            x=x+F.softmax(self.proj_vectors[i](x),dim=1)@self.MTG_vectors[i];x=conv(x,e)
            if i<len(model.conv_layers)-1:x=act(x)
        return global_mean_pool(x,batch)

class DAPromptStage(nn.Module):
    """Author discrete graph construction, replacing scipy coalescing by torch.
    Keeps two structure tokens, C semantic tokens, broadcast, diversity loss.
    Backbone freezing is a benchmark change (original main optimizer includes it).
    """
    def __init__(self,input_dim,C):
        super().__init__();self.structure_prompt=nn.Parameter(torch.empty(2,input_dim));self.semantics_prompt=nn.Parameter(torch.empty(C,input_dim));nn.init.xavier_uniform_(self.structure_prompt);nn.init.xavier_uniform_(self.semantics_prompt)
    @staticmethod
    def dedup(e1,e2,n,threshold=.8):
        e=torch.cat([e1,e2],1);idx,w=coalesce(e,torch.ones(e.shape[1],device=e.device),num_nodes=n,reduce='sum');return idx[:,torch.sigmoid(w)>threshold]
    def manipulate(self,batch):
        graphs=[]
        for g in batch.to_data_list():
            x=torch.cat([self.structure_prompt,g.x]);normal=len(x);n=normal+len(self.semantics_prompt)
            sim=torch.sigmoid(F.normalize(x,dim=1)@F.normalize(x,dim=1).T);candidates=(sim>.2).nonzero().T
            e=self.dedup(g.edge_index+2,candidates,n)
            target=int((g.node_ids==int(g.index)).nonzero().flatten()[0])+2
            semantic=torch.arange(normal,n,device=x.device);target_edges=to_undirected(torch.stack([torch.full_like(semantic,target),semantic]))
            sources=semantic.repeat_interleave(normal-2);dest=torch.arange(2,normal,device=x.device).repeat(len(semantic));broadcast=torch.stack([sources,dest])
            edges=torch.cat([e,broadcast,target_edges],1);graphs.append(Data(x=torch.cat([x,self.semantics_prompt]),edge_index=edges))
        return Batch.from_data_list(graphs)
    def diversity(self):
        loss=0
        for p in [self.structure_prompt,self.semantics_prompt]:
            p=F.normalize(p,dim=1);sim=p@p.T;loss=loss+.5*sim[~torch.eye(len(p),device=p.device,dtype=torch.bool)].mean()
        return loss
    def embed(self,model,batch):
        b=self.manipulate(batch);return global_mean_pool(F.relu(model(b.x,b.edge_index)),b.batch)

def train_baseline(method,model,data,support,sy,query,C,seed,steps,caches):
    seed_all(seed);model.eval();x=data.x.cuda();sy=sy.cuda();kind='induced'
    if kind not in caches:caches[kind]=graph_cache(data,torch.cat([support.cpu(),query.cpu()]).unique(),kind)
    missing=[int(i) for i in support.tolist() if int(i) not in caches[kind]]
    if missing:caches[kind].update(graph_cache(data,torch.tensor(missing),kind))
    b=batch_from(caches[kind],support,x,sy)
    if method=='MTG':prompt=MTGPrompt([x.shape[1],128]).cuda();lr=.001;wd=1e-5
    elif method=='DAPrompt-fixed':prompt=DAPromptStage(x.shape[1],C).cuda();lr=.01;wd=5e-4
    else:raise ValueError(method)
    head=nn.Linear(128,C).cuda();pars=list(prompt.parameters())+list(head.parameters());opt=torch.optim.Adam(pars,lr=lr,weight_decay=wd);losses=[];grads=[]
    def emb(b):return prompt.embed(model,b.x,b.edge_index,b.batch) if method=='MTG' else prompt.embed(model,b)
    for t in range(steps):
        opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(emb(b)),sy)
        if method=='DAPrompt-fixed':loss=loss+prompt.diversity()
        assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in pars if p.grad is not None);grads.append(sum(float(p.grad.square().sum()) for p in pars if p.grad is not None)**.5);opt.step();losses.append(float(loss.detach()))
    prompt.eval();head.eval();out=[]
    with torch.no_grad():
        for q in query.split(16):out.append(head(emb(batch_from(caches[kind],q,x))))
    scores=torch.zeros(len(x),C,device=x.device);scores[query.cuda()]=torch.cat(out)
    state={'prompt':{k:v.detach().cpu() for k,v in prompt.state_dict().items()},'head':{k:v.detach().cpu() for k,v in head.state_dict().items()}}
    return scores,state,{'training_loss':losses,'gradient_norm':grads,'updates':steps,'query_labels_received':False,'checkpoint_frozen':True,'lr':lr,'weight_decay':wd,'trainable_parameters':sum(p.numel() for p in pars),'support_batch_size':len(support),'readout':'native subgraph','adaptation':'frozen checkpoint; DAPrompt prompt-stage only' if method=='DAPrompt-fixed' else 'official MTG operators with fixed support protocol'}
