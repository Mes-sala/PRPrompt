"""Fixed-protocol adapters. Native prompt/readout operators, frozen ProG GCN.
See unified-comparison provenance report for deliberate benchmark adaptations.
Training receives support labels only; query labels are never accepted.
"""
import random, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import coalesce
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from prompt_graph.prompt.GPF import GPF,GPF_plus
from prompt_graph.prompt.GPrompt import Gprompt
from prompt_graph.prompt.GPPTPrompt import GPPTPrompt
from prompt_graph.prompt.UniPrompt import UniPrompt
from prompt_graph.prompt.GraphTOP import Rewiring,get_topology_subgraphs,prompted_edge_weights,topology_losses,annealing_temperature
from prompt_graph.data.induced_graph import _build_single_induced_graph
from prompt_graph.utils import act,constraint,Gprompt_tuning_loss

class ExactEdgePrompt(nn.Module):
    """Global edge prompt equals x+p before each normalized GCN linear map.
    No degree multiplication: every edge receives p once, including self-loops.
    """
    def __init__(self,dim_list):
        super().__init__();self.global_prompt=nn.ParameterList([nn.Parameter(torch.empty(1,d)) for d in dim_list])
        for p in self.global_prompt:nn.init.xavier_uniform_(p)
    def embed(self,model,x,edge_index):
        for i,conv in enumerate(model.conv_layers):
            x=conv(x+self.global_prompt[i],edge_index)
            if i<len(model.conv_layers)-1:x=act(x)
        return x

def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def graph_cache(data,nodes,kind):
    clean=Data(x=data.x.cpu(),edge_index=data.edge_index.cpu(),y=torch.zeros(data.num_nodes,dtype=torch.long),num_nodes=data.num_nodes)
    out={}
    for i in nodes.tolist():
        if kind=='topology':g=get_topology_subgraphs(clean,[i])[0]
        else:g=_build_single_induced_graph(clean,i,100,300,label=0)
        del g.x;g.y=torch.tensor([-1]);out[int(i)]=g
    return out

def batch_from(cache,ids,x,labels=None):
    gs=[]
    for pos,i in enumerate(ids.tolist()):
        g=cache[int(i)].clone();ix=g.node_idx if hasattr(g,'node_idx') else g.node_ids
        g.x=x[ix.to(x.device)];g.y=torch.tensor([int(labels[pos]) if labels is not None else -1],device=x.device)
        gs.append(g.to(x.device))
    return Batch.from_data_list(gs)

def train_baseline(method,model,data,support,sy,query,C,seed,steps,caches):
    seed_all(seed);model.eval();assert all(not p.requires_grad for p in model.parameters())
    x=data.x.to('cuda');e=data.edge_index.to('cuda');s=support.to('cuda');sy=sy.to('cuda');device=x.device
    with torch.no_grad():z0=model(x,e).detach()
    head=None;lr=.001;wd=1e-5
    if method=='GPPT':
        prompt=GPPTPrompt(128,C,C,device).to(device);masked=torch.full((len(x),),-1,device=device,dtype=torch.long);masked[s]=sy
        prompt.weight_init(z0,e,masked,s);lr=.002;wd=5e-4
    elif method=='GPF':prompt=GPF(x.shape[1]).to(device)
    elif method=='GPF+':prompt=GPF_plus(x.shape[1],20).to(device)
    elif method=='GPrompt':prompt=Gprompt(128).to(device)
    elif method=='EdgePrompt':prompt=ExactEdgePrompt([x.shape[1],128]).to(device);wd=0
    elif method=='GraphTOP':prompt=Rewiring(128).to(device);wd=0
    elif method=='UniPrompt':prompt=UniPrompt(x,50,'cosine',6,len(x)).to(device)
    else:raise ValueError(method)
    if method not in ['GPPT','GPrompt']:head=nn.Linear(128,C).to(device)
    pars=list(prompt.parameters())+(list(head.parameters()) if head is not None else [])
    optimizer=torch.optim.Adam(pars,lr=lr,weight_decay=wd)
    kind='topology' if method=='GraphTOP' else 'induced'
    if method in ['GPF','GPF+','GPrompt','EdgePrompt','GraphTOP']:
        if kind not in caches:caches[kind]=graph_cache(data,torch.cat([support.cpu(),query.cpu()]).unique(),kind)
        missing=[int(i) for i in support.tolist() if int(i) not in caches[kind]]
        if missing:caches[kind].update(graph_cache(data,torch.tensor(missing),kind))
        batch=batch_from(caches[kind],support,x,sy)
    if method=='GPrompt':
        with torch.no_grad():base=model(batch.x,batch.edge_index,batch.batch).detach()
    if method=='UniPrompt':comb_index,comb_weight=gcn_norm(e,num_nodes=len(x),add_self_loops=True)
    losses=[];gradnorm=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if method=='GPPT':
            logits=prompt(z0,e);loss=F.cross_entropy(logits[s],sy)+.001*constraint(device,prompt.get_TaskToken())
        elif method=='GPrompt':
            emb=prompt(base);centers=torch.stack([emb[sy==c].mean(0) for c in range(C)])
            loss=Gprompt_tuning_loss()(emb,centers,sy)
        elif method=='GraphTOP':
            weights,prob=prompted_edge_weights(prompt,z0,batch,annealing_temperature(step+1,steps),True)
            emb=model(batch.x,batch.edge_index,edge_weight=weights)[batch.target_node_index]
            ent,sparse=topology_losses(prob,batch);loss=F.cross_entropy(head(emb),sy)+2*ent+10*sparse
        elif method=='UniPrompt':
            pi,pw=prompt();comb_index,comb_weight=prompt.edge_fuse(comb_index,comb_weight,pi,pw,.99)
            logits=head(model(x,comb_index,edge_weight=comb_weight));loss=F.cross_entropy(logits[s],sy)
        else:
            if method=='EdgePrompt':emb=global_mean_pool(prompt.embed(model,batch.x,batch.edge_index),batch.batch)
            else:emb=model(prompt.add(batch.x),batch.edge_index,batch.batch)
            loss=F.cross_entropy(head(emb),sy)
        assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in pars if p.grad is not None)
        gradnorm.append(sum(float(p.grad.detach().square().sum()) for p in pars if p.grad is not None)**.5)
        optimizer.step();losses.append(float(loss.detach()))
        if method=='GPPT':prompt.update_StructureToken_weight(prompt.get_mid_h())
    prompt.eval()
    if head is not None:head.eval()
    with torch.no_grad():
        if method=='GPPT':qscore=prompt(z0,e)[query.to(device)]
        elif method=='UniPrompt':
            # Official runner evaluates last constructed graph with final classifier.
            qscore=head(model(x,comb_index.detach(),edge_weight=comb_weight.detach()))[query.to(device)]
        else:
            if method=='GPrompt':
                emb=prompt(base);centers=torch.stack([emb[sy==c].mean(0) for c in range(C)])
            chunks=[];torch.manual_seed(seed+1000000);torch.cuda.manual_seed_all(seed+1000000)
            for q in query.split(32):
                b=batch_from(caches[kind],q,x)
                if method=='GraphTOP':
                    weights,_=prompted_edge_weights(prompt,z0,b,.03,False);emb=model(b.x,b.edge_index,edge_weight=weights)[b.target_node_index];out=head(emb)
                elif method=='GPrompt':
                    emb=prompt(model(b.x,b.edge_index,b.batch));out=F.cosine_similarity(emb[:,None,:],centers[None,:,:],dim=-1)/.1
                elif method=='EdgePrompt':out=head(global_mean_pool(prompt.embed(model,b.x,b.edge_index),b.batch))
                else:out=head(model(prompt.add(b.x),b.edge_index,b.batch))
                chunks.append(out)
            qscore=torch.cat(chunks)
    scores=torch.zeros(len(x),C,device=device);scores[query.to(device)]=qscore
    state={'prompt':{k:v.detach().cpu() for k,v in prompt.state_dict().items()}}
    if head is not None:state['head']={k:v.detach().cpu() for k,v in head.state_dict().items()}
    if method=='UniPrompt':state['final_graph']={'edge_index':comb_index.cpu(),'edge_weight':comb_weight.detach().cpu()}
    return scores,state,{'method_family':'ProG fixed protocol reproduction','training_loss':losses,'gradient_norm':gradnorm,'updates':steps,'trainable_parameters':sum(p.numel() for p in pars),'lr':lr,'weight_decay':wd,'readout':'native','query_labels_received':False,'support_batch_size':len(s),'checkpoint_frozen':True}
