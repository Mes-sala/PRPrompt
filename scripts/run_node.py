"""Fit final input-node PRPrompt on an explicit, externally prepared task."""
import argparse
from pathlib import Path
import torch
from prprompt import TwoLayerGCN,fit_fullgraph,fit_subgraph,SubgraphConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,help='Tensor dict: x,edge_index,support,support_labels,classes; optional query/subgraphs')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['fullgraph','subgraph'],default='fullgraph')
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    task=torch.load(args.task,map_location='cpu',weights_only=True);x=task['x'].to(args.device);e=task['edge_index'].to(args.device)
    si,sy=task['support'],task['support_labels'];c=int(task['classes'])
    if 'query' in task and torch.isin(torch.as_tensor(si).cpu(),torch.as_tensor(task['query']).cpu()).any():raise ValueError('Support/query overlap')
    base=TwoLayerGCN.from_checkpoint(args.checkpoint,x.shape[1],device=args.device)
    if args.mode=='fullgraph':scores,z,state,audit=fit_fullgraph(base,x,e,si,sy,c,seed=args.seed)
    else:
        subgraphs=task['subgraphs']
        scores,z,state,audit=fit_subgraph(base,x,e,si,sy,c,lambda i:subgraphs[i],config=SubgraphConfig(seed=args.seed))
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(scores=scores.cpu(),embeddings=z.cpu(),state=state,audit=audit,query=task.get('query')),dest)

if __name__=='__main__':main()
