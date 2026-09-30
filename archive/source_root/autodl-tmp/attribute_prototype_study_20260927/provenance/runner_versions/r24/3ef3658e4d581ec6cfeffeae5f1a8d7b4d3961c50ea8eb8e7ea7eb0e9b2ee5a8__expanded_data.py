"""Explicit 14-dataset loading and encoder input contracts for expanded grids."""
from pathlib import Path
import json
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import run_study as r
OLD=list(r.DATASETS)
NEW=['Squirrel','WikiCS','CS','Physics','Roman-empire','Amazon-ratings']
ALL=OLD+NEW

def load_data(name):
 if name in OLD:return r.load_data(name)
 if name not in NEW:raise ValueError(f'Unknown or unsupported dataset: {name}')
 records={}
 for f in ['expanded_datasets.json','expanded_heterophily_datasets.json']:
  records.update(json.loads((r.ROOT/'provenance'/f).read_text()))
 meta=records[name];assert meta['status']=='ready';path=Path(meta['canonical_path']);assert r.sha(path)==meta['sha256']
 a=torch.load(path,map_location='cpu',weights_only=True);d=Data(x=a['x'],edge_index=a['edge_index'],y=a['y']);pool=a['pool'];query=a['query'];C=int(d.y.max())+1
 assert r.thash(d.x)==meta['input_hash'] and r.thash(d.edge_index)==meta['edge_hash']
 assert not torch.isin(pool,query).any() and torch.bincount(d.y[pool],minlength=C).min()>=5
 xn=NormalizeFeatures()(Data(x=d.x.clone(),edge_index=d.edge_index)).x
 ei,ew=gcn_norm(d.edge_index,num_nodes=len(d.x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(d.x),len(d.x))).coalesce().cuda()
 return d,xn.cuda(),adj,pool,query,C,meta['protocol']

def encoder_input(d,metadata):
 mode=metadata.get('input_normalization')
 if mode=='NormalizeFeatures':return NormalizeFeatures()(Data(x=d.x.clone(),edge_index=d.edge_index)).x.cuda()
 if mode is not None:raise ValueError(f'Unsupported input preprocessing {mode}')
 # Historical checkpoints use their original dataset loader contract.
 return d.x.cuda()

if __name__=='__main__':
 audit={}
 for name in ALL:
  d,x,adj,pool,query,C,protocol=load_data(name)
  audit[name]=dict(nodes=len(x),features=x.shape[1],classes=C,pool=len(pool),query=len(query),protocol=protocol,attribute_hash=r.thash(x),query_hash=r.thash(query),pool_hash=r.thash(pool))
  if name in NEW:
   expected=NormalizeFeatures()(Data(x=d.x.clone(),edge_index=d.edge_index)).x
   assert torch.equal(encoder_input(d,{'input_normalization':'NormalizeFeatures'}).cpu(),expected)
  print(name,'passed',flush=True);del d,x,adj;torch.cuda.empty_cache()
 r.save(r.ROOT/'provenance/expanded_loader_audit.json',audit)
