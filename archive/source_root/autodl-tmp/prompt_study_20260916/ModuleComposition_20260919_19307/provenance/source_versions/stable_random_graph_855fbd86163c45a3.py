"""Deterministic CPU float64 frozen encoding and centered two-pass pooling."""
import argparse,json,sys
from pathlib import Path
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool,global_max_pool
import run_graph1 as r
def stats(z,batch):
 mu=global_mean_pool(z,batch);var=global_mean_pool((z-mu[batch]).square(),batch);std=var.sqrt();rms=global_mean_pool(z.square(),batch).sqrt();std=torch.where(std<=1e-10*rms,torch.zeros_like(std),std)
 return [mu,global_max_pool(z,batch),std]
def prepare(ds,assets,prog,out):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);sys.path.insert(0,prog)
 from prompt_graph.model import GCN
 root=Path(out);root.mkdir(parents=True,exist_ok=True);data=TUDataset(root=str(Path(assets)/'raw_datasets'),name=ds,use_node_attr=True);graphs=list(data);basis=r.wl(graphs,data.num_node_labels);raw=[];structure=[];prov={}
 for g in graphs:
  raw.append(torch.cat(stats(g.x.double(),torch.zeros(g.num_nodes,dtype=torch.long)),1)[0]);deg=torch.bincount(g.edge_index[0],minlength=g.num_nodes).double();hist=torch.bincount(deg.long().clamp_max(10),minlength=11).double();structure.append(torch.cat([torch.tensor([g.num_nodes,g.num_edges/2,deg.mean(),deg.std(unbiased=False),deg.max()],dtype=torch.double).log1p(),hist.log1p()]))
 for name,z in [('attributes',torch.stack(raw)),('structure',torch.stack(structure))]:
  z=r.standardized(z);basis['attributes_linear' if name=='attributes' else 'structure']=r.featurek(z);d=torch.cdist(z,z).square();median=d[d>0].median().clamp_min(1e-8)
  for suffix,scale in [('_01',.1),('',1.),('_10',10.)]:basis[name+'_rbf'+suffix]=torch.exp(-d/(2*median*scale))
 for seed in [11,23,47,71]:
  torch.manual_seed(seed);model=GCN(data.num_features,128,128,num_layer=2).eval().requires_grad_(False);fp32={n:r.digest(v) for n,v in model.state_dict().items()};model=model.double();before={n:r.digest(v) for n,v in model.state_dict().items()};blocks=[[] for _ in range(6)]
  with torch.no_grad():
   for batch in DataLoader(graphs,batch_size=64,shuffle=False):
    hs=model(batch.x.double(),batch.edge_index,prompt_type='DAGPrompT')[1:]
    for dst,value in zip(blocks,[v for h in hs for v in stats(h,batch.batch)]):dst.append(value)
  assert before=={n:r.digest(v) for n,v in model.state_dict().items()};name='random'+str(seed);joined=[torch.cat(b) for b in blocks];basis[name+'_mean']=r.featurek(joined[3]);basis[name+'_multi']=r.normk(sum(r.featurek(b) for b in joined)/6)
  prov[name]={'fp32_init_state':fp32,'float64_state':before,'frozen':True,'zero_std_graphs_by_layer':[int((joined[i].norm(dim=1)==0).sum()) for i in [2,5]]}
 basis['RH']=r.normk(sum(basis[n+'_mean'] for n in r.RANDOM)/4);basis['RM']=r.normk(sum(basis[n+'_multi'] for n in r.RANDOM)/4)
 torch.save(basis,root/(ds+'_kernels.pt'));r.save(root/(ds+'_provenance.json'),{'kernel_sha':{n:r.digest(k) for n,k in basis.items()},'weights':prov,'precision':'FP32 initialized weights promoted unchanged to float64; deterministic single-thread CPU encoding','variance':'centered two-pass; std <= 1e-10 * per-channel RMS set to zero','source_sha':r.sha(__file__),'model_sources':{str(p):r.sha(p) for p in (Path(prog)/'prompt_graph/model').glob('*.py')},'raw_sha':{p.name:r.sha(p) for p in Path(data.raw_dir).glob('*') if p.is_file()}})
 return basis
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='ENZYMES');ap.add_argument('--assets',required=True);ap.add_argument('--prog',default='/root/ProG');ap.add_argument('--output',required=True);a=ap.parse_args();prepare(a.dataset,a.assets,a.prog,a.output)
