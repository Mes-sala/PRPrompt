"""Download and audit additional node-classification datasets; no model selection."""
from pathlib import Path
import json,os,time,traceback
import torch
from torch_geometric.datasets import WebKB,Coauthor,WikiCS,WikipediaNetwork
import run_study as r

def main():
 out=r.ROOT/'provenance/expanded_datasets.json';records={}
 for name in ['Texas','Wisconsin','Squirrel','WikiCS','CS','Physics']:
  try:
   root=str(r.ROOT/'data'/name)
   if name in ['Texas','Wisconsin']:ds=WebKB(root,name)
   elif name in ['CS','Physics']:ds=Coauthor(root,name)
   elif name=='WikiCS':ds=WikiCS(root)
   else:ds=WikipediaNetwork(root,'squirrel',geom_gcn_preprocess=True)
   d=ds[0];C=int(d.y.max())+1;tr=getattr(d,'train_mask',None);te=getattr(d,'test_mask',None)
   if tr is not None and te is not None:
    tr=tr[:,0] if tr.ndim==2 else tr;te=te[:,0] if te.ndim==2 else te;pool=tr.nonzero().flatten();query=te.nonzero().flatten();protocol='official_train_pool_test_query_fold0'
   else:
    gen=torch.Generator().manual_seed(20260924);pool=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:20]] for c in range(C) for ix in [(d.y==c).nonzero().flatten()]]);mask=torch.ones(len(d.y),dtype=torch.bool);mask[pool]=False;query=mask.nonzero().flatten();protocol='fixed20_perclass_pool_all_remaining_query'
   counts=torch.bincount(d.y[pool],minlength=C);assert counts.min()>=5 and not torch.isin(pool,query).any();assert torch.isfinite(d.x).all()
   dest=r.ROOT/'data'/name/'canonical.pt';torch.save(dict(x=d.x,edge_index=d.edge_index,y=d.y,pool=pool,query=query),dest)
   records[name]=dict(status='ready',nodes=len(d.x),edges=d.edge_index.shape[1],features=d.x.shape[1],classes=C,pool_per_class=counts.tolist(),query=len(query),protocol=protocol,canonical_path=str(dest),sha256=r.sha(dest),input_hash=r.thash(d.x),edge_hash=r.thash(d.edge_index))
  except Exception:records[name]=dict(status='failed',traceback=traceback.format_exc())
  r.save(out,records);print(name,records[name]['status'],flush=True)
 rows=[]
 for base in ['/root/ProG/Experiment','/root/autodl-tmp']:
  for root,ds,fs in os.walk(base):
   ds[:]=[d for d in ds if d not in ['prog_env','data','.git','node_modules','results','predictions','miniconda3']]
   for f in fs:
    if f.endswith(('.pth','.ckpt')):rows.append(dict(path=str(Path(root)/f),bytes=(Path(root)/f).stat().st_size))
 r.save(r.ROOT/'provenance/expanded_checkpoint_file_inventory.json',rows);print('checkpoint files',len(rows),flush=True)
if __name__=='__main__':main()
