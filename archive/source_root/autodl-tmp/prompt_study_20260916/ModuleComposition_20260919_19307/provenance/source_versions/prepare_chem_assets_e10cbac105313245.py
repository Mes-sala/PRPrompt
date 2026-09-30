"""Fetch official molecular CSVs and audit a compatibility-only frozen GIN forward."""
import argparse,csv,hashlib,importlib.util,json,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem,rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data,Batch
from torch_geometric.nn import global_mean_pool
import run_graph1 as r
URLS={'BBBP':('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv','smiles','p_np'),'BACE':('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv','mol','Class')}
CHIRAL=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,Chem.rdchem.ChiralType.CHI_OTHER]
BONDS=[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]
DIRS=[Chem.rdchem.BondDir.NONE,Chem.rdchem.BondDir.ENDUPRIGHT,Chem.rdchem.BondDir.ENDDOWNRIGHT]
def graph(m):
 x=torch.tensor([[a.GetAtomicNum()-1,CHIRAL.index(a.GetChiralTag())] for a in m.GetAtoms()],dtype=torch.long);assert len(x)>0 and int(x[:,0].min())>=0 and int(x[:,0].max())<118 and int(x[:,1].max())<3
 edges=[];attrs=[]
 for b in m.GetBonds():
  i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();v=[BONDS.index(b.GetBondType()),DIRS.index(b.GetBondDir())];edges += [(i,j),(j,i)];attrs += [v,v]
 return Data(x=x,edge_index=torch.tensor(edges,dtype=torch.long).reshape(-1,2).T.contiguous(),edge_attr=torch.tensor(attrs,dtype=torch.long).reshape(-1,2))
def fixed_forward(self,x,edge_index,edge_attr):
 loops=torch.arange(len(x),device=x.device);ei=torch.cat([edge_index,torch.stack([loops,loops])],1);ea=torch.cat([edge_attr,torch.stack([torch.full_like(loops,4),torch.zeros_like(loops)],1)],0);msg=x[ei[0]]+self.edge_embedding1(ea[:,0])+self.edge_embedding2(ea[:,1]);total=torch.zeros_like(x);total.index_add_(0,ei[1],msg);return self.mlp(total)
def main(a):
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True);meta={};sample=[]
 for ds,(url,scol,lcol) in URLS.items():
  p=root/(ds+'.csv')
  if not p.exists() or p.stat().st_size==0:
   with urllib.request.urlopen(url,timeout=60) as response:payload=response.read()
   assert len(payload)>100;temp=p.with_suffix('.download');temp.write_bytes(payload);temp.replace(p)
  rows=list(csv.DictReader(p.open()));items=[];rejected=[]
  for i,row in enumerate(rows):
   m=Chem.MolFromSmiles(row[scol])
   if m is None:rejected.append({'row':i,'reason':'invalid_smiles'});continue
   try:g=graph(m)
   except (AssertionError,ValueError) as e:rejected.append({'row':i,'reason':'input_contract_'+type(e).__name__});continue
   key=Chem.MolToSmiles(m,isomericSmiles=True,canonical=True);scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False);items.append({'row':i,'canonical':key,'scaffold':scaffold,'label':int(row[lcol]),'atoms':g.num_nodes})
   if len(sample)<16:sample.append(g)
  groups={}
  for item in items:groups.setdefault(item['canonical'],[]).append(item)
  meta[ds]={'url':url,'sha256':r.sha(p),'rows':len(rows),'accepted':len(items),'rejected':rejected,'smiles_field':scol,'label_field':lcol,'classes':{str(cl):sum(v['label']==cl for v in items) for cl in [0,1]},'canonical_unique':len(groups),'conflicting_label_canonical_groups':sum(len({v['label'] for v in vs})>1 for vs in groups.values()),'scaffold_groups':len({v['scaffold'] for v in items}),'empty_scaffold':sum(v['scaffold']=='' for v in items)}
  r.save(root/(ds+'_input_inventory.json'),items)
 spec=importlib.util.spec_from_file_location('gpf_chem_model',root/'source/gpf_chem_model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.GINConv.forward=fixed_forward;model=mod.GNN(5,300,JK='last',drop_ratio=0,gnn_type='gin');state=torch.load(root/'contextpred.pth',map_location='cpu',weights_only=True);loaded=model.load_state_dict(state,strict=True);model=model.double().eval().requires_grad_(False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);before={n:r.digest(v) for n,v in model.state_dict().items()}
 batch=Batch.from_data_list(sample)
 with torch.no_grad():
  z=model(batch.x,batch.edge_index,batch.edge_attr);pooled=global_mean_pool(z,batch.batch);second=model(batch.x,batch.edge_index,batch.edge_attr);single=torch.stack([model(g.x,g.edge_index,g.edge_attr).mean(0) for g in sample]);perm=torch.randperm(len(batch.x),generator=torch.Generator().manual_seed(19308600));inverse=torch.argsort(perm);zp=model(batch.x[perm],inverse[batch.edge_index],batch.edge_attr);permerr=float((zp[inverse]-z).abs().max())
 assert before=={n:r.digest(v) for n,v in model.state_dict().items()};assert torch.equal(z,second) and torch.allclose(pooled,single,atol=1e-9,rtol=1e-9) and permerr<1e-9
 # Compare one convolution against a transparent per-target Python sum.
 conv=model.gnns[0];g=sample[0];x=model.x_embedding1(g.x[:,0])+model.x_embedding2(g.x[:,1]);ref=[]
 with torch.no_grad():
  for i in range(g.num_nodes):
   acc=x[i]+conv.edge_embedding1.weight[4]+conv.edge_embedding2.weight[0]
   for e in torch.where(g.edge_index[1]==i)[0].tolist():acc=acc+x[g.edge_index[0,e]]+conv.edge_embedding1(g.edge_attr[e,0])+conv.edge_embedding2(g.edge_attr[e,1])
   ref.append(conv.mlp(acc))
  looperr=float((torch.stack(ref)-conv(x,g.edge_index,g.edge_attr)).abs().max());assert looperr<1e-9
 report={'datasets':meta,'rdkit':rdBase.rdkitVersion,'torch':torch.__version__,'source_sha':r.sha(__file__),'model_source_sha':r.sha(root/'source/gpf_chem_model.py'),'checkpoint_sha':r.sha(root/'contextpred.pth'),'strict_load':str(loaded),'sample_graphs':len(sample),'repeat_bitwise_equal':torch.equal(z,second),'batch_vs_single_max_error':float((pooled-single).abs().max()),'node_permutation_max_error':permerr,'reference_conv_max_error':looperr,'state_unchanged':True,'scope':'Compatibility engineering only, no predictive scoring; original PyG1 GIN edge+loop sum explicitly ported. Pretraining sample overlap not yet audited.'}
 r.save(root/'preparation_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);main(ap.parse_args())
