"""Checkpointed channel-chunked GAT aggregation, preserving attention and graph."""
import sys,json,copy,types
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter
from torch_geometric.nn import GATConv
import run_study as r
sys.path.insert(0,str(r.ROOT/'source/prog_expansion'))
import prompt_graph.pretrain as pre
import expand_pretraining as original

def chunk_propagate(self,edge_index,x,alpha,size=None):
 if x[0].shape[-1]<=512:return self._dense_propagate(edge_index,x=x,alpha=alpha,size=size)
 assert isinstance(edge_index,torch.Tensor) and edge_index.layout==torch.strided and self.flow=='source_to_target' and self.aggr=='add'
 n=len(x[1]) if x[1] is not None else (size[1] if size else len(x[0]));src,dst=edge_index
 def aggregate(part,weights):return scatter(part[src]*weights.unsqueeze(-1),dst,dim=0,dim_size=n,reduce='sum')
 outs=[]
 for part in x[0].split(256,dim=-1):
  if torch.is_grad_enabled() and (part.requires_grad or alpha.requires_grad):out=checkpoint(aggregate,part,alpha,use_reentrant=False,preserve_rng_state=False)
  else:out=aggregate(part,alpha)
  outs.append(out)
 return torch.cat(outs,dim=-1)

def patch(module):
 module._dense_propagate=module.propagate;module.propagate=types.MethodType(chunk_propagate,module)

def audit():
 torch.manual_seed(391);edge=torch.randint(0,53,(2,250),device='cuda');x=torch.randn(53,16,device='cuda',requires_grad=True);xx=x.detach().clone().requires_grad_(True);a=GATConv(16,640,heads=2,dropout=0).cuda();b=copy.deepcopy(a);patch(b)
 ya=a(x,edge);yb=b(xx,edge);ya.square().mean().backward();yb.square().mean().backward();diff=float((ya-yb).abs().max());grad=max(float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(a.parameters(),b.parameters()));ig=float((x.grad-xx.grad).abs().max());assert torch.allclose(ya,yb,atol=2e-6,rtol=2e-5) and max(grad,ig)<2e-6
 r.save(r.ROOT/'provenance/gat_chunk_checkpoint_audit.json',dict(output_max_error=diff,parameter_gradient_max_error=grad,input_gradient_max_error=ig,chunk_width=256,heads=2,scope='GAT attention unchanged; checkpointed aggregation forward and parameter/input backward'))
Base=pre.GraphMAE2
class ChunkedGraphMAE2(Base):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  for module in self.decoder.modules():
   if isinstance(module,GATConv):patch(module)
if __name__=='__main__':
 audit();pre.GraphMAE2=ChunkedGraphMAE2;job=json.loads(Path(sys.argv[1]).read_text());assert job['family']=='GraphMAE2' and job['architecture']=='GAT';original.train(job)
 f=r.ROOT/'checkpoints'/job['id']/'ready.json';m=json.loads(f.read_text());m['memory_execution']='GAT checkpointed aggregation chunk_width=256';m['repair_runner_sha256']=r.sha(__file__);r.save(f,m)
