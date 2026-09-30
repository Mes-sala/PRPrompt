"""Chunk mean aggregation features without altering SAGE layers or graph."""
import sys,json,copy,types
from pathlib import Path
import torch
import torch.nn.functional as F
import run_study as r
sys.path.insert(0,str(r.ROOT/'source/prog_expansion'))
import prompt_graph.pretrain as pre
from torch_geometric.nn import SAGEConv
import expand_pretraining as original

def chunk_forward(self,x,edge_index,size=None):
 assert self.aggr=='mean'
 if isinstance(x,torch.Tensor):x=(x,x)
 if self.project and hasattr(self,'lin'):x=(self.lin(x[0]).relu(),x[1])
 # Destination features determine target node count; SAGE message uses x_j only.
 parts=[self.propagate(edge_index,x=(part,x[1]),size=size) for part in x[0].chunk(16,-1)]
 out=self.lin_l(torch.cat(parts,-1))
 if self.root_weight and x[1] is not None:out=out+self.lin_r(x[1])
 if self.normalize:out=F.normalize(out,p=2.,dim=-1)
 return out

def audit():
 torch.manual_seed(813);edge=torch.randint(0,73,(2,350),device='cuda');x=torch.randn(73,64,device='cuda',requires_grad=True);xx=x.detach().clone().requires_grad_(True);a=SAGEConv(64,128).cuda();b=copy.deepcopy(a);b.forward=types.MethodType(chunk_forward,b)
 ya=a(x,edge);yb=b(xx,edge);ya.square().mean().backward();yb.square().mean().backward();diff=float((ya-yb).abs().max());grad=max(float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(a.parameters(),b.parameters()));ig=float((x.grad-xx.grad).abs().max());assert torch.allclose(ya,yb,atol=2e-6,rtol=2e-5) and max(grad,ig)<2e-6
 r.save(r.ROOT/'provenance/sage_chunk_aggregation_audit.json',dict(output_max_error=diff,parameter_gradient_max_error=grad,input_gradient_max_error=ig,feature_chunks=16,scope='Mean SAGEConv chunked aggregation; forward and parameter/input backward'))
Base=pre.GraphMAE2
class ChunkedGraphMAE2(Base):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  for module in self.modules():
   if isinstance(module,SAGEConv):module.forward=types.MethodType(chunk_forward,module)
if __name__=='__main__':
 audit();pre.GraphMAE2=ChunkedGraphMAE2;job=json.loads(Path(sys.argv[1]).read_text());assert job['family']=='GraphMAE2' and job['architecture']=='GraphSAGE';original.train(job)
 f=r.ROOT/'checkpoints'/job['id']/'ready.json';m=json.loads(f.read_text());m['memory_execution']='SAGE mean aggregation feature_chunks=16';m['repair_runner_sha256']=r.sha(__file__);r.save(f,m)
