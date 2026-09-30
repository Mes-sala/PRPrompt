"""GCN feature decomposition repair; preserves mathematical message passing."""
import sys,json,copy
from pathlib import Path
import torch
import run_study as r
sys.path.insert(0,str(r.ROOT/'source/prog_expansion'))
import prompt_graph.pretrain as pre
from torch_geometric.nn import GCNConv
import expand_pretraining as original

def audit():
 torch.manual_seed(732);edge=torch.randint(0,73,(2,350),device='cuda');x=torch.randn(73,16,device='cuda');a=GCNConv(16,128).cuda();b=copy.deepcopy(a);b.decomposed_layers=8
 ya=a(x,edge);yb=b(x,edge);la=ya.square().mean();lb=yb.square().mean();la.backward();lb.backward();diff=float((ya-yb).abs().max());grad=max(float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(a.parameters(),b.parameters()));assert torch.allclose(ya,yb,atol=2e-6,rtol=2e-5) and grad<2e-6
 r.save(r.ROOT/'provenance/gcn_decomposition_audit.json',dict(output_max_error=diff,gradient_max_error=grad,decomposed_layers=8,scope='GCNConv feature decomposition, small graph forward and backward'))
Base=pre.GraphMAE2
class ChunkedGraphMAE2(Base):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  for module in self.modules():
   if isinstance(module,GCNConv):module.decomposed_layers=8
if __name__=='__main__':
 audit();pre.GraphMAE2=ChunkedGraphMAE2;job=json.loads(Path(sys.argv[1]).read_text());assert job['family']=='GraphMAE2' and job['architecture']=='GCN';original.train(job)
 f=r.ROOT/'checkpoints'/job['id']/'ready.json';m=json.loads(f.read_text());m['memory_execution']='GCNConv decomposed_layers=8';m['repair_runner_sha256']=r.sha(__file__);r.save(f,m)
