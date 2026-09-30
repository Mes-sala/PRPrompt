import sys,ast,types,json,torch,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'prog_src'))
from prompt_graph.tasker.unified_comparison import ExactEdgePrompt
from prompt_graph.prompt.UniPrompt import UniPrompt
from prompt_graph.prompt.GraphTOP import Rewiring,sample_edges,topology_losses
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce
from sklearn.neighbors import kneighbors_graph
from torch import nn
from torch.nn import functional as F

def classes(path,names,ns):
 tree=ast.parse(path.read_text());tree.body=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in names];exec(compile(tree,str(path),'exec'),ns);return ns
from torch_geometric.nn import MessagePassing,inits
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops,degree
ns=classes(R/'upstream/EdgePrompt/node/model.py',['GCNConv'],dict(torch=torch,nn=nn,MessagePassing=MessagePassing,Linear=Linear,inits=inits,add_self_loops=add_self_loops,degree=degree))
torch.manual_seed(17);e=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]]);x=torch.randn(4,5,dtype=torch.float64);p=torch.randn(1,5,dtype=torch.float64,requires_grad=True)
a=ns['GCNConv'](5,3).double();b=GCNConv(5,3).double();b.load_state_dict(a.state_dict());a.requires_grad_(False);b.requires_grad_(False)
y=a(x,e,p);z=b(x+p,e);torch.testing.assert_close(y,z,rtol=1e-10,atol=1e-10)
ga=torch.autograd.grad(y.square().sum(),p)[0];gb=torch.autograd.grad(z.square().sum(),p)[0];torch.testing.assert_close(ga,gb,rtol=1e-10,atol=1e-10)
ns=classes(R/'upstream/UniPrompt/model/UniPrompt.py',['UniPrompt'],dict(torch=torch,np=np,nn=nn,F=F,kneighbors_graph=kneighbors_graph,coalesce=coalesce))
x=torch.randn(9,5);a=ns['UniPrompt'](x,3,'cosine',6,9);b=UniPrompt(x,3,'cosine',6,9);b.edge_weight.data.copy_(a.edge_weight.data);torch.testing.assert_close(a.base_edge_index,b.base_edge_index);ia,wa=e,torch.ones(e.shape[1]);ib,wb=ia,wa
for t in range(3):
 pa,va=a();pb,vb=b();ia,wa=a.edge_fuse(ia,wa,pa,va,.99);ib,wb=b.edge_fuse(ib,wb,pb,vb,.99)
 torch.testing.assert_close(ia,ib);torch.testing.assert_close(wa,wb)
 torch.testing.assert_close(torch.autograd.grad(wa.square().sum(),a.edge_weight)[0],torch.autograd.grad(wb.square().sum(),b.edge_weight)[0])
ns=classes(R/'upstream/GraphTOP/prompt.py',['Rewiring'],dict(torch=torch,nn=nn,F=F));a=ns['Rewiring'](5);b=Rewiring(5);b.load_state_dict(a.state_dict());x=torch.randn(7,5,requires_grad=True);torch.testing.assert_close(a(x),b(x));torch.testing.assert_close(torch.autograd.grad(a(x).sum(),x)[0],torch.autograd.grad(b(x).sum(),x)[0])
p=torch.sigmoid(torch.randn(7));torch.manual_seed(18);u0,u1=torch.rand_like(p),torch.rand_like(p);eps=1e-10;noise=-torch.log(torch.log(u1+eps)/torch.log(u0+eps)+eps);expected=torch.sigmoid((torch.log(p+eps)-torch.log(1-p+eps)+noise)/.7);torch.manual_seed(18);torch.testing.assert_close(expected,sample_edges(p,.7,True))
receipt={'passed':True,'EdgePrompt_official_conv_forward_gradient':True,'UniPrompt_official_three_step_recurrence_forward_gradient':True,'GraphTOP_official_rewiring_forward_gradient_and_sampling':True}
(R/'operator_audit.json').write_text(json.dumps(receipt,indent=2));print(receipt)
