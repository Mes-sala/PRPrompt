from pathlib import Path
import sys,importlib.util,json,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'prog_src'))
from prompt_graph.tasker.unified_2026 import MTGPrompt,DAPromptStage
from prompt_graph.model import GCN
from torch_geometric.data import Data,Batch
from torch_geometric.nn import global_mean_pool

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
mtg=load('official_mtg',R/'upstream/MTG/message_tuning/MTGCN.py')
da=load('official_da',R/'upstream/DAPrompt/prompt.py')
torch.manual_seed(14);model=GCN(5,8,8,num_layer=2).double().eval();model.requires_grad_(False);a=mtg.MTGCN(5,8,8,num_layer=2,m_num=10).double().eval();b=MTGPrompt([5,8]).double()
for i,conv in enumerate(a.conv_layers):conv.load_state_dict(model.conv_layers[i].state_dict());conv.requires_grad_(False)
b.load_state_dict({k:v for k,v in a.state_dict().items() if k.startswith(('MTG_vectors','proj_vectors'))})
e=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]]);x=torch.randn(4,5,dtype=torch.float64);batch=torch.zeros(4,dtype=torch.long)
y=a(x,e,batch);z=b.embed(model,x,e,batch);torch.testing.assert_close(y,z)
ga=torch.autograd.grad(y.square().sum(),list(a.MTG_vectors)+list(a.proj_vectors.parameters()));gb=torch.autograd.grad(z.square().sum(),list(b.MTG_vectors)+list(b.proj_vectors.parameters()))
for u,v in zip(ga,gb):torch.testing.assert_close(u,v)
# Original DAPrompt graph construction versus torch implementation on the same prompt.
model=model.cuda().float();p=DAPromptStage(5,3).cuda();ref=da.DAPrompt(encoder=model,structure_token_number=2,semantics_token_number=3,input_dim=5,hid_dim=8,device='cuda',outer_thre=.2,lamda1=.5,lamda2=.5,structure_threshold=.8,broadcast=True)
ref.structure_prompt.data.copy_(p.structure_prompt.data);ref.semantics_prompt.data.copy_(p.semantics_prompt.data)
g=Data(x=x.float().cuda(),edge_index=e.cuda(),node_ids=torch.arange(4,device='cuda'),index=1,center_id=1);bch=Batch.from_data_list([g]);a=ref.create_manipulated_graph(bch);b=p.manipulate(bch)
torch.testing.assert_close(a.x,b.x);torch.testing.assert_close(a.edge_index,b.edge_index)
y=global_mean_pool(torch.relu(model(a.x,a.edge_index)),a.batch);z=p.embed(model,bch);torch.testing.assert_close(y,z)
la=.5*ref.prompt_diversity_loss(ref.structure_prompt)+.5*ref.prompt_diversity_loss(ref.semantics_prompt);lb=p.diversity();torch.testing.assert_close(la,lb)
ga=torch.autograd.grad(y.square().sum()+la,[ref.structure_prompt,ref.semantics_prompt]);gb=torch.autograd.grad(z.square().sum()+lb,[p.structure_prompt,p.semantics_prompt])
for u,v in zip(ga,gb):torch.testing.assert_close(u,v)
report={'passed':True,'MTG_official_forward_gradient':True,'DAPrompt_official_manipulated_edges_features_diversity_forward_gradient':True,'scope':'same frozen encoder; dedicated DAPrompt pretraining excluded and explicitly labelled'};(R/'operator_audit.json').write_text(json.dumps(report,indent=2));print(report)
