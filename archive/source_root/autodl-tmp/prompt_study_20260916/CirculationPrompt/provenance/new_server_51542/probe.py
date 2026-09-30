import json,hashlib,sys,tempfile
from pathlib import Path
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

root=Path('/root/autodl-tmp/prompt_study_20260916')
expected=json.loads((root/'CirculationPrompt/provenance/new_server_51542/expected.json').read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
sys.path.insert(0,str(root/'AnchorPrompt'))
from run import digest
sys.path.insert(0,'/root/ProG')
from prompt_graph.model import GCN
for f,h in expected['sources'].items():assert sha(f)==h,(f,'source drift')
for f,h in expected['checkpoints'].items():assert sha(root/f)==h,(f,'snapshot drift')
with tempfile.TemporaryDirectory(prefix='prompt_access_probe_',dir=root) as td:
 p=Path(td)/'probe.txt';p.write_text('read-write');assert p.read_text()=='read-write'
inventory={};tests=[]
for ds in ['Cora','CiteSeer']:
 data=Planetoid('/root/ProG/data/Planetoid',ds,transform=NormalizeFeatures())[0]
 assert digest(data.x)==expected['data'][ds]['x_sha256']
 assert digest(data.edge_index)==expected['data'][ds]['edge_sha256']
 assert sha(f'/root/ProG/data/Planetoid/{ds}/processed/data.pt')==expected['data'][ds]['processed_file_sha256']
 for f in sorted(Path(f'/root/ProG/Experiment/pre_trained_model/{ds}').glob('*.GCN.128hidden_dim.pth')):inventory[str(f)]=sha(f)
 for ck in ['DGI','GraphMAE','GraphCL','GraphMAE2']:
  path=Path(f'/root/ProG/Experiment/pre_trained_model/{ds}/{ck}.GCN.128hidden_dim.pth')
  assert sha(path)==sha(root/'CrossfitGradientPrompt/provenance'/ds/path.name)
  model=GCN(data.num_features,128,128,num_layer=2).cuda().eval().requires_grad_(False)
  model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
  x=data.x.cuda().requires_grad_(True);h=model(x,data.edge_index.cuda());g=torch.autograd.grad(h.square().mean(),x)[0]
  assert torch.isfinite(h).all() and torch.isfinite(g).all() and g.norm()>0
  tests.append({'dataset':ds,'checkpoint':ck,'shape':list(h.shape),'input_gradient_norm':float(g.norm())})
out={'passed':True,'port':51542,'python':sys.executable,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'source_hashes_match':True,'checkpoint_snapshot_hashes_match':True,'data_hashes_match':True,'temporary_readwrite_cleaned':not Path(td).exists(),'checkpoint_inventory':inventory,'smoke_tests':tests}
dest=root/'CirculationPrompt/provenance/new_server_51542/migration_audit.json'
dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
