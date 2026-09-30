"""Diagnostic only: repeated raw cosine readout, float32 versus float64 arithmetic."""
import sys,json,sqlite3,zlib
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
import run_study as r
import expanded_data as data
uid=sys.argv[1];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
u=next(x for x in json.loads((r.ROOT/'results/expanded_grid_dispatch/coverage_verified.json').read_text())['units'] if x['id']==uid)
d,x,adj,pool,query,C,protocol=data.load_data(u['dataset']);model=None
if u['kind']=='frozen':
 sys.path.insert(0,u['source_root'])
 if u.get('legacy'):
  from prompt_graph.model import GCN
  model=GCN(d.x.shape[1],128,128,num_layer=2).cuda().eval()
 else:
  from prompt_graph.model import build_gnn
  model=build_gnn(u['architecture'],d.x.shape[1],128,2).cuda().eval()
 model.load_state_dict(torch.load(u['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False)
con=sqlite3.connect(r.ROOT/'results'/('default_anchor_'+uid)/(uid+'.sqlite'));support=torch.tensor(json.loads(con.execute('SELECT support FROM splits WHERE seed=271400').fetchone()[0]),device='cuda');sy=d.y[support.cpu()].cuda();cfg='h2_raw_no_refine' if u['kind']=='attributes' else 'raw_no_refine';gold=np.frombuffer(zlib.decompress(con.execute('SELECT prediction FROM results WHERE seed=271400 AND config=?',(cfg,)).fetchone()[0]),dtype=np.uint8);con.close();records=[];preds=[]
for repeat in range(3):
 with torch.no_grad():
  base=model(data.encoder_input(d,u),d.edge_index.cuda()) if model is not None else torch.sparse.mm(adj,torch.sparse.mm(adj,x))
  for dtype in [torch.float32,torch.float64]:
   z=F.normalize(base.to(dtype),dim=1);oh=F.one_hot(sy,C).to(dtype);mu=(oh.T@z[support])/oh.sum(0)[:,None];score=z@F.normalize(mu,dim=1).T/.2;q=score[query.cuda()];pred=q.argmax(1).cpu().numpy();top=q.topk(2,dim=1).values;margin=(top[:,0]-top[:,1]).cpu().numpy();prob_pred=q.softmax(-1).argmax(1).cpu().numpy();records.append(dict(repeat=repeat,dtype=str(dtype),base_sha=r.thash(base),score_sha=r.thash(q),accuracy=float((pred==d.y[query].numpy()).mean()*100),different_from_sealed=int((pred!=gold).sum()),softmax_argmax_changes=int((pred!=prob_pred).sum()),median_margin=float(np.median(margin)),margin_below_1e6=int((margin<1e-6).sum()),zero_margin=int((margin==0).sum()),query_nodes=len(pred)));preds.append(pred)
result=dict(unit=uid,records=records,f32_repeat_label_changes=[int((preds[0]!=preds[i]).sum()) for i in [2,4]],f64_repeat_label_changes=[int((preds[1]!=preds[i]).sum()) for i in [3,5]],scope='raw no-refinement precision diagnostic; no original result overwritten');r.save(r.ROOT/'provenance'/('precision_'+uid+'.json'),result);print(json.dumps(result))
