"""Predeclared follow-up: shared teacher graph separates representation value from readout."""
from run_core import *

def run(job):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();xn=x if job['dataset'] in g.NODES[:5] else g.NormalizeFeatures()(data.clone()).x.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,xn)).detach();rc=r-r.mean(0,keepdim=True);teacher_op,teacher_ids,teacher_weights=knn(rc);filters=A.make_filters(x,edges)
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach();z0-=z0.mean(0,keepdim=True)
 out=ROOT/'readout_confirmation'/job['id'];out.mkdir(parents=True,exist_ok=True)
 for seed in range(26092700,26092720):
  folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'result.json'
  if done.exists():
   row=json.loads(done.read_text());assert row['runner_sha256']==sha(__file__) and row['prediction_sha256']==sha(folder/'prediction.npz');continue
  gen=torch.Generator().manual_seed(seed);si=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]).cuda();sy=data.y[si.cpu()].cuda();split=hashlib.sha256((A.thash(si)+A.thash(query)).encode()).hexdigest();save(ROOT/'readout_confirmation_current.json',dict(job=job['id'],seed=seed,time=time.time()))
  z,p,t=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',100,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True)
  with torch.no_grad():
   score=A.logits(z,si,sy,C);adapted_op,_,_=knn(z);tu=A.rectified_teacher(r,si,sy,C,'rect_teacher_ref50',seed)[0];tc=A.rectified_teacher(rc,si,sy,C,'rect_teacher_ref50',seed)[0]
   scores=dict(Prompt_adapted=propagate(score,adapted_op,si,sy,C),Prompt_teachergraph=propagate(score,teacher_op,si,sy,C),Frozen_teachergraph=propagate(A.logits(z0,si,sy,C),teacher_op,si,sy,C),Teacher_uncentered=tu,Teacher_centered=tc,Teacher_uncentered_teachergraph=propagate(tu,teacher_op,si,sy,C),Teacher_centered_teachergraph=propagate(tc,teacher_op,si,sy,C),AttributeProto_teachergraph=propagate(A.logits(rc,si,sy,C),teacher_op,si,sy,C))
   arrays={k:s[query.cuda()].double().softmax(-1).cpu().numpy() for k,s in scores.items()}
  np.savez_compressed(folder/'prediction.npz',support=si.cpu().numpy(),query=query.numpy(),**arrays);ph=sha(folder/'prediction.npz');torch.save({'tokens':p['tokens']},folder/'parameters.pt');assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
  y=data.y[query].numpy();np.save(out/'query_labels.npy',y);row=dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],seed=seed,split_sha256=split,runner_sha256=sha(__file__),prediction_sha256=ph,teacher_graph_ids_sha256=A.thash(teacher_ids),teacher_graph_weights_sha256=A.thash(teacher_weights),frozen_verified=True,metrics={k:metrics(y,p) for k,p in arrays.items()});save(done,row);print(json.dumps(dict(job=job['id'],seed=seed,metrics={k:v['accuracy'] for k,v in row['metrics'].items()})),flush=True)
 save(out/'complete.json',dict(expected=20,count=len(list(out.glob('*/result.json'))),finished=time.time()))
if __name__=='__main__':
 lock=json.loads((ROOT/'readout_confirmation_plan.json').read_text());assert lock['runner_sha256']==sha(__file__)
 for job in json.loads((ROOT/'plan.json').read_text())['jobs']:run(job)
 save(ROOT/'readout_confirmation_complete.json',dict(finished=time.time(),fits=420,prediction_views=3360))
