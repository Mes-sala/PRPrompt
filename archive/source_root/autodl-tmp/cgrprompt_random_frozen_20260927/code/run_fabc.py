from run_core import *
from run_full import local
import fcntl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'fabc';OUT.mkdir(exist_ok=True)
lock=open(OUT/'execution.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
plan=json.loads((OUT/'plan.json').read_text());assert all(sha(ROOT/k)==v for k,v in plan['sources'].items());identity=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
for ds in ['Cora','CiteSeer']:
 data,pool,q,C,protocol=g.data_load(ds);ck=ROOT/'checkpoints'/f'{ds}__random_20260927.pth';ch=sha(ck);assert ch==plan['checkpoints'][ds]
 model=g.GCN(data.x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();op,_,_=knn(r-r.mean(0,keepdim=True));filters=A.make_filters(x,edges);gh=hashlib.sha256((A.thash(data.x)+A.thash(data.edge_index)).encode()).hexdigest()
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 for seed in plan['support_seeds']:
  folder=OUT/'results'/ds/str(seed);folder.mkdir(parents=True,exist_ok=True);oldfolder=ROOT/'results'/ds/str(seed);old=json.loads((oldfolder/'result.json').read_text());previous=np.load(oldfolder/'prediction.npz')
  assert sha(oldfolder/'prediction.npz')==old['prediction_sha256'] and old['checkpoint_sha256']==ch and old['graph_sha256']==gh and old['frozen_verified']
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);assert np.array_equal(support.numpy(),previous['support']) and np.array_equal(q.numpy(),previous['query']);si=support.cuda();sy=data.y[support].cuda();split=old['split_sha256']
  if (folder/'result.json').exists():
   done=json.loads((folder/'result.json').read_text());assert done['identity']==identity and done['checkpoint_sha256']==ch and done['split_sha256']==split and sha(folder/'prediction.npz')==done['prediction_sha256'];continue
  save(OUT/'status.json',dict(dataset=ds,seed=seed,stage='running'));start=time.perf_counter();arrays={'F110':previous['Prompt__none'],'F111':previous['Prompt__localtemp']}
  for sharp,matrix in [(0,0),(0,1),(1,0)]:
   z,params,tr=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',100,filters,'none',.1 if sharp else .2,'raw','gshampoo' if matrix else 'adam',.01);z=z-z.mean(0,keepdim=True)
   with torch.no_grad():
    score=A.logits(z,si,sy,C);views={f'F{sharp}{matrix}0':score,f'F{sharp}{matrix}1':local(score,op,si,sy,C,seed)}
    arrays.update({k:v[q.cuda()].detach().double().softmax(-1).cpu().numpy() for k,v in views.items()})
   torch.save(params,folder/f'parameters_F{sharp}{matrix}.pt')
  assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None and not p.requires_grad for p in model.parameters())
  assert set(arrays)=={f'F{i:03b}' for i in range(8)} and all(np.isfinite(v).all() and abs(v.sum(1)-1).max()<1e-10 for v in arrays.values())
  np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=q.numpy(),**arrays);ph=sha(folder/'prediction.npz');y=data.y[q].numpy()
  save(folder/'result.json',dict(dataset=ds,seed=seed,identity=identity,checkpoint_sha256=ch,graph_sha256=gh,split_sha256=split,prediction_sha256=ph,frozen_verified=True,steps=100,pretraining_steps=0,reused=['F110','F111'],reused_prediction_sha256=old['prediction_sha256'],reused_result_sha256=sha(oldfolder/'result.json'),seconds=time.perf_counter()-start,metrics={k:metrics(y,v) for k,v in arrays.items()}));print(ds,seed,flush=True)
assert len(list((OUT/'results').glob('*/*/result.json')))==80
save(OUT/'complete.json',dict(records=80,prediction_views=640,new_fits=240,reused_fits=80,identity=identity,time=time.time()));save(OUT/'status.json',dict(stage='completed'))
