from run_core import *
from run_full import local
import random,fcntl
ROOT=Path(__file__).resolve().parents[1]
lock=open(ROOT/'execution.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
contract=json.loads((ROOT/'plan.json').read_text())
assert all(sha(ROOT/k)==v for k,v in contract['sources'].items())
identity=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
for ds in ['Cora','CiteSeer']:
 data,pool,q,C,protocol=g.data_load(ds);seed_init=contract['initialization_seed']
 random.seed(seed_init);np.random.seed(seed_init);torch.manual_seed(seed_init);torch.cuda.manual_seed_all(seed_init)
 model=g.GCN(data.x.shape[1],128,128,num_layer=2).cuda().eval();model.requires_grad_(False)
 dest=ROOT/'checkpoints';dest.mkdir(exist_ok=True);ck=dest/f'{ds}__random_{seed_init}.pth'
 if ck.exists():
  saved=torch.load(ck,map_location='cuda',weights_only=True);assert all(torch.equal(saved[k],v) for k,v in model.state_dict().items())
 else:torch.save(model.state_dict(),ck)
 ch=sha(ck);before={k:A.thash(v) for k,v in model.state_dict().items()};out=ROOT/'results'/ds;out.mkdir(parents=True,exist_ok=True)
 x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);filters=A.make_filters(x,edges)
 gh=hashlib.sha256((A.thash(data.x)+A.thash(data.edge_index)).encode()).hexdigest()
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach()
 for seed in contract['support_seeds']:
  folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'result.json'
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda();assert not torch.isin(support,q).any();split=hashlib.sha256((A.thash(support)+A.thash(q)).encode()).hexdigest()
  if done.exists():
   old=json.loads(done.read_text());assert old['identity']==identity and old['split_sha256']==split and old['checkpoint_sha256']==ch and sha(folder/'prediction.npz')==old['prediction_sha256'];continue
  save(ROOT/'status.json',dict(dataset=ds,seed=seed,stage='running'));start=time.perf_counter()
  z,params,tr=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',100,filters,'none',.1,'raw','gshampoo',.01);z=z-z.mean(0,keepdim=True)
  with torch.no_grad():
   score=A.logits(z,si,sy,C);adapted,_,_=knn(z)
   views=dict(Prompt__localtemp=local(score,op,si,sy,C,seed),Prompt__fixed=propagate(score,op,si,sy,C),Prompt__original_M3=propagate(score,adapted,si,sy,C),Prompt__none=score,Frozen=A.logits(z0-z0.mean(0,keepdim=True),si,sy,C))
   for name,rr in [('Teacher_uncentered',r),('Teacher_centered',rc)]:
    sc=A.rectified_teacher(rr,si,sy,C,'rect_teacher_ref50',seed)[0]
    views[name+'__none']=sc;views[name+'__fixed']=propagate(sc,op,si,sy,C);views[name+'__localtemp']=local(sc,op,si,sy,C,seed)
   arrays={k:v[q.cuda()].detach().double().softmax(-1).cpu().numpy() for k,v in views.items()}
  assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None and not p.requires_grad for p in model.parameters())
  assert all(np.isfinite(p).all() and abs(p.sum(1)-1).max()<1e-10 for p in arrays.values())
  np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=q.numpy(),**arrays);ph=sha(folder/'prediction.npz');torch.save(params,folder/'parameters.pt');y=data.y[q].numpy();np.save(out/'query_labels.npy',y)
  save(done,dict(dataset=ds,seed=seed,initialization_seed=seed_init,pretraining_steps=0,steps=100,identity=identity,checkpoint_sha256=ch,graph_sha256=gh,split_sha256=split,prediction_sha256=ph,frozen_verified=True,seconds=time.perf_counter()-start,metrics={k:metrics(y,p) for k,p in arrays.items()}));print(ds,seed,flush=True)
 assert len(list(out.glob('*/result.json')))==40
save(ROOT/'complete.json',dict(records=80,identity=identity,time=time.time()));save(ROOT/'status.json',dict(stage='completed',records=80))
