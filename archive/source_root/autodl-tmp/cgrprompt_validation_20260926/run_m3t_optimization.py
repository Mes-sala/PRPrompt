"""M3-T support-only readout search; no new learnable module or query-based selection."""
from run_core import *
KS=[10,25,50,100];ALPHAS=[.5,.9,.95]
CONFIGS=[dict(k=0,alpha=0.)]+[dict(k=k,alpha=a) for k in KS for a in ALPHAS]
SOURCES=['Prompt','Teacher_uncentered','Teacher_centered','AttributeProto','Frozen']

def knn_k(z,k):
 zn=F.normalize(z,dim=1);ids=[];weights=[]
 for st in range(0,len(z),512):
  en=min(st+512,len(z));sim=zn[st:en]@zn.T;sim[torch.arange(en-st,device=z.device),torch.arange(st,en,device=z.device)]=-torch.inf
  v,j=sim.topk(min(k,len(z)-1),dim=1);ids.append(j);weights.append((v/.2).softmax(1))
 ids=torch.cat(ids);weights=torch.cat(weights);rows=torch.arange(len(z),device=z.device)[:,None].expand_as(ids)
 op=torch.sparse_coo_tensor(torch.stack([rows.flatten(),ids.flatten()]),weights.flatten(),(len(z),len(z))).coalesce()
 return op,dict(ids_sha256=A.thash(ids),weights_sha256=A.thash(weights))

@torch.no_grad()
def initial(z,z0,r,rc,si,sy,C,seed):
 return dict(Prompt=A.logits(z,si,sy,C),Teacher_uncentered=A.rectified_teacher(r,si,sy,C,'rect_teacher_ref50',seed)[0],Teacher_centered=A.rectified_teacher(rc,si,sy,C,'rect_teacher_ref50',seed)[0],AttributeProto=A.logits(rc,si,sy,C),Frozen=A.logits(z0,si,sy,C))

@torch.no_grad()
def readout(score,ops,si,sy,C,config):
 return score if config['k']==0 else propagate(score,ops[config['k']],si,sy,C,alpha=config['alpha'])

def run(job,smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();xn=x if job['dataset'] in g.NODES[:5] else g.NormalizeFeatures()(data.clone()).x.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,xn)).detach();rc=r-r.mean(0,keepdim=True)
 ops={};graphhash={};sync();graphstart=time.perf_counter()
 for k in KS:ops[k],graphhash[str(k)]=knn_k(rc,k)
 sync();graphseconds=time.perf_counter()-graphstart
 if smoke:
  old,_,_=knn(rc);assert torch.equal(old.indices(),ops[50].indices()) and torch.equal(old.values(),ops[50].values())
 filters=A.make_filters(x,edges);ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach();z0-=z0.mean(0,keepdim=True)
 out=ROOT/('m3t_smoke' if smoke else 'm3t_optimization')/job['id'];out.mkdir(parents=True,exist_ok=True);seeds=list(range(26092740,26092741 if smoke else 26092760));steps=2 if smoke else 100
 for seed in seeds:
  folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'result.json'
  if done.exists():
   row=json.loads(done.read_text());assert row['runner_sha256']==sha(__file__) and row['prediction_sha256']==sha(folder/'prediction.npz');continue
  save(ROOT/'m3t_current.json',dict(job=job['id'],seed=seed,smoke=smoke,time=time.time()));gen=torch.Generator().manual_seed(seed);chunks=[p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]];support=torch.cat(chunks);train=torch.cat([p[:4] for p in chunks]);valid=torch.cat([p[4:] for p in chunks]);assert not torch.isin(support,query).any();si=support.cuda();sy=data.y[support].cuda();ti=train.cuda();ty=data.y[train].cuda();vi=valid.cuda();vy=data.y[valid].cuda();split=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest()
  sync();start=time.perf_counter();z,_,_=A.optimizer_prompt(u,r,adj,model,edges,ti,ty,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True);inner=initial(z,z0,r,rc,ti,ty,C,seed);trials={};selected={}
  for source in SOURCES:
   trials[source]=[]
   for config in CONFIGS:
    score=readout(inner[source],ops,ti,ty,C,config);ce=float(F.cross_entropy(score[vi],vy));assert np.isfinite(ce);trials[source].append(dict(**config,validation_ce=ce))
   selected[source]=min(trials[source],key=lambda t:(t['validation_ce'],t['alpha'],t['k']))
  sync();selection_seconds=time.perf_counter()-start;save(folder/'selection.json',dict(inner_train=train.tolist(),inner_valid=valid.tolist(),trials=trials,selected=selected,query_labels_used=False,tie_rule='CE, then smaller alpha, then smaller k'));del inner
  sync();start=time.perf_counter();z,p,t=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True);scores=initial(z,z0,r,rc,si,sy,C,seed);arrays={}
  with torch.no_grad():
   for source in SOURCES:
    for variant,config in [('fixed',dict(k=50,alpha=.9)),('selected',selected[source])]:arrays[source+'_'+variant]=readout(scores[source],ops,si,sy,C,config)[query.cuda()].double().softmax(-1).cpu().numpy()
   adapted,_,_=knn(z);arrays['Prompt_original_M3']=propagate(scores['Prompt'],adapted,si,sy,C)[query.cuda()].double().softmax(-1).cpu().numpy()
   for source in ['Prompt','Teacher_uncentered','Teacher_centered']:arrays[source+'_no_propagation']=scores[source][query.cuda()].double().softmax(-1).cpu().numpy()
  sync();final_seconds=time.perf_counter()-start;assert len(arrays)==14;np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=query.numpy(),**arrays);ph=sha(folder/'prediction.npz');torch.save(dict(tokens=p['tokens']),folder/'parameters.pt');assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
  y=data.y[query].numpy();np.save(out/'query_labels.npy',y);row=dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],seed=seed,steps=steps,smoke=smoke,split_sha256=split,runner_sha256=sha(__file__),core_runner_sha256=sha(ROOT/'run_core.py'),prediction_sha256=ph,frozen_verified=True,selection_query_label_free=True,selection_seconds=selection_seconds,final_seconds=final_seconds,graph_construction_seconds_once_per_job=graphseconds,graph_hashes=graphhash,selected=selected,metrics={k:metrics(y,p) for k,p in arrays.items()});save(done,row);print(json.dumps(dict(job=job['id'],seed=seed,accuracy={k:v['accuracy'] for k,v in row['metrics'].items()})),flush=True)
 save(out/'complete.json',dict(expected=len(seeds),count=len(list(out.glob('*/result.json'))),finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();lock=json.loads((ROOT/'m3t_optimization_plan.json').read_text());assert lock['runner_sha256']==sha(__file__) and lock['core_runner_sha256']==sha(ROOT/'run_core.py');jobs=json.loads((ROOT/'plan.json').read_text())['jobs']
 for job in jobs[:1] if args.smoke else jobs:run(job,args.smoke)
 save(ROOT/('m3t_smoke_complete.json' if args.smoke else 'm3t_optimization_complete.json'),dict(finished=time.time(),fits=1 if args.smoke else 420,prediction_views=14 if args.smoke else 5880))
