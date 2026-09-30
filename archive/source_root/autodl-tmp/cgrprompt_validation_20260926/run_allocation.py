from run_core import *
from allocation import fit

def run(job,smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();xn=x if job['dataset'] in g.NODES[:5] else g.NormalizeFeatures()(data.clone()).x.cuda()
 ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,xn)).detach()
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 out=ROOT/('allocation_smoke' if smoke else 'allocation')/job['id'];out.mkdir(parents=True,exist_ok=True)
 configs=[('fixed','adam'),('fixed','gshampoo'),('learned_hidden','adam'),('learned_teacher','adam')]
 if job['phase']=='main':configs += [('permuted','adam'),('random','adam'),('learned_budget','adam'),('fixed_rank','adam')]
 for seed in job['seeds'][:1] if smoke else job['seeds']:
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda();split=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest();steps=2 if smoke else 100
  for route,opt in configs:
   name=route+'__'+opt;folder=out/f'{seed}__{name}';folder.mkdir(exist_ok=True)
   if (folder/'result.json').exists():
    old=json.loads((folder/'result.json').read_text());assert old['operator_sha256']==sha(ROOT/'allocation.py') and old['runner_sha256']==sha(__file__) and old['prediction_sha256']==sha(folder/'prediction.npz');continue
   save(ROOT/'allocation_current.json',dict(job=job['id'],seed=seed,method=name,time=time.time()));torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
   z,trace=fit(u,r,adj,model,edges,si,sy,C,seed,route,opt,.01,steps,(steps,))[steps]
   if smoke and route=='fixed':
    zz,pp,tt=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,A.make_filters(x,edges),'none',.1,'raw',opt,.01);zz-=zz.mean(0,keepdim=True)
    err=float((z-zz).abs().max());assert torch.allclose(z,zz,atol=5e-5,rtol=1e-5),(name,err);trace['source_equivalence_max_error']=err
   with torch.no_grad():
    score=A.logits(z,si,sy,C);op,ix,weights=knn(z);prop=propagate(score,op,si,sy,C)
   sync();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();probs=score[query.cuda()].double().softmax(-1).cpu().numpy();pprop=prop[query.cuda()].double().softmax(-1).cpu().numpy()
   np.savez_compressed(folder/'prediction.npz',probabilities=probs,propagated_probabilities=pprop,support=support.numpy(),query=query.numpy());ph=sha(folder/'prediction.npz')
   torch.save(dict(prompt=trace.pop('prompt'),router=trace.pop('router')),folder/'parameters.pt');save(folder/'trace.json',trace)
   assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and all(p.grad is None for p in model.parameters()) and not model.training
   y=data.y[query].numpy();np.save(out/'query_labels.npy',y)
   row=dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],seed=seed,method=name,steps=steps,phase=job['phase'],smoke=smoke,split_sha256=split,runner_sha256=sha(__file__),operator_sha256=sha(ROOT/'allocation.py'),prediction_sha256=ph,frozen_verified=True,parameters=trace['parameters'],routing_parameters=trace['routing_parameters'],seconds=elapsed,fit_seconds=trace['seconds'],peak_gpu_bytes=peak,prototype_metrics=metrics(y,probs),propagated_metrics=metrics(y,pprop));save(folder/'result.json',row);print(json.dumps(dict(job=job['id'],seed=seed,method=name,accuracy=row['propagated_metrics']['accuracy'],seconds=elapsed)),flush=True)
 save(out/'complete.json',dict(expected=len(configs)*(1 if smoke else len(job['seeds'])),count=len(list(out.glob('*/result.json'))),finished=time.time()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');args=ap.parse_args();plan=json.loads((ROOT/'plan.json').read_text());lock=json.loads((ROOT/'allocation_plan.json').read_text());assert lock['runner_sha256']==sha(__file__) and lock['operator_sha256']==sha(ROOT/'allocation.py')
 for j in plan['jobs']:
  if args.job and args.job!=j['id']:continue
  try:run(j,args.smoke)
  except Exception as e:save(ROOT/'allocation_failure.json',dict(error=repr(e),job=j['id'],traceback=traceback.format_exc()));raise
 save(ROOT/('allocation_smoke_complete.json' if args.smoke else 'allocation_complete.json'),dict(finished=time.time(),job=args.job))
