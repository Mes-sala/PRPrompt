"""Support-only inner holdout selection; 4 train +1 validation per class, refit on all5."""
from run_core import *
from allocation import fit
import inspect
import prompt_graph.tasker.unified_comparison as native
# Explicit, recorded adapter: only adds a learning-rate argument to archived native runner.
code=inspect.getsource(native.train_baseline).replace('def train_baseline(method,model,data,support,sy,query,C,seed,steps,caches):','def train_baseline_lr(method,model,data,support,sy,query,C,seed,steps,caches,lr_override):').replace('head=None;lr=.001;wd=1e-5','head=None;lr=lr_override;wd=1e-5')
namespace=dict(native.__dict__);exec(compile(code,'native_learning_rate_adapter','exec'),namespace);native_lr=namespace['train_baseline_lr']
CONFIGS=[('fixed_adam','fixed','adam'),('fixed_gshampoo','fixed','gshampoo'),('fixed_ungrafted','fixed','shampoo'),('learned_hidden','learned_hidden','adam'),('learned_teacher','learned_teacher','adam'),('GPF+','native','GPF+'),('GPrompt','native','GPrompt')]

def run(job,smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach()
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 safe=g.Data(x=data.x,edge_index=data.edge_index,num_nodes=data.num_nodes)
 out=ROOT/('tuning_smoke' if smoke else 'tuning')/job['id'];out.mkdir(parents=True,exist_ok=True)
 budgets=[2,3] if smoke else [100,300,1000];rates=[.01] if smoke else [.001,.01,.1];seeds=job['seeds'][:1] if smoke else job['seeds'];caches={'induced':native.graph_cache(safe,torch.cat([pool,query]).unique(),'induced')}
 for seed in seeds:
  gen=torch.Generator().manual_seed(seed);chunks=[p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]];support=torch.cat(chunks);train=torch.cat([p[:4] for p in chunks]);valid=torch.cat([p[4:] for p in chunks]);sy=data.y[support].cuda();ty=data.y[train].cuda();vy=data.y[valid].cuda();split=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest()
  for name,route,opt in CONFIGS:
   folder=out/f'{seed}__{name}';folder.mkdir(exist_ok=True);done=folder/'result.json'
   if done.exists():
    old=json.loads(done.read_text());assert old['runner_sha256']==sha(__file__) and old['operator_sha256']==sha(ROOT/'allocation.py') and old['prediction_sha256']==sha(folder/'prediction.npz');continue
   save(ROOT/'tuning_current.json',dict(job=job['id'],seed=seed,method=name,time=time.time()));trials=[]
   for lr in rates:
    trialfile=folder/f'trials_{lr}.json'
    if trialfile.exists():
     cached=json.loads(trialfile.read_text());assert cached['runner_sha256']==sha(__file__);trials+=cached['trials'];continue
    rows=[]
    if route!='native':
     snaps=fit(u,r,adj,model,edges,train.cuda(),ty,C,seed,route,opt,lr,max(budgets),budgets)
     for step in budgets:
      z,t=snaps[step]
      with torch.no_grad():op,_,_=knn(z);s=propagate(A.logits(z,train.cuda(),ty,C),op,train.cuda(),ty,C)[valid.cuda()];loss=float(F.cross_entropy(s,vy));acc=float((s.argmax(-1)==vy).float().mean())
      rows.append(dict(lr=lr,steps=step,validation_ce=loss,validation_accuracy=acc,fit_seconds=t['seconds'],train_loss=t['loss']))
    else:
     for step in budgets:
      sync();start=time.perf_counter();s,p,t=native_lr(opt,model,safe,train,ty,valid,C,seed,step,caches,lr);sync();loss=float(F.cross_entropy(s[valid.cuda()],vy));acc=float((s[valid.cuda()].argmax(-1)==vy).float().mean());rows.append(dict(lr=lr,steps=step,validation_ce=loss,validation_accuracy=acc,fit_seconds=time.perf_counter()-start,train_loss=t['training_loss'][-1]))
    assert all(np.isfinite(row['validation_ce']) for row in rows);save(trialfile,dict(runner_sha256=sha(__file__),trials=rows));trials+=rows
   selected=min(trials,key=lambda t:(t['validation_ce'],t['steps'],t['lr']));save(folder/'selection.json',dict(rule='minimum support holdout CE; tie smaller budget then LR',selected=selected,trials=trials,inner_train=train.tolist(),inner_valid=valid.tolist(),query_labels_used=False))
   torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter();lr=selected['lr'];steps=selected['steps']
   if route!='native':
    z,t=fit(u,r,adj,model,edges,support.cuda(),sy,C,seed,route,opt,lr,steps,(steps,))[steps]
    with torch.no_grad():op,_,_=knn(z);scores=propagate(A.logits(z,support.cuda(),sy,C),op,support.cuda(),sy,C)
    torch.save(dict(prompt=t.pop('prompt'),router=t.pop('router')),folder/'parameters.pt');parameters=t['parameters']
   else:
    scores,p,t=native_lr(opt,model,safe,support,sy,query,C,seed,steps,caches,lr);torch.save(p,folder/'parameters.pt');parameters=t['trainable_parameters']
   sync();elapsed=time.perf_counter()-start;probs=scores[query.cuda()].detach().cpu().double().softmax(-1).numpy();np.savez_compressed(folder/'prediction.npz',probabilities=probs,support=support.numpy(),query=query.numpy());ph=sha(folder/'prediction.npz')
   assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
   y=data.y[query].numpy();np.save(out/'query_labels.npy',y);row=dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],seed=seed,method=name,steps=steps,lr=lr,parameters=parameters,smoke=smoke,selection_query_label_free=True,frozen_verified=True,split_sha256=split,runner_sha256=sha(__file__),operator_sha256=sha(ROOT/'allocation.py'),prediction_sha256=ph,refit_seconds=elapsed,tuning_fit_seconds=(sum(max(t['fit_seconds'] for t in trials if t['lr']==rate) for rate in rates) if route!='native' else sum(t['fit_seconds'] for t in trials)),peak_gpu_bytes=torch.cuda.max_memory_allocated(),**metrics(y,probs));save(done,row);print(json.dumps(dict(job=job['id'],seed=seed,method=name,steps=steps,lr=lr,accuracy=row['accuracy'])),flush=True)
 save(out/'complete.json',dict(expected=len(seeds)*len(CONFIGS),count=len(list(out.glob('*/result.json'))),finished=time.time()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');args=ap.parse_args();plan=json.loads((ROOT/'plan.json').read_text());lock=json.loads((ROOT/'tuning_plan.json').read_text());assert lock['runner_sha256']==sha(__file__) and lock['operator_sha256']==sha(ROOT/'allocation.py')
 for j in plan['jobs']:
  if j['phase']!='main' or (args.job and args.job!=j['id']):continue
  try:run(j,args.smoke)
  except Exception as e:save(ROOT/'tuning_failure.json',dict(error=repr(e),job=j['id'],traceback=traceback.format_exc()));raise
 save(ROOT/('tuning_smoke_complete.json' if args.smoke else 'tuning_complete.json'),dict(finished=time.time(),job=args.job))
