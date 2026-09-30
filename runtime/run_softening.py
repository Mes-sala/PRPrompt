"""PRPrompt basic M3: fixed adapted kNN, change only initial probability temperature."""
import os,sys,json,time,hashlib,argparse,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent; BASE=ROOT/'base'
sys.path.insert(0,str(BASE))
import grid as g
A=g.a3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):g.dump(p,x)
def metrics(y,p):
 pred=p.argmax(1);oh=np.eye(p.shape[1])[y]
 return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),micro_f1=float(f1_score(y,pred,average='micro',zero_division=0)),roc_auc_macro_ovr=float(roc_auc_score(oh,p,average='macro')),auprc_macro=float(average_precision_score(oh,p,average='macro')))
def knn(z):
 zn=F.normalize(z,dim=1);ix=[];ww=[]
 for st in range(0,len(z),1024):
  en=min(st+1024,len(z));s=zn[st:en]@zn.T;s[torch.arange(en-st,device=z.device),torch.arange(st,en,device=z.device)]=-torch.inf
  v,j=s.topk(min(50,len(z)-1),dim=1);ix.append(j);ww.append((v/.2).softmax(1))
 ix=torch.cat(ix);w=torch.cat(ww);w=w/w.sum(1,keepdim=True);rows=torch.arange(len(z),device=z.device)[:,None].expand_as(ix)
 return torch.sparse_coo_tensor(torch.stack([rows.flatten(),ix.flatten()]),w.flatten(),(len(z),len(z))).coalesce(),ix,w

def prop(p0,op,si,sy,C):
 p=p0.clone();target=F.one_hot(sy,C).float();p[si]=target
 for _ in range(20):
  p=(1-.9)*p0+.9*torch.sparse.mm(op,p);p[si]=target
 assert torch.equal(p[si],target);assert torch.allclose(p.sum(1),torch.ones(len(p),device=p.device),atol=3e-6)
 # Match historical log followed by double precision softmax for stored probabilities.
 return p.clamp_min(1e-12).log()

def run(job,smoke):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 plan=json.loads((ROOT/'plan.json').read_text());assert sha(__file__)==plan['runner_sha256']
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 cp=ROOT/'checkpoints'/f"{job['id']}.pth";assert sha(cp)==job['checkpoint']['sha256']
 out=ROOT/('smoke' if smoke else 'results')/job['id'];out.mkdir(parents=True,exist_ok=True)
 data=g.Planetoid(str(ROOT/'data/Planetoid'),job['dataset'],transform=g.NormalizeFeatures())[0]
 pool=data.train_mask.nonzero().flatten();query=data.test_mask.nonzero().flatten();C=int(data.y.max())+1
 assert not torch.isin(pool,query).any()
 x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True)
 adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();filters=A.make_filters(x,edges)
 model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(cp,weights_only=True,map_location='cpu'));model.requires_grad_(False)
 before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 graphhash=hashlib.sha256((A.thash(data.x)+A.thash(data.edge_index)).encode()).hexdigest()
 seeds=job['seeds'][:1] if smoke else job['seeds'];steps=2 if smoke else 100
 for seed in seeds:
  folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'result.json'
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda()
  assert not torch.isin(support,query).any();assert torch.all(torch.bincount(sy,minlength=C)==5)
  split=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest()
  if done.exists():
   old=json.loads(done.read_text());assert old['runner_sha256']==sha(__file__) and old['split_sha256']==split and old['prediction_sha256']==sha(folder/'sealed.npz');continue
  dump(out/'current.json',dict(seed=seed,status='training',time=time.time(),pid=os.getpid()))
  start=time.time();torch.cuda.reset_peak_memory_stats()
  z,p,trace=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw','gshampoo',.01)
  z=z-z.mean(0,keepdim=True);fit=time.time()-start
  with torch.no_grad():
   score=A.logits(z,si,sy,C);op,ix,w=knn(z);initial={};scores={};diagnostics={}
   assert not (ix==torch.arange(len(z),device=z.device)[:,None]).any()
   for temp in plan['temperatures']:
    key=f'T{temp:.1f}';scaled=score*(.2/temp);p0=scaled.softmax(-1);initial[key]=p0
    assert torch.equal(p0.argmax(1),score.argmax(1))
    scores[key]=prop(p0,op,si,sy,C)
    diagnostics[key]=dict(initial_entropy=float((-(p0*p0.clamp_min(1e-12).log()).sum(1))[query.cuda()].mean()),initial_max_probability=float(p0[query.cuda()].max(1).values.mean()))
   scores['no_propagation']=score
   # On first seed of every job, compare complete baseline to untouched historical readout.
   if seed==seeds[0]:
    ref,_=A.diffusion_readout(z,si,sy,C,'qeps-50',seed,filters['plain'],{},folder/'parity')
    err=float((ref.softmax(-1)-scores['T0.2'].softmax(-1)).abs().max());assert err<2e-6,err
   else:err=None
   for i,t in enumerate(plan['temperatures'][1:],1):assert diagnostics[f'T{t:.1f}']['initial_entropy']>=diagnostics[f'T{plan["temperatures"][i-1]:.1f}']['initial_entropy']-1e-6
   probs={k:v[query.cuda()].detach().cpu().double().softmax(-1).numpy() for k,v in scores.items()}
  assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(v.grad is None for v in model.parameters())
  np.savez_compressed(folder/'sealed.npz',support=support.numpy(),query=query.numpy(),**probs)
  ph=sha(folder/'sealed.npz')
  torch.save(dict(z=z.cpu(),prompt=p,support=support,query=query,neighbor_indices=ix.cpu(),neighbor_weights=w.cpu(),base_logits=score.cpu()),folder/'state.pt')
  # Query labels enter only after all candidates for this seed have been sealed.
  y=data.y[query].numpy();np.save(out/'query_labels.npy',y)
  rows={k:metrics(y,v) for k,v in probs.items()};basepred=probs['T0.2'].argmax(1)
  for key in diagnostics:
   pred=probs[key].argmax(1);diagnostics[key].update(corrected_vs_base=int(((basepred!=y)&(pred==y)).sum()),broken_vs_base=int(((basepred==y)&(pred!=y)).sum()))
  row=dict(job=job['id'],dataset=job['dataset'],checkpoint=job['checkpoint']['family'],seed=seed,smoke=smoke,steps=steps,metrics=rows,diagnostics=diagnostics,split_sha256=split,graph_sha256=graphhash,checkpoint_sha256=sha(cp),runner_sha256=sha(__file__),prediction_sha256=ph,state_sha256=sha(folder/'state.pt'),frozen_verified=True,readout_parity_max_error=err,fit_seconds=fit,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda)
  dump(done,row);print(json.dumps(dict(job=job['id'],seed=seed,accuracy={k:v['accuracy'] for k,v in rows.items()},seconds=row['seconds'])),flush=True)
  del z,p,op,ix,w,scores,initial;torch.cuda.empty_cache()
 assert len(list(out.glob('*/result.json')))==len(seeds)
 dump(out/'complete.json',dict(seeds=len(seeds),predictions=len(seeds)*4,runner_sha256=sha(__file__),finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 plan=json.loads((ROOT/'plan.json').read_text());job=next(j for j in plan['jobs'] if j['id']==args.job)
 try:run(job,args.smoke)
 except Exception as e:dump(ROOT/f'failure_{args.job}_{"smoke" if args.smoke else "run"}.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
