"""Exact historical operators; additional fixed readout controls; seal before evaluation."""
import os,sys,json,time,hashlib,traceback,argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
BASE=Path('/root/autodl-tmp/unified_three_backbone_20260924')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import grid as g
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
A=g.a3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):g.dump(p,x)
def sync():torch.cuda.synchronize()
def metrics(y,p):
 C=p.shape[1];pred=p.argmax(1);oh=np.eye(C)[y]
 return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),micro_f1=float(f1_score(y,pred,average='micro',zero_division=0)),roc_auc_macro_ovr=float(roc_auc_score(oh,p,average='macro')),auprc_macro=float(average_precision_score(oh,p,average='macro')))

def knn(z):
 zn=F.normalize(z,dim=1);ix=[];ww=[]
 for st in range(0,len(z),512):
  en=min(st+512,len(z));sim=zn[st:en]@zn.T;sim[torch.arange(en-st,device=z.device),torch.arange(st,en,device=z.device)]=-torch.inf
  v,j=sim.topk(min(50,len(z)-1),dim=1);ix.append(j);ww.append((v/.2).softmax(1))
 ix=torch.cat(ix);w=torch.cat(ww);rows=torch.arange(len(z),device=z.device)[:,None].expand_as(ix)
 return torch.sparse_coo_tensor(torch.stack([rows.flatten(),ix.flatten()]),w.flatten(),(len(z),len(z))).coalesce(),ix,w

def propagate(score,op,si,sy,C,alpha=.9,clamp=True):
 p0=score.softmax(-1);p=p0.clone();target=F.one_hot(sy,C).float()
 if clamp:p[si]=target
 for _ in range(20):
  p=(1-alpha)*p0+alpha*torch.sparse.mm(op,p)
  if clamp:p[si]=target
 return p.clamp_min(1e-12).log()

def run(job,smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];name=job['dataset'];out=ROOT/('smoke' if smoke else 'core')/job['id'];out.mkdir(parents=True,exist_ok=True)
 data,pool,query,C,protocol=g.data_load(name);x=data.x.cuda();edges=data.edge_index.cuda();xn=x if name in g.NODES[:5] else g.NormalizeFeatures()(data.clone()).x.cuda()
 ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,xn)).detach();filters=A.make_filters(x,edges)
 model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach()
 graphhash=hashlib.sha256((A.thash(data.x)+A.thash(data.edge_index)).encode()).hexdigest();steps=2 if smoke else 100
 seeds=job['seeds'][:1] if smoke else job['seeds'];all_methods=[f'F{i:03b}' for i in range(8)]+['M1_normmatched','M1_amplitude_only','M2_diagonal','M2_ungrafted','rect_uncentered','A2_center','rawprop','frozen','frozen_prop','M3_original','M3_frozen_knn','M3_teacher_knn','M3_no_clamp','M3_no_restart']
 if job['phase']=='extension':all_methods=['F000','F111','rect_uncentered','A2_center','rawprop','frozen','frozen_prop']
 for seed in seeds:
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda();assert not torch.isin(support,query).any()
  split=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest();cache={};opcache={};probcache={}
  for method in all_methods:
   folder=out/f'{seed}__{method}';folder.mkdir(exist_ok=True);done=folder/'result.json'
   if done.exists():
    old=json.loads(done.read_text());assert old['split_sha256']==split and old['runner_sha256']==sha(__file__) and sha(folder/'prediction.npz')==old['prediction_sha256'];continue
   save(ROOT/'current.json',dict(job=job['id'],seed=seed,method=method,time=time.time(),smoke=smoke));sync();start=time.perf_counter();torch.cuda.reset_peak_memory_stats();trace={};fit_seconds=0.;params=0
   with torch.no_grad():
    if method in ['rect_uncentered','A2_center','rawprop']:
     rr=r-r.mean(0,keepdim=True) if method=='A2_center' else r
     scores=A.logits(rr,si,sy,C) if method=='rawprop' else A.rectified_teacher(rr,si,sy,C,'rect_teacher_ref50',seed)[0]
   if method not in ['rect_uncentered','A2_center','rawprop']:
    if method.startswith('frozen'):z=z0-z0.mean(0,keepdim=True);fitkey='frozen'
    else:
     key=method if method.startswith('F') or method.startswith('M1') or method.startswith('M2') else 'F111'
     if key.startswith('F'):sharp,pre,prop=map(int,key[1:]);temp=.1 if sharp else .2;mode='raw';optimizer='gshampoo' if pre else 'adam';fitkey=key[:3]
     else:temp=.1;mode={'M1_normmatched':'matched','M1_amplitude_only':'amplitude'}.get(key,'raw');optimizer={'M2_diagonal':'gdiagonal','M2_ungrafted':'shampoo'}.get(key,'gshampoo');fitkey=key
     if fitkey not in cache:
      sync();ft=time.perf_counter();zz,p,t=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',temp,mode,optimizer,.01);sync();elapsed=time.perf_counter()-ft
      cache[fitkey]=(zz-zz.mean(0,keepdim=True),t,elapsed)
      torch.save({k:v for k,v in p.items() if k in ['tokens','amplitude']},out/f'{seed}__{fitkey}_prompt.pt')
     z,trace,fit_seconds=cache[fitkey];params=C*128
    with torch.no_grad():
     scores=A.logits(z,si,sy,C);prop=(method.startswith('F') and method[-1]=='1') or method.startswith('M') or method=='frozen_prop'
     if prop:
      if method=='M3_original':op=filters['plain']
      else:
       okey='frozen' if method=='M3_frozen_knn' else 'teacher' if method=='M3_teacher_knn' else fitkey
       if okey not in opcache:
        oz=z0-z0.mean(0,keepdim=True) if okey=='frozen' else r-r.mean(0,keepdim=True) if okey=='teacher' else z
        opcache[okey]=knn(oz)
       op,ix,w=opcache[okey]
      scores=propagate(scores,op,si,sy,C,alpha=1. if method=='M3_no_restart' else .9,clamp=method!='M3_no_clamp')
   if smoke and method=='F111':
    ref,_=A.diffusion_readout(z,si,sy,C,'qeps-50',seed,filters['plain'],{},out/'parity')
    assert torch.allclose(scores.softmax(-1),ref.softmax(-1),atol=2e-6,rtol=1e-5),'readout parity'
   sync();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();qscore=scores[query.cuda()].detach().cpu().double();probs=qscore.softmax(-1).numpy();assert np.isfinite(probs).all()
   np.savez_compressed(folder/'prediction.npz',scores=qscore.numpy(),probabilities=probs,query=query.numpy(),support=support.numpy())
   ph=sha(folder/'prediction.npz');assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and all(p.grad is None for p in model.parameters()) and not model.training
   # First query-label access for this method occurs only after prediction file sealing.
   np.save(out/'query_labels.npy',data.y[query].numpy())
   met=metrics(data.y[query].numpy(),probs)
   row=dict(dataset=name,checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],job=job['id'],seed=seed,method=method,steps=steps,phase=job['phase'],smoke=smoke,split_sha256=split,graph_sha256=graphhash,runner_sha256=sha(__file__),prediction_sha256=ph,frozen_verified=True,seconds=elapsed,fit_seconds=fit_seconds,peak_gpu_bytes=peak,parameters=params,protocol=protocol,**met)
   save(done,row);print(json.dumps({k:row[k] for k in ['job','seed','method','accuracy','seconds']}),flush=True)
  del cache,opcache;torch.cuda.empty_cache()
 save(out/'complete.json',dict(expected=len(seeds)*len(all_methods),count=len(list(out.glob('*/result.json'))),methods=all_methods,finished=time.time()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--job');args=ap.parse_args();plan=json.loads((ROOT/'plan.json').read_text());assert sha(__file__)==plan['core_runner_sha256']
 for job in plan['jobs']:
  if args.job and args.job!=job['id']:continue
  try:run(job,args.smoke)
  except Exception as ex:save(ROOT/'failure.json',dict(job=job['id'],error=repr(ex),traceback=traceback.format_exc()));raise
 save(ROOT/('smoke_complete.json' if args.smoke else 'core_complete.json'),dict(finished=time.time(),job=args.job))
