"""R09: identical four-geometry support CV budget for each inference readout."""
import argparse,json,os,time,fcntl,traceback
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
import run_study as r
import external_readouts as ext

GEOMETRIES=[(0,0),(0,1),(2,0),(2,1)]
READOUTS=['anchored','simpleshot_style','soft_kmeans_adapted','laplacianshot_adapted']

def predict(features,emb,neighbors,adj,key,si,sy,C,method,seed):
 if method=='anchored':
  return r.predict(features[key[0]],adj,si,sy,C,dict(hops=key[0],center=key[1],op='prototype',beta=.5,tau=.05,rounds=3),seed).softmax(-1).cpu().numpy()
 return ext.scores(emb[key],si,sy,C,method,neighbors[key])

def choose(features,emb,neighbors,adj,si,sy,C,method,seed):
 # No query labels accepted. Fold validation nodes remain visible without labels.
 k=int(torch.bincount(sy,minlength=C).min());assert k>=2
 classix=[(sy==c).nonzero().flatten() for c in range(C)];records=[]
 for key in GEOMETRIES:
  pp=[];yy=[]
  for fold in range(k):
   vi=torch.stack([ix[fold] for ix in classix]);mask=torch.ones(len(si),device=si.device,dtype=torch.bool);mask[vi]=False
   p=predict(features,emb,neighbors,adj,key,si[mask],sy[mask],C,method,seed)
   pp.append(p[si[vi].cpu().numpy()]);yy.append(sy[vi].cpu().numpy())
  p=np.concatenate(pp);y=np.concatenate(yy)
  records.append(dict(hops=key[0],center=key[1],accuracy=float((p.argmax(1)==y).mean()),nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,None)).mean())))
 i=min(range(4),key=lambda j:(-records[j]['accuracy'],records[j]['nll'],j))
 return GEOMETRIES[i],records

def run(plan):
 assert r.sha(__file__)==plan['runner_sha256'] and r.sha(ext.__file__)==plan['external_sha256'] and r.sha(r.__file__)==plan['operator_sha256']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()));count=0
 for name in plan['datasets']:
  d,x,adj,pool,query,C,protocol=r.load_data(name);features={0:x,2:torch.sparse.mm(adj,torch.sparse.mm(adj,x))};emb={};neighbors={};prep={}
  for key in GEOMETRIES:
   z=features[key[0]];emb[key]=F.normalize(z-z.mean(0,keepdim=True) if key[1] else z,dim=1)
   torch.cuda.synchronize();t=time.perf_counter();neighbors[key]=ext.nearest_cache(emb[key],plan['shots']*C);torch.cuda.synchronize();prep[key]=time.perf_counter()-t
  for seed in plan['seeds']:
   done=out/name/f'{seed}__selection'/'complete.json'
   if done.exists():
    assert json.loads(done.read_text())['plan_sha256']==r.sha(plan['_path']);count+=8;continue
   assert __import__('shutil').disk_usage(r.ROOT).free>5*1024**3
   gen=torch.Generator().manual_seed(seed);support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:plan['shots']]] for c in range(C) for ix in [pool[d.y[pool]==c]]]);si=support.cuda();sy=d.y[support].cuda();split=r.thash(torch.cat([support,query]));assert not torch.isin(support,query).any()
   r.save(out/'current.json',dict(dataset=name,seed=seed,completed=count,time=time.time()))
   with torch.no_grad():
    selected={};records={};cvtime={}
    for method in READOUTS:
     torch.cuda.synchronize();t=time.perf_counter();selected[method],records[method]=choose(features,emb,neighbors,adj,si,sy,C,method,seed);torch.cuda.synchronize();cvtime[method]=time.perf_counter()-t
    jobs=[('cv_'+m,m,selected[m],m) for m in READOUTS]+[('anchored_on_'+m,'anchored',selected[m],m) for m in READOUTS[1:]]+[('c0_base','anchored',(2,0),None)]
    for tag,method,key,selector in jobs:
     folder=out/name/f'{seed}__{tag}';folder.mkdir(parents=True,exist_ok=True);pf=folder/'prediction.npz';torch.cuda.synchronize();t=time.perf_counter();prob=predict(features,emb,neighbors,adj,key,si,sy,C,method,seed);torch.cuda.synchronize();secs=time.perf_counter()-t
     p=prob[query.numpy()];assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);np.savez_compressed(pf,probability=p,support=support.numpy(),query=query.numpy());ph=r.sha(pf)
     r.save(folder/'result.json',dict(dataset=name,representation='attributes',seed=seed,shots=plan['shots'],method=tag,selected_config=dict(hops=key[0],center=key[1],readout=method),selector=selector,selection_records=records.get(selector,[]),selection_seconds=cvtime.get(selector,0),seconds=secs,knn_preparation_seconds=sum(prep.values()) if selector=='laplacianshot_adapted' else 0,knn_cost_scope='One-time dataset cache for all four geometries; not included in selection_seconds',protocol=protocol,split_sha256=split,prediction_sha256=ph,plan_sha256=r.sha(plan['_path']),**r.measures(d.y[query].numpy(),p)));count+=1
   r.save(done,dict(plan_sha256=r.sha(plan['_path']),time=time.time()));print(json.dumps(dict(dataset=name,seed=seed,completed=count)),flush=True)
  del features,emb,neighbors,d,x,adj;torch.cuda.empty_cache()
 expected=len(plan['datasets'])*len(plan['seeds'])*8;assert count==expected;r.save(out/'complete.json',dict(expected=expected,completed=count,finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan
 try:run(p)
 except Exception:r.save(r.ROOT/'results'/p['round']/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
