import argparse,json,os,time,fcntl,traceback
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
import run_study as r
import external_readouts as ext

def run(plan):
 assert r.sha(__file__)==plan['runner_sha256'] and r.sha(ext.__file__)==plan['external_sha256'] and r.sha(r.__file__)==plan['operator_sha256']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()));count=0
 for name in plan['datasets']:
  d,x,adj,pool,query,C,protocol=r.load_data(name);features={0:x};features[2]=torch.sparse.mm(adj,torch.sparse.mm(adj,x));emb={};neighbors={};prep={}
  for hops in [0,2]:
   for center in [0,1]:
    key=(hops,center);z=features[hops];emb[key]=F.normalize(z-z.mean(0,keepdim=True) if center else z,dim=1)
    torch.cuda.synchronize();start=time.perf_counter();neighbors[key]=ext.nearest_cache(emb[key],plan['shots']*C);torch.cuda.synchronize();prep[key]=time.perf_counter()-start
  for seed in plan['seeds']:
   gen=torch.Generator().manual_seed(seed);support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:plan['shots']]] for c in range(C) for ix in [pool[d.y[pool]==c]]]);si=support.cuda();sy=d.y[support].cuda();split=r.thash(torch.cat([support,query]));assert not torch.isin(support,query).any()
   for cfg in plan['configs']:
    method=cfg['id'];folder=out/name/f'{seed}__{method}';pf=folder/'prediction.npz';rf=folder/'result.json'
    if rf.exists():
     old=json.loads(rf.read_text());assert old['plan_sha256']==r.sha(plan['_path']) and old['prediction_sha256']==r.sha(pf);count+=1;continue
    if __import__('shutil').disk_usage(r.ROOT).free<5*1024**3:raise RuntimeError('Disk threshold')
    r.save(out/'current.json',dict(dataset=name,seed=seed,method=method,completed=count,time=time.time()));folder.mkdir(parents=True,exist_ok=True)
    key=(cfg['hops'],cfg['center']);torch.cuda.synchronize();t=time.perf_counter()
    with torch.no_grad():
     if cfg['readout']=='anchored':prob=r.predict(features[cfg['hops']],adj,si,sy,C,dict(cfg,op='prototype',beta=.5,tau=.05,rounds=3),seed).softmax(-1).cpu().numpy()
     else:prob=ext.scores(emb[key],si,sy,C,cfg['readout'],neighbors[key])
    torch.cuda.synchronize();secs=time.perf_counter()-t;p=prob[query.numpy()];assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    np.savez_compressed(pf,probability=p,support=support.numpy(),query=query.numpy());ph=r.sha(pf);metrics=r.measures(d.y[query].numpy(),p)
    r.save(rf,dict(dataset=name,representation='attributes',seed=seed,shots=plan['shots'],method=method,config=cfg,seconds=secs,knn_preparation_seconds=prep[key] if cfg['readout']=='laplacianshot_adapted' else 0,protocol=protocol,split_sha256=split,prediction_sha256=ph,plan_sha256=r.sha(plan['_path']),**metrics));count+=1
   print(json.dumps(dict(dataset=name,seed=seed,completed=count)),flush=True)
  del emb,neighbors,features,x,adj,d;torch.cuda.empty_cache()
 expected=len(plan['datasets'])*len(plan['seeds'])*len(plan['configs']);assert count==expected;r.save(out/'complete.json',dict(expected=expected,completed=count,finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan
 try:run(p)
 except Exception:r.save(r.ROOT/'results'/p['round']/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
