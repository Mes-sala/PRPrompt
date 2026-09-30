"""Support-only cross-validation for attribute geometry and anchored refinement."""
import argparse,json,time,os,fcntl,traceback
from pathlib import Path
import numpy as np
import torch
import run_study as r

def choose(features,adj,si,sy,C,candidates,seed,shuffle=False):
 # Each fold holds one support from each class; no query labels are accepted.
 k=int(torch.bincount(sy,minlength=C).min());assert k>=2
 classix=[(sy==c).nonzero().flatten() for c in range(C)]
 records=[]
 for cfg in candidates:
  truths=[];probs=[]
  for fold in range(k):
   vi=torch.stack([ix[fold] for ix in classix]);mask=torch.ones(len(si),device=si.device,dtype=torch.bool);mask[vi]=False
   score=r.predict(features[cfg['hops']],adj,si[mask],sy[mask],C,cfg,seed)
   probs.append(score[si[vi]].softmax(-1));truths.append(sy[vi])
  p=torch.cat(probs);y=torch.cat(truths)
  if shuffle:
   gen=torch.Generator(device=y.device).manual_seed(seed+991);y=y[torch.randperm(len(y),generator=gen,device=y.device)]
  acc=float((p.argmax(1)==y).float().mean());nll=float(-p[torch.arange(len(y),device=y.device),y].clamp_min(1e-12).log().mean())
  records.append(dict(id=cfg['id'],accuracy=acc,nll=nll))
 best=min(range(len(records)),key=lambda i:(-records[i]['accuracy'],records[i]['nll'],i))
 return candidates[best],records

def run(plan):
 assert r.sha(__file__)==plan['runner_sha256'] and r.sha(Path(r.__file__))==plan['operator_sha256']
 out=r.ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()))
 count=0
 for name in plan['datasets']:
  d,x,adj,pool,query,C,protocol=r.load_data(name);features={0:x};features[1]=torch.sparse.mm(adj,x);features[2]=torch.sparse.mm(adj,features[1])
  for seed in plan['seeds']:
   if __import__('shutil').disk_usage(r.ROOT).free<5*1024**3:raise RuntimeError('Disk safety threshold')
   folder=out/name/f'{seed}__selection';done=folder/'complete.json'
   if done.exists():
    z=json.loads(done.read_text());assert z['plan_sha256']==r.sha(plan['_path']);count+=len(plan['outputs']);continue
   gen=torch.Generator().manual_seed(seed);support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:plan['shots']]] for c in range(C) for ix in [pool[d.y[pool]==c]]]);si=support.cuda();sy=d.y[support].cuda();split=r.thash(torch.cat([support,query]));assert not torch.isin(support,query).any()
   r.save(out/'current.json',dict(dataset=name,seed=seed,completed=count,time=time.time()))
   with torch.no_grad():
    torch.cuda.synchronize();t=time.perf_counter();best,records=choose(features,adj,si,sy,C,plan['candidates'],seed);torch.cuda.synchronize();cvseconds=time.perf_counter()-t
    # Permuted-label control reuses the same CV predictions conceptually, recomputed to avoid retaining tensors.
    sham,shamrecords=choose(features,adj,si,sy,C,plan['candidates'],seed,True)
    no=[(c,v) for c,v in zip(plan['candidates'],records) if c['rounds']==0];nobest=min(no,key=lambda t:(-t[1]['accuracy'],t[1]['nll']))[0]
    baselines={c['id']:c for c in plan['fixed']}
    choices=dict(cv_prototype=best,cv_no_refine=nobest,cv_label_shuffle=sham,**baselines)
    for tag in plan['outputs']:
     cfg=choices[tag];dest=out/name/f'{seed}__{tag}';dest.mkdir(parents=True,exist_ok=True);pf=dest/'prediction.npz'
     torch.cuda.synchronize();start=time.perf_counter();scores=r.predict(features[cfg['hops']],adj,si,sy,C,cfg,seed);torch.cuda.synchronize();infer=time.perf_counter()-start
     p=scores[query.cuda()].softmax(-1).cpu().numpy();np.savez_compressed(pf,probability=p,support=support.numpy(),query=query.numpy());ph=r.sha(pf)
     m=r.measures(d.y[query].numpy(),p)
     r.save(dest/'result.json',dict(dataset=name,representation='attributes',seed=seed,method=tag,selected_config=cfg,selection_records=records if tag!='cv_label_shuffle' else shamrecords,selection_seconds=cvseconds if tag.startswith('cv') else 0,seconds=infer,shots=plan['shots'],protocol=protocol,split_sha256=split,prediction_sha256=ph,plan_sha256=r.sha(plan['_path']),**m));count+=1
   r.save(done,dict(plan_sha256=r.sha(plan['_path']),finished=time.time()));print(json.dumps(dict(dataset=name,seed=seed,selected=best['id'],completed=count)),flush=True)
  del features,x,adj,d;torch.cuda.empty_cache()
 expected=len(plan['datasets'])*len(plan['seeds'])*len(plan['outputs']);assert count==expected;r.save(out/'complete.json',dict(expected=expected,completed=count,finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan
 try:run(p)
 except Exception:r.save(r.ROOT/'results'/p['round']/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
