"""Frozen representation transfer; no Prompt modules imported or fitted."""
import sys,json,time,os,fcntl,traceback,argparse
from pathlib import Path
import torch,numpy as np
import run_study as r
sys.path.insert(0,'/root/autodl-tmp/unified_three_backbone_20260924/prog_src')
from prompt_graph.model import GCN
ROOT=r.ROOT

def run(plan):
 assert r.sha(__file__)==plan['runner_sha256'] and r.sha(Path(r.__file__))==plan['operator_sha256']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()))
 count=0;skips=[]
 for ck in plan['checkpoints']:
  assert ck['family']!='DGI'
  name=ck['dataset'];d,x,adj,pool,query,C,protocol=r.load_data(name)
  assert r.sha(ck['path'])==ck['sha256']
  model=GCN(d.x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
  before={k:r.thash(v) for k,v in model.state_dict().items()}
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.no_grad():z=model(d.x.cuda(),d.edge_index.cuda()).detach()
  torch.cuda.synchronize();feature_time=time.perf_counter()-t;zh=r.thash(z)
  for seed in plan['seeds']:
   gen=torch.Generator().manual_seed(seed);support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:plan['shots']]] for c in range(C) for ix in [pool[d.y[pool]==c]]]);si=support.cuda();sy=d.y[support].cuda();split=r.thash(torch.cat([support,query]));assert not torch.isin(support,query).any()
   for cfg in plan['configs']:
    folder=out/ck['id']/f'{seed}__{cfg["id"]}';dest=folder/'result.json';pf=folder/'prediction.npz'
    if dest.exists():
     old=json.loads(dest.read_text());assert old['plan_sha256']==r.sha(plan['_path']) and old['prediction_sha256']==r.sha(pf);count+=1;continue
    if __import__('shutil').disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk safety threshold')
    folder.mkdir(parents=True,exist_ok=True);r.save(out/'current.json',dict(checkpoint=ck['id'],seed=seed,method=cfg['id'],completed=count,time=time.time()))
    torch.cuda.synchronize();t=time.perf_counter()
    with torch.no_grad():scores=r.predict(z,adj,si,sy,C,cfg,seed)
    torch.cuda.synchronize();secs=time.perf_counter()-t
    p=scores[query.cuda()].softmax(-1).cpu().numpy();assert np.isfinite(p).all();np.savez_compressed(pf,probability=p,support=support.numpy(),query=query.numpy());ph=r.sha(pf)
    metrics=r.measures(d.y[query].numpy(),p)
    r.save(dest,dict(dataset=name,representation='frozen_GCN_output',checkpoint_id=ck['id'],family=ck['family'],checkpoint_sha256=ck['sha256'],provenance=ck.get('provenance'),feature_sha256=zh,feature_seconds=feature_time,seed=seed,method=cfg['id'],config=cfg,shots=plan['shots'],protocol=protocol,split_sha256=split,prediction_sha256=ph,plan_sha256=r.sha(plan['_path']),seconds=secs,**metrics));count+=1
  assert before=={k:r.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
  r.save(out/ck['id']/'frozen_verified.json',dict(frozen=True,checkpoint_sha256=ck['sha256'],feature_sha256=zh))
  print(json.dumps(dict(checkpoint=ck['id'],completed=count)),flush=True)
  del model,z,d,x,adj;torch.cuda.empty_cache()
 expected=len(plan['checkpoints'])*len(plan['seeds'])*len(plan['configs']);assert count==expected
 r.save(out/'complete.json',dict(expected=expected,completed=count,finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();p=json.loads(Path(args.plan).read_text());p['_path']=args.plan
 try:run(p)
 except Exception:r.save(ROOT/'results'/p['round']/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
