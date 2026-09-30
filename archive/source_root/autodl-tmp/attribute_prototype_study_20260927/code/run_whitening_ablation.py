import sys,json,time,os,fcntl,argparse,traceback
from pathlib import Path
import torch,numpy as np
import run_study as r
from run_selection import choose
sys.path.insert(0,'/root/autodl-tmp/unified_three_backbone_20260924/prog_src')
from prompt_graph.model import GCN

def run(plan):
 assert r.sha(__file__)==plan['runner_sha256'] and r.sha(r.__file__)==plan['operator_sha256']
 assert r.sha(Path(__file__).parent/'run_selection.py')==plan['selector_sha256']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(out/'process.json',dict(pid=os.getpid(),started=time.time()));count=0
 for ck in plan['checkpoints']:
  assert ck['family']!='DGI' and r.sha(ck['path'])==ck['sha256']
  name=ck['dataset'];d,x,adj,pool,query,C,protocol=r.load_data(name)
  model=GCN(d.x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False);before={k:r.thash(v) for k,v in model.state_dict().items()}
  with torch.no_grad():
   z=model(d.x.cuda(),d.edge_index.cuda()).detach();zc=z-z.mean(0,keepdim=True);var=zc.square().mean(0);cov=zc.T@zc/len(z);ev,u=torch.linalg.eigh(cov);features={0:z}
   for index,strength in enumerate([.01,.1,1.],1):
    reg=strength*var.mean().clamp_min(1e-12);features[index]=zc@((u*(ev.clamp_min(0)+reg).rsqrt()[None,:])@u.T)
  for seed in plan['seeds']:
   done=out/ck['id']/f'{seed}__selection'/'complete.json'
   if done.exists():
    assert json.loads(done.read_text())['plan_sha256']==r.sha(plan['_path']);count+=len(plan['outputs']);continue
   if __import__('shutil').disk_usage(r.ROOT).free<5*1024**3:raise RuntimeError('Disk threshold')
   gen=torch.Generator().manual_seed(seed);support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:plan['shots']]] for c in range(C) for ix in [pool[d.y[pool]==c]]]);si=support.cuda();sy=d.y[support].cuda();assert not torch.isin(support,query).any();split=r.thash(torch.cat([support,query]))
   r.save(out/'current.json',dict(checkpoint=ck['id'],seed=seed,completed=count,time=time.time()))
   with torch.no_grad():
    torch.cuda.synchronize();t=time.perf_counter();best,records=choose(features,adj,si,sy,C,plan['candidates'],seed);torch.cuda.synchronize();selection_time=time.perf_counter()-t
    choices={c['id']:c for c in plan['candidates']};choices.update(cv_frozen=best,cv_frozen_no_refine=dict(best,rounds=0))
    for method in plan['outputs']:
     cfg=choices[method];folder=out/ck['id']/f'{seed}__{method}';folder.mkdir(parents=True,exist_ok=True);pf=folder/'prediction.npz';torch.cuda.synchronize();t=time.perf_counter();score=r.predict(features[cfg['hops']],adj,si,sy,C,cfg,seed);torch.cuda.synchronize();secs=time.perf_counter()-t
     p=score[query.cuda()].softmax(-1).cpu().numpy();assert np.isfinite(p).all();np.savez_compressed(pf,probability=p,support=support.numpy(),query=query.numpy());ph=r.sha(pf)
     r.save(folder/'result.json',dict(dataset=name,representation='frozen_GCN_geometry',checkpoint_id=ck['id'],family=ck['family'],checkpoint_sha256=ck['sha256'],provenance=ck.get('provenance'),seed=seed,shots=plan['shots'],method=method,selected_config=cfg,selection_records=records,selection_seconds=selection_time if method.startswith('cv') else 0,seconds=secs,protocol=protocol,split_sha256=split,prediction_sha256=ph,plan_sha256=r.sha(plan['_path']),**r.measures(d.y[query].numpy(),p)));count+=1
   r.save(done,dict(plan_sha256=r.sha(plan['_path']),time=time.time()))
  assert before=={k:r.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters());r.save(out/ck['id']/'frozen_verified.json',dict(frozen=True,checkpoint_sha256=ck['sha256']));print(json.dumps(dict(checkpoint=ck['id'],completed=count)),flush=True)
  del model,z,zc,features,d,x,adj;torch.cuda.empty_cache()
 expected=len(plan['checkpoints'])*len(plan['seeds'])*len(plan['outputs']);assert count==expected;r.save(out/'complete.json',dict(expected=expected,completed=count,finished=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());p['_path']=a.plan
 try:run(p)
 except Exception:r.save(r.ROOT/'results'/p['round']/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
