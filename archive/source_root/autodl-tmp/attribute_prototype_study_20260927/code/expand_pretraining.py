"""Label-free ProG pretraining expansion in an isolated source snapshot."""
import os,sys,json,time,argparse,subprocess,traceback,fcntl
from pathlib import Path
import run_study as r

def train(job):
 import torch
 from torch_geometric.data import Data
 from torch_geometric.transforms import NormalizeFeatures
 sys.path.insert(0,str(r.ROOT/'source/prog_expansion'))
 from prompt_graph.pretrain import GraphMAE2,S2GAE,Bandana
 from prompt_graph.model import build_gnn
 from prompt_graph.utils import seed_everything
 seed_everything(job['seed']);torch.set_num_threads(4)
 name=job['dataset'];canonical=r.ROOT/'data'/name/'canonical.pt'
 if canonical.exists():
  a=torch.load(canonical,weights_only=True,map_location='cpu');d=Data(x=a['x'],edge_index=a['edge_index'])
 else:
  old,*_=r.load_data(name);d=Data(x=old.x.cpu(),edge_index=old.edge_index.cpu());del old,_;torch.cuda.empty_cache()
 d=NormalizeFeatures()(d);assert d.y is None
 target=r.ROOT/'checkpoints'/job['id'];target.mkdir(parents=True,exist_ok=True);out=target/'encoder.pth'
 cls={'GraphMAE2':GraphMAE2,'S2GAE':S2GAE,'Bandana':Bandana}[job['family']]
 r.save(target/'input.json',dict(labels_used=False,input_normalization='NormalizeFeatures',x_sha256=r.thash(d.x),edge_sha256=r.thash(d.edge_index),job=job,runner_sha256=r.sha(__file__)))
 task=cls(dataset_name=name,gnn_type=job['architecture'],hid_dim=128,gln=2,num_epoch=job['epochs'],device=0,num_workers=0,data=d,output_path=str(out));task.pretrain()
 check=build_gnn(job['architecture'],d.x.shape[1],128,2).cuda().eval();check.load_state_dict(torch.load(out,weights_only=True,map_location='cpu'),strict=True);check.requires_grad_(False)
 with torch.no_grad():z=check(d.x.cuda(),d.edge_index.cuda());assert z.shape==(len(d.x),128) and torch.isfinite(z).all()
 r.save(target/'ready.json',dict(**job,path=str(out),sha256=r.sha(out),input_normalization='NormalizeFeatures',labels_used=False,embedding_shape=list(z.shape),finished=time.time(),source_parity='ProG-adapted; not full author parity verified'))

def queue(plan):
 assert plan['runner_sha256']==r.sha(__file__)
 folder=r.ROOT/'results/expanded_pretraining';folder.mkdir(parents=True,exist_ok=True);lock=open(folder/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.save(folder/'process.json',dict(pid=os.getpid(),started=time.time()));records=[]
 for job in plan['jobs']:
  target=r.ROOT/'checkpoints'/job['id'];target.mkdir(parents=True,exist_ok=True)
  if (target/'ready.json').exists():
   meta=json.loads((target/'ready.json').read_text());assert r.sha(meta['path'])==meta['sha256'];records.append(dict(id=job['id'],status='ready'));continue
  if (target/'failure.json').exists():records.append(dict(id=job['id'],status='failed_recorded'));continue
  assert __import__('shutil').disk_usage(r.ROOT).free>8*1024**3
  r.save(folder/'current.json',dict(job=job,completed=len(records),total=len(plan['jobs']),time=time.time()))
  jf=target/'job.json';r.save(jf,job)
  with (target/'train.log').open('w') as log:
   try:
    z=subprocess.run([sys.executable,__file__,'--job',str(jf)],stdout=log,stderr=subprocess.STDOUT,timeout=2700);status='ready' if z.returncode==0 and (target/'ready.json').exists() else 'failed'
   except subprocess.TimeoutExpired:status='timeout'
  if status!='ready':r.save(target/'failure.json',dict(status=status,job=job,time=time.time()))
  records.append(dict(id=job['id'],status=status));r.save(folder/'status.json',records);print(job['id'],status,flush=True)
 r.save(folder/'complete.json',dict(total=len(plan['jobs']),ready=sum(x['status']=='ready' for x in records),records=records))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--job');ap.add_argument('--plan');a=ap.parse_args()
 if a.job:train(json.loads(Path(a.job).read_text()))
 else:queue(json.loads(Path(a.plan).read_text()))
