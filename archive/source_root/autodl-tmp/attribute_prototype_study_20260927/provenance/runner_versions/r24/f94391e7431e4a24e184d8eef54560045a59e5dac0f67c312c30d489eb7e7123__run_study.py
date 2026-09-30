"""Independent attribute/prototype research. Query labels enter only after prediction sealing."""
import os,sys,json,time,hashlib,argparse,fcntl,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,Amazon,Actor,WebKB,WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent.parent
DATASETS=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,p)
def load_data(name):
 assets='/root/autodl-tmp/accuracy_prompt_20260923/assets5090/data'
 if name in ['Cora','CiteSeer','PubMed']:d=Planetoid('/root/ProG/data/Planetoid',name,transform=NormalizeFeatures())[0]
 elif name in ['Computers','Photo']:d=Amazon('/root/ProG/data/Amazon',name,transform=NormalizeFeatures())[0]
 elif name=='Actor':d=Actor(assets+'/Actor')[0]
 elif name=='Cornell':d=WebKB(assets+'/Cornell','Cornell')[0]
 else:d=WikipediaNetwork(assets+'/WikipediaNetwork','chameleon',geom_gcn_preprocess=True)[0]
 C=int(d.y.max())+1
 if name in ['Computers','Photo']:
  gen=torch.Generator().manual_seed(20260924);pool=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:20]] for c in range(C) for ix in [(d.y==c).nonzero().flatten()]])
  mask=torch.ones(len(d.y),dtype=torch.bool);mask[pool]=False;query=mask.nonzero().flatten();protocol='fixed20_perclass_pool_all_remaining_query'
 else:
  tr=d.train_mask[:,0] if d.train_mask.ndim==2 else d.train_mask;te=d.test_mask[:,0] if d.test_mask.ndim==2 else d.test_mask
  pool=tr.nonzero().flatten();query=te.nonzero().flatten();protocol='official_train_pool_test_query_fold0'
 assert not torch.isin(pool,query).any()
 xn=d.x if name in DATASETS[:5] else NormalizeFeatures()(d.clone()).x
 ei,ew=gcn_norm(d.edge_index,num_nodes=len(d.x),add_self_loops=True)
 adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(d.x),len(d.x))).coalesce().cuda()
 return d,xn.cuda(),adj,pool,query,C,protocol

def configs():
 out=[]
 for center in [0,1]:
  base=dict(center=center,hops=2,beta=.5,tau=.05,rounds=3,op='prototype')
  def add(tag,**kw):out.append(dict(base,**kw,id=f'c{center}_{tag}'))
  add('base');add('no_refine',rounds=0)
  for k in [0,1,4]:add(f'hop{k}',hops=k)
  for b in [.25,.75,1.]:add(f'beta{b}',beta=b)
  for t in [.02,.1,.2]:add(f'tau{t}',tau=t)
  for n in [1,10]:add(f'iter{n}',rounds=n)
  for mod in ['confidence','confidence_shuffle','neighbor','neighbor_shuffle']:add(mod,module=mod)
  add('ridge',op='ridge');add('1nn',op='1nn')
 out.append(dict(id='labelprop',op='labelprop',hops=2,center=0))
 return out

def predict(r,adj,si,sy,C,cfg,seed):
 z=F.normalize(r-r.mean(0,keepdim=True) if cfg['center'] else r,dim=1)
 oh=F.one_hot(sy,C).float();mu0=(oh.T@z[si])/oh.sum(0)[:,None]
 if cfg['op']=='1nn':return F.one_hot(sy[(z@z[si].T).argmax(1)],C).float()*10
 if cfg['op']=='ridge':
  a=z[si];w=torch.linalg.solve(a@a.T+torch.eye(len(si),device=z.device),oh);return z@a.T@w
 if cfg['op']=='labelprop':
  f=torch.zeros((len(z),C),device=z.device);f[si]=oh
  for _ in range(50):f=.9*torch.sparse.mm(adj,f);f[si]=oh
  return f.clamp_min(1e-12).log()
 mask=torch.ones(len(z),dtype=torch.bool,device=z.device);mask[si]=False;zu=z[mask];mu=mu0.clone()
 gen=torch.Generator(device=z.device).manual_seed(seed+33000);perm=torch.randperm(len(zu),generator=gen,device=z.device)
 for _ in range(cfg['rounds']):
  q=(zu@F.normalize(mu,dim=1).T/cfg['tau']).softmax(-1)
  mod=cfg.get('module','none')
  if mod.startswith('confidence'):
   weights=(1+(q*q.clamp_min(1e-12).log()).sum(1)/np.log(C)).clamp_min(1e-6)
   if mod.endswith('shuffle'):weights=weights[perm]
   q=q*weights[:,None]
  if mod.startswith('neighbor'):
   allq=(z@F.normalize(mu,dim=1).T/cfg['tau']).softmax(-1);smooth=torch.sparse.mm(adj,allq)[mask];smooth=smooth/smooth.sum(1,keepdim=True).clamp_min(1e-12)
   if mod.endswith('shuffle'):smooth=smooth[perm]
   q=.5*q+.5*smooth
  pseudo=(q.T@zu)/q.sum(0).clamp_min(1e-8)[:,None]
  mu=(1-cfg['beta'])*mu0+cfg['beta']*pseudo
 return z@F.normalize(mu,dim=1).T/.2

def measures(y,p):
 C=p.shape[1];pr=p.argmax(1);oh=np.eye(C)[y]
 m=dict(accuracy=accuracy_score(y,pr)*100,macro_f1=f1_score(y,pr,average='macro',zero_division=0)*100,micro_f1=f1_score(y,pr,average='micro',zero_division=0)*100)
 present=[c for c in range(C) if len(np.unique(oh[:,c]))==2]
 m['roc_auc']=float(np.mean([roc_auc_score(oh[:,c],p[:,c]) for c in present])) if present else None
 m['average_precision']=float(np.mean([average_precision_score(oh[:,c],p[:,c]) for c in present])) if present else None
 m['auc_classes']=present;return m

def run(plan,smoke=False):
 assert sha(__file__)==plan['runner_sha256'], 'Runner changed after freeze'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'results'/plan['round'];out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(out/'process.json',dict(pid=os.getpid(),started=time.time(),plan_sha256=sha(plan['_path'])))
 methods=plan['configs'];count=0
 for name in plan['datasets']:
  d,x,adj,pool,query,C,protocol=load_data(name)
  # Preparation has no access to query labels.
  features={0:x}
  for k in range(1,max(c['hops'] for c in methods)+1):features[k]=torch.sparse.mm(adj,features[k-1])
  gh=hashlib.sha256((thash(d.x)+thash(d.edge_index)).encode()).hexdigest()
  for seed in plan['seeds']:
   gen=torch.Generator().manual_seed(seed);support=torch.cat([ix[torch.randperm(len(ix),generator=gen)[:plan['shots']]] for c in range(C) for ix in [pool[d.y[pool]==c]]])
   assert len(support)==C*plan['shots'] and not torch.isin(support,query).any()
   si=support.cuda();sy=d.y[support].cuda();split=thash(torch.cat([support,query]))
   for cfg in methods:
    tag=f'{name}/{seed}__{cfg["id"]}';folder=out/tag;result=folder/'result.json';predfile=folder/'prediction.npz'
    if result.exists():
     old=json.loads(result.read_text());assert old['plan_sha256']==sha(plan['_path']) and old['prediction_sha256']==sha(predfile);count+=1;continue
    if __import__('shutil').disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Less than 5GiB free; stop before new work')
    save(out/'current.json',dict(dataset=name,seed=seed,method=cfg['id'],completed=count,time=time.time()))
    folder.mkdir(parents=True,exist_ok=True);torch.cuda.synchronize();started=time.perf_counter()
    with torch.no_grad():scores=predict(features[cfg['hops']],adj,si,sy,C,cfg,seed)
    torch.cuda.synchronize();secs=time.perf_counter()-started
    p=scores[query.cuda()].softmax(-1).cpu().numpy();assert np.isfinite(p).all()
    np.savez_compressed(predfile,probability=p,support=support.numpy(),query=query.numpy())
    ph=sha(predfile)
    # All predictions are sealed before query labels are read for scoring.
    metrics=measures(d.y[query].numpy(),p)
    row=dict(dataset=name,representation='attributes',seed=seed,method=cfg['id'],config=cfg,shots=plan['shots'],protocol=protocol,graph_sha256=gh,split_sha256=split,prediction_sha256=ph,plan_sha256=sha(plan['_path']),runner_sha256=sha(__file__),seconds=secs,**metrics)
    save(result,row);count+=1
   print(json.dumps(dict(dataset=name,seed=seed,completed=count)),flush=True)
  del features,adj,x,d;torch.cuda.empty_cache()
 expected=len(plan['datasets'])*len(plan['seeds'])*len(methods)
 assert count==expected
 save(out/'complete.json',dict(completed=count,expected=expected,finished=time.time(),plan_sha256=sha(plan['_path'])))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);args=ap.parse_args();p=json.loads(Path(args.plan).read_text());p['_path']=args.plan
 try:run(p)
 except Exception:
  save(ROOT/'results'/p['round']/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
