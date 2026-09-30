"""Fixed five-line cross-checkpoint screen; original scientific operators imported unchanged."""
import os,sys,json,hashlib,time,argparse,traceback,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for p in (ROOT/'sources').iterdir():
 if p.is_dir():sys.path.insert(0,str(p))
sys.path.insert(0,'/root/ProG')
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,Amazon,Actor,WebKB,WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import run_round53_focal as a1
import run_round83_graft as a3
import run_round39_vat as a4
import run_round32_confirm as a5
from prompt_graph.model import GCN
from prompt_graph.utils import act
import operators as ab
NODES=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell']
GRAPHS=['MUTAG','PROTEINS','ENZYMES']
FAMILIES=['DGI','GraphCL','GraphMAE','GraphMAE2','Bandana','S2GAE','Edgepred_GPPT','Edgepred_Gprompt','SimGRACE','MultiGprompt']
METHODS=['Combo_temp02_mean']
DIRECT=[]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(t,p)
def verify_sources():
 for r in json.loads((ROOT/'source_manifest.json').read_text()):assert sha(ROOT/r['path'])==r['sha256'],r
 return sha(ROOT/'source_manifest.json')
def data_load(name):
 assets='/root/autodl-tmp/accuracy_prompt_20260923/assets5090/data'
 if name in ['Cora','CiteSeer','PubMed']:data=Planetoid('/root/ProG/data/Planetoid',name,transform=NormalizeFeatures())[0]
 elif name in ['Computers','Photo']:data=Amazon('/root/ProG/data/Amazon',name,transform=NormalizeFeatures())[0]
 elif name=='Actor':data=Actor(assets+'/Actor')[0]
 elif name=='Cornell':data=WebKB(assets+'/Cornell','Cornell')[0]
 elif name=='Chameleon':data=WikipediaNetwork(assets+'/WikipediaNetwork','chameleon',geom_gcn_preprocess=True)[0]
 else:raise ValueError(name)
 if name in ['Computers','Photo']:
  g=torch.Generator().manual_seed(20260924);pool=[]
  for c in range(int(data.y.max())+1):
   ix=(data.y==c).nonzero().flatten();assert len(ix)>=10;pool.append(ix[torch.randperm(len(ix),generator=g)[:20]])
  pool=torch.cat(pool);mask=torch.ones(len(data.y),dtype=torch.bool);mask[pool]=False;query=mask.nonzero().flatten();protocol='fixed20_perclass_pool_all_remaining_query'
 else:
  train=data.train_mask[:,0] if data.train_mask.ndim==2 else data.train_mask
  test=data.test_mask[:,0] if data.test_mask.ndim==2 else data.test_mask
  pool=train.nonzero().flatten();query=test.nonzero().flatten();protocol='official_train_pool_test_query_fold0'
 assert len(query)>0 and not torch.isin(pool,query).any()
 C=int(data.y.max())+1;assert torch.all(torch.bincount(data.y[pool],minlength=C)>=5)
 return data,pool,query,C,protocol

def run_job(jobid,smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 plan=json.loads((ROOT/'plan.json').read_text());assert verify_sources()==plan['source_manifest_sha256'];assert sha(__file__)==plan['runner_sha256'],'Runner changed after plan freeze'
 job=next(j for j in plan['jobs'] if j['id']==jobid);name=job['dataset'];out=ROOT/('smoke' if smoke else 'results')/jobid;out.mkdir(parents=True,exist_ok=True)
 steps=plan['steps'];seeds=plan['seeds'][:1] if smoke else plan['seeds']
 data,pool,query,C,protocol=data_load(name);x=data.x.cuda();edges=data.edge_index.cuda();xn=x if name in NODES[:5] else NormalizeFeatures()(data.clone()).x.cuda()
 ei,ew=gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,xn)).detach()
 gh=hashlib.sha256((a1.thash(data.x)+a1.thash(data.edge_index)).encode()).hexdigest();filters=a1.make_filters(x,edges)
 if not job['direct']:
  ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False);before={k:a1.thash(v) for k,v in model.state_dict().items()}
  with torch.no_grad():
   u=model.conv_layers[0](x,edges).detach();h=act(u).detach();z0=model.conv_layers[1](h,edges).detach();ref=model(x,edges);assert torch.allclose(z0,ref,atol=5e-5,rtol=1e-5)
 else:ck=dict(family='Direct',sha256=None,path=None,provenance={'checkpoint_independent':True})
 methods=DIRECT if job['direct'] else METHODS
 for seed in seeds:
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);sy=data.y[support].cuda();si=support.cuda();assert torch.all(torch.bincount(sy,minlength=C)==5)
  split=hashlib.sha256((a1.thash(support)+a1.thash(query)).encode()).hexdigest();cache={};neighbors={}
  def embedding(key):
   parent=Path(plan['ablation_remote'])/'results'/jobid/f'{seed}__A3_temp02'
   refs=json.loads((ROOT/'reference_manifest.json').read_text())
   ref=next(v for v in refs if v['origin']=='ablation' and v['path']==str((parent/'result.json').relative_to(Path(plan['ablation_remote']))))
   assert sha(parent/'result.json')==ref['sha256'] and sha(parent/'sealed.pt')==ref['sealed_sha256']
   old=json.loads((parent/'result.json').read_text());saved=torch.load(parent/'sealed.pt',map_location='cpu',weights_only=True)
   assert old['split_sha256']==split and old['checkpoint_sha256']==ck['sha256'] and old['graph_sha256']==gh
   assert torch.equal(saved['support'],support) and torch.equal(saved['query'],query)
   assert old['trace']['gate_temperature']==.2 and not old['trace']['kd_active']
   prompt=saved['prompt'];assert prompt['amplitude']==10.
   with torch.no_grad():
    gate=prompt['gate'].cuda();tokens=prompt['tokens'].cuda();scale=u.std(0).clamp_min(1e-6)
    raw=model.conv_layers[1](a3.act(u+10.*(gate@tokens.tanh())*scale),edges)
    center=prompt['center'].cuda();assert torch.allclose(raw.mean(0,keepdim=True),center,atol=2e-5,rtol=1e-5)
    z=raw-center
    replay,rt=a1.diffusion_readout(z,si,sy,C,'qeps-50',seed,filters['plain'],{},out/'neighbors'/f'{seed}__replay')
    actual=replay[query.cuda()].cpu();error=float((actual-saved['scores']).abs().max())
    assert torch.allclose(actual,saved['scores'],atol=2e-5,rtol=1e-5),error
    assert torch.equal(actual.argmax(-1),saved['prediction']),'Replay predictions differ'
   return z,prompt,dict(rt,readout='mean',replay_max_error=error,replay_prediction_identical=True,parent_result_sha256=ref['sha256'],parent_prediction_sha256=ref['sealed_sha256'],new_training_steps=0,historical_training_steps=100,gate_temperature=.2,kd_active=False)
  for method in methods:
   folder=out/f'{seed}__{method}';folder.mkdir(exist_ok=True);resultfile=folder/'result.json'
   if resultfile.exists():
    old=json.loads(resultfile.read_text());assert old['split_sha256']==split and old['checkpoint_sha256']==ck['sha256'] and old['runner_sha256']==sha(__file__);assert sha(folder/'sealed.pt')==old['prediction_sha256'];continue
   start=time.time();dump(out/'current.json',dict(seed=seed,method=method,started=start))
   try:
    if job['direct']:
     scores,trace=ab.direct_ablate(r,xn,si,sy,C,seed,method);prompt={}
    else:
     z,prompt,trace=embedding(method)
     kind='mean'
     scores,readtrace=a1.diffusion_readout(z,si,sy,C,kind,seed,filters['plain'],{},out/'neighbors'/f'{seed}__{method}')
     trace=dict(trace,**readtrace)
    ss=scores[query.cuda()].detach().cpu();assert torch.isfinite(ss).all();pred=ss.argmax(-1)
    # No query labels are passed to operators; serialize before scoring.
    torch.save(dict(query=query,support=support,scores=ss,prediction=pred,prompt=prompt),folder/'sealed.pt');sealed_hash=sha(folder/'sealed.pt')
    if not job['direct']:assert before=={k:a1.thash(v) for k,v in model.state_dict().items()} and all(p.grad is None for p in model.parameters()) and not model.training
    acc=float((pred==data.y[query]).float().mean())
    row=dict(dataset=name,checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],checkpoint_path=ck['path'],seed=seed,method=method,accuracy=acc,seconds=time.time()-start,shots=5,steps=steps,smoke=smoke,graph_sha256=gh,split_sha256=split,prediction_sha256=sealed_hash,runner_sha256=sha(__file__),source_manifest_sha256=plan['source_manifest_sha256'],support_size=len(support),query_size=len(query),protocol=protocol,input=plan['datasets'][name]['input'],frozen_verified=not job['direct'],provenance=ck['provenance'],trace=trace)
    dump(resultfile,row);print(json.dumps({k:row[k] for k in ['dataset','checkpoint','seed','method','accuracy','seconds']}),flush=True)
   except Exception as e:
    dump(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seed=seed,method=method));raise
  del cache,neighbors;gc.collect();torch.cuda.empty_cache()
 expected=len(seeds)*len(methods);files=list(out.glob('*/result.json'));assert len(files)==expected
 dump(out/'complete.json',dict(expected=expected,results=len(files),finished=time.time(),job=jobid,smoke=smoke))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run']);ap.add_argument('--job');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 run_job(args.job,args.smoke)
