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
NODES=['Cora','CiteSeer','PubMed','Computers','Photo','Actor','Chameleon','Cornell']
GRAPHS=['MUTAG','PROTEINS','ENZYMES']
FAMILIES=['DGI','GraphCL','GraphMAE','GraphMAE2','Bandana','S2GAE','Edgepred_GPPT','Edgepred_Gprompt','SimGRACE','MultiGprompt']
METHODS=['frozen','graphprompt','base_qeps','A1_token_mean','A1_token_qeps','A1_scale2_mean','A1_scale2_qeps','A3_tr1_mean','A3_tr1_qeps','A3_gsh_mean','A3_gsh_qeps','A4_none','A4_vat03','A5_self100','A5_selfonly','A5_center','A5_diag10']
DIRECT=['rawprop','rect_uncentered','A2_center']
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

def inventory():
 assert not (ROOT/'plan.json').exists(),'Plan immutable; use another run directory'
 verify_sources();rows={};roots=['/root/ProG/Experiment/pre_trained_model','/root/autodl-tmp/accuracy_prompt_20260923/assets5090','/root/autodl-tmp/pubmed_5090_transfer','/root/autodl-tmp/prompt_study_20260916']
 for root in roots:
  for p,ds,fs in os.walk(root):
   for n in fs:
    if not n.endswith('.pth'):continue
    parts=Path(p).parts;name=next((x for x in NODES+GRAPHS if x in parts),None)
    fam=n.split('.')[0]
    if name is None or fam not in FAMILIES:continue
    path=Path(p)/n;h=sha(path);key=(name,fam,h)
    if key not in rows:rows[key]=dict(dataset=name,family=fam,sha256=h,paths=[])
    rows[key]['paths'].append(str(path))
 audits={}
 for name in NODES:
  try:
   data,pool,query,C,protocol=data_load(name)
   audits[name]=dict(status='ready',nodes=len(data.x),edges=data.edge_index.shape[1],features=data.x.shape[1],classes=C,pool=len(pool),query=len(query),pool_per_class=torch.bincount(data.y[pool],minlength=C).tolist(),protocol=protocol,input='NormalizeFeatures' if name in NODES[:5] else 'raw',teacher_input='NormalizeFeatures')
  except Exception as e:audits[name]=dict(status='data_failure',error=repr(e),traceback=traceback.format_exc())
 entries=[]
 for (name,fam,h),r in sorted(rows.items()):
  r['id']=name+'__'+fam+'__'+h[:12];r['path']=r['paths'][0]
  if name in GRAPHS:r.update(status='unsupported_task',reason='Five retained operators use node support indices and node-level transduction; graph-level pooling requires a separate method adaptation.')
  elif audits[name]['status']!='ready':r.update(status='data_failure',reason=audits[name]['error'])
  else:
   try:
    s=torch.load(r['path'],map_location='cpu',weights_only=True);model=GCN(audits[name]['features'],128,128,num_layer=2);model.load_state_dict(s,strict=True)
    side=Path(r['path']+'.json');r['provenance']=json.loads(side.read_text()) if side.exists() else {'source_parity':'legacy_unverified'}
    if 'checkpoint_sha256' in r['provenance']:assert r['provenance']['checkpoint_sha256']==h,'sidecar hash mismatch'
    r.update(status='ready',reason='Strict two-layer GCN state_dict load passed')
   except Exception as e:r.update(status='unsupported_architecture' if fam=='MultiGprompt' else 'checkpoint_load_failure',reason=repr(e))
  entries.append(r)
 jobs=[dict(id='Direct__'+n,dataset=n,direct=True) for n in NODES if audits[n]['status']=='ready']
 # Small datasets first; round-robin is not based on accuracy.
 for r in sorted(entries,key=lambda r:(audits.get(r['dataset'],{}).get('nodes',10**9),r['dataset'],r['family'],r['sha256'])):
  if r['status']=='ready':jobs.append(dict(id=r['id'],dataset=r['dataset'],direct=False,checkpoint=r))
 plan=dict(schema='five-mainline-grid-v1',created=time.time(),seeds=[24092400+i for i in range(5)],shots=5,steps=100,methods=METHODS,direct_methods=DIRECT,datasets=audits,checkpoints=entries,jobs=jobs,source_manifest_sha256=verify_sources(),runner_sha256=sha(__file__),scope='all unique named pretrained checkpoints found, including legacy hashes; graph tasks and incompatible architecture explicit',interpretation='retrospective fixed recipe screen; no query-selected hyperparameters or held-out confirmation claim')
 plan['expected_results']=sum((len(DIRECT) if j['direct'] else len(METHODS))*len(plan['seeds']) for j in jobs)
 dump(ROOT/'plan.json',plan);print(json.dumps({'jobs':len(jobs),'expected_results':plan['expected_results'],'checkpoint_statuses':{s:sum(r['status']==s for r in entries) for s in sorted({r['status'] for r in entries})},'datasets':audits},ensure_ascii=False),flush=True)

def run_job(jobid,smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 plan=json.loads((ROOT/'plan.json').read_text());assert verify_sources()==plan['source_manifest_sha256'];assert sha(__file__)==plan['runner_sha256'],'Runner changed after plan freeze'
 job=next(j for j in plan['jobs'] if j['id']==jobid);name=job['dataset'];out=ROOT/('smoke' if smoke else 'results')/jobid;out.mkdir(parents=True,exist_ok=True)
 steps=2 if smoke else plan['steps'];seeds=plan['seeds'][:1] if smoke else plan['seeds']
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
   if key in cache:return cache[key]
   if key=='base':v=(z0,{}, {})
   elif key=='gp':v=a1.train(h,z0,model,edges,si,sy,C,'graphprompt',seed,steps)
   elif key=='token':v=a1.token_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plainlate',steps,filters)
   elif key=='scale2':v=a1.focal_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plainlate',steps,filters,2,'scale')
   elif key in ['tr1','gsh']:
    if key=='tr1':z,p,t=a3.temperature_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw')
    else:z,p,t=a3.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw','gshampoo',.01)
    center=z.mean(0,keepdim=True);v=(z-center,dict(p,center=center.cpu()),dict(t,geometry='post_center',raw_embedding_sha256=a1.thash(z)))
   elif key in ['none','vat03']:v=a4.vat_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_kd3_ref50',steps,filters,'none' if key=='none' else 'vat',0. if key=='none' else .3)
   elif key in ['self100','selfonly']:v=a5.structural_prompt(model,x,edges,xn,si,sy,C,key,steps)
   elif key in ['center','diag10']:v=a5.geometry_prompt(model,x,edges,xn,si,sy,C,seed,'geom_'+key+'_base',steps,out/'geometry'/f'{key}.pt')
   else:raise ValueError(key)
   cache[key]=v;return v
  for method in methods:
   folder=out/f'{seed}__{method}';folder.mkdir(exist_ok=True);resultfile=folder/'result.json'
   if resultfile.exists():
    old=json.loads(resultfile.read_text());assert old['split_sha256']==split and old['checkpoint_sha256']==ck['sha256'] and old['runner_sha256']==sha(__file__);assert sha(folder/'sealed.pt')==old['prediction_sha256'];continue
   start=time.time();dump(out/'current.json',dict(seed=seed,method=method,started=start))
   try:
    if job['direct']:
     if method=='rawprop':scores=a1.logits(r,si,sy,C);trace={}
     else:scores,trace=a1.rectified_teacher(r-r.mean(0,keepdim=True) if method=='A2_center' else r,si,sy,C,'rect_teacher_ref50',seed)
     prompt={}
    else:
     kind='mean'
     if method=='frozen':key='base'
     elif method=='graphprompt':key='gp'
     elif method=='base_qeps':key='base';kind='qeps-50'
     elif method.startswith(('A1_','A3_')):
      _,key,k=method.split('_');kind='qeps-50' if k=='qeps' else 'mean'
     else:key=method.split('_')[1]
     z,prompt,trace=embedding(key)
     scores,readtrace=a1.diffusion_readout(z,si,sy,C,kind,seed,filters['plain'],neighbors.setdefault(key,{}),out/'neighbors'/f'{seed}__{key}')
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
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['inventory','run']);ap.add_argument('--job');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 if args.command=='inventory':inventory()
 else:run_job(args.job,args.smoke)
