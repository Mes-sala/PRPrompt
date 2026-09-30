"""Frozen corrected-backbone CGRPrompt evaluation; all views predeclared."""
from run_core import *
from allocation import fit as allocation_fit
import fcntl
from torch_geometric.data import Data
ROOT=Path(__file__).resolve().parents[1]
BASELINES=['GPPT','GPF','GPF+','GPrompt','EdgePrompt','GraphTOP','UniPrompt']
ROUTES=['permuted','random','learned_hidden','learned_teacher','learned_budget','fixed_rank']

def local(score,op,si,sy,C,seed,mode='local',alpha=.9,clamp=True):
 labels=score.argmax(1);ix=op.indices();a=torch.zeros(len(score),device=score.device).scatter_add_(0,ix[0],op.values()*(labels[ix[0]]==labels[ix[1]]).float());t=1+3*(1-a)
 if mode=='shuffle':t=t[torch.randperm(len(t),generator=torch.Generator(device=t.device).manual_seed(seed+701),device=t.device)]
 if mode=='mean':t=t.mean().expand_as(t)
 return propagate(score/t[:,None],op,si,sy,C,alpha=alpha,clamp=clamp)

def run(ds,fam,smoke):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 pre=json.loads((ROOT/'pretraining'/f'{ds}__{fam}'/'complete.json').read_text());ck=pre['checkpoint_path'];assert sha(ck)==pre['checkpoint_sha256']
 contract=json.loads((ROOT/'downstream_ready.json').read_text());assert all(sha(ROOT/k)==v for k,v in contract['sources'].items())
 identity=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest();job=f'{ds}__{fam}';out=ROOT/('smoke' if smoke else 'results')/job;out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'execution.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
 data,pool,q,C,protocol=g.data_load(ds);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);filters=A.make_filters(x,edges)
 gh=hashlib.sha256((A.thash(data.x)+A.thash(data.edge_index)).encode()).hexdigest();model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach()
 seeds=[26093220] if smoke else contract['seeds'];steps=2 if smoke else 100;bcache={}
 for seed in seeds:
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda();assert not torch.isin(support,q).any();split=hashlib.sha256((A.thash(support)+A.thash(q)).encode()).hexdigest()
  def begin(name):
   folder=out/str(seed)/name;folder.mkdir(parents=True,exist_ok=True)
   if (folder/'result.json').exists():
    row=json.loads((folder/'result.json').read_text());assert row['identity']==identity and row['checkpoint_sha256']==pre['checkpoint_sha256'] and row['graph_sha256']==gh and row['split_sha256']==split and sha(folder/'prediction.npz')==row['prediction_sha256'];return None
   save(ROOT/'downstream_status.json',dict(job=job,seed=seed,bundle=name,smoke=smoke,time=time.time()));torch.cuda.reset_peak_memory_stats();return folder,time.perf_counter()
  def seal(ticket,views,params=None,trace=None):
   folder,start=ticket
   with torch.no_grad():arrays={k:v[q.cuda()].detach().double().softmax(-1).cpu().numpy() for k,v in views.items()}
   assert all(np.isfinite(v).all() and abs(v.sum(1)-1).max()<1e-10 for v in arrays.values());assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and all(p.grad is None for p in model.parameters()) and not model.training
   np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=q.numpy(),**arrays);ph=sha(folder/'prediction.npz')
   if params is not None:torch.save(params,folder/'parameters.pt')
   sync();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();y=data.y[q].numpy();np.save(out/'query_labels.npy',y)
   save(folder/'result.json',dict(job=job,dataset=ds,family=fam,seed=seed,bundle=folder.name,steps=steps,smoke=smoke,identity=identity,checkpoint_sha256=pre['checkpoint_sha256'],graph_sha256=gh,split_sha256=split,prediction_sha256=ph,frozen_verified=True,seconds=elapsed,peak_gpu_bytes=peak,trace=trace or {},metrics={k:metrics(y,p) for k,p in arrays.items()}));print(json.dumps(dict(job=job,seed=seed,bundle=folder.name,seconds=elapsed)),flush=True)
  ticket=begin('teachers')
  if ticket:
   with torch.no_grad():
    views={}
    for name,rr in [('Teacher_uncentered',r),('Teacher_centered',rc)]:
     sc=A.rectified_teacher(rr,si,sy,C,'rect_teacher_ref50',seed)[0];views[name+'__none']=sc;views[name+'__fixed']=propagate(sc,op,si,sy,C);views[name+'__localtemp']=local(sc,op,si,sy,C,seed)
    sc=A.logits(r,si,sy,C);views['AttributeProto']=sc
    sc=A.logits(z0-z0.mean(0,keepdim=True),si,sy,C);views['Frozen']=sc;views['Frozen_teachergraph']=propagate(sc,op,si,sy,C)
   seal(ticket,views)
  for sharp in [0,1]:
   for matrix in [0,1]:
    name=f'fit_F{sharp}{matrix}';ticket=begin(name)
    if not ticket:continue
    z,params,tr=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1 if sharp else .2,'raw','gshampoo' if matrix else 'adam',.01);z=z-z.mean(0,keepdim=True)
    with torch.no_grad():
     sc=A.logits(z,si,sy,C);views={f'F{sharp}{matrix}0':sc,f'F{sharp}{matrix}1':local(sc,op,si,sy,C,seed)}
     if sharp==matrix==1:
      adapted,_,_=knn(z);views.update(Prompt__none=sc,Prompt__original_M3=propagate(sc,adapted,si,sy,C),Prompt__fixed=propagate(sc,op,si,sy,C),Prompt__localtemp=views['F111'],Prompt__localtemp_shuffled=local(sc,op,si,sy,C,seed,'shuffle'),Prompt__mean_temperature=local(sc,op,si,sy,C,seed,'mean'),Prompt__uniform=propagate(torch.zeros_like(sc),op,si,sy,C),Prompt__no_clamp=local(sc,op,si,sy,C,seed,clamp=False),Prompt__no_restart=local(sc,op,si,sy,C,seed,alpha=1.))
      np.savez_compressed(ticket[0]/'initial_scores.npz',Prompt=sc.cpu().numpy())
    seal(ticket,views,params,{'optimizer': 'gshampoo' if matrix else 'adam','temperature':.1 if sharp else .2,'parameters':C*128,'prompt_noop_error':tr['prompt_noop_norm_error']})
  for name,mode,opt in [('M1_normmatched','matched','gshampoo'),('M1_amplitude_only','amplitude','gshampoo'),('M2_diagonal','raw','gdiagonal'),('M2_ungrafted','raw','shampoo')]:
   ticket=begin(name)
   if not ticket:continue
   z,params,tr=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,mode,opt,.01);z=z-z.mean(0,keepdim=True)
   with torch.no_grad():sc=A.logits(z,si,sy,C);views={name+'__none':sc,name+'__LC':local(sc,op,si,sy,C,seed)}
   seal(ticket,views,params,{'parameters':C*128,'optimizer':opt,'mode':mode})
  for route in ROUTES:
   ticket=begin('allocation_'+route)
   if not ticket:continue
   z,tr=allocation_fit(u,r,adj,model,edges,si,sy,C,seed,route,'adam',.01,steps,(steps,))[steps]
   with torch.no_grad():sc=A.logits(z,si,sy,C);views={'allocation_'+route+'__none':sc,'allocation_'+route+'__LC':local(sc,op,si,sy,C,seed)}
   params={k:tr.pop(k) for k in ['prompt','router']};seal(ticket,views,params,tr)
  for method in BASELINES:
   ticket=begin('baseline_'+method)
   if not ticket:continue
   safe=Data(x=data.x,edge_index=data.edge_index,num_nodes=data.num_nodes);scores,params,tr=g.train_baseline(method,model,safe,support,sy,q,C,seed,steps,bcache);seal(ticket,{method:scores},params,tr)
  torch.cuda.empty_cache()
 files=list(out.glob('*/*/result.json'));expected=len(seeds)*22;assert len(files)==expected,(len(files),expected);save(out/'complete.json',dict(records=len(files),expected=expected,identity=identity,checkpoint_sha256=pre['checkpoint_sha256'],time=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--family',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();run(a.dataset,a.family,a.smoke)
