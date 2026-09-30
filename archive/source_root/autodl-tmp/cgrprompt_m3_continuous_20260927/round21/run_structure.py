"""Finite, predeclared M3 structural screen. Query labels only after sealing."""
import sys
from pathlib import Path
OLD=Path('/root/autodl-tmp/cgrprompt_validation_20260926')
sys.path.insert(0,str(OLD))
from run_core import *
ROOT=Path(__file__).resolve().parent
VARIANTS=['huber01','huber03']
SOURCES=['Prompt','Teacher_uncentered','Teacher_centered']

@torch.no_grad()
def calibrated(score,op,si,sy,C):
 labels=score.argmax(1);ix=op.indices();agree=torch.zeros(len(score),device=score.device).scatter_add_(0,ix[0],op.values()*(labels[ix[0]]==labels[ix[1]]).float());return propagate(score/(1+3*(1-agree))[:,None],op,si,sy,C)

@torch.no_grad()
def structural(score,op,si,sy,C,variant,seed,shuffle=False,zero=False,source=None,scores=None,mean=False,fixed=False):
 ix=op.indices();n=len(score);assert op._nnz()==n*50 and torch.equal(ix[0],torch.arange(n,device=score.device).repeat_interleave(50))
 neighbors=ix[1].reshape(n,50);weights=op.values().reshape(n,50,1);labels=score.argmax(1)
 agree=(weights.squeeze(-1)*(labels[neighbors]==labels[:,None]).float()).sum(1)
 initial=(score if fixed else score/(1+3*(1-agree))[:,None]).softmax(-1);p=initial.clone();y=F.one_hot(sy,C).float();p[si]=y
 delta=2. if zero else .1 if variant=='huber01' else .3
 gen=torch.Generator(device=score.device).manual_seed(seed+701);perm=torch.rand((n,50),generator=gen,device=score.device).argsort(1)
 for _ in range(20):
  values=p[neighbors];center=(weights*values).sum(1)
  for _ in range(3):
   distance=torch.linalg.vector_norm(values-center[:,None,:],dim=-1,keepdim=True);factor=(delta/distance.clamp_min(1e-8)).clamp_max(1.)
   if shuffle:factor=factor.gather(1,perm[:,:,None])
   if mean:factor=factor.mean(1,keepdim=True).expand_as(factor)
   w=weights*factor;center=(w*values).sum(1)/w.sum(1).clamp_min(1e-12)
  p=.1*initial+.9*center;p[si]=y
 assert torch.isfinite(p).all() and p.min()>=0 and (p.sum(1)-1).abs().max()<2e-6
 return p.clamp_min(1e-12).log()


def run(job,stage,variants):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert job['phase']=='main' and job['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);filters=A.make_filters(x,edges)
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 seeds=range(26094800,26094801) if stage=='smoke' else range(26094800,26094810) if stage=='screen' else range(26094820,26094840)
 out=ROOT/stage/job['id'];out.mkdir(parents=True,exist_ok=True);steps=2 if stage=='smoke' else 100
 for seed in seeds:
  folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'result.json'
  if done.exists():
   old=json.loads(done.read_text());assert old['runner_sha256']==sha(__file__) and old['prediction_sha256']==sha(folder/'prediction.npz');continue
  save(ROOT/'current.json',dict(stage=stage,job=job['id'],seed=seed,time=time.time()));gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);assert not torch.isin(support,query).any();si=support.cuda();sy=data.y[support].cuda();start=time.perf_counter()
  z,params,trace=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True)
  with torch.no_grad():
   scores=dict(Prompt=A.logits(z,si,sy,C),Teacher_uncentered=A.rectified_teacher(r,si,sy,C,'rect_teacher_ref50',seed)[0],Teacher_centered=A.rectified_teacher(rc,si,sy,C,'rect_teacher_ref50',seed)[0]);arrays={};checks={}
   for source,score in scores.items():
    views={'fixed':propagate(score,op,si,sy,C),'none':score,'localtemp':calibrated(score,op,si,sy,C)}
    for v in variants:
     views[v]=structural(score,op,si,sy,C,v,seed,source=source,scores=scores)
     if source!='Prompt':views[v+'_fixed']=structural(score,op,si,sy,C,v,seed,source=source,scores=scores,fixed=True)
    if source=='Prompt':
     adapted,_,_=knn(z);views['original_M3']=propagate(score,adapted,si,sy,C)
     for v in variants:
      views[v+'_shuffled']=structural(score,op,si,sy,C,v,seed,shuffle=True,source=source,scores=scores)
      views[v+'_constant']=structural(score,op,si,sy,C,v,seed,mean=True,source=source,scores=scores)
     for v in variants:
      noop=structural(score,op,si,sy,C,v,seed,zero=True,source=source,scores=scores);err=float((noop.softmax(-1)-views['localtemp'].softmax(-1)).abs().max());assert err<2e-6;checks[v+'_identity_error']=err
    for v,value in views.items():arrays[source+'__'+v]=value[query.cuda()].double().softmax(-1).cpu().numpy()
   assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
  np.savez_compressed(folder/'initial_scores.npz',**{k:v.cpu().numpy() for k,v in scores.items()});np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=query.numpy(),**arrays);ph=sha(folder/'prediction.npz');torch.save(params,folder/'parameters.pt');sync();elapsed=time.perf_counter()-start
  y=data.y[query].numpy();np.save(out/'query_labels.npy',y)
  row=dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],stage=stage,seed=seed,steps=steps,runner_sha256=sha(__file__),core_runner_sha256=sha(OLD/'run_core.py'),prediction_sha256=ph,split_sha256=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest(),frozen_verified=True,query_label_free_operator=True,identity_checks=checks,seconds=elapsed,metrics={k:metrics(y,p) for k,p in arrays.items()});save(done,row);print(json.dumps(dict(job=job['id'],seed=seed,seconds=elapsed)),flush=True)
 save(out/'complete.json',dict(count=len(list(out.glob('*/result.json'))),expected=len(seeds)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','screen','confirm'],required=True);ap.add_argument('--variant',choices=VARIANTS);args=ap.parse_args();assert args.stage!='confirm' or args.variant
 variants=[args.variant] if args.variant else VARIANTS;jobs=[j for j in json.loads((OLD/'plan.json').read_text())['jobs'] if j['phase']=='main' and j['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']];assert len(jobs)==4
 save(ROOT/(args.stage+'_plan.json'),dict(jobs=jobs,variants=variants,runner_sha256=sha(__file__),core_runner_sha256=sha(OLD/'run_core.py'),design_sha256=sha(ROOT/'设计与判定.md')))
 try:
  for job in jobs[:1] if args.stage=='smoke' else jobs:run(job,args.stage,variants)
  save(ROOT/(args.stage+'_complete.json'),dict(finished=time.time(),jobs=1 if args.stage=='smoke' else 4))
 except Exception as ex:save(ROOT/(args.stage+'_failure.json'),dict(error=repr(ex),traceback=traceback.format_exc()));raise
