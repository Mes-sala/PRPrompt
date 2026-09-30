"""Finite, predeclared M3 structural screen. Query labels only after sealing."""
import sys
from pathlib import Path
OLD=Path('/root/autodl-tmp/cgrprompt_validation_20260926')
sys.path.insert(0,str(OLD))
from run_core import *
ROOT=Path(__file__).resolve().parent
VARIANTS=['multiscale','adaptive','logit','balanced']
SOURCES=['Prompt','Teacher_uncentered','Teacher_centered']

@torch.no_grad()
def structural(score,op,si,sy,C,variant,seed,shuffle=False,zero=False):
 if zero:return propagate(score,op,si,sy,C)
 gen=torch.Generator(device=score.device).manual_seed(seed+701)
 if shuffle and variant in ['multiscale','logit']:
  ids=torch.arange(len(score),device=score.device);ids=ids[~torch.isin(ids,si)];score=score.clone();score[ids]=score[ids[torch.randperm(len(ids),generator=gen,device=score.device)]]
 p0=score.softmax(-1);target=F.one_hot(sy,C).float()
 if variant=='balanced':
  p=propagate(score,op,si,sy,C).softmax(-1);mass=p.mean(0)
  if shuffle:mass=mass[torch.randperm(C,generator=gen,device=score.device)]
  p=p/mass.clamp_min(1e-8).sqrt();p=p/p.sum(1,keepdim=True);return p.clamp_min(1e-12).log()
 if variant=='logit':
  p0=score-score.mean(1,keepdim=True);p0=p0/p0.std(1,keepdim=True,unbiased=False).clamp_min(1e-6);target=C**.5*(target-1/C)
 alpha=.9
 if variant=='adaptive':
  entropy=-(p0*p0.clamp_min(1e-12).log()).sum(1,keepdim=True)/np.log(C);alpha=.5+.45*entropy
  if shuffle:alpha=alpha[torch.randperm(len(alpha),generator=gen,device=score.device)]
 p=p0.clone();p[si]=target;scales=[]
 for step in range(1,21):
  p=(1-alpha)*p0+alpha*torch.sparse.mm(op,p);p[si]=target
  if step in [1,5,20]:scales.append(p.clone())
 if variant=='multiscale':p=torch.stack(scales).mean(0)
 return p if variant=='logit' else p.clamp_min(1e-12).log()


def run(job,stage,variants):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert job['phase']=='main' and job['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);filters=A.make_filters(x,edges)
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 seeds=range(26092900,26092901) if stage=='smoke' else range(26092900,26092910) if stage=='screen' else range(26092920,26092940)
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
    views={'fixed':propagate(score,op,si,sy,C),'none':score}
    for v in variants:views[v]=structural(score,op,si,sy,C,v,seed)
    if source=='Prompt':
     adapted,_,_=knn(z);views['original_M3']=propagate(score,adapted,si,sy,C)
     for v in variants:views[v+'_shuffled']=structural(score,op,si,sy,C,v,seed,shuffle=True)
     for v in variants:
      noop=structural(score,op,si,sy,C,v,seed,zero=True);err=float((noop.softmax(-1)-views['fixed'].softmax(-1)).abs().max());assert err<2e-6;checks[v+'_identity_error']=err
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
