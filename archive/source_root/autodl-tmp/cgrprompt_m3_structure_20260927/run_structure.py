"""Finite, predeclared M3 structural screen. Query labels only after sealing."""
import sys
from pathlib import Path
OLD=Path('/root/autodl-tmp/cgrprompt_validation_20260926')
sys.path.insert(0,str(OLD))
from run_core import *
ROOT=Path(__file__).resolve().parent
VARIANTS=['correction05','correction10','affinity1','affinity2']
SOURCES=['Prompt','Teacher_uncentered','Teacher_centered']

@torch.no_grad()
def structural(score,op,si,sy,C,variant,seed,shuffle=False,zero=False):
 p=score.softmax(-1);target=F.one_hot(sy,C).float()
 gen=torch.Generator(device=score.device).manual_seed(seed+701)
 if variant.startswith('correction'):
  e0=torch.zeros_like(p);residual=target-p[si]
  if shuffle:residual=residual[torch.randperm(len(si),generator=gen,device=score.device)]
  e0[si]=residual;m0=torch.zeros_like(p[:,:1]);m0[si]=1;e=e0.clone();mass=m0.clone()
  for _ in range(50):
   e=.1*e0+.9*torch.sparse.mm(op,e);mass=.1*m0+.9*torch.sparse.mm(op,mass);e[si]=e0[si];mass[si]=1
  beta=0 if zero else .5 if variant=='correction05' else 1.
  p=(p+beta*e/mass.clamp_min(1e-8)).clamp_min(1e-8);p=p/p.sum(1,keepdim=True)
  return propagate(p.log(),op,si,sy,C)
 q=(score/.25).softmax(-1);q[si]=target
 if shuffle:q=q[torch.randperm(len(q),generator=gen,device=score.device)]
 ix=op.indices();agreement=(q[ix[0]].sqrt()*q[ix[1]].sqrt()).sum(1).clamp(0,1)
 factor=torch.ones_like(agreement) if zero else .1+.9*agreement.pow(1 if variant=='affinity1' else 2)
 w=op.values()*factor;den=torch.zeros(len(p),device=p.device).scatter_add_(0,ix[0],w);w=w/den[ix[0]]
 newop=torch.sparse_coo_tensor(ix,w,op.shape).coalesce()
 return propagate(score,newop,si,sy,C)


def run(job,stage,variants):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert job['phase']=='main' and job['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);filters=A.make_filters(x,edges)
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 seeds=range(26092800,26092801) if stage=='smoke' else range(26092800,26092810) if stage=='screen' else range(26092820,26092840)
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
     for v in ['correction05','affinity1']:
      noop=structural(score,op,si,sy,C,v,seed,zero=True);err=float((noop.softmax(-1)-views['fixed'].softmax(-1)).abs().max());assert err<2e-6;checks[v+'_identity_error']=err
    for v,value in views.items():arrays[source+'__'+v]=value[query.cuda()].double().softmax(-1).cpu().numpy()
   assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
  np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=query.numpy(),**arrays);ph=sha(folder/'prediction.npz');torch.save(params,folder/'parameters.pt');sync();elapsed=time.perf_counter()-start
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
