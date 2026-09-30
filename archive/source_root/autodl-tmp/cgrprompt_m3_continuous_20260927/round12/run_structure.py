"""Finite, predeclared M3 structural screen. Query labels only after sealing."""
import sys
from pathlib import Path
OLD=Path('/root/autodl-tmp/cgrprompt_validation_20260926')
sys.path.insert(0,str(OLD))
from run_core import *
ROOT=Path(__file__).resolve().parent
VARIANTS=['repulse05','repulse10','attract10','perm10']
SOURCES=['Prompt','Teacher_uncentered','Teacher_centered']

@torch.no_grad()
def calibrated(score,op,si,sy,C):
 labels=score.argmax(1);ix=op.indices();agree=torch.zeros(len(score),device=score.device).scatter_add_(0,ix[0],op.values()*(labels[ix[0]]==labels[ix[1]]).float());return propagate(score/(1+3*(1-agree))[:,None],op,si,sy,C)

@torch.no_grad()
def farthest(z):
 zn=F.normalize(z,dim=1);ids=[];weights=[]
 for st in range(0,len(z),512):
  en=min(st+512,len(z));sim=-(zn[st:en]@zn.T);sim[torch.arange(en-st,device=z.device),torch.arange(st,en,device=z.device)]=-torch.inf;v,j=sim.topk(min(50,len(z)-1),dim=1);ids.append(j);weights.append((v/.2).softmax(1))
 ids=torch.cat(ids);w=torch.cat(weights);rows=torch.arange(len(z),device=z.device)[:,None].expand_as(ids);return torch.sparse_coo_tensor(torch.stack([rows.flatten(),ids.flatten()]),w.flatten(),(len(z),len(z))).coalesce()

@torch.no_grad()
def structural(score,op,si,sy,C,variant,seed,shuffle=False,zero=False,source=None,scores=None,negative=None):
 if source=='Prompt':
  ix=op.indices();labels=score.argmax(1);agree=torch.zeros(len(score),device=score.device).scatter_add_(0,ix[0],op.values()*(labels[ix[0]]==labels[ix[1]]).float());score=score/(1+3*(1-agree))[:,None]
 beta=0 if zero else .05 if variant=='repulse05' else -.1 if variant=='attract10' else .1
 neg=negative
 if shuffle or variant=='perm10':
  gen=torch.Generator(device=score.device).manual_seed(seed+701);perm=torch.randperm(len(score),generator=gen,device=score.device);neg=torch.sparse_coo_tensor(perm[negative.indices()],negative.values(),negative.shape).coalesce()
 p0=score.softmax(-1);p=p0.clone();target=F.one_hot(sy,C).float();p[si]=target
 for _ in range(20):
  p=(.1*p0+.9*torch.sparse.mm(op,p)-beta*(torch.sparse.mm(neg,p)-1/C)).clamp_min(1e-8);p=p/p.sum(1,keepdim=True);p[si]=target
 assert torch.isfinite(p).all() and (p.sum(1)-1).abs().max()<2e-6
 return p.clamp_min(1e-12).log()


def run(job,stage,variants):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert job['phase']=='main' and job['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);negative=farthest(rc);filters=A.make_filters(x,edges)
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 seeds=range(26093900,26093901) if stage=='smoke' else range(26093900,26093910) if stage=='screen' else range(26093920,26093940)
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
    for v in variants:views[v]=structural(score,op,si,sy,C,v,seed,source=source,scores=scores,negative=negative)
    if source=='Prompt':
     adapted,_,_=knn(z);views['original_M3']=propagate(score,adapted,si,sy,C)
     for v in variants:views[v+'_shuffled']=structural(score,op,si,sy,C,v,seed,shuffle=True,source=source,scores=scores,negative=negative)
     for v in variants:
      noop=structural(score,op,si,sy,C,v,seed,zero=True,source=source,scores=scores,negative=negative);err=float((noop.softmax(-1)-views['localtemp'].softmax(-1)).abs().max());assert err<2e-6;checks[v+'_identity_error']=err
    for v,value in views.items():arrays[source+'__'+v]=value[query.cuda()].double().softmax(-1).cpu().numpy()
   assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
  np.savez_compressed(folder/'initial_scores.npz',**{k:v.cpu().numpy() for k,v in scores.items()});np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=query.numpy(),**arrays);ph=sha(folder/'prediction.npz');torch.save(params,folder/'parameters.pt');sync();elapsed=time.perf_counter()-start
  y=data.y[query].numpy();np.save(out/'query_labels.npy',y)
  row=dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],stage=stage,seed=seed,steps=steps,runner_sha256=sha(__file__),core_runner_sha256=sha(OLD/'run_core.py'),prediction_sha256=ph,split_sha256=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest(),frozen_verified=True,query_label_free_operator=True,identity_checks=checks,negative_graph_hash=dict(indices=A.thash(negative.indices()),weights=A.thash(negative.values())),seconds=elapsed,metrics={k:metrics(y,p) for k,p in arrays.items()});save(done,row);print(json.dumps(dict(job=job['id'],seed=seed,seconds=elapsed)),flush=True)
 save(out/'complete.json',dict(count=len(list(out.glob('*/result.json'))),expected=len(seeds)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','screen','confirm'],required=True);ap.add_argument('--variant',choices=VARIANTS);args=ap.parse_args();assert args.stage!='confirm' or args.variant
 variants=[args.variant] if args.variant else VARIANTS;jobs=[j for j in json.loads((OLD/'plan.json').read_text())['jobs'] if j['phase']=='main' and j['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']];assert len(jobs)==4
 save(ROOT/(args.stage+'_plan.json'),dict(jobs=jobs,variants=variants,runner_sha256=sha(__file__),core_runner_sha256=sha(OLD/'run_core.py'),design_sha256=sha(ROOT/'设计与判定.md')))
 try:
  for job in jobs[:1] if args.stage=='smoke' else jobs:run(job,args.stage,variants)
  save(ROOT/(args.stage+'_complete.json'),dict(finished=time.time(),jobs=1 if args.stage=='smoke' else 4))
 except Exception as ex:save(ROOT/(args.stage+'_failure.json'),dict(error=repr(ex),traceback=traceback.format_exc()));raise
