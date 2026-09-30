from pathlib import Path
import sys
OLD=Path('/root/autodl-tmp/cgrprompt_validation_20260926');sys.path.insert(0,str(OLD))
from run_core import *
import optimizer_graph as G
ROOT=Path(__file__).resolve().parent
VARIANTS=['graph50','fold50','shuffle50']
@torch.no_grad()
def calibrated(score,op,si,sy,C):
 ix=op.indices();labels=score.argmax(1);agree=torch.zeros(len(score),device=score.device).scatter_add_(0,ix[0],op.values()*(labels[ix[0]]==labels[ix[1]]).float());return propagate(score/(1+3*(1-agree))[:,None],op,si,sy,C)
def run(job,stage):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert job['phase']=='main' and job['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']
 assert g.verify_sources()==json.loads((BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=knn(rc);filters=A.make_filters(x,edges)
 ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 steps=2 if stage=='smoke' else 100;seeds=range(26093800,26093801) if stage=='smoke' else range(26093800,26093810) if stage=='screen' else range(26093820,26093840)
 out=ROOT/stage/job['id'];out.mkdir(parents=True,exist_ok=True)
 for seed in seeds:
  folder=out/str(seed);folder.mkdir(exist_ok=True);done=folder/'result.json'
  if done.exists():
   previous=json.loads(done.read_text());assert previous['runner_sha256']==sha(__file__) and previous['optimizer_sha256']==sha(ROOT/'optimizer_graph.py') and previous['prediction_sha256']==sha(folder/'prediction.npz');continue
  gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda();assert not torch.isin(support,query).any();traces={};zs={};times={};parity={}
  gpu_gen=torch.Generator(device=x.device).manual_seed(seed+777);perm=torch.randperm(len(x),generator=gpu_gen,device=x.device);shuffled=torch.sparse_coo_tensor(perm[op.indices()],op.values(),op.shape).coalesce()
  args=(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',steps,filters,'none',.1,'raw','gshampoo',.01)
  for variant in ['baseline']+VARIANTS:
   save(ROOT/'current.json',dict(stage=stage,job=job['id'],seed=seed,variant=variant,time=time.time()));sync();start=time.perf_counter()
   if variant=='baseline':z,p,trace=A.optimizer_prompt(*args)
   else:z,p,trace=G.optimizer_prompt(*args,graph_op=shuffled if variant=='shuffle50' else op,graph_weight=.5,graph_kind='fold' if variant=='fold50' else 'graph')
   if stage=='smoke' and variant=='baseline':
    zero,zp,_=G.optimizer_prompt(*args,graph_op=op,graph_weight=0);parity=dict(output_error=float((z-zero).abs().max()),token_error=float((p['tokens']-zp['tokens']).abs().max()));assert max(parity.values())<2e-6
   assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters());sync();times[variant]=time.perf_counter()-start;zs[variant]=z-z.mean(0,keepdim=True);traces[variant]=trace;torch.save(p,folder/(variant+'_parameters.pt'))
  with torch.no_grad():
   arrays={};scores={};base=A.logits(zs['baseline'],si,sy,C);scores['Prompt']=base
   values={'Prompt__fixed':propagate(base,op,si,sy,C),'Prompt__localtemp':calibrated(base,op,si,sy,C),'Prompt__none':base};adapted,_,_=knn(zs['baseline']);values['Prompt__original_M3']=propagate(base,adapted,si,sy,C)
   for source,representation in [('Teacher_uncentered',r),('Teacher_centered',rc)]:
    s=A.rectified_teacher(representation,si,sy,C,'rect_teacher_ref50',seed)[0];scores[source]=s;values[source+'__fixed']=propagate(s,op,si,sy,C);values[source+'__none']=s
   for variant in VARIANTS:
    s=A.logits(zs[variant],si,sy,C);scores[variant]=s;values['Prompt__'+variant]=calibrated(s,op,si,sy,C)
   for k,v in values.items():arrays[k]=v[query.cuda()].double().softmax(-1).cpu().numpy()
  np.savez_compressed(folder/'prediction.npz',support=support.numpy(),query=query.numpy(),**arrays);ph=sha(folder/'prediction.npz');np.savez_compressed(folder/'initial_scores.npz',**{k:v.cpu().numpy() for k,v in scores.items()});save(folder/'training_traces.json',traces)
  y=data.y[query].numpy();np.save(out/'query_labels.npy',y);save(done,dict(job=job['id'],dataset=job['dataset'],checkpoint=ck['family'],checkpoint_sha256=ck['sha256'],seed=seed,stage=stage,steps=steps,prompt_fits=4,runner_sha256=sha(__file__),optimizer_sha256=sha(ROOT/'optimizer_graph.py'),core_runner_sha256=sha(OLD/'run_core.py'),prediction_sha256=ph,split_sha256=hashlib.sha256((A.thash(support)+A.thash(query)).encode()).hexdigest(),frozen_verified=True,query_label_free_operator=True,zero_parity=parity,times=times,metrics={k:metrics(y,p) for k,p in arrays.items()}));print(json.dumps(dict(job=job['id'],seed=seed,times=times)),flush=True)
 save(out/'complete.json',dict(count=len(list(out.glob('*/result.json'))),expected=len(seeds)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['smoke','screen','confirm']);args=ap.parse_args();jobs=[j for j in json.loads((OLD/'plan.json').read_text())['jobs'] if j['phase']=='main' and j['checkpoint']['family'] in ['Edgepred_Gprompt','Edgepred_GPPT']];assert len(jobs)==4
 save(ROOT/(args.stage+'_plan.json'),dict(jobs=jobs,variants=VARIANTS,candidate='graph50',runner_sha256=sha(__file__),optimizer_sha256=sha(ROOT/'optimizer_graph.py'),design_sha256=sha(ROOT/'设计与判定.md')))
 try:
  for job in jobs[:1] if args.stage=='smoke' else jobs:run(job,args.stage)
  save(ROOT/(args.stage+'_complete.json'),dict(finished=time.time(),episodes=1 if args.stage=='smoke' else 40 if args.stage=='screen' else 80))
 except Exception as ex:save(ROOT/(args.stage+'_failure.json'),dict(error=repr(ex),traceback=traceback.format_exc()));raise
