"""Dedicated M3-T cost audit. Run only after all accuracy experiments finish."""
from run_m3t_optimization import *

def run_cost(smoke=False):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 jobs=[j for j in json.loads((ROOT/'plan.json').read_text())['jobs'] if j['phase']=='main' or j['dataset']=='PubMed'];rows=[]
 if smoke:jobs=jobs[:1]
 for job in jobs:
  data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);filters=A.make_filters(x,edges);ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:A.thash(v) for k,v in model.state_dict().items()}
  with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach();z0-=z0.mean(0,keepdim=True)
  def score_for(source,si,sy,seed):
   if source=='Prompt':
    z,_,_=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',2 if smoke else 100,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True)
    return A.logits(z,si,sy,C).detach(),z
   with torch.no_grad():
    if source.startswith('Teacher'):score=A.rectified_teacher(rc if source=='Teacher_centered' else r,si,sy,C,'rect_teacher_ref50',seed)[0]
    else:score=A.logits(rc if source=='AttributeProto' else z0,si,sy,C)
    return score,None
  configs=[('Prompt','original')]+[(source,variant) for source in SOURCES for variant in ['fixed','selected']]
  for seed in range(26092740,26092741 if smoke else 26092745):
   gen=torch.Generator().manual_seed(seed);chunks=[p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]];support=torch.cat(chunks);train=torch.cat([p[:4] for p in chunks]);valid=torch.cat([p[4:] for p in chunks]);si=support.cuda();sy=data.y[support].cuda();ti=train.cuda();ty=data.y[train].cuda();vi=valid.cuda();vy=data.y[valid].cuda()
   for source,variant in configs:
    torch.cuda.empty_cache();sync();base_mem=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter();ops={};graph_seconds=0.;selection_seconds=0.;config=dict(k=50,alpha=.9)
    if variant!='original':
     t=time.perf_counter()
     for k in KS if variant=='selected' else [50]:ops[k]=knn_k(rc,k)[0]
     sync();graph_seconds=time.perf_counter()-t
    if variant=='selected':
     t=time.perf_counter();inner,inner_z=score_for(source,ti,ty,seed);trials=[]
     with torch.no_grad():
      for candidate in CONFIGS:
       v=readout(inner,ops,ti,ty,C,candidate);ce=float(F.cross_entropy(v[vi],vy));assert np.isfinite(ce);trials.append(dict(**candidate,validation_ce=ce))
     config=min(trials,key=lambda t:(t['validation_ce'],t['alpha'],t['k']));del inner,inner_z,v;sync();selection_seconds=time.perf_counter()-t
    t=time.perf_counter();score,z=score_for(source,si,sy,seed);sync();final_score_seconds=time.perf_counter()-t;t=time.perf_counter()
    with torch.no_grad():
     if variant=='original':
      op,graph_ids,graph_weights=knn(z);prediction=propagate(score,op,si,sy,C);del op,graph_ids,graph_weights
     else:prediction=readout(score,ops,si,sy,C,config)
    sync();final_readout_seconds=time.perf_counter()-t;total=time.perf_counter()-t0;assert torch.isfinite(prediction).all();assert before=={k:A.thash(v) for k,v in model.state_dict().items()} and all(p.grad is None for p in model.parameters())
    row=dict(job=job['id'],seed=seed,method=source+'_'+variant,smoke=smoke,total_seconds=total,graph_seconds=graph_seconds,selection_seconds=selection_seconds,final_score_seconds=final_score_seconds,final_readout_seconds=final_readout_seconds,without_graph_construction_seconds=total-graph_seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-base_mem,selected=config,runner_sha256=sha(__file__),operator_sha256=sha(ROOT/'run_m3t_optimization.py'),frozen_verified=True);rows.append(row);save(ROOT/('m3t_cost_smoke_results.json' if smoke else 'm3t_cost_results.json'),rows);print(json.dumps(row),flush=True);del prediction,score,z,ops;torch.cuda.empty_cache()
  del model,u,z0,filters;torch.cuda.empty_cache()
 expected=11 if smoke else 495;assert len(rows)==expected;save(ROOT/('m3t_cost_smoke_complete.json' if smoke else 'm3t_cost_complete.json'),dict(finished=time.time(),rows=len(rows),scope='6main+3PubMed,5seeds,11methods; preloaded data and frozen forward; cold graph timing and phase decomposition; no separate cached rerun'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();plan=json.loads((ROOT/'m3t_cost_plan.json').read_text());assert plan['runner_sha256']==sha(__file__) and plan['operator_sha256']==sha(ROOT/'run_m3t_optimization.py');run_cost(args.smoke)
