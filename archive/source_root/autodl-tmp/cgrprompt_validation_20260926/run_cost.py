"""Dedicated sequential cost pass after all other experiment processes finish."""
from run_core import *
from allocation import fit

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 plan=json.loads((ROOT/'plan.json').read_text());rows=[]
 for job in plan['jobs']:
  if job['phase']!='main' and job['dataset']!='PubMed':continue
  data,pool,query,C,protocol=g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();ck=job['checkpoint'];assert sha(ck['path'])==ck['sha256'];model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False)
  with torch.no_grad():u=model.conv_layers[0](x,edges).detach();z0=model.conv_layers[1](g.act(u),edges).detach()
  for idx,seed in enumerate(job['seeds'][:5]):
   gen=torch.Generator().manual_seed(seed);si=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]).cuda();sy=data.y[si.cpu()].cuda()
   for name,route,opt in [('CGRPrompt','fixed','gshampoo'),('AdamPrompt','fixed','adam'),('LearnedRouter','learned_hidden','adam'),('DirectTeacher',None,None),('FrozenPropagation',None,None)]:
    sync();torch.cuda.reset_peak_memory_stats();base_mem=torch.cuda.memory_allocated();t0=time.perf_counter();fit_seconds=0.;construction=0.;readout=0.;params=0
    if route:
     snaps=fit(u,r,adj,model,edges,si,sy,C,seed,route,opt,.01,100,(100,));z,tr=snaps[100];sync();fit_total=time.perf_counter()-t0;fit_seconds=tr['seconds'];construction=fit_total-fit_seconds;params=tr['parameters']
    elif name=='DirectTeacher':
     score=A.rectified_teacher(r,si,sy,C,'rect_teacher_ref50',seed)[0];sync();construction=time.perf_counter()-t0
    else:z=z0-z0.mean(0,keepdim=True)
    if name!='DirectTeacher':
     rt=time.perf_counter()
     with torch.no_grad():op,ids,weights=knn(z);score=propagate(A.logits(z,si,sy,C),op,si,sy,C)
     sync();readout=time.perf_counter()-rt
    sync();row=dict(job=job['id'],seed=seed,repeat=idx,method=name,total_seconds=time.perf_counter()-t0,construction_seconds=construction,fit_seconds=fit_seconds,readout_seconds=readout,peak_gpu_bytes=torch.cuda.max_memory_allocated(),incremental_peak_bytes=torch.cuda.max_memory_allocated()-base_mem,parameters=params,runner_sha256=sha(__file__),operator_sha256=sha(ROOT/'allocation.py'),graph_nodes=len(x));rows.append(row);save(ROOT/'cost_results.json',rows);print(json.dumps(row),flush=True)
    del score
    if name!='DirectTeacher':del op,ids,weights,z
    if route:del snaps,tr
    torch.cuda.empty_cache()
  del model,u,z0;torch.cuda.empty_cache()
 assert len(rows)==225,len(rows);save(ROOT/'cost_complete.json',dict(rows=len(rows),finished=time.time(),scope='6 main and3 PubMed;5 repeats;5 methods;preloaded graph/model;sequential post-experiments'))
if __name__=='__main__':run()
