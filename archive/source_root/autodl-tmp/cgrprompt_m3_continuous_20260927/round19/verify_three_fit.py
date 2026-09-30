"""Engineering parity only: train exactly two folds plus the full model."""
from pathlib import Path
import run_structure as m
import torch,numpy as np,json,time
R=Path(__file__).resolve().parent
plan=json.loads((R/'confirm_plan.json').read_text());results=[];seed=26094620
for job in plan['jobs']:
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 assert m.g.verify_sources()==json.loads((m.BASE/'plan.json').read_text())['source_manifest_sha256']
 data,pool,query,C,protocol=m.g.data_load(job['dataset']);x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=m.g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();rc=r-r.mean(0,keepdim=True);op,_,_=m.knn(rc);filters=m.A.make_filters(x,edges)
 ck=job['checkpoint'];assert m.sha(ck['path'])==ck['sha256'];model=m.g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(ck['path'],weights_only=True,map_location='cpu'));model.requires_grad_(False);before={k:m.A.thash(v) for k,v in model.state_dict().items()}
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 gen=torch.Generator().manual_seed(seed);support=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]);si=support.cuda();sy=data.y[support].cuda()
 refdir=R/'confirm'/job['id']/str(seed);record=json.loads((refdir/'result.json').read_text());assert m.sha(refdir/'prediction.npz')==record['prediction_sha256'];ref=np.load(refdir/'prediction.npz');assert np.array_equal(support.numpy(),ref['support']) and np.array_equal(query.numpy(),ref['query'])
 start=time.perf_counter();preds=[];fits=0
 for fold in [seed%5,(seed+2)%5]:
  held=torch.arange(len(si),device=x.device)%5==fold;ti=si[~held];ty=sy[~held]
  z,_,_=m.A.optimizer_prompt(u,r,adj,model,edges,ti,ty,C,seed+fold+1,'token_ridge_kd10plain',100,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True);fits+=1
  with torch.no_grad():preds.append(m.calibrated(m.A.logits(z,ti,ty,C),op,ti,ty,C).softmax(-1))
 z,_,_=m.A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',100,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True);fits+=1
 with torch.no_grad():
  full=m.calibrated(m.A.logits(z,si,sy,C),op,si,sy,C).softmax(-1);q=.5*full+.5*torch.stack(preds).mean(0);prob=q.clamp_min(1e-8).log()[query.cuda()].double().softmax(-1).cpu().numpy()
 m.sync();seconds=time.perf_counter()-start;assert fits==3;assert before=={k:m.A.thash(v) for k,v in model.state_dict().items()} and not model.training and all(p.grad is None for p in model.parameters())
 err=float(np.max(np.abs(prob-ref['Prompt__budget2'])));same=bool(np.array_equal(prob.argmax(1),ref['Prompt__budget2'].argmax(1)));assert err<2e-6 and same
 out=R/'three_fit_parity'/job['id'];out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'prediction.npz',probability=prob,support=support.numpy(),query=query.numpy())
 results.append(dict(job=job['id'],seed=seed,actual_optimizer_calls=fits,max_probability_error=err,argmax_identical=same,seconds=seconds,reference_prediction_sha256=record['prediction_sha256'],prediction_sha256=m.sha(out/'prediction.npz')))
m.save(R/'three_fit_parity.json',dict(status='pass',engineering_only=True,script_sha256=m.sha(__file__),source_runner_sha256=m.sha(R/'run_structure.py'),results=results));print(json.dumps(results,indent=2))
