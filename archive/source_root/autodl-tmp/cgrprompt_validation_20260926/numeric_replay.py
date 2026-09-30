from run_core import *
job=next(j for j in json.loads((ROOT/'plan.json').read_text())['jobs'] if j['dataset']=='Cora' and j['checkpoint']['family']=='DGI')
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
data,pool,query,C,_=g.data_load('Cora');x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,x)).detach();model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(job['checkpoint']['path'],weights_only=True));model.requires_grad_(False);filters=A.make_filters(x,edges);records=[]
for seed in [24092803,24092806,24092809]:
 gen=torch.Generator().manual_seed(seed);si=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]).cuda();sy=data.y[si.cpu()].cuda();previous=None
 for repeat in range(3):
  with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
  z,p,t=A.optimizer_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plain',100,filters,'none',.1,'raw','gshampoo',.01);z-=z.mean(0,keepdim=True)
  with torch.no_grad():op,_,_=knn(z);scores=propagate(A.logits(z,si,sy,C),op,si,sy,C);reference,_=A.diffusion_readout(z,si,sy,C,'qeps-50',seed,filters['plain'],{},ROOT/'numeric_neighbors');a=scores[query.cuda()].double().softmax(-1).cpu().numpy();b=reference[query.cuda()].double().softmax(-1).cpu().numpy()
  np.savez_compressed(ROOT/f'numeric_replay_{seed}_{repeat}.npz',probabilities=a,original_readout_probabilities=b)
  row=dict(seed=seed,repeat=repeat,token_sha256=A.thash(p['tokens']),u_sha256=A.thash(u),knn_chunk_changed_predictions=int((a.argmax(1)!=b.argmax(1)).sum()),knn_chunk_probability_error=float(abs(a-b).max()),relative_token_difference_to_previous=float((p['tokens']-previous['tokens']).norm()/previous['tokens'].norm()) if previous is not None else 0,changed_predictions_to_previous=int((a.argmax(1)!=previous['prob'].argmax(1)).sum()) if previous is not None else 0)
  previous=dict(tokens=p['tokens'],prob=a);records.append(row);save(ROOT/'numeric_sensitivity.json',records);print(json.dumps(row),flush=True)
