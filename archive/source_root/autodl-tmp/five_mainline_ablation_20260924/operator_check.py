import json,torch
from pathlib import Path
import grid as g
R=Path(__file__).resolve().parent;checks=[];torch.set_num_threads(4)
plan=json.loads((R/'plan.json').read_text())
for name in g.NODES:
 j=next(j for j in plan['jobs'] if j['dataset']==name and not j['direct'] and j['checkpoint']['family']=='GraphMAE');data,pool,query,C,_=g.data_load(name);x=data.x.cuda();edges=data.edge_index.cuda();xn=x if name in g.NODES[:5] else g.NormalizeFeatures()(data.clone()).x.cuda();ei,ew=g.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce();r=torch.sparse.mm(adj,torch.sparse.mm(adj,xn)).detach();seed=plan['seeds'][0];gen=torch.Generator().manual_seed(seed);si=torch.cat([p[torch.randperm(len(p),generator=gen)[:5]] for c in range(C) for p in [pool[data.y[pool]==c]]]).cuda();sy=data.y.cuda()[si]
 model=g.GCN(x.shape[1],128,128,num_layer=2).cuda().eval();model.load_state_dict(torch.load(j['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False)
 with torch.no_grad():u=model.conv_layers[0](x,edges).detach()
 filters=g.a1.make_filters(x,edges)
 z0,_,_=g.a1.focal_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plainlate',2,filters,2,'scale');z1,_,_=g.ab.a1_ablate(u,r,adj,model,edges,si,sy,C,seed,'token_ridge_kd10plainlate',2,filters,2,'scale',ablation='none');assert torch.allclose(z0,z1,atol=5e-5,rtol=1e-5)
 z0,_,_=g.a4.vat_prompt(u,r,adj,model,edges,si,sy,C,seed,'token_kd3_ref50',2,filters,'vat',.3);z1,_,_=g.ab.a4_ablate(u,r,adj,model,edges,si,sy,C,seed,'token_kd3_ref50',2,filters,'vat',.3,ablation='none');assert torch.allclose(z0,z1,atol=5e-5,rtol=1e-5)
 s0,_=g.a1.rectified_teacher(r-r.mean(0,keepdim=True),si,sy,C,'rect_teacher_ref50',seed);s1,_=g.ab.direct_ablate(r,xn,si,sy,C,seed,'reference');assert torch.allclose(s0,s1,atol=1e-5,rtol=1e-5)
 s0,_=g.a1.rectified_teacher(r,si,sy,C,'rect_teacher_ref0',seed);s1=g.a1.logits(r,si,sy,C);assert torch.allclose(s0,s1,atol=1e-5,rtol=1e-5)
 zb=g.a5.selfloop_forward(model,x,edges,x.new_ones(2),True);mu,w,_=g.a5.geometry_stats(zb,'diag10');z0=(zb-mu)@w;z1,_,_=g.ab.geometry_ablate(model,x,'reference');assert torch.allclose(z0,z1,atol=5e-5,rtol=1e-5)
 checks.append(dict(dataset=name,unchanged_A1_forward=True,unchanged_A4_forward=True,unchanged_A2_teacher=True,beta0_equals_prototype=True,unchanged_diag10=True))
 del model;torch.cuda.empty_cache()
audit=dict(passed=True,checks=checks,scope='unablated implementations equivalent to historical operators on all eight datasets; smoke steps only');g.dump(R/'operator_audit.json',audit);print(json.dumps(audit))
