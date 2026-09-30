from pathlib import Path
import json,argparse
import torch
import torch.nn.functional as F
import run_round61_transfer as m
root=Path(__file__).resolve().parent
@torch.no_grad()
def remove_pc(x):
 v=torch.ones(x.shape[1],device=x.device);v=F.normalize(v,dim=0)
 for i in range(300):
  v1=F.normalize(x.T@(x@v),dim=0);delta=(v1-v).norm();v=v1
  if delta<1e-7:break
 av=x.T@(x@v);ev=v@av;res=(av-ev*v).norm()/av.norm().clamp_min(1e-12);assert res<1e-5,res
 z=x-(x@v)[:,None]*v[None,:];orth=(z@v).norm()/x.norm().clamp_min(1e-12);assert orth<1e-5
 return z,v,dict(iterations=i+1,eigenvalue=float(ev),relative_residual=float(res),projection_residual=float(orth))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--config',default='round63_config.json');args=ap.parse_args();torch.set_num_threads(4);c=json.loads((root/args.config).read_text());out=Path(args.out);out.mkdir(exist_ok=True);assert not list(out.glob('*/result.json'))
 seeds=[95199] if args.smoke else c['seeds'];m.save_json(out/'config.json',dict(**c,actual_seeds=seeds,smoke=args.smoke,source_sha256=m.sha(__file__),dependency_sha256=m.sha(m.__file__)))
 raw=m.Planetoid('/root/ProG/data/Planetoid','CiteSeer')[0];assert ((raw.x==0)|(raw.x==1)).all();data=m.Planetoid('/root/ProG/data/Planetoid','CiteSeer',transform=m.NormalizeFeatures())[0];x=data.x.cuda();edges=data.edge_index.cuda();ei,ew=m.gcn_norm(edges,num_nodes=len(x),add_self_loops=True);adj=torch.sparse_coo_tensor(ei.flip(0),ew,(len(x),len(x))).coalesce()
 def prop(a):return torch.sparse.mm(adj,torch.sparse.mm(adj,a))
 original=prop(x);assert torch.allclose(prop(x*torch.ones(x.shape[1],device=x.device)),original,atol=1e-7)
 df=(raw.x>0).sum(0).float().cuda();prob=df/df.sum();reps={'base':original};meta={};artifacts={};(out/'geometry').mkdir(exist_ok=True)
 for kind,a in [('sif3',1e-3),('sif4',1e-4)]:
  w=a/(a+prob);reps[kind]=prop(x*w);expected=a/(a+(raw.x>0).sum(0).double()/float(df.sum()));assert torch.allclose(w.cpu().double(),expected,atol=1e-7);artifacts[kind]=dict(weight=w.cpu(),frequency=prob.cpu());meta[kind]=dict(a=a,weight_min=float(w.min()),weight_max=float(w.max()))
 for parent,kind in [('base','pc'),('sif3','sif3pc'),('sif4','sif4pc')]:
  reps[kind],v,info=remove_pc(reps[parent]);artifacts[kind]=dict(pc=v.cpu(),parent=parent);meta[kind]=info
 mean=original.mean(0);reps['center']=original-mean;artifacts['center']=dict(mean=mean.cpu());meta['center']=dict(mean_residual=float(reps['center'].mean(0).norm()));assert reps['center'].mean(0).norm()<1e-6
 if args.smoke:
  toy=torch.rand(37,17,device='cuda',generator=torch.Generator(device='cuda').manual_seed(63));z,v,info=remove_pc(toy);vv=torch.linalg.svd(toy,full_matrices=False).Vh[0];assert abs(float(v@vv))>1-1e-5
 files={}
 for kind,z in reps.items():
  file=out/'geometry'/f'{kind}.pt';torch.save(dict(**artifacts.get(kind,{}),embedding_sha256=m.thash(z),input_sha256=m.thash(x)),file);files[kind]=file
 query=data.test_mask.nonzero().flatten();query=query[torch.randperm(len(query),generator=torch.Generator().manual_seed(20260923))];query=query[:500] if c['phase']=='dev' else query[500:];pool=data.train_mask.nonzero().flatten();gh=m.hashlib.sha256((m.thash(data.x)+m.thash(data.edge_index)).encode()).hexdigest();n=0
 for seed in seeds:
  gen=torch.Generator().manual_seed(seed);support=torch.cat([q[torch.randperm(len(q),generator=gen)[:5]] for cl in range(6) for q in [pool[data.y[pool]==cl]]]);si=support.cuda();sy=data.y[support].cuda();assert not set(support.tolist())&set(query.tolist());split=m.hashlib.sha256((m.thash(support)+m.thash(query)).encode()).hexdigest()
  for method in c['methods']:
   kind='base' if method=='raw' else method;z=reps[kind];scores=m.logits(z,si,sy,6) if method=='raw' else m.rectified_teacher(z,si,sy,6,'rect_teacher_ref50',seed)[0];ss=scores[query].cpu();folder=out/f'CiteSeer__Direct__{seed}__{method}';folder.mkdir();torch.save(dict(query=query,support=support,scores=ss,prediction=ss.argmax(-1),prompt={}),folder/'sealed.pt');acc=(ss.argmax(-1)==data.y[query]).float().mean().item();m.save_json(folder/'result.json',dict(dataset='CiteSeer',checkpoint='Direct',checkpoint_sha256=None,seed=seed,phase=c['phase'],method=method,accuracy=acc,graph_sha256=gh,split_sha256=split,source_sha256=m.sha(__file__),dependency_sha256=m.sha(m.__file__),prediction_sha256=m.sha(folder/'sealed.pt'),geometry_file=str(files[kind].relative_to(out)),geometry_sha256=m.sha(files[kind]),embedding_sha256=m.thash(z)));n+=1
 m.save_json(out/'operator_audit.json',dict(rows=n,passed=True,raw_binary=True,frequency_sum=float(prob.sum()),representation_diagnostics=meta));print(n,meta)
if __name__=='__main__':main()
