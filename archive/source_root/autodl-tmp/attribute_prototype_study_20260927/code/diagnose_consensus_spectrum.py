"""Fixed-temperature consensus diagnosis; no accuracy/model/parameter selection."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
import run_study as r
import expanded_data as data

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results/consensus_spectrum';out.mkdir(parents=True,exist_ok=True)
 for name in data.ALL:
  dst=out/(name+'.json')
  if dst.exists():
   prior=json.loads(dst.read_text())
   if all(t['classification']!='unresolved' for t in prior['rows']):continue
  d,x,adj,pool,query,C,protocol=data.load_data(name);x=x.double();adj=adj.double();z=F.normalize(adj@(adj@x),dim=1)
  geom=json.loads((r.ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text());assert r.thash(x)==geom['input_sha256'] and r.thash(d.edge_index)==geom['graph_sha256']
  zz=z.T@z;total=z.sum(0);rows=[]
  for seed in range(271700,271720):
   pf=r.ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__graph_unanchored'/'prediction.npz'
   with np.load(pf) as f:si=torch.tensor(f['support'],device='cuda')
   n=len(z)-len(si);zs=z[si];g=(total-zs.sum(0))/n;gn=g.norm();v=g/gn
   # Tangential covariance P Sigma P equals P E[zz^T] P because P g=0.
   def proj(t):return t-v*(v@t)
   def mv(t):
    t=proj(t);return proj((zz@t-zs.T@(zs@t))/n)
   trace=float((z.square().sum()-zs.square().sum())/n-v@((zz@v-zs.T@(zs@v))/n))
   gen=torch.Generator(device='cuda').manual_seed(9817);q=proj(torch.randn(z.shape[1],dtype=z.dtype,device='cuda',generator=gen));q=q/q.norm()
   for k in range(120):
    a=mv(q);q=a/a.norm().clamp_min(1e-30)
   a=mv(q);lam=float(q@a);res=float((a-lam*q).norm());scale=float(.05*gn)
   # PSD block bound: complement operator norm <= its trace.
   rem=max(0.,trace-lam);upper=min(trace,.5*(lam+rem+((lam-rem)**2+4*res**2)**.5))
   lo=lam/scale;hi=upper/scale
   exact=None
   if lo<=1+1e-8 and hi>=1-1e-8:
    M=(zz-zs.T@zs)/n;w=M@v;b=v@w;M=M-torch.outer(v,w)-torch.outer(w,v)+b*torch.outer(v,v);M=(M+M.T)/2
    exact=float(torch.linalg.eigvalsh(M)[-1])/scale;lo=exact-1e-9;hi=exact+1e-9
    del M
   rows.append(dict(seed=seed,rho_dense_eigvalsh=exact,n=n,g_norm=float(gn),tau=.05,lambda_rayleigh=lam,residual=res,trace=trace,rho_lower=lo,rho_upper=hi,classification='stable' if hi<1-1e-8 else 'unstable' if lo>1+1e-8 else 'unresolved',support_hash=r.thash(si),source_prediction_sha256=r.sha(pf)))
  r.save(dst,dict(dataset=name,runner_sha256=r.sha(__file__),geometry=geom,rows=rows,method='120-step power Rayleigh lower bound; PSD trace/block upper bound; float64',finished=time.time()));print(name,min(a['rho_lower'] for a in rows),max(a['rho_upper'] for a in rows),set(a['classification'] for a in rows),flush=True)
  del zz,z,x,adj,d;torch.cuda.empty_cache()
if __name__=='__main__':main()
