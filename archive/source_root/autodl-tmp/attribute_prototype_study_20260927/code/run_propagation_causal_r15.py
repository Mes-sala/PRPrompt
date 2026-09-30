"""R15 shift x propagation mechanism controls; fixed settings, no query labels."""
import json,time,fcntl,os,inspect
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
import run_study as r
import expanded_data as data

def spectrum(z,si,zz,total):
 zs=z[si];n=len(z)-len(si);g=(total-zs.sum(0))/n;gn=g.norm();assert gn>1e-10;v=g/gn
 def proj(t):return t-v*(v@t)
 def mv(t):
  t=proj(t);return proj((zz@t-zs.T@(zs@t))/n)
 trace=float((z.square().sum()-zs.square().sum())/n-v@((zz@v-zs.T@(zs@v))/n))
 gen=torch.Generator(device='cuda').manual_seed(9817);q=proj(torch.randn(z.shape[1],dtype=z.dtype,device='cuda',generator=gen));q=q/q.norm()
 for _ in range(120):
  a=mv(q);q=a/a.norm().clamp_min(1e-30)
 a=mv(q);lam=float(q@a);res=float((a-lam*q).norm());scale=float(.05*gn);rem=max(0,trace-lam);upper=min(trace,.5*(lam+rem+((lam-rem)**2+4*res**2)**.5));lo=lam/scale;hi=upper/scale;exact=None
 if lo<=1+1e-8 and hi>=1-1e-8:
  M=(zz-zs.T@zs)/n;w=M@v;b=v@w;M=M-torch.outer(v,w)-torch.outer(w,v)+b*torch.outer(v,v);M=(M+M.T)/2
  exact=float(torch.linalg.eigvalsh(M)[-1])/scale;lo=exact-1e-9;hi=exact+1e-9
 return dict(g_norm=float(gn),lambda_rayleigh=lam,trace=trace,residual=res,rho_lower=lo,rho_upper=hi,rho_dense_eigvalsh=exact,classification='stable' if hi<1-1e-8 else 'unstable' if lo>1+1e-8 else 'unresolved')

def trajectory(z,si,sy,C,beta):
 oh=F.one_hot(sy,C).double();a=oh.T@z[si]/oh.sum(0)[:,None];m=a.clone();mask=torch.ones(len(z),dtype=torch.bool,device='cuda');mask[si]=False;zu=z[mask]
 def stats(m):
  v=float((m-m.mean(0)).square().sum(1).mean());u=F.normalize(m,dim=1);av=float((u-u.mean(0)).square().sum(1).mean());return dict(separation=v,angular_separation=av,min_norm=float(m.norm(dim=1).min()))
 hist=[stats(m)]
 for _ in range(3):
  q=(zu@F.normalize(m,dim=1).T/.05).softmax(-1);mass=q.sum(0);assert float(mass.min())>1e-8
  u=q.T@zu/mass[:,None];m=(1-beta)*a+beta*u;hist.append(stats(m))
 for t in hist:t['ratio']=t['separation']/hist[0]['separation']
 return a,m,hist

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=r.ROOT/'results/propagation_causal_r15';out.mkdir(parents=True,exist_ok=True);lock=open(out/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r.save(out/'process.json',dict(pid=os.getpid(),started=time.time(),runner_sha256=r.sha(__file__)))
 (out/'NormalizeFeatures_source.txt').write_text(inspect.getsource(NormalizeFeatures.forward))
 for name in data.ALL:
  d,x,adj,pool,query,C,protocol=data.load_data(name);adj=adj.double();raw=d.x.cuda();conditions={'original':x.double()};mi=float(raw.min())
  if mi<0:
   shifted=d.x-d.x.min();den=shifted.sum(1,keepdim=True).clamp_min(1);reconstructed=shifted/den;assert torch.equal(reconstructed,x.cpu())
   conditions['remove_shift_same_denominator']=(d.x/den).cuda().double()
  oldgeom=json.loads((r.ROOT/'results/attribute_mechanism_r13'/name/'geometry.json').read_text());assert r.thash(x.double())==oldgeom['input_sha256'] and r.thash(d.edge_index)==oldgeom['graph_sha256']
  for mode,inp in conditions.items():
   for hops in [0,2]:
    dst=out/f'{name}__{mode}__h{hops}.json'
    if dst.exists():continue
    z=F.normalize(inp if hops==0 else adj@(adj@inp),dim=1);zero_rows=int((z.norm(dim=1)<=1e-10).sum())
    reuse=mode=='original' and hops==2
    spectra={t['seed']:t for t in json.loads((r.ROOT/'results/consensus_spectrum'/f'{name}.json').read_text())['rows']} if reuse else None
    zz=None if reuse else z.T@z;total=z.sum(0);rows=[]
    for seed in range(271700,271720):
     tag='raw_no_update' if hops==0 else 'graph_unanchored';pf=r.ROOT/'results/attribute_mechanism_r13'/name/f'{seed}__{tag}'/'prediction.npz'
     with np.load(pf) as f:support=f['support'].copy();oldanchor=f['anchor'].copy();oldmu=f['prototype'].copy()
     si=torch.tensor(support,device='cuda');sy=d.y[support].cuda();s=spectra[seed].copy() if reuse else spectrum(z,si,zz,total)
     s.update(seed=seed,spectrum_reused=reuse,source_prediction_sha256=r.sha(pf),support_hash=r.thash(si));tra={}
     for beta in [.5,1.]:
      a,m,hist=trajectory(z,si,sy,C,beta);tra[str(beta)]=hist
      if mode=='original':assert np.allclose(a.cpu().numpy(),oldanchor,rtol=1e-10,atol=1e-12)
      if reuse and beta==1:assert np.allclose(m.cpu().numpy(),oldmu,rtol=1e-10,atol=1e-12)
     s['trajectory']=tra;rows.append(s)
    r.save(dst,dict(dataset=name,mode=mode,hops=hops,raw_min=mi,zero_representation_rows=zero_rows,input_sha256=r.thash(inp),representation_sha256=r.thash(z),graph_sha256=r.thash(d.edge_index),runner_sha256=r.sha(__file__),rows=rows,finished=time.time()))
    print(name,mode,hops,min(t['rho_lower'] for t in rows),max(t['rho_upper'] for t in rows),set(t['classification'] for t in rows),flush=True)
    del zz,z
  del d,x,adj,raw,conditions;torch.cuda.empty_cache()
 files=list(out.glob('*__h*.json'));r.save(out/'complete.json',dict(geometries=len(files),rows=sum(len(json.loads(f.read_text())['rows']) for f in files),runner_sha256=r.sha(__file__),finished=time.time()))
if __name__=='__main__':main()
