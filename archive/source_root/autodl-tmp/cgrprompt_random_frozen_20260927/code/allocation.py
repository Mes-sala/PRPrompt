"""Matched hidden-interface allocation controls. No evaluation-label argument."""
import torch
import torch.nn.functional as F
from run_core import A,g,sync
from optimizer83 import GraftedMatrix
from optimizer82 import MatrixPreconditioner

def fit(u,r,adj,model,edges,si,sy,C,seed,route='fixed',optimizer='adam',lr=.01,steps=100,snapshots=(100,),amplitude=10.):
 torch.manual_seed(seed);device=u.device;d=u.shape[1]
 with torch.no_grad():
  scores=A.rectified_teacher(r-r.mean(0,keepdim=True),si,sy,C,'rect_teacher_ref50',seed)[0]
  gate,_=A.temperature_gate(scores,adj,.1,'raw');target_norm=gate.norm();hfeat=F.normalize(u.detach(),dim=1)
  if route=='permuted':gate=gate[torch.randperm(len(gate),device=device)]
  if route=='random':
   q=torch.randn((len(u),C),device=device).softmax(-1);gate=torch.sparse.mm(adj,q-q.mean(0,keepdim=True));gate=gate*(target_norm/gate.norm())
 rank=d//2 if route in ['learned_budget','fixed_rank'] else d
 if rank!=d:
  proj=torch.linalg.qr(torch.randn(d,rank,device=device),mode='reduced')[0];features=hfeat@proj
 else:proj=None;features=hfeat
 P=torch.nn.Parameter(torch.zeros(C,rank,device=device));router=None
 if route in ['learned_hidden','learned_teacher','learned_budget']:
  router=torch.nn.Parameter(torch.zeros(C,rank,device=device) if route=='learned_teacher' else torch.randn(C,rank,device=device)/rank**.5)
 params=[P]+([router] if router is not None else [])
 opt=torch.optim.Adam([P],lr=lr) if optimizer=='adam' else GraftedMatrix([P],lr,optimizer[1:]) if optimizer.startswith('g') else MatrixPreconditioner([P],lr,optimizer)
 ropt=torch.optim.Adam([router],lr=lr) if router is not None else None
 scale=u.std(0).clamp_min(1e-6).detach()
 def forward():
  if router is not None:
   logits=features@router.T/.2
   if route=='learned_teacher':logits=logits+scores*2
   q=logits.softmax(-1);q=torch.sparse.mm(adj,q-q.mean(0,keepdim=True))
  else:q=gate
  token=P.tanh() if proj is None else P.tanh()@proj.T
  return model.conv_layers[1](g.act(u+amplitude*(q@token)*scale),edges)
 with torch.no_grad():
  frozen=model.conv_layers[1](g.act(u),edges);assert torch.allclose(forward(),frozen,atol=1e-6,rtol=1e-5)
 result={};history=[];sync();import time
 start=time.perf_counter()
 for step in range(1,steps+1):
  opt.zero_grad()
  if ropt is not None:ropt.zero_grad()
  z=forward();ce=F.cross_entropy(A.logits(z,si,sy,C,loo=True),sy);loss=ce+.001*sum(p.square().mean() for p in params)
  assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in model.parameters());opt.step()
  if ropt is not None:ropt.step()
  if step==1 or step%100==0:history.append(dict(step=step,loss=float(loss.detach()),ce=float(ce.detach()),gradient_norm=float(P.grad.norm())))
  if step in snapshots:
   sync()
   with torch.no_grad():
    raw=forward().detach();z=raw-raw.mean(0,keepdim=True);result[step]=(z,dict(seconds=time.perf_counter()-start,loss=float(loss.detach()),history=list(history),route=route,optimizer=optimizer,lr=lr,parameters=sum(p.numel() for p in params),residual_parameters=P.numel(),routing_parameters=router.numel() if router is not None else 0,prompt=P.detach().cpu(),router=router.detach().cpu() if router is not None else None))
 return result
