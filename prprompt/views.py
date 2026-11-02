"""The locked MUTAG equal-view algorithm, generalized only in class count."""
import torch
from torch.nn import functional as F
@torch.no_grad()
def build_view_routes(h,si,labels,support_batch,d,classes):
 assert h.shape[1]==3+2*d and sorted(labels.unique().tolist())==list(range(classes))
 views=[];raw_rms=[];final_rms=[];active=[]
 for j in range(d):
  hd=h[:,[0,1,2,3+j,3+d+j]]
  q=torch.stack([hd[si][labels==c].mean(0) for c in range(classes)])
  p=(F.normalize(hd,dim=-1)@F.normalize(q,dim=-1).T/.1).softmax(-1)
  u=p-p[si].mean(0);rms=u[si][support_batch].square().mean().sqrt();raw_rms.append(float(rms))
  if rms<1e-6:
   views.append(torch.zeros_like(u));final_rms.append(0.);active.append(False);continue
  v=(u/rms).tanh();v=v-v[si].mean(0);norm=v[si][support_batch].square().mean().sqrt().clamp_min(1e-6)
  views.append(v/norm);final_rms.append(float(norm));active.append(True)
 channel=torch.stack(views,dim=-1);mean=channel.mean(-1);mean_center=mean[si].mean(0);mean=mean-mean_center
 mean_rms=mean[si][support_batch].square().mean().sqrt().clamp_min(1e-6);mean=mean/mean_rms
 return channel,mean,dict(raw_view_rms=raw_rms,bounded_view_rms=final_rms,active_views=active,view_mean_center=mean_center.tolist(),view_mean_rms=float(mean_rms),view_bounding='tanh after initial support normalization, then support recenter/RMS',input_dimension=d,classes=classes)
