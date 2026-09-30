"""Documented inference adaptations; not full original image-benchmark reproductions."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

def normalize(a):
 a=a-a.max(1,keepdims=True);p=np.exp(a);return p/p.sum(1,keepdims=True)
def laplacian_update(unary,kernel,lam=1.,iterations=20):
 # Matches official lshot_update.py bound_update, but also returns probabilities.
 y=normalize(-unary);old=float('inf')
 for i in range(iterations):
  y=normalize(-unary+lam*kernel.dot(y))
  energy=(y*np.log(np.maximum(y,1e-20))+unary*y-lam*kernel.dot(y)*y).sum()
  if i>1 and abs(energy-old)<=1e-6*abs(old):break
  old=energy.copy()
 return y

def nearest_cache(z,max_support):
 k=min(max_support+2,len(z)-1);parts=[]
 for start in range(0,len(z),256):
  a=z[start:start+256]@z.T;a[torch.arange(len(a),device=z.device),torch.arange(start,start+len(a),device=z.device)]=-torch.inf
  parts.append(a.topk(k,dim=1).indices)
 return torch.cat(parts)

def scores(z,si,sy,C,method,neighbors=None):
 oh=F.one_hot(sy,C).float();sums=oh.T@z[si];counts=oh.sum(0)[:,None];mu=sums/counts
 mask=torch.ones(len(z),device=z.device,dtype=torch.bool);mask[si]=False
 def distance(a,b):return ((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T).clamp_min(0)
 if method=='simpleshot_style':return (-distance(z,mu)).softmax(-1).cpu().numpy()
 if method=='soft_kmeans_adapted':
  for _ in range(3):
   q=(-distance(z[mask],mu)/.05).softmax(-1);mu=(sums+q.T@z[mask])/(counts+q.sum(0)[:,None])
  return (-distance(z,mu)/.2).softmax(-1).cpu().numpy()
 assert method=='laplacianshot_adapted' and neighbors is not None
 # Original authors use knn=3 including self, giving 2 directed non-self neighbors.
 cand=neighbors[mask];valid=mask[cand];keep=valid & (valid.long().cumsum(1)<=2);assert torch.all(keep.sum(1)==2)
 mapping=torch.full((len(z),),-1,dtype=torch.long,device=z.device);mapping[mask]=torch.arange(int(mask.sum()),device=z.device)
 col=mapping[cand[keep]].cpu().numpy();n=int(mask.sum());row=np.repeat(np.arange(n),2)
 kernel=sparse.csc_matrix((np.ones(len(row)),(row,col)),shape=(n,n))
 unary=distance(z[mask],mu).cpu().double().numpy();y=laplacian_update(unary,kernel)
 p=np.zeros((len(z),C));p[mask.cpu().numpy()]=y;p[si.cpu().numpy()]=oh.cpu().numpy();return p
