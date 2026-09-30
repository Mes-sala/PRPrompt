"""Audit compressed ridge and seal SVM native predictions before loading query labels."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
import run_graph1 as r
from run_graph2 import simplex
from run_graph3 import DIRECT,CS
from score_graph1 import numpy_predict

def main(a):
 torch.set_num_threads(2);source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 assert json.loads((source/'complete.json').read_text())['count']==60
 r.save(out/'manifest.json',{'source_manifest_sha':r.sha(source/'manifest.json'),'checker_source_sha':r.sha(__file__),'purpose':'Independent ridge reconstruction and SVM native predict tie convention; no query labels loaded.'})
 audits={};filesha={};checks=0
 for ds in ['ENZYMES','NCI1','NCI109']:
  basis=torch.load(source/(ds+'_basis.pt'),weights_only=True);prov=json.loads((source/(ds+'_provenance.json')).read_text())
  for n,k in basis.items():assert r.digest(k)==prov['kernel_sha'][n] and torch.isfinite(k).all() and torch.allclose(k,k.T,atol=1e-12);checks+=1
  stats={'svm_argmax_vs_native_differences':0,'raw_expert_max_error':0.,'projected_mean_max_error':0.,'expert_mean_argmax_differences':0}
  for count,p in enumerate(sorted((source/'sealed'/ds).glob('*.pt'))):
   rec=torch.load(p,weights_only=True);s,y,q=rec['support'],rec['support_labels'],rec['query'];c=int(y.max())+1
   assert not torch.isin(s,q).any() and r.digest(s)==rec['support_sha'] and r.digest(q)==rec['query_sha'];checks+=3
   for n,sc in rec['scores'].items():assert r.digest(sc)==rec['scores_sha'][n] and torch.isfinite(sc).all();checks+=1
   for fold in rec['folds']:
    tr,val=fold['train'],fold['heldout'];assert not torch.isin(tr,val).any() and set(tr.tolist()+val.tolist())==set(s.tolist());yy=y[torch.tensor([s.tolist().index(i) for i in tr.tolist()])];assert r.digest(yy)==fold['train_labels_sha'];checks+=2
   errors=(rec['oof']-F.one_hot(y,c)).square().mean((-1,-2));assert torch.allclose(errors,rec['ridge_cv'],atol=1e-12)
   for i,n in enumerate(rec['names']):
    j=min(range(6),key=lambda j:(round(float(errors[i,j]),12),-r.LAM[j]));assert rec['choices']['ridge_cv_'+n]['lambda']==r.LAM[j];checks+=1
   mean=simplex(rec['experts']).mean(1)
   for n,w in rec['weights'].items():assert torch.allclose(torch.einsum('m,mqc->qc',w,mean),rec['scores'][n],atol=1e-12);checks+=1
   if count==0:
    for i,n in enumerate(rec['names']):
     k=basis[n].numpy();preds=np.stack([numpy_predict(k,s.numpy(),y.numpy(),c,lam)[q.numpy()] for lam in r.LAM]);err=float(abs(preds-rec['experts'][i].numpy()).max());stats['raw_expert_max_error']=max(stats['raw_expert_max_error'],err)
     assert np.allclose(preds,rec['experts'][i].numpy(),atol=1e-4,rtol=1e-5),(ds,n,err)
     projected=simplex(torch.from_numpy(preds)).mean(0);stats['projected_mean_max_error']=max(stats['projected_mean_max_error'],float((projected-mean[i]).abs().max()));stats['expert_mean_argmax_differences']+=int((projected.argmax(1)!=mean[i].argmax(1)).sum())
    assert stats['expert_mean_argmax_differences']==0
   # SVC's multiclass predict() and argmax(OVR scores) can differ on vote ties.
   # CV used native predict(), so final prediction must follow exactly that rule.
   rec['svm_original_margin_scores']={};delta={}
   for n in ['svm_C1_'+v for v in DIRECT]+['svm_direct_selected']:
    if n=='svm_direct_selected':kernel=rec['choices'][n]['kernel'];C=rec['choices'][n]['C']
    else:kernel=n[len('svm_C1_'):];C=1.
    k=basis[kernel].numpy();model=SVC(C=C,kernel='precomputed').fit(k[np.ix_(s.numpy(),s.numpy())],y.numpy());pred=model.predict(k[np.ix_(q.numpy(),s.numpy())]);delta[n]=int((pred!=rec['scores'][n].argmax(1).numpy()).sum());stats['svm_argmax_vs_native_differences']+=delta[n]
    rec['svm_original_margin_scores'][n]=rec['scores'][n];rec['scores'][n]=F.one_hot(torch.from_numpy(pred),c).double()
   best=min(rec['svm_cv'],key=lambda key:(-rec['svm_cv'][key],float(key.split('|')[1]),key.split('|')[0]));kernel,C=best.split('|');assert rec['choices']['svm_direct_selected']=={'kernel':kernel,'C':float(C),'cv_accuracy':rec['svm_cv'][best]};checks+=1
   if count==0:
    for kernel in DIRECT:
     k=basis[kernel].numpy()[np.ix_(s.numpy(),s.numpy())];yn=y.numpy()
     for C in CS:
      hits=0
      for fold in range(5):
       val=np.array([np.where(yn==cl)[0][fold] for cl in range(c)]);tr=np.array([i for i in range(len(y)) if i not in val]);m=SVC(C=C,kernel='precomputed').fit(k[np.ix_(tr,tr)],yn[tr]);hits+=int((m.predict(k[np.ix_(val,tr)])==yn[val]).sum())
      assert hits/len(y)==rec['svm_cv'][kernel+'|'+str(C)];checks+=1
   rec['svm_native_rule_audit']=delta;rec['scores_sha']={n:r.digest(v) for n,v in rec['scores'].items()};dest=out/'sealed'/ds/p.name;dest.parent.mkdir(parents=True,exist_ok=True);torch.save(rec,dest);filesha[str(dest.relative_to(out))]=r.sha(dest)
  audits[ds]=stats;print('CHECKED',ds,stats,flush=True)
 r.save(out/'audit.json',{'checks':checks,'datasets':audits,'sealed_sha':filesha});r.save(out/'complete.json',{'count':len(filesha),'finished':time.time()})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);main(ap.parse_args())
