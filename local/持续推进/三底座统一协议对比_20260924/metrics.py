"""Post-sealing metrics; cannot influence fitting or model selection."""
from pathlib import Path
import json,hashlib,sys,argparse
import torch,numpy as np
torch.set_num_threads(1)
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def compute(transport=False):
 if not transport:
  sys.path.insert(0,str(R));from grid import data_load
  truths={d:data_load(d)[0].y.cpu() for d in ['Cora','CiteSeer']}
  torch.save(truths,R/'audit_only_labels.pt')
 else:truths=torch.load(R/'audit_only_labels.pt',weights_only=True)
 roots=[R]+([R/'extension2026'] if (R/'extension2026/results').exists() else [])
 out=[]
 for root in roots:
  for p in sorted((root/'results').glob('*/*/result.json')):
   row=json.loads(p.read_text());q=p.parent/'sealed.pt';assert sha(q)==row['prediction_sha256'];s=torch.load(q,map_location='cpu',weights_only=False)
   y=truths[row['dataset']][s['query']].numpy();local_probs=torch.softmax(s['scores'].double(),dim=1).numpy();probpath=p.parent/'evaluation_probabilities.npy'
   if transport:
    probs=np.load(probpath,allow_pickle=False);assert probs.shape==local_probs.shape;assert np.max(np.abs(probs-local_probs))<=8*np.finfo(np.float64).eps
   else:
    probs=local_probs;np.save(probpath,probs,allow_pickle=False)
   pred=s['prediction'].numpy();C=probs.shape[1];onehot=np.eye(C)[y]
   m={'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro',labels=list(range(C)),zero_division=0),'micro_f1':f1_score(y,pred,average='micro',labels=list(range(C)),zero_division=0),'roc_auc_macro_ovr':roc_auc_score(y,probs,multi_class='ovr',average='macro',labels=list(range(C))),'auprc_macro':average_precision_score(onehot,probs,average='macro')}
   assert abs(m['accuracy']-row['accuracy'])<1e-7;assert abs(m['micro_f1']-m['accuracy'])<1e-10
   assert all(np.isfinite(v) for v in m.values())
   out.append({k:row[k] for k in ['dataset','checkpoint','checkpoint_sha256','seed','method','split_sha256','prediction_sha256']}|{'metrics':m,'result_path':str(p.relative_to(R)),'evaluation_probabilities_sha256':sha(probpath)})
 result={'definition':{'f1':'macro and micro; all C classes; zero_division=0','roc_auc':'macro one-versus-rest from softmax float64 sealed scores','auprc':'macro class-wise average precision; not trapezoidal PR integral','unit':'fraction','scope':'post-sealing evaluation only','numerics':'canonical server float64 softmax probabilities saved; transport checks normalization within 8 float64 eps, then recomputes exact ranking metrics from canonical probabilities'},'audit_labels_sha256':sha(R/'audit_only_labels.pt'),'rows':out}
 target=R/('metrics_transport_verified.json' if transport else 'all_metrics.json')
 if transport:
  old=json.loads((R/'all_metrics.json').read_text());assert old['audit_labels_sha256']==result['audit_labels_sha256'];assert len(old['rows'])==len(out)
  for a,b in zip(old['rows'],out):
   assert {k:v for k,v in a.items() if k!='metrics'}=={k:v for k,v in b.items() if k!='metrics'}
   assert all(abs(a['metrics'][k]-b['metrics'][k])<1e-12 for k in b['metrics'])
  target.write_text(json.dumps({'passed':True,'rows':len(out),'labels_sha256':result['audit_labels_sha256']},indent=2))
 else:target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print('metrics verified',len(out))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--transport',action='store_true');a=ap.parse_args();compute(a.transport)
