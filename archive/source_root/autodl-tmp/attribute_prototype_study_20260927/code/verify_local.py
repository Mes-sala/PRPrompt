"""Independent numpy metric recomputation from sealed probabilities; no model code."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np

def binary(y,s):
 ix=np.argsort(-s,kind='stable');yy=y[ix];ss=s[ix];ends=np.r_[np.where(np.diff(ss)!=0)[0],len(ss)-1]
 tp=np.cumsum(yy)[ends];fp=ends+1-tp;nt=yy.sum();nf=len(yy)-nt
 ap=np.sum(np.diff(np.r_[0,tp])/nt*(tp/(ends+1)))
 auc=np.trapezoid(np.r_[0,tp]/nt,np.r_[0,fp]/nf)
 return float(auc),float(ap)
def verify(p,tag):
 name={'r01':'r01_development','r02':'r02_checkpoint_transfer','r03':'r03_confirmation','r04':'r04_support_selection','r05_5shot':'r05_selection_confirm_5shot','r05_3shot':'r05_selection_confirm_3shot'}[tag]
 base=p/'results'/name;manifest=json.loads((p/'provenance'/f'{tag}_remote_manifest.json').read_text())
 for x in manifest:
  f=base/x['path'];assert f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==x['sha256'],str(f)
 labels=np.load(p/'provenance/query_labels_for_independent_scoring.npz');errs={k:0. for k in ['accuracy','macro_f1','micro_f1','roc_auc','average_precision']};count=0
 for f in base.glob('*/*/result.json'):
  row=json.loads(f.read_text());z=np.load(f.parent/'prediction.npz');prob=z['probability'];y=labels[row['dataset']];pred=prob.argmax(1);C=prob.shape[1]
  cm=np.zeros((C,C),dtype=np.int64);np.add.at(cm,(y,pred),1);tp=cm.diagonal();den=cm.sum(0)+cm.sum(1);valid=den>0
  acc=float(np.mean(pred==y)*100);mf=float(np.mean(2*tp[valid]/den[valid])*100)
  vals=[binary((y==c).astype(int),prob[:,c]) for c in range(C) if 0<(y==c).sum()<len(y)]
  metrics=dict(accuracy=acc,micro_f1=acc,macro_f1=mf,roc_auc=float(np.mean([a for a,b in vals])),average_precision=float(np.mean([b for a,b in vals])))
  for k,v in metrics.items():errs[k]=max(errs[k],abs(v-row[k]))
  count+=1
 assert max(errs.values())<1e-8,errs
 out=dict(files_verified=len(manifest),predictions_independently_rescored=count,metric_max_errors=errs,bytes_verified=sum(x['bytes'] for x in manifest))
 (p/'provenance'/f'{tag.upper()}本地五指标独立复算.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(tag,out,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('tag',nargs='+');a=ap.parse_args();p=Path(__file__).resolve().parent.parent
 for tag in a.tag:verify(p,tag)
