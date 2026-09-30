from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'CGRPrompt_叙事补验_20260926'))
from audit_local import metrics,paired,sha
stage=sys.argv[1];rows=[];checked=0;maxerr=0.
for f in sorted((ROOT/stage).glob('*/*/result.json')):
 r=json.loads(f.read_text());assert sha(f.parent/'prediction.npz')==r['prediction_sha256'];assert sha(ROOT/'run_structure.py')==r['runner_sha256'];assert r['frozen_verified'] and r['query_label_free_operator'];ar=np.load(f.parent/'prediction.npz');y=np.load(f.parent.parent/'query_labels.npy');assert not np.intersect1d(ar['support'],ar['query']).size
 sh=hashlib.sha256((hashlib.sha256(ar['support'].tobytes()).hexdigest()+hashlib.sha256(ar['query'].tobytes()).hexdigest()).encode()).hexdigest();assert sh==r['split_sha256']
 for k,ref in r['metrics'].items():
  p=ar[k];assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-10
  for m,v in metrics(y,p).items():err=abs(v-ref[m]);maxerr=max(maxerr,err);assert err<1e-11;checked+=1
 rows.append(r)
expected=40 if stage=='screen' else 80
assert len(rows)==expected,(len(rows),expected)
plan=json.loads((ROOT/(stage+'_plan.json')).read_text());variants=plan['variants'];jobs=sorted(set(r['job'] for r in rows));summary={};pairs={}
for job in jobs:
 rs=sorted([r for r in rows if r['job']==job],key=lambda r:r['seed']);summary[job]={k:{m:float(np.mean([r['metrics'][k][m] for r in rs]))*100 for m in rs[0]['metrics'][k]} for k in rs[0]['metrics']};pairs[job]={}
 for v in variants:
  a='Prompt__'+v;pairs[job][v]={}
  for b in ['Prompt__localtemp','Prompt__fixed','Prompt__original_M3','Teacher_uncentered__fixed','Teacher_centered__fixed','Teacher_uncentered__none','Teacher_centered__none','Teacher_uncentered__'+v,'Teacher_centered__'+v,a+'_shuffled',a+'_mean','Teacher_uncentered__'+v+'_fixed','Teacher_centered__'+v+'_fixed','Teacher_uncentered__'+v+'_raw','Teacher_centered__'+v+'_raw']:
   pairs[job][v][b]=paired([r['metrics'][a]['accuracy'] for r in rs],[r['metrics'][b]['accuracy'] for r in rs])
ranking=[]
for v in variants:
 gains=[pairs[j][v]['Prompt__localtemp']['mean'] for j in jobs];gaps=[min(value['mean'] for name,value in pairs[j][v].items() if name.startswith('Teacher_')) for j in jobs]
 ranking.append(dict(variant=v,mean_gain=float(np.mean(gains)),positive_settings=sum(g>0 for g in gains),mean_matched_teacher_gap=float(np.mean(gaps)),passes=float(np.mean(gains))>=.2 and sum(g>0 for g in gains)>=3 and float(np.mean(gaps))>0))
selected=sorted([r for r in ranking if r['passes']],key=lambda r:(-r['mean_matched_teacher_gap'],r['variant']))
out=dict(stage=stage,records=len(rows),checked_metrics=checked,max_metric_error=maxerr,summary=summary,pairs=pairs,ranking=ranking,selected=selected[0]['variant'] if selected else None)
(ROOT/(stage+'_analysis.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:out[k] for k in ['stage','records','checked_metrics','max_metric_error','ranking','selected']},indent=2))
for job in jobs:
 print(job,{k:round(v['accuracy'],3) for k,v in summary[job].items() if not k.endswith('shuffled')})
