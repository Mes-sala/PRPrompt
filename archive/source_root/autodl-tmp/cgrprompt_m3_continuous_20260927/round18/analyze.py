from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'CGRPrompt_叙事补验_20260926'))
from audit_local import metrics,paired,sha
stage=sys.argv[1];rows=[];checked=0;maxerr=0.;max_ensemble_error=0.
plan=json.loads((ROOT/(stage+'_plan.json')).read_text());assert sha(ROOT/'设计与判定.md')==plan['design_sha256'];assert sha(ROOT/'run_structure.py')==plan['runner_sha256'];assert (ROOT/(stage+'_complete.json')).exists()
for f in sorted((ROOT/stage).glob('*/*/result.json')):
 r=json.loads(f.read_text());assert sha(f.parent/'prediction.npz')==r['prediction_sha256'];assert sha(ROOT/'run_structure.py')==r['runner_sha256'];assert r['frozen_verified'] and r['query_label_free_operator'];ar=np.load(f.parent/'prediction.npz');y=np.load(f.parent.parent/'query_labels.npy');assert not np.intersect1d(ar['support'],ar['query']).size
 sh=hashlib.sha256((hashlib.sha256(ar['support'].tobytes()).hexdigest()+hashlib.sha256(ar['query'].tobytes()).hexdigest()).encode()).hexdigest();assert sh==r['split_sha256']
 for k,ref in r['metrics'].items():
  p=ar[k];assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum(1)-1).max()<1e-10
  for m,v in metrics(y,p).items():err=abs(v-ref[m]);maxerr=max(maxerr,err);assert err<1e-11;checked+=1
 assert sha(f.parent/'fold_predictions.npz')==r['fold_predictions_sha256'];assert sha(f.parent/'folds.json')==r['folds_sha256'];assert sha(f.parent/'oof_probabilities.npz')==r['oof_sha256']
 fp=np.load(f.parent/'fold_predictions.npz');oo=np.load(f.parent/'oof_probabilities.npz');folds=json.loads((f.parent/'folds.json').read_text());seen=[]
 for i,fold in enumerate(folds):
  tr=np.array(fold['train']);vi=np.array(fold['heldout']);assert not np.intersect1d(tr,vi).size;assert set(tr)|set(vi)==set(ar['support']);seen+=vi.tolist();positions=[int(np.where(ar['support']==v)[0][0]) for v in vi]
  for src in ['Prompt','Teacher_uncentered','Teacher_centered']:assert np.array_equal(fp[src][i,vi],oo[src][positions])
 assert sorted(seen)==sorted(ar['support'].tolist()) and r['prompt_fits']==6
 for src in ['Prompt','Teacher_uncentered','Teacher_centered']:
  q=fp[src].mean(0)[ar['query']];full=ar[src+'__'+('localtemp' if src=='Prompt' else 'fixed')]
  for v in plan['variants']:
   w=.5 if v=='ens50correct' else 1.;expected_plain=(1-w)*full+w*q;err=float(np.abs(expected_plain-ar[src+'__'+v+'_plain']).max());assert err<3e-7;max_ensemble_error=max(max_ensemble_error,err)
 rows.append(r)
expected=len(plan['jobs'])*(10 if stage=='screen' else 20)
assert len(rows)==expected,(len(rows),expected)
plan=json.loads((ROOT/(stage+'_plan.json')).read_text());variants=plan['variants'];jobs=sorted(set(r['job'] for r in rows));summary={};pairs={}
for job in jobs:
 rs=sorted([r for r in rows if r['job']==job],key=lambda r:r['seed']);assert [r['seed'] for r in rs]==list(range(26094500,26094510) if stage=='screen' else range(26094520,26094540));summary[job]={k:{m:float(np.mean([r['metrics'][k][m] for r in rs]))*100 for m in rs[0]['metrics'][k]} for k in rs[0]['metrics']};pairs[job]={}
 for v in variants:
  a='Prompt__'+v;pairs[job][v]={}
  for b in ['Prompt__localtemp','Prompt__fixed','Prompt__original_M3','Teacher_uncentered__fixed','Teacher_centered__fixed','Teacher_uncentered__none','Teacher_centered__none','Teacher_uncentered__'+v,'Teacher_centered__'+v,a+'_shuffled',a+'_plain','Prompt__oof50l01','Teacher_uncentered__oof50l01','Teacher_centered__oof50l01','Teacher_uncentered__'+v+'_plain','Teacher_centered__'+v+'_plain']:
   pairs[job][v][b]=paired([r['metrics'][a]['accuracy'] for r in rs],[r['metrics'][b]['accuracy'] for r in rs])
ranking=[]
for v in variants:
 gains=[pairs[j][v]['Prompt__oof50l01']['mean'] for j in jobs];gaps=[min(value['mean'] for name,value in pairs[j][v].items() if name.startswith('Teacher_')) for j in jobs]
 ranking.append(dict(variant=v,mean_gain=float(np.mean(gains)),positive_settings=sum(g>0 for g in gains),mean_matched_teacher_gap=float(np.mean(gaps)),passes=float(np.mean(gains))>=.2 and sum(g>0 for g in gains)>=3 and float(np.mean(gaps))>0))
conditional=[]
for v in variants:
 js=[j for j in jobs if j.startswith('Cora')];gs=[pairs[j][v]['Prompt__oof50l01']['mean'] for j in js];tg=[min(value['mean'] for name,value in pairs[j][v].items() if name.startswith('Teacher_')) for j in js];pg=[pairs[j][v]['Prompt__localtemp']['mean'] for j in js]
 conditional.append(dict(scope='Cora',variant=v,mean_gain=float(np.mean(gs)),strong_teacher_gap=float(np.mean(tg)),passes=bool(all(x>0 for x in gs) and np.mean(gs)>=.2 and np.mean(tg)>0 and all(x>0 for x in pg))))
selected=sorted([r for r in ranking if r['passes']],key=lambda r:(-r['mean_matched_teacher_gap'],r['variant']))
out=dict(max_ensemble_replay_error=max_ensemble_error,conditional=conditional,stage=stage,records=len(rows),checked_metrics=checked,max_metric_error=maxerr,summary=summary,pairs=pairs,ranking=ranking,selected=selected[0]['variant'] if selected else None)
(ROOT/(stage+'_analysis.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:out[k] for k in ['stage','records','checked_metrics','max_metric_error','ranking','selected','conditional']},indent=2))
for job in jobs:
 print(job,{k:round(v['accuracy'],3) for k,v in summary[job].items() if not k.endswith('shuffled')})
