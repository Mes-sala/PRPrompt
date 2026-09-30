from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'CGRPrompt_叙事补验_20260926'))
from audit_local import metrics,paired,sha
stage='confirm';rows=[];checked=0;maxerr=0.;max_ensemble_error=0.
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
 assert r['candidate_required_fits']==3 and r['selected_folds']==[r['seed']%5,(r['seed']+2)%5]
 for src in ['Prompt','Teacher_uncentered','Teacher_centered']:
  full=ar[src+'__'+('localtemp' if src=='Prompt' else 'fixed')]
  for key,ix in [('budget2',r['selected_folds']),('ensemble6',list(range(5)))]:
   q=fp[src][ix].mean(0)[ar['query']];expected_prob=.5*full+.5*q;err=float(np.abs(expected_prob-ar[src+'__'+key]).max());assert err<3e-7;max_ensemble_error=max(max_ensemble_error,err)

 rows.append(r)
expected=len(plan['jobs'])*(10 if stage=='screen' else 20)
assert len(rows)==expected,(len(rows),expected)


a='Prompt__budget2';jobs=sorted({r['job'] for r in rows});assert len(jobs)==2 and all(j.startswith('CiteSeer') for j in jobs)
teachers=['Teacher_uncentered__fixed','Teacher_centered__fixed','Teacher_uncentered__none','Teacher_centered__none','Teacher_uncentered__budget2','Teacher_centered__budget2']
summary={};pairs={};gains=[];cost=[];gaps=[]
for j in jobs:
 rs=sorted([r for r in rows if r['job']==j],key=lambda r:r['seed']);assert [r['seed'] for r in rs]==list(range(26094620,26094640))
 summary[j]={k:{m:float(np.mean([r['metrics'][k][m] for r in rs]))*100 for m in rs[0]['metrics'][k]} for k in rs[0]['metrics']}
 pairs[j]={b:paired([r['metrics'][a]['accuracy'] for r in rs],[r['metrics'][b]['accuracy'] for r in rs]) for b in ['Prompt__localtemp','Prompt__fixed','Prompt__ensemble6','Prompt__budget2_shuffled','Teacher_uncentered__ensemble6','Teacher_centered__ensemble6']+teachers}
 gains.append(pairs[j]['Prompt__localtemp']['mean']);cost.append(pairs[j]['Prompt__ensemble6']['mean']);gaps.append(min(pairs[j][t]['mean'] for t in teachers))
out=dict(records=len(rows),candidate_required_fits=3,experiment_fits=len(rows)*6,checked_metrics=checked,max_metric_error=maxerr,max_ensemble_replay_error=max_ensemble_error,mean_gain=float(np.mean(gains)),change_vs_six=float(np.mean(cost)),strong_teacher_gap=float(np.mean(gaps)),passes=bool(all(g>0 for g in gains) and np.mean(gains)>=.2 and np.mean(cost)>=-.1 and np.mean(gaps)>0),summary=summary,pairs=pairs)
(ROOT/'confirm_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
lines=['# 三次拟合M3-E2确认结果','','每设置20次新支持抽样；候选部署需3次拟合，验证为了配对六次版本仍实际训练6次/包，共240次。','','|设置|M3-LC|M3-E六次|M3-E2三次|比M3-LC及95%区间|比六次版本及95%区间|','|---|---:|---:|---:|---:|---:|']
for j in jobs:
 b=pairs[j]['Prompt__localtemp'];c=pairs[j]['Prompt__ensemble6'];lines.append('|'+j+'|'+ '|'.join(f"{summary[j][k]['accuracy']:.3f}" for k in ['Prompt__localtemp','Prompt__ensemble6',a])+'|'+f"{b['mean']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]|{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]|")
lines+=['',f"预设采用条件：{out['passes']}；平均较M3-LC{out['mean_gain']:+.3f}pp，较六次版本{out['change_vs_six']:+.3f}pp，强教师差{out['strong_teacher_gap']:+.3f}pp。",'','|设置|最强匹配教师|教师Accuracy|差值及95%区间|','|---|---|---:|---:|']
for j in jobs:
 t=max(teachers,key=lambda k:summary[j][k]['accuracy']);q=pairs[j][t];lines.append('|'+j+'|'+t+'|'+f"{summary[j][t]['accuracy']:.3f}|{q['mean']:+.3f} [{q['ci_low']:+.3f}, {q['ci_high']:+.3f}]|")
lines+=['',f"独立复算{checked}项指标通过，最大误差{maxerr:.3g}；指定两折与全部五折重算误差{max_ensemble_error:.3g}。",'','拟合数量减半不等于已实测端到端耗时减半。平均−0.1pp采用门槛不是正式统计非劣检验；同图新支持抽样不等于独立图泛化。']
(ROOT/'确认结果.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:out[k] for k in ['records','mean_gain','change_vs_six','strong_teacher_gap','passes']},ensure_ascii=False));print('\n'.join(lines))
