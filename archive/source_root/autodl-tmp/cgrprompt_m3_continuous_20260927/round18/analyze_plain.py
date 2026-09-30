from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'CGRPrompt_叙事补验_20260926'))
from audit_local import metrics,paired,sha
stage='plain_confirm';rows=[];checked=0;maxerr=0.;max_ensemble_error=0.
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

assert sha(ROOT/'纯集成确认设计.md')==plan['conditional_design_sha256'];assert sha(ROOT/'run_plain_confirm.py')==plan['wrapper_sha256']
a='Prompt__ens50correct_plain';jobs=sorted({r['job'] for r in rows});assert len(jobs)==2 and all(j.startswith('CiteSeer') for j in jobs)
summary={};pairs={};gains=[];gaps=[]
teachers=['Teacher_uncentered__fixed','Teacher_centered__fixed','Teacher_uncentered__none','Teacher_centered__none','Teacher_uncentered__ens50correct_plain','Teacher_centered__ens50correct_plain']
for j in jobs:
 rs=sorted([r for r in rows if r['job']==j],key=lambda r:r['seed']);assert [r['seed'] for r in rs]==list(range(26094520,26094540))
 summary[j]={k:{m:float(np.mean([r['metrics'][k][m] for r in rs]))*100 for m in rs[0]['metrics'][k]} for k in rs[0]['metrics']}
 pairs[j]={b:paired([r['metrics'][a]['accuracy'] for r in rs],[r['metrics'][b]['accuracy'] for r in rs]) for b in ['Prompt__localtemp','Prompt__fixed','Prompt__oof50l01','Prompt__ens50correct']+teachers}
 gains.append(pairs[j]['Prompt__localtemp']['mean']);gaps.append(min(pairs[j][t]['mean'] for t in teachers))
result=dict(candidate=a,scope='CiteSeer',records=len(rows),prompt_fits=6*len(rows),checked_metrics=checked,max_metric_error=maxerr,max_ensemble_replay_error=max_ensemble_error,mean_gain=float(np.mean(gains)),strong_teacher_gap=float(np.mean(gaps)),passes=bool(all(g>0 for g in gains) and np.mean(gains)>=.2 and np.mean(gaps)>0),summary=summary,pairs=pairs)
(ROOT/'纯集成确认结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# CiteSeer纯集成确认结果','','候选为完整5shot模型与五个4shot留出模型平均预测的等权组合，不含OOF误差校正。20次新支持抽样/设置，40包共240次Prompt拟合；不能与单次拟合基线宣称相同训练成本。','','|设置|M3-LC|纯集成|差值及95%区间|同批最强教师|教师差及95%区间|','|---|---:|---:|---:|---|---:|']
for j in jobs:
 t=max(teachers,key=lambda k:summary[j][k]['accuracy']);b=pairs[j]['Prompt__localtemp'];q=pairs[j][t]
 lines.append('|'+j+'|'+f"{summary[j]['Prompt__localtemp']['accuracy']:.3f}|{summary[j][a]['accuracy']:.3f}|{b['mean']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]|{t} {summary[j][t]['accuracy']:.3f}|{q['mean']:+.3f} [{q['ci_low']:+.3f}, {q['ci_high']:+.3f}]|")
lines+=['',f"预设条件是否通过：{result['passes']}；相对M3-LC平均{result['mean_gain']:+.4f}pp，相对最强教师平均{result['strong_teacher_gap']:+.4f}pp。",'',f"五指标独立复算{checked}项通过，最大误差{maxerr:.3g}；折身份/OOF缓存来源核验通过，集成重算概率最大差{max_ensemble_error:.3g}。",'','同一张图的新支持抽样确认，不能视作独立图泛化；该假设来自第十八轮开发对照，选择历史保留。']
(ROOT/'纯集成确认结果.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:result[k] for k in ['records','mean_gain','strong_teacher_gap','passes']},ensure_ascii=False));print('\n'.join(lines))
