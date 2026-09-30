"""Independently recompute all sealed probabilities, paired deltas and transfer parity."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text());summary=[];verification=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
 oh=np.eye(p.shape[1])[y];pr=p.argmax(1)
 return dict(accuracy=float(accuracy_score(y,pr)),macro_f1=float(f1_score(y,pr,average='macro',zero_division=0)),micro_f1=float(f1_score(y,pr,average='micro',zero_division=0)),roc_auc_macro_ovr=float(roc_auc_score(oh,p,average='macro')),auprc_macro=float(average_precision_score(oh,p,average='macro')))
for j in plan['jobs']:
 d=R/'results'/j['id'];rows=[];history=[]
 for seed in j['seeds']:
  p=d/str(seed)/'result.json'
  if not p.exists():continue
  row=json.loads(p.read_text());f=p.parent/'sealed.npz';assert sha(f)==row['prediction_sha256'];assert row['runner_sha256']==plan['runner_sha256']
  a=np.load(f);y=np.load(d/'query_labels.npy');assert not np.isin(a['support'],a['query']).any()
  for key in row['metrics']:
   prob=a[key];assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1)
   m=metric(y,prob)
   assert max(abs(m[k]-row['metrics'][key][k]) for k in m)<1e-12
  old=R/'reference/core'/j['id']/f'{seed}__F111/result.json'
  if old.exists():
   prev=json.loads(old.read_text());assert prev['split_sha256']==row['split_sha256'] and prev['graph_sha256']==row['graph_sha256'];history.append(100*(row['metrics']['T0.2']['accuracy']-prev['accuracy']))
  rows.append(row)
 if not rows:continue
 for key in rows[0]['metrics']:
  stats={}
  for name in rows[0]['metrics'][key]:
   arr=np.array([x['metrics'][key][name] for x in rows])*100;ref=np.array([x['metrics']['T0.2'][name] for x in rows])*100;diff=arr-ref
   rng=np.random.default_rng(20260928);boot=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1);lo,hi=np.quantile(boot,[.025,.975])
   stats[name]=dict(mean=arr.mean(),sd=arr.std(ddof=1) if len(arr)>1 else None,delta_pp=diff.mean(),ci95=[lo,hi])
  summary.append(dict(job=j['id'],method=key,n=len(rows),metrics=stats))
 verification.append(dict(job=j['id'],verified_seeds=len(rows),expected_seeds=len(j['seeds']),historical_mean_delta_pp=float(np.mean(history)) if history else None,historical_max_absolute_delta_pp=float(np.max(np.abs(history))) if history else None))
(R/'summary.json').write_text(json.dumps(summary,indent=2));(R/'independent_verification.json').write_text(json.dumps(verification,indent=2))
lines=['# PRPrompt M3 初始概率软化结果','', '固定适配表示近邻，T=0.2为配对基线。Accuracy为百分数，差值为百分点。区间基于同图支持抽样，未校正多重比较。','', '|设置|温度/对照|n|Accuracy|相对T=0.2|95%配对区间|','|---|---|---:|---:|---:|---|']
for x in summary:
 m=x['metrics']['accuracy'];lines.append(f"|{x['job']}|{x['method']}|{x['n']}|{m['mean']:.3f}|{m['delta_pp']:+.3f}|[{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]|")
lines+=['','## 跨机器原始分支对照','']
for v in verification:lines.append(f"- {v['job']}：复算 {v['verified_seeds']}/{v['expected_seeds']}；相对历史原始M3均值差 {v['historical_mean_delta_pp']} pp，最大单抽样绝对差 {v['historical_max_absolute_delta_pp']} pp。")
(R/'结果汇总.md').write_text('\n'.join(lines)+'\n');print(json.dumps(verification))
