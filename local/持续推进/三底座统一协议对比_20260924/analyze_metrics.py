from pathlib import Path
import re
import json,statistics as st,random,csv
import numpy as np
R=Path(__file__).resolve().parent;d=json.loads((R/'all_metrics.json').read_text());plan=json.loads((R/'plan.json').read_text());methods=plan['methods']+['MTG','DAPrompt-fixed'];metric_names=['accuracy','macro_f1','micro_f1','roc_auc_macro_ovr','auprc_macro'];labels=['Accuracy','Macro-F1','Micro-F1','AUC-ROC (macro OVR)','AUPRC (macro AP)'];g=random.Random(20260924);draws=np.array([[g.randrange(20) for _ in range(20)] for _ in range(10000)])
assert len(d['rows'])==1680,len(d['rows']);out=[];lines=['# 统一协议：三底座、14方法、多指标对照','两数据集、六checkpoint、20共同支持seed、5-shot、100优化更新。新增2026年模型为MTG和DAPrompt-fixed；后者只复现固定底座上的提示阶段，不包含专用结构预训练和主干更新。新方法算子与官方代码对照通过，但本表是统一协议移植结果，不冒称原论文数值复刻。','指标单位均为百分数，表中为20seed均值±样本标准差。ROC采用多类one-vs-rest宏平均，PR采用每类average precision宏平均；从封存分数统一softmax后计算。Micro-F1在本任务等于Accuracy。预测封存后才计算指标。']
for j in plan['jobs']:
 ds,fam=j['dataset'],j['checkpoint']['family'];v={};splits={}
 for m in methods:
  rows=sorted([r for r in d['rows'] if r['dataset']==ds and r['checkpoint']==fam and r['method']==m],key=lambda r:r['seed']);assert [r['seed'] for r in rows]==plan['seeds'];assert all(r['checkpoint_sha256']==j['checkpoint']['sha256'] for r in rows)
  v[m]={k:np.array([r['metrics'][k]*100 for r in rows]) for k in metric_names};splits[m]=[r['split_sha256'] for r in rows]
 assert all(x==splits['F000'] for x in splits.values())
 lines+=['## '+ds+' / '+fam,'| 方法 | '+' | '.join(labels)+' |','|'+'---|'*6]
 for m in methods:
  lines+=['| '+m+' | '+' | '.join(f'{v[m][k].mean():.3f} ± {v[m][k].std(ddof=1):.3f}' for k in metric_names)+' |']
  for k in metric_names:
   diff=v['F111'][k]-v[m][k];boots=np.sort(diff[draws].mean(1));out.append({'dataset':ds,'checkpoint':fam,'method':m,'metric':k,'mean':float(v[m][k].mean()),'sd':float(v[m][k].std(ddof=1)),'full_minus_method':float(diff.mean()),'ci_low':float(boots[249]),'ci_high':float(boots[9749]),'positive':int((diff>1e-8).sum()),'tie':int((abs(diff)<=1e-8).sum()),'negative':int((diff< -1e-8).sum())})
 lines+=['### 完整方案相对对照的配对差值（pp）','各单元格为差值及95% bootstrap区间；正值表示完整方案更高。','| 对照 | Accuracy | Macro-F1 | AUC-ROC | AUPRC |','|---|---|---|---|---|']
 for m in methods:
  if m=='F111':continue
  vals=[]
  for k in ['accuracy','macro_f1','roc_auc_macro_ovr','auprc_macro']:
   q=next(x for x in out if x['dataset']==ds and x['checkpoint']==fam and x['method']==m and x['metric']==k);vals.append(f"{q['full_minus_method']:+.3f} [{q['ci_low']:+.3f}, {q['ci_high']:+.3f}]")
  lines.append('| '+m+' | '+' | '.join(vals)+' |')
lines+=['## 统计与复现边界','配对区间为固定图、固定查询集条件下的支持采样bootstrap，10000次，种子20260924，未校正多重比较。三底座是前轮筛选，不能作为独立外部验证。不同方法保留原生读出与不同计算量，100优化更新并不等于100相同FLOPs。论文方法或底座不兼容的改动均见机制与协议说明.md和extension2026/模型选择与复现边界.md。完整方案失败的格照常报告，不删行、不补调query。']
(R/'统一协议多指标对比.md').write_text(re.sub(r'(?<=\|)\n\n(?=\|)', '\n', '\n\n'.join(lines))+'\n')
(R/'all_metric_pairs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with (R/'统一协议多指标对比.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('all metrics cells',len(out))
