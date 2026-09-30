from pathlib import Path
import re
import json,statistics,random,csv
R=Path(__file__).resolve().parent;plan=json.loads((R/'plan.json').read_text());assert json.loads((R/'final_audit.json').read_text())['passed']
g=random.Random(20260924);draws=[[g.randrange(20) for _ in range(20)] for _ in range(10000)]
def paired(v):
 a=sorted(sum(v[i] for i in ix)/20 for ix in draws)
 return {'mean_pp':statistics.mean(v),'ci95':[a[249],a[9749]],'positive':sum(x>1e-8 for x in v),'negative':sum(x< -1e-8 for x in v),'tie':sum(abs(x)<=1e-8 for x in v)}
cells=[];flat=[]
for job in plan['jobs']:
 c={'dataset':job['dataset'],'checkpoint':job['checkpoint']['family'],'checkpoint_sha256':job['checkpoint']['sha256'],'methods':{}}
 for m in plan['methods']:
  rows=[json.loads((R/'results'/job['id']/f'{s}__{m}'/'result.json').read_text()) for s in plan['seeds']]
  scores=[100*r['accuracy'] for r in rows];c['methods'][m]={'mean':statistics.mean(scores),'sd':statistics.stdev(scores),'per_seed':scores,'mean_seconds':statistics.mean(r['seconds'] for r in rows)}
 for m,d in c['methods'].items():
  d['minus_000']=paired([v-b for v,b in zip(d['per_seed'],c['methods']['F000']['per_seed'])]);d['full_minus_method']=paired([v-b for v,b in zip(c['methods']['F111']['per_seed'],d['per_seed'])]);flat.append(dict(dataset=c['dataset'],checkpoint=c['checkpoint'],method=m,mean_accuracy=d['mean'],sd_accuracy=d['sd'],full_minus_method_pp=d['full_minus_method']['mean_pp'],ci_low=d['full_minus_method']['ci95'][0],ci_high=d['full_minus_method']['ci95'][1],mean_seconds=d['mean_seconds']))
 cells.append(c)
(R/'paired_results.json').write_text(json.dumps({'cells':cells,'bootstrap_seed':20260924,'bootstrap_replicates':10000,'scope':'same graph support sampling; no multiple comparison adjustment; selected backbones'},ensure_ascii=False,indent=2))
with (R/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
lines=['# 三底座统一协议重跑结果','所有方法同一checkpoint、同一20支持seed、同一查询集、5-shot和100优化更新。保留方法各自输入/读出机制；不是原论文完整协议复刻。来源与移植差异见机制与协议说明.md。','## Accuracy均值±样本标准差（%）','| 方法 | '+' | '.join(c['dataset']+'/'+c['checkpoint'] for c in cells)+' |','|'+'---|'*7]
for m in plan['methods']:lines.append('| '+m+' | '+' | '.join(f"{c['methods'][m]['mean']:.3f} ± {c['methods'][m]['sd']:.3f}" for c in cells)+' |')
for c in cells:
 lines+=['## '+c['dataset']+'/'+c['checkpoint'],'| 对照 | 完整方案−对照 pp | 配对95%区间 | 正/平/负seed |','|---|---:|---|---|']
 for m,d in c['methods'].items():
  if m=='F111':continue
  q=d['full_minus_method'];lines.append(f"| {m} | {q['mean_pp']:+.3f} | [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}] | {q['positive']}/{q['tie']}/{q['negative']} |")
lines+=['## 解读边界','正负结果全部列出，不因本方法失败删除基线。不将20支持seed当独立图；预训练底座是前轮筛选，未校正多重比较。F011/F101/F110分别去M1/M2/M3，F000是共同基础；不能把胜过000当作胜过已有Prompt。GraphTOP包含随机推理，固定单次推理seed，无query挑选。不同方法FLOPs/可训练参数量不等，时间与参数记录保存在结果制品。']
(R/'统一协议结果对比.md').write_text(re.sub(r'(?<=\|)\n\n(?=\|)', '\n', '\n\n'.join(lines))+'\n');print('analyzed',len(flat),'cells')
