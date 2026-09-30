from pathlib import Path
import json,statistics,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;out=R/'chapter_assets';out.mkdir(exist_ok=True)
stages=['stage2_confirmation','stage2b_gppt_confirmation'];cells=[];costs=[]
for stage in stages:
 p=R/stage;plan=json.loads((p/'plan.json').read_text());assert json.loads((p/'final_audit.json').read_text())['passed'];assert json.loads((p/'pair_audit.json').read_text())['passed']
 j=json.loads((p/'confirmation_pairs.json').read_text())
 for c in j['cells']:cells.append(dict(c,stage=stage))
 for job in plan['jobs']:
  if job['direct']:continue
  for m in ['F101','F110','F111','M2_diagonal','M2_ungrafted']:
   rows=[json.loads((p/'results'/job['id']/f'{seed}__{m}'/'result.json').read_text()) for seed in plan['seeds']]
   tr=[r['trace']['training_seconds'] for r in rows];rd=[r['trace']['readout_seconds'] for r in rows]
   costs.append(dict(stage=stage,dataset=job['dataset'],checkpoint=job['checkpoint']['family'],sha256=job['checkpoint']['sha256'],method=m,trainable_parameters=rows[0]['trace']['trainable_parameters'],training_seconds_mean=statistics.mean(tr),readout_seconds_mean=statistics.mean(rd),fit_plus_readout_seconds_mean=statistics.mean(a+b for a,b in zip(tr,rd)),warning='Per-method latency estimate; do not sum shared fits. Excludes dataset/model loading and serialization. Same steps are not equal time budgets.'))
(out/'costs.json').write_text(json.dumps(costs,ensure_ascii=False,indent=2));(out/'consolidated_results.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2))
fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
for i,ax in enumerate(axes,1):
 for y,c in enumerate(cells):
  r=c['contrasts']['remove_M'+str(i)];v=r['mean_pp'];lo,hi=r['ci95'];color='#2166ac' if lo>0 else ('#b2182b' if v<0 else '#777777')
  ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=4,markersize=5)
 ax.axvline(0,color='black',lw=.8,ls='--');ax.set_title(['M1: gate sharpening','M2: matched optimization','M3: probability propagation'][i-1]);ax.set_xlabel('Full minus ablated accuracy (pp)');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(cells)),[c['dataset']+' / '+c['checkpoint'].replace('Edgepred_','EP-') for c in cells]);axes[0].invert_yaxis();fig.suptitle('Fixed-recipe confirmation: 20 paired support seeds per setting');fig.tight_layout();fig.savefig(out/'module_effects.png',dpi=200);fig.savefig(out/'module_effects.svg');plt.close(fig)
t='# 实验成本\n\n单位秒，20个seed均值；不含数据/权重加载与预测序列化。F101为锐化Adam+传播，F110为锐化匹配Shampoo无传播，F111为完整方法。共享训练不能重复计数，统计来源是运行时计时，不是严格隔离GPU性能基准。\n\n| 数据集/底座 | 方法 | 可训练参数 | 训练秒 | 读出秒 | 合计秒 |\n|---|---|---:|---:|---:|---:|\n'
for c in costs:t+=f"| {c['dataset']}/{c['checkpoint']} | {c['method']} | {c['trainable_parameters']} | {c['training_seconds_mean']:.3f} | {c['readout_seconds_mean']:.3f} | {c['fit_plus_readout_seconds_mean']:.3f} |\n"
(out/'成本表.md').write_text(t)
(out/'manifest.json').write_text(json.dumps([dict(path=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.iterdir()) if f.is_file() and f.name!='manifest.json'],indent=2));print('chapter_assets_created')
