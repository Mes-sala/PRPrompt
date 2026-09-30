from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text())
assert json.loads((R/'final_audit.json').read_text())['passed'];assert json.loads((R/'reference_audit.json').read_text())['passed']
rows={}
for ref in json.loads((R/'reference_manifest.json').read_text()):
 b=Path(p[ref['origin']+'_remote']);b=b if b.exists() else Path(p[ref['origin']+'_local']);r=json.loads((b/ref['path']).read_text());rows[r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']]=r
for f in (R/'results').glob('*/*/result.json'):
 r=json.loads(f.read_text());rows[r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']]=r
metrics=['combo_minus_neither','combo_minus_sharp_only','combo_minus_prop_only','interaction','combo_minus_gsh','combo_minus_A1_scale','combo_minus_A1_CE','combo_minus_direct_uncentered','combo_minus_direct_centered'];cells=[]
for j in p['jobs']:
 ds=j['dataset'];h=j['checkpoint']['sha256'];rr=[]
 for seed in p['seeds']:
  def get(m,direct=False):return rows[ds,None if direct else h,seed,m]
  methods=['Combo_temp02_mean','A3_tr1_mean','A3_temp02','A3_tr1_qeps','A3_gsh_qeps','A1_scale2_qeps','A1_token_qeps']
  values=[get(m) for m in methods]+[get('rect_uncentered',True),get('A2_center',True)];assert len({r['split_sha256'] for r in values})==1
  a,b,c,d,g,aa,ac,tu,tc=[100*r['accuracy'] for r in values]
  rr.append(dict(seed=seed,neither=a,sharp_only=b,prop_only=c,combo=d,**dict(zip(metrics,[d-a,d-b,d-c,d-b-c+a,d-g,d-aa,d-ac,d-tu,d-tc]))))
 cells.append(dict(dataset=ds,checkpoint=j['checkpoint']['family'],sha256=h,per_seed=rr,mean={m:statistics.mean(r[m] for r in rr) for m in metrics+['neither','sharp_only','prop_only','combo']},std={m:statistics.stdev(r[m] for r in rr) for m in metrics}))
stats={}
for m in metrics:
 byds={ds:statistics.mean(c['mean'][m] for c in cells if c['dataset']==ds) for ds in p['datasets']};vs=[c['mean'][m] for c in cells]
 stats[m]=dict(dataset_macro=statistics.mean(byds.values()),checkpoint_macro=statistics.mean(vs),by_dataset=byds,positive=sum(x>1e-5 for x in vs),tie=sum(abs(x)<=1e-5 for x in vs),negative=sum(x< -1e-5 for x in vs))
o=dict(cells=cells,summary=stats,descriptive_joint_wins=sum(c['mean']['combo_minus_sharp_only']>1e-5 and c['mean']['combo_minus_prop_only']>1e-5 for c in cells),joint_and_direct_wins=sum(all(c['mean'][m]>1e-5 for m in ['combo_minus_sharp_only','combo_minus_prop_only','combo_minus_direct_uncentered','combo_minus_direct_centered']) for c in cells))
(R/'组合配对.json').write_text(json.dumps(o,ensure_ascii=False,indent=2))
t='# 锐化与概率传播：2×2组合结果\n\n315条新增读出、1970条历史引用通过服务器审计；新增训练次数为0。沿用8数据集、63份checkpoint、5-shot、5支持seed及100步历史训练。组合分支是已有A3_tr1_qeps，补齐普通温度无传播后分析交互，不冒充新训练或独立确认。\n\n差值单位pp，正值表示组合更好；交互是差中之差。5个seed不是独立图。\n\n| 对比 | 数据集宏均 | checkpoint宏均 | 正/平/负格 |\n|---|---:|---:|---|\n'
for m,r in stats.items():t+=f"| {m} | {r['dataset_macro']:+.3f} | {r['checkpoint_macro']:+.3f} | {r['positive']}/{r['tie']}/{r['negative']} |\n"
t+=f"\n同时超过两个单组件的格数：{o['descriptive_joint_wins']}/63；还同时超过两个直接教师：{o['joint_and_direct_wins']}/63。此处只描述固定格，不选择新配置。\n"
for m,r in stats.items():
 t+=f'\n## {m}\n\n| 数据集 | 平均差pp |\n|---|---:|\n'+''.join(f'| {ds} | {v:+.3f} |\n' for ds,v in r['by_dataset'].items())
t+='\n## 逐checkpoint四格准确率\n\n| 数据集 | checkpoint | SHA前缀 | 无组件 | 仅锐化 | 仅传播 | 组合 | 交互pp |\n|---|---|---|---:|---:|---:|---:|---:|\n'
for c in cells:
 v=c['mean'];t+=f"| {c['dataset']} | {c['checkpoint']} | {c['sha256'][:12]} | {v['neither']:.3f} | {v['sharp_only']:.3f} | {v['prop_only']:.3f} | {v['combo']:.3f} | {v['interaction']:+.3f} |\n"
t+='\n输出中心化固定保留，KD关闭；本轮没有测试删除中心化的新精简组合。锐化包含原raw门控的幅度变化，并非纯熵效应。传播是末端读出，不是主干内第二个Prompt。无图分类、无MultiGprompt架构覆盖。完整配对与5种子差值见组合配对.json；最终本地归档状态另见传输审计。\n'
(R/'组合结果汇总.md').write_text(t);print(json.dumps(o['summary'],ensure_ascii=False))
