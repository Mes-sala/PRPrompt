from pathlib import Path
import json,statistics,itertools
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text())
for name in ['final_audit.json','reference_audit.json']:assert json.loads((R/name).read_text())['passed']
rows={}
for ref in json.loads((R/'reference_manifest.json').read_text()):
 b=Path(p[ref['origin']+'_remote']);b=b if b.exists() else Path(p[ref['origin']+'_local']);r=json.loads((b/ref['path']).read_text());rows[r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']]=r
for f in (R/'results').glob('*/*/result.json'):
 r=json.loads(f.read_text());rows[r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']]=r
mapping={'000':'Combo_temp02_mean','001':'A3_temp02','010':'T3_normal_gsh_mean','011':'T3_normal_gsh_qeps','100':'A3_tr1_mean','101':'A3_tr1_qeps','110':'A3_gsh_mean','111':'A3_gsh_qeps'}
comparisons={'remove_M1':{'111':1,'011':-1},'remove_M2':{'111':1,'101':-1},'remove_M3':{'111':1,'110':-1},'full_vs_base':{'111':1,'000':-1}}
for i in range(3):
 bits=['0']*3;bits[i]='1';comparisons['M'+str(i+1)+'_alone']={''.join(bits):1,'000':-1}
for subset in [(0,1),(0,2),(1,2),(0,1,2)]:
 for fixed in ([0,1] if len(subset)==2 else [0]):
  weights={}
  for active in itertools.product([0,1],repeat=len(subset)):
   bits=[fixed]*3
   for i,v in zip(subset,active):bits[i]=v
   weights[''.join(map(str,bits))]=(-1)**(len(subset)-sum(active))
  comparisons['interaction_'+''.join(str(i+1) for i in subset)+'_other'+str(fixed)]=weights
cells=[]
for job in p['jobs']:
 ds=job['dataset'];sha=job['checkpoint']['sha256'];seeds=[]
 for seed in p['seeds']:
  rr={k:rows[ds,sha,seed,m] for k,m in mapping.items()};assert len({r['split_sha256'] for r in rr.values()})==1
  a={k:100*r['accuracy'] for k,r in rr.items()};metrics={k:sum(a[cell]*w for cell,w in weights.items()) for k,weights in comparisons.items()}
  for m in ['rect_uncentered','A2_center','graphprompt','frozen','A1_scale2_qeps','A1_token_qeps','A3_diagonal','A3_ungrafted']:
   ref=rows[ds,None if m in ['rect_uncentered','A2_center'] else sha,seed,m];assert ref['split_sha256']==rr['111']['split_sha256'];metrics['full_minus_'+m]=a['111']-100*ref['accuracy']
  seeds.append(dict(seed=seed,accuracy=a,contrasts=metrics))
 cells.append(dict(dataset=ds,checkpoint=job['checkpoint']['family'],sha256=sha,per_seed=seeds,means={k:statistics.mean(r['contrasts'][k] for r in seeds) for k in seeds[0]['contrasts']}))
summary={}
for k in cells[0]['means']:
 d={ds:statistics.mean(c['means'][k] for c in cells if c['dataset']==ds) for ds in p['datasets']};v=[c['means'][k] for c in cells];summary[k]=dict(dataset_macro=statistics.mean(d.values()),checkpoint_macro=statistics.mean(v),by_dataset=d,positive=sum(x>1e-5 for x in v),tie=sum(abs(x)<=1e-5 for x in v),negative=sum(x< -1e-5 for x in v))
out=dict(mapping=mapping,contrasts=comparisons,summary=summary,cells=cells,three_modules_positive_macro=all(summary['remove_M'+str(i)]['dataset_macro']>0 for i in [1,2,3]))
(R/'factorial_pairs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
t='# 阶段1三模块因子结果\n\n探索性全因子补齐，不是新支持seed确认。M1锐化、M2步长匹配Shampoo替换Adam、M3概率传播。每个差值同checkpoint同支持种子配对；正值为完整方案较好，交互为差中之差。\n\n| 对照 | 数据集宏均pp | checkpoint宏均pp | 正/平/负 |\n|---|---:|---:|---|\n'
for k,r in summary.items():t+=f"| {k} | {r['dataset_macro']:+.3f} | {r['checkpoint_macro']:+.3f} | {r['positive']}/{r['tie']}/{r['negative']} |\n"
t+='\n## 完整方案逐模块移除\n\n| 数据集 | 去掉M1 | 去掉M2 | 去掉M3 | 相对未中心化直接教师 |\n|---|---:|---:|---:|---:|\n'
for ds in p['datasets']:t+='| '+ds+' | '+' | '.join(f"{summary[k]['by_dataset'][ds]:+.3f}" for k in ['remove_M1','remove_M2','remove_M3','full_minus_rect_uncentered'])+' |\n'
t+='\n三个模块宏均移除增量是否全正：'+str(out['three_modules_positive_macro'])+'。此布尔量不是毕业设计完成标准；还需固定方案确认、强基线竞争力、复杂性对照及完整章节文档。全8格逐seed准确率与全部交互见factorial_pairs.json。\n';(R/'阶段1结果.md').write_text(t);print(json.dumps(summary,ensure_ascii=False))
