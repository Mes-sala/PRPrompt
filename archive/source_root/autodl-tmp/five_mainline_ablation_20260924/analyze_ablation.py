from pathlib import Path
import json,statistics,collections
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());B=Path(p['parent_remote']);B=B if B.exists() else Path(p['parent_local'])
assert json.loads((R/'final_audit.json').read_text())['passed'];assert json.loads((R/'reference_audit.json').read_text())['passed']
old=[json.loads(f.read_text()) for f in (B/'results').glob('*/*/result.json')];new=[json.loads(f.read_text()) for f in (R/'results').glob('*/*/result.json')];assert len(old)==5475 and len(new)==4610
rows={(r['dataset'],r['checkpoint_sha256'],r['seed'],r['method']):r for r in old+new};assert len(rows)==10085
comparisons=[
('01','动态CE缩放','A1_scale2_qeps','A1_token_qeps'),('01','概率传播','A1_scale2_qeps','A1_scale2_mean'),('01','Prompt整体','A1_scale2_qeps','base_qeps'),('01','KD','A1_scale2_qeps','A1_nokd'),('01','语义门控对应','A1_scale2_qeps','A1_permgate'),
('02','中心化','A2_center','rect_uncentered'),('02','软原型修正','A2_center','A2_norect'),('02','两跳传播','A2_center','A2_noprop'),('02','3次相对1次迭代','A2_center','A2_oneiter'),('02','自适应类别分配','A2_center','A2_uniform'),('02','伪标签节点对应','A2_center','A2_permassign'),
('03','门控锐化','A3_tr1_qeps','A3_temp02'),('03','输出中心化','A3_tr1_qeps','A3_nocenter'),('03','条件Prompt','A3_tr1_qeps','A3_noprompt'),('03','Adam步长匹配','A3_gsh_qeps','A3_ungrafted'),('03','完整矩阵相对对角','A3_gsh_qeps','A3_diagonal'),('03','匹配Shampoo相对Adam','A3_gsh_qeps','A3_tr1_qeps'),('03','概率传播','A3_gsh_qeps','A3_gsh_mean'),
('04','VAT一致性','A4_vat03','A4_none'),('04','KD','A4_vat03','A4_nokd'),('04','教师修正','A4_vat03','A4_norect_vat'),('04','VAT方向相对等epsilon随机噪声','A4_vat03','A4_randomnoise'),
('05','自环增强100','A5_self100','A5_self1'),('05','self-only结构','A5_selfonly','frozen'),('05','单独中心化','A5_center','A5_selfonly'),('05','中心化条件下对角缩放','A5_diag10','A5_center'),('05','对角缩放条件下中心化','A5_diag10','A5_diag_nocenter'),('05','方差收缩','A5_diag10','A5_diag0')]
interactions=[('01','动态CE×概率传播',['A1_scale2_qeps','A1_scale2_mean','A1_token_qeps','A1_token_mean']),('02','中心化×修正',['A2_center','A2_norect','rect_uncentered','rawprop']),('04','教师修正×VAT',['A4_vat03','A4_none','A4_norect_vat','A4_norect_none']),('05','中心化×对角缩放',['A5_diag10','A5_center','A5_diag_nocenter','A5_selfonly'])]
result=[]
for line,label,a,b in comparisons:
 cells=[]
 for j in p['jobs']:
  if j['direct']!=(line=='02'):continue
  ds=j['dataset'];h=None if j['direct'] else j['checkpoint']['sha256'];dsrows=[]
  for seed in p['seeds']:
   x=rows[ds,h,seed,a];y=rows[ds,h,seed,b];assert x['split_sha256']==y['split_sha256'];dsrows.append(dict(seed=seed,full=100*x['accuracy'],ablated=100*y['accuracy'],delta_pp=100*(x['accuracy']-y['accuracy'])))
  delta=[v['delta_pp'] for v in dsrows];cells.append(dict(dataset=ds,checkpoint='Direct' if j['direct'] else j['checkpoint']['family'],sha256=h,n=5,mean_delta_pp=statistics.mean(delta),std_delta_pp=statistics.stdev(delta),per_seed=dsrows))
 byds={ds:statistics.mean(c['mean_delta_pp'] for c in cells if c['dataset']==ds) for ds in p['datasets']};v=[c['mean_delta_pp'] for c in cells]
 result.append(dict(line=line,module=label,full=a,ablation=b,cells=cells,dataset_macro_pp=statistics.mean(byds.values()),checkpoint_macro_pp=statistics.mean(v),by_dataset=byds,positive=sum(x>1e-5 for x in v),tie=sum(abs(x)<=1e-5 for x in v),negative=sum(x< -1e-5 for x in v)))
ints=[]
for line,label,methods in interactions:
 cells=[]
 for j in p['jobs']:
  if j['direct']!=(line=='02'):continue
  ds=j['dataset'];h=None if j['direct'] else j['checkpoint']['sha256'];v=[]
  for seed in p['seeds']:
   rr=[rows[ds,h,seed,m] for m in methods];assert len({r['split_sha256'] for r in rr})==1;acc=[r['accuracy']*100 for r in rr];v.append(dict(seed=seed,interaction_pp=acc[0]-acc[1]-acc[2]+acc[3]))
  cells.append(dict(dataset=ds,checkpoint='Direct' if j['direct'] else j['checkpoint']['family'],sha256=h,mean_interaction_pp=statistics.mean(t['interaction_pp'] for t in v),per_seed=v))
 ints.append(dict(line=line,name=label,methods=methods,cells=cells))
# Equal nominal epsilon does not automatically imply equal realized RMS (near-zero VAT gradients).
rms=[]
for j in p['jobs']:
 if j['direct']:continue
 for seed in p['seeds']:
  a=rows[j['dataset'],j['checkpoint']['sha256'],seed,'A4_vat03'];b=rows[j['dataset'],j['checkpoint']['sha256'],seed,'A4_randomnoise'];aa=[x.get('perturb_rms') for x in a['trace']['history']];bb=[x.get('perturb_rms') for x in b['trace']['history']]
  rms.append(dict(dataset=j['dataset'],sha256=j['checkpoint']['sha256'],seed=seed,vat_rms=aa,random_rms=bb,matched_recorded_steps=all(abs(x-y)<1e-5 for x,y in zip(aa,bb))))
(R/'消融配对.json').write_text(json.dumps(dict(comparisons=result,interactions=ints,vat_rms_audit=rms),ensure_ascii=False,indent=2))
text='# 五主线消融结果\n\n4610条新结果及5475条旧对照引用已完成服务器审计。传输核验另见transport_audit.json。相同8数据集、63份checkpoint、5-shot、5种子和100步；非独立外部确认，不能把支持采样seed当独立图。主线01是γ=2动态CE缩放，不是固定乘2。\n\n所有差值为完整分支减去消融分支；正值表示该条件下保留模块更好，负值表示该模块有害。不要跨不同完整分支混用基准。数据集宏均值对8个数据集等权，checkpoint宏均值对63格等权。主线02只有8个直接方法格，不按checkpoint重复计数。\n\n| 主线 | 检验模块 | 数据集宏均值pp | checkpoint宏均值pp | 正/平/负格 |\n|---|---|---:|---:|---|\n'
for r in result:text+=f"| {r['line']} | {r['module']} | {r['dataset_macro_pp']:+.3f} | {r['checkpoint_macro_pp']:+.3f} | {r['positive']}/{r['tie']}/{r['negative']} |\n"
text+='\n## VAT方向对照的幅度审核\n\n两分支设定epsilon均为.3，但近零梯度可能使VAT实际扰动RMS更小。只有保存的首末步RMS均一致，才可把结果近似解释为方向对照；保存首末步不等于全轨迹匹配。记录的RMS通过格数：'+str(sum(x['matched_recorded_steps'] for x in rms))+'/'+str(len(rms))+'。不匹配项必须注明幅度混杂，不能单独归因于方向。\n'
for line in ['01','02','03','04','05']:
 t=f'# 主线{line}消融结果\n\n每项按同seed、同checkpoint配对。差值=完整分支−消融分支。正值是保留模块收益；并非统计显著性结论。\n'
 for r in [x for x in result if x['line']==line]:
  t+=f"\n## {r['module']}\n\n完整：`{r['full']}`；消融：`{r['ablation']}`。\n\n| 数据集 | checkpoint | SHA前缀 | 配对增量均值±标准差pp |\n|---|---|---|---:|\n"
  for c in r['cells']:t+=f"| {c['dataset']} | {c['checkpoint']} | {(c['sha256'] or 'direct')[:12]} | {c['mean_delta_pp']:+.3f} ± {c['std_delta_pp']:.3f} |\n"
 for it in [x for x in ints if x['line']==line]:
  t+=f"\n## {it['name']}的2×2交互\n\n定义：`{it['methods'][0]} - {it['methods'][1]} - {it['methods'][2]} + {it['methods'][3]}`。描述性交互，正值不代表已证明普遍协同。\n\n| 数据集 | checkpoint | SHA前缀 | 交互pp |\n|---|---|---|---:|\n"
  for c in it['cells']:t+=f"| {c['dataset']} | {c['checkpoint']} | {(c['sha256'] or 'direct')[:12]} | {c['mean_interaction_pp']:+.3f} |\n"
 (R/f'主线{line}_消融结果.md').write_text(t)
 text+=f'\n- [主线{line}逐checkpoint消融结果](主线{line}_消融结果.md)\n'
(R/'消融结果汇总.md').write_text(text);print('ablation report created')
