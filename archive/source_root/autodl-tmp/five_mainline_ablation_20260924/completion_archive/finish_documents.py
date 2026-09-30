from pathlib import Path
import json,statistics,fcntl,os,time,shutil,hashlib
root=Path('/Users/shq/Research/GP/持续推进');p=root/'五主线消融_20260924'
for name in ['final_audit.json','reference_audit.json','transport_audit.json','artifact_transport_audit.json']:
 assert json.loads((p/name).read_text())['passed'],name
j=json.loads((p/'消融配对.json').read_text());snap=p/'documentation_snapshots';snap.mkdir(exist_ok=True)
stamp=time.strftime('%Y%m%d_%H%M%S')
def append(f,s):
 with f.open('a+b') as h:
  fcntl.flock(h,fcntl.LOCK_EX);h.seek(0);old=h.read()
  if s.encode() in old:return
  (snap/(hashlib.sha256(str(f).encode()).hexdigest()[:10]+'_'+stamp+'_'+f.name)).write_bytes(old)
  h.seek(0,2);h.write(s.encode());h.flush();os.fsync(h.fileno());h.seek(0);assert h.read(len(old))==old
texts={
'01':'概率传播的数据集宏均增益为 +0.745 pp（38/63 checkpoint格为正），但在 CiteSeer、Actor、Chameleon 的均值为负。打乱门控对应后下降很大，完整对应的宏均增益 +18.070 pp；这支持节点与门控对应的重要性，不证明机制原创。动态CE缩放相对普通CE为 −0.633 pp，8个数据集均值全部为负，不能继续作为通用增强项。KD为 −0.155 pp，收益依数据集变化。保留条件Prompt和传播作为有条件方向，动态缩放应降级为历史对照。',
'02':'两跳传播宏均 +4.509 pp；Cora +22.240、CiteSeer +17.240 pp，但 Actor −3.816、Cornell −10.270 pp。中心化宏均 −6.180 pp，Computers −23.498、Photo −20.700 pp；只在 Cora、CiteSeer、Actor 均值为正。软原型修正在 CiteSeer +4.760、Cora +2.760 pp，整体 −0.591 pp。额外迭代、自适应分配与伪标签对应均无普遍收益。本线保留 CiteSeer 条件下的修正教师，不能推广为所有数据集默认教师。它仍是直接读出方案，不是可学习Prompt。',
'03':'锐化门控宏均 +1.271 pp（42/63格为正），7个数据集均值为正、Cornell为零。Adam步长匹配相对无匹配矩阵更新 +11.718 pp，说明这套矩阵更新对更新尺度高度敏感；不能把这一差值全部解释为矩阵预条件的贡献。完整矩阵相对对角仅 −0.013 pp，25正/1平/37负；匹配Shampoo相对Adam +0.454 pp。输出中心化 −1.726 pp，仅Actor均值为正。保留锐化和匹配步长；完整矩阵必要性证据不足，中心化不宜默认打开。',
'04':'VAT相对无VAT宏均 +0.132 pp，25正/14平/24负，属于小且不稳定的增益。教师修正 +0.631 pp（46/63格为正），其中 PubMed +1.596 pp；KD −0.035 pp，接近零。VAT相对同epsilon随机噪声 +0.121 pp，但315个checkpoint×seed记录中只有207个首末步RMS匹配，108个不匹配，存在方向和幅度混杂，不能宣称纯方向优势。保留修正教师作为本线较清晰贡献，VAT仅作为有条件小增益。',
'05':'self-only相对原图宏均 −2.679 pp，自环100为 −1.401 pp，均非通用增强；self-only在Actor +1.382、Chameleon +1.228、Cornell +8.198 pp，保留结构依赖的适用范围。self-only上的中心化在Actor +2.943、Chameleon +3.121 pp，而Cornell −3.604 pp。中心化之后加对角缩放整体仅 +0.171 pp，Actor −0.007 pp；方差收缩仅 +0.002 pp。Actor收益主要支持固定结构与中心化，对角缩放及收缩的额外贡献很弱。本线属于固定结构/几何对照。'
}
common='\n\n口径：以下增量均为预定义完整分支减消融分支，单位为准确率百分点；总体采用8数据集等权宏均值。每个checkpoint内配对5个相同支持采样seed，再对checkpoint汇总。5个seed不是5张独立图，未宣称统计显著性或独立外部确认。主线03的tr1与gsh完整分支不同，不能混用消融分母。\n'
report='# 五主线消融：结论与完成状态\n\n71/71任务完成，0失败；4610条新增结果、5475条旧对照引用均通过审计。覆盖8个节点数据集、63份按SHA去重的兼容checkpoint，5-shot、5种子、100步。全部服务器归档文件已下载并逐文件SHA256通过，随后本地预测传输审计与跨轮引用审计均通过。\n'+common
for line,s in texts.items():report+=f'\n## 主线{line}\n\n{s}\n'
report+='\n## 预定义2×2交互\n\n交互为四个配对分支的差中之差，正值只表示本批的描述性交互，不是已证明的普遍协同。\n\n| 交互 | 数据集宏均值pp |\n|---|---:|\n'
for r in j['interactions']:
 ds={c['dataset'] for c in r['cells']};v=statistics.mean(statistics.mean(c['mean_interaction_pp'] for c in r['cells'] if c['dataset']==d) for d in ds)
 report+=f"| {r['name']} | {v:+.3f} |\n"
report+='\n## 解释与覆盖边界\n\nscale2是gamma=2的focal梯度系数均值对CE进行动态缩放，并非固定乘2。其缩放系数停止梯度；已保存的训练代码及预测没有因文档更正而变化。VAT的108个RMS不匹配记录完整保存在消融配对.json；即使首末步匹配也不意味着全轨迹匹配。\n\n主线02只训练8个数据集直接方法，不按63份checkpoint重复统计。3份MultiGprompt权重架构不兼容、MUTAG/PROTEINS/ENZYMES的16份图分类权重任务不匹配，均未计入成功；本轮不能推断这些项目的效果。\n\n本轮共28组预定义模块比较、4组2×2交互，保留全部正负结果；结论是适用范围分析，没有据此改动方案、seed或超参数。没有额外启动新一轮训练。\n\n## 审计与原始表\n\n- [全部模块与汇总表](消融结果汇总.md)\n- [逐seed配对与VAT RMS](消融配对.json)\n- [服务器预测审计](final_audit.json)\n- [本地预测传输审计](transport_audit.json)\n- [5475条旧引用审计](reference_audit.json)\n- [全部下载文件SHA核验](artifact_transport_audit.json)\n'
(p/'结论与完成状态.md').write_text(report)
for line,s in texts.items():
 d=next(root.glob(f'主线{line}_*'));t=(p/f'主线{line}_消融结果.md').read_text()
 t='\n'.join([t.splitlines()[0],'',s,common,'\n详细配对表如下。\n','\n'.join(t.splitlines()[1:])])
 if line=='04':t+='\nVAT幅度边界：等epsilon不等于等实际RMS；207/315首末步匹配，108不匹配，不能据此单独归因于方向。\n'
 if line=='01':t+='\nscale2是gamma=2的focal梯度系数均值动态缩放CE，绝非固定乘2。\n'
 (d/'消融结果_20260924.md').write_text(t)
 for f in [d/'方案说明.md',d/'README.md',d/'消融方案_20260924.md']:
  if f.exists():append(f,'\n\n## 2026-09-24 消融完成\n\n'+s+'\n\n已完成全部71任务、4610条新增结果及5475条引用审计，完整归档通过本地SHA核验。[本线完整消融结果](消融结果_20260924.md)。\n')
entry='\n\n## 2026-09-24 五主线消融完成并归档\n\n71/71任务完成、0失败；新增4610条，审计旧引用5475条。8节点数据集、63兼容checkpoint、5-shot、5种子、100步。服务器与本地预测/引用审计通过，全文件SHA通过。\n\n'+ '\n\n'.join('主线'+k+'：'+v for k,v in texts.items())+'\n\n全部正负模块比较与2×2交互已归档；5个seed不是独立图，VAT有108/315个实际RMS不匹配记录。见[消融结论](../五主线消融_20260924/结论与完成状态.md)。\n'
append(root/'AccuracyPrompt_20260923_51542'/'总进展.md',entry)
append(root/'五组优先主线索引.md','\n\n## 2026-09-24 全消融完成\n\n[结论与审计](五主线消融_20260924/结论与完成状态.md)：71任务、4610新增、5475引用，全部审计及本地SHA通过。五条主线文件夹各有消融结果_20260924.md。\n')
f=root/'AccuracyPrompt_20260923_51542'/'state.json';old=f.read_bytes();(snap/('state_'+stamp+'.json')).write_bytes(old);st=json.loads(old)
st['five_mainline_ablation'].update(status='complete_audited_synced',new_results_completed=4610,reused_references_audited=5475,completed_jobs=71,failed_jobs=0,report=str(p/'结论与完成状态.md'),artifact_sha_verified=True)
st.update(running=None,running_jobs=[],next_action='Five-mainline grid and ablations completed and documented; no additional training launched',local_sync_pending=None)
for k in ['pid_observed','remote_pid','local_sync_pid']:st[k]=None
with f.with_suffix('.tmp').open('w') as h:json.dump(st,h,ensure_ascii=False,indent=2);h.flush();os.fsync(h.fileno())
os.replace(f.with_suffix('.tmp'),f)
(p/'completion.json').write_text(json.dumps(dict(status='complete_audited_synced_documented',finished=time.time(),new_rows=4610,reference_rows=5475,jobs=71,failures=0),indent=2))
print('documentation_and_state_complete')
