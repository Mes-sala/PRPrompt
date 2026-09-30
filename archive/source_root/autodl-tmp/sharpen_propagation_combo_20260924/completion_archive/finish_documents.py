from pathlib import Path
import json,time,fcntl,os,hashlib,shutil
root=Path('/Users/shq/Research/GP/持续推进');p=root/'锐化与传播组合验证_20260924'
for n in ['final_audit.json','reference_audit.json','transport_audit.json','artifact_transport_audit.json']:assert json.loads((p/n).read_text())['passed']
o=json.loads((p/'组合配对.json').read_text());s=o['summary'];snap=p/'documentation_snapshots';snap.mkdir(exist_ok=True)
text='''# 锐化门控与概率传播组合：结论与完成状态

63/63任务完成，零正式任务失败；315条新增读出及1970条历史引用均通过服务器与本地审计。908个服务器归档文件、154926029字节逐文件SHA256验证通过。新增训练次数0，使用原100步训练封存的Prompt；保留v1 smoke失败和v2修复资料，分数重放阈值未放宽。

## 判断

可以组合，并有条件性增益，但没有普遍协同或强基线竞争优势。保留为精简候选与诊断基线，不升级为已确认的新主方法，不继续盲目叠加组件。

以下为8数据集等权平均，单位pp：

| 对照 | 组合增量 | 正/平/负checkpoint格 |
|---|---:|---|
'''
labels={'combo_minus_neither':'普通温度且无传播','combo_minus_sharp_only':'仅锐化','combo_minus_prop_only':'仅传播','interaction':'2×2交互','combo_minus_gsh':'原匹配Shampoo完整分支','combo_minus_A1_scale':'原A1动态CE缩放分支','combo_minus_A1_CE':'原A1普通CE分支','combo_minus_direct_uncentered':'未中心化直接教师','combo_minus_direct_centered':'中心化直接教师'}
for m,label in labels.items():
 r=s[m];text+=f"| {label} | {r['dataset_macro']:+.3f} | {r['positive']}/{r['tie']}/{r['negative']} |\n"
text+='''
32/63格同时超过两个单组件；只有3/63格同时超过两个单组件及两个直接教师。组合的改善主要集中在Cora、CiteSeer、Computers、Photo；对仅锐化加入传播，PubMed和Cornell的均值为负。不能据此只挑胜出的checkpoint声称普遍有效。

交互宏均−0.151 pp，26正/2平/35负。两组件一起用可以比单独用更好，但本批平均收益低于两个单组件效应的简单相加，不支持普遍正协同。该负交互也不意味着每个数据集都负交互。

相对A1动态缩放分支，数据集等权−1.313 pp，而checkpoint等权+1.006 pp；相对A1普通CE分别−1.946和+0.219 pp。CiteSeer等数据集checkpoint较多，因此不能只展示checkpoint等权的正均值。

## 解释边界

组合就是已有A3_tr1_qeps；本轮补齐普通温度无传播这一格，使交互可识别，并非发现一个新训练方法。输出中心化固定保留；KD关闭，但教师仍用于门控。锐化采用raw gate，也改变门控幅度，未隔离纯熵效应；传播属于末端读出，不是主干内第二个Prompt。这里的“无组件”仅指无锐化和无概率传播，仍有条件Prompt等共同部分。

5个支持采样seed不代表5张独立图，没有独立确认或显著性结论。8个节点数据集、63份兼容checkpoint，不含架构不支持的MultiGprompt及图分类任务。固定原seed、5-shot、100步历史训练，无查询驱动调参。未联合验证删除中心化、更换优化器或添加VAT的其他组合。

## 后续决策

本轮不支持继续以“组合协同”作为主要论点。若用户决定开展下一轮，应先提出独立假设并预先固定方案，同时保留两个直接教师；不能把对弱基线的大增量当成整体竞争力。当前不自动追加新训练。

[完整表及逐checkpoint四格准确率](组合结果汇总.md) · [逐seed配对](组合配对.json) · [文件核验](artifact_transport_audit.json) · [预测核验](transport_audit.json) · [引用核验](reference_audit.json)
'''
(p/'结论与完成状态.md').write_text(text)
def append(f,t):
 with f.open('a+b') as h:
  fcntl.flock(h,fcntl.LOCK_EX);h.seek(0);old=h.read()
  if t.encode() in old:return
  (snap/(hashlib.sha256(str(f).encode()).hexdigest()[:12]+'_'+f.name)).write_bytes(old);h.seek(0,2);h.write(t.encode());h.flush();os.fsync(h.fileno());h.seek(0);assert h.read(len(old))==old
brief='\n\n## 2026-09-24 锐化×概率传播组合验证完成\n\n63任务、315新增读出、1970历史引用全部审计及本地SHA通过，新增训练0。组合相对仅锐化+0.562 pp、仅传播+1.271 pp，但交互−0.151 pp；相对未中心化直接教师−9.695 pp。32/63格胜两个单组件，仅3/63格还同时胜两个直接教师。不支持普遍协同或升级为已确认新方法。数据集等权，5支持seed不是独立图。\n'
append(root/'AccuracyPrompt_20260923_51542'/'总进展.md',brief+'\n[完整组合结论](../锐化与传播组合验证_20260924/结论与完成状态.md)。\n')
for line in ['01','03']:
 d=next(root.glob('主线'+line+'_*'));shutil.copy2(p/'结论与完成状态.md',d/'组合验证结论_20260924.md')
 # Relative links in the copied document target the central experiment directory.
 f=d/'组合验证结论_20260924.md';z=f.read_text()
 for n in ['组合结果汇总.md','组合配对.json','artifact_transport_audit.json','transport_audit.json','reference_audit.json']:z=z.replace(']('+n+')','](../锐化与传播组合验证_20260924/'+n+')')
 f.write_text(z)
 for n in ['方案说明.md','README.md']:append(d/n,brief+'\n[组合验证结论](组合验证结论_20260924.md)。\n')
f=root/'AccuracyPrompt_20260923_51542'/'state.json';old=f.read_bytes();(snap/'state_before_completion.json').write_bytes(old);st=json.loads(old);st['combo_2x2'].update(status='complete_audited_synced_documented',completed_jobs=63,new_readout_rows=315,reference_rows_audited=1970,failures=0,report=str(p/'结论与完成状态.md'));st.update(running=None,running_jobs=[],next_action='Combination validation complete; no further training launched',local_sync_pending=None);t=f.with_suffix('.tmp');t.write_text(json.dumps(st,ensure_ascii=False,indent=2));t.replace(f)
(p/'completion.json').write_text(json.dumps(dict(status='complete_audited_synced_documented',jobs=63,new_readout_rows=315,reference_rows=1970,new_training_fits=0,time=time.time()),indent=2))
print('documents complete')
