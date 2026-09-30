from pathlib import Path
import json,csv
ROOT=Path('/Users/shq/Research/GP/持续推进'); OUT=ROOT/'PRPrompt_原型路由图提示_20260928'
U=ROOT/'三底座统一协议对比_20260924'; B=ROOT/'CGRPrompt_叙事补验_20260926'
rows=list(csv.DictReader((U/'统一协议多指标对比.csv').open()))
S=json.loads((B/'本地复核/summary.json').read_text()); P=json.loads((B/'本地复核/paired.json').read_text())
jobs=['Cora__Edgepred_Gprompt__87bd546f7897','CiteSeer__Edgepred_Gprompt__7bf67dc55e3a','Cora__Edgepred_GPPT__a279caf82ae2','CiteSeer__Edgepred_GPPT__e5488f6ef537']
names=['Cora / Gprompt','CiteSeer / Gprompt','Cora / GPPT','CiteSeer / GPPT']
f=lambda x:f'{float(x):.3f}'; sg=lambda x:f'{float(x):+.3f}'
def ur(job,m,metric='accuracy'):
 d,c,_=job.split('__');return next(x for x in rows if x['dataset']==d and x['checkpoint']==c and x['method']==m and x['metric']==metric)
def ss(job,m,stage='core',view='default'):
 return next(x for x in S if x['job']==job and x['method']==m and x['stage']==stage and x['view']==view)
def pp(job,m,stage='core',ref='F111'):
 return next(x for x in P if x['job']==job and x['comparator']==m and x['reference']==ref and x['stage']==stage and x['metric']=='accuracy')
def ci(x):return f"{sg(x['mean'])} [{sg(x['ci_low'])}, {sg(x['ci_high'])}]"
def table(head,data):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in data])
a=['# PRPrompt基础M3版本：证据索引与数值表\n','日期：2026-09-28。本文与[基础版本叙事](PRPrompt_基础M3版本_论文叙事_20260928.md)配套。主文使用四个边预测设置，已有其他设置作为范围记录，不新增实验。\n','## 1. 版本与数据口径\n','- 本版本M3在最终适配表示上构图，由同一表示形成支持原型概率，执行概率回注与支持标签钳制。\n- 统一协议采用14方法、六份checkpoint、20共同支持抽样的正式多指标CSV；早期summary.json只记录12方法/1440条，不能作为最终覆盖数。\n- 补验仅选用core、allocation、tuning及基础版本的全因子分析。每张表保持自己的实验批次，不与其他批次拼接。\n- core中的F111才是完整基础版本；`M3_original`在该运行器中表示**改用原图传播的对照**，不能按名称误当作基础版本本身。已核对run_core.py第74行。\n- 教师对照来自core中的rect_uncentered、A2_center，是直接属性原型分数；这些分支没有执行后续近邻概率传播。\n- 20次重复来自同一张图的支持抽样；所有区间为相应批次配对bootstrap，未作多重比较校正。Accuracy、F1、AUC、AP按百分数报告，差值单位pp。\n','## 2. 统一协议主结果\n']
ext=[m for m in sorted(set(x['method'] for x in rows)) if not m.startswith('F')]
data=[]
for j,n in zip(jobs,names):
 full=ur(j,'F111');base=ur(j,'F000');best=max([ur(j,m) for m in ext],key=lambda x:float(x['mean']))
 data.append([n,f(full['mean'])+' ± '+f(full['sd']),f(base['mean']),sg(base['full_minus_method']),best['method']+': '+f(best['mean']),sg(float(full['mean'])-float(best['mean']))])
a +=[table(['设置','完整Accuracy','F000','完整−F000','本轮最高外部对照','完整−该对照'],data),'\n外部最高按本轮均值描述性排序；不能解释为所有方法各自充分调参后的最好结果。DAPrompt-fixed有明确移植限制。\n','### 2.1 完整五指标\n',table(['设置','Accuracy','Macro-F1','Micro-F1','ROC-AUC','宏AP'],[[n]+[f(ur(j,'F111',m)['mean'])+' ± '+f(ur(j,'F111',m)['sd']) for m in ['accuracy','macro_f1','micro_f1','roc_auc_macro_ovr','auprc_macro']] for j,n in zip(jobs,names)]),'\n### 2.2 条件移除\n']
def uci(x):return f"{sg(x['full_minus_method'])} [{sg(x['ci_low'])}, {sg(x['ci_high'])}]"
a +=[table(['设置','相对F000','M1锐化开关 F011','M2更新开关 F101','M3传播开关 F110'],[[n]+[uci(ur(j,m)) for m in ['F000','F011','F101','F110']] for j,n in zip(jobs,names)]),'\nF000仍有类别条件残差；F011保留类别条件构造，只关闭额外锐化。因此这些开关不等于删除整个M1结构。\n','## 3. 固定类别分配对照：补验allocation\n',table(['设置','固定条件Adam','属性学习路由','教师初始化学习路由'],[[n]+[f(ss(j,m,'allocation','propagated_metrics')['accuracy']['mean']) for m in ['fixed__adam','learned_hidden__adam','learned_teacher__adam']] for j,n in zip(jobs,names)]),'\n共用隐藏注入、损失与读出，基础比较采用Adam。学习路由有额外参数；等参数低秩对照还改变残差维度。固定优于这些路由不自动等于证明过拟合原因，训练损失与优化难度须分别解释。\n','## 4. 形状、幅度与矩阵更新：补验core\n',table(['设置','完整−等范数锐化','完整−仅幅度替代','完整−匹配对角','完整−无范数匹配'],[[n]+[ci(pp(j,m)) for m in ['M1_normmatched','M1_amplitude_only','M2_diagonal','M2_ungrafted']] for j,n in zip(jobs,names)]),'\n## 5. 支持内等网格选参：补验tuning\n',table(['设置','矩阵更新','Adam','无匹配','矩阵−Adam及区间'],[[n]+[f(ss(j,m,'tuning')['accuracy']['mean']) for m in ['fixed_gshampoo','fixed_adam','fixed_ungrafted']]+[ci(pp(j,'fixed_adam','tuning','fixed_gshampoo'))] for j,n in zip(jobs,names)]),'\n每类5支持拆为4训练/1验证，以支持内验证CE从3学习率×3更新预算中选择，再用全部支持重训。它是独立选择协议，不能与固定100更新表混为同一实验。\n','## 6. 基础M3内部对照：补验core\n',table(['设置','完整−原图传播','完整−冻结表示近邻','完整−不钳制','完整−不回注'],[[n]+[ci(pp(j,m)) for m in ['M3_original','M3_frozen_knn','M3_no_clamp','M3_no_restart']] for j,n in zip(jobs,names)]),'\nM3整体相对无传播有正向收益，不意味着所有内部选择单独最优。原图传播有正负差异；CiteSeer两组取消回注更高，支持钳制在四个主设置中也没有一致的单独正收益。将回注与钳制写为明确的预测约束，不写成消融已经证明不可替代。\n','## 7. 属性原型直接预测：补验core\n',table(['设置','完整Prompt','未中心化教师','中心化教师','完整−未中心化及区间','完整−中心化及区间'],[[n]+[f(ss(j,m)['accuracy']['mean']) for m in ['F111','rect_uncentered','A2_center']]+[ci(pp(j,m)) for m in ['rect_uncentered','A2_center']] for j,n in zip(jobs,names)]),'\nCora/GPPT的core完整结果为71.620，统一主表为71.615。保留本地重放的小数差异；配对差和区间只使用本批结果。直接教师是内部属性原型路径，不是外部已发表模型的正式名称；其参数无需预训练checkpoint，同图同抽样跨底座重复不构成新增独立证据。\n','## 8. 已有范围记录\n','三底座统一协议还包含两组DGI：完整Accuracy为51.230与48.620，低于本轮GPrompt的55.490与58.485。本稿不以此否定边预测范围内的结果，也不将已有其他范围结果删除。后续不新增DGI。扩展图与其他预训练的基础F111记录位于补验core，未合并进本文四组主表。\n','## 9. 原始来源\n','- [统一协议最终五指标CSV](../三底座统一协议对比_20260924/统一协议多指标对比.csv)\n- [统一协议结论](../三底座统一协议对比_20260924/结论与完成状态.md)\n- [外部方法移植协议](../三底座统一协议对比_20260924/机制与协议说明.md)\n- [2026方法移植边界](../三底座统一协议对比_20260924/extension2026/模型选择与复现边界.md)\n- [补验独立复算均值](../CGRPrompt_叙事补验_20260926/本地复核/summary.json)\n- [补验独立配对区间](../CGRPrompt_叙事补验_20260926/本地复核/paired.json)\n- [基础运行器与读出对照定义](../CGRPrompt_叙事补验_20260926/服务器回传/run_core.py)\n- [全因子交互](../CGRPrompt_叙事补验_20260926/本地复核/interactions.json)\n- [基础M3说明](三个模块Framework详解_20260926/M3_说明.md)\n']
(OUT/'PRPrompt_基础M3版本_证据索引_20260928.md').write_text('\n'.join(a))
print('Evidence generated from',len(rows),'unified rows;',len(S),'summary rows;',len(P),'paired rows')
