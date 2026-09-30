from pathlib import Path
import json
R=Path(__file__).resolve().parent
p=json.loads((R/'all_metric_pairs.json').read_text());plan=json.loads((R/'plan.json').read_text());baselines=['GPPT','GPF','GPF+','GPrompt','EdgePrompt','GraphTOP','UniPrompt','MTG','DAPrompt-fixed']
def row(ds,ck,m,k='accuracy'):return next(x for x in p if x['dataset']==ds and x['checkpoint']==ck and x['method']==m and x['metric']==k)
def fmt(x):return f"{x['mean']:.3f} ± {x['sd']:.3f}"
lines=['# 统一协议多指标实验结论','', '1680条正式结果，覆盖Cora、CiteSeer的DGI、Edgepred_Gprompt、Edgepred_GPPT六份checkpoint，14方法、20共同支持seed、5-shot、100更新；1560次拟合。前轮结果未混入本轮表格。','', '## 完整方案与本轮对照','', '以下最优对照只是在这组固定配置中按平均Accuracy描述性排序，不代表调参后最优，也不是新选定的训练配置。F1为Macro-F1，AUC为macro OVR，AP为macro average precision；单位%。','', '| 数据集 / 底座 | 完整Accuracy | 完整F1 | 完整AUC | 完整AP | 对照最高Accuracy | 完整−该对照(pp) |','|---|---|---|---|---|---|---|']
wins=0
for j in plan['jobs']:
 ds,ck=j['dataset'],j['checkpoint']['family'];best=max(baselines,key=lambda m:row(ds,ck,m)['mean']);b=row(ds,ck,best);wins+=b['full_minus_method']>0
 lines.append('| '+ds+' / '+ck+' | '+' | '.join(fmt(row(ds,ck,'F111',k)) for k in ['accuracy','macro_f1','roc_auc_macro_ovr','auprc_macro'])+f" | {best}: {fmt(b)} | {b['full_minus_method']:+.2f} |")
lines+=['',f'完整方案在六格中的{wins}格高于全部九项外部对照的平均Accuracy。该计数仅为当前预算与适配协议的描述，不作为显著性或普适结论。','','## 三模块移除实验','', 'F011去M1（锐化）；F101去M2（Adam范数匹配Shampoo）；F110去M3（概率传播）。下表为完整方案减对应对照的Accuracy配对差值及95% bootstrap区间。完整多指标差值见主结果表。','','| 数据集 / 底座 | 相对000 | M1移除差 | M2移除差 | M3移除差 |','|---|---|---|---|---|']
for j in plan['jobs']:
 ds,ck=j['dataset'],j['checkpoint']['family'];cells=[]
 for m in ['F000','F011','F101','F110']:
  x=row(ds,ck,m);cells.append(f"{x['full_minus_method']:+.2f} [{x['ci_low']:+.2f}, {x['ci_high']:+.2f}]")
 lines.append('| '+ds+' / '+ck+' | '+' | '.join(cells)+' |')
lines+=['','多指标观察：相对000，完整方案在六格的Macro-F1和AP均提高，配对区间下界均为正；AUC均值也都提高，但Cora/Edgepred_GPPT的区间跨0。加入M1在Cora/Edgepred_Gprompt使AUC下降0.575 pp、AP下降1.277 pp，在Cora/Edgepred_GPPT使AUC下降0.221 pp，三项区间均为负。因此不能声称三个组件在每个指标上都有效。','','本轮三个组件的Accuracy移除差区间均正仅出现在DGI的两格；CiteSeer/Edgepred_Gprompt的M2为+0.460 pp，区间[-0.025,+0.950]，这轮没有明确重现其单独增益。DGI内部组件证据较强，但完整方案仍低于GPrompt；内部有效不等于外部领先。','','## 新增2026模型','','| 数据集 / 底座 | 方法 | Accuracy | Macro-F1 | AUC-ROC | AP |','|---|---|---|---|---|---|']
for j in plan['jobs']:
 ds,ck=j['dataset'],j['checkpoint']['family']
 for m in ['MTG','DAPrompt-fixed']:lines.append('| '+ds+' / '+ck+' | '+m+' | '+' | '.join(fmt(row(ds,ck,m,k)) for k in ['accuracy','macro_f1','roc_auc_macro_ovr','auprc_macro'])+' |')
lines+=['','MTG为ICML2026消息调优方法，已对照作者算子与提示梯度；保留作者把Softmax概率送入CE的训练行为。DAPrompt为AAAI2026，本轮明确标为DAPrompt-fixed：只移植提示阶段，固定既定checkpoint，未复现专用结构预训练和主干更新；默认阈值下离散几何恒定。不能以该表断言原论文DAPrompt无效。出处、commit、工程修复与未纳入模型见[2026复现边界](extension2026/模型选择与复现边界.md)。','','## 解读边界','','本轮未重跑上一轮强直接教师，Prompt对照排名不能外推为超过全部强基线。多项外部方法在本轮固定配置下准确率较低；本轮未做基于支持集的等额超参数搜索或收敛性扫描，不能据此宣称超过原论文最佳水平。模块移除有增益和战胜其他方法是两个不同问题。AUC/AP评价分数排序，F1评价固定argmax分类，可能出现指标方向不一致；不以某个更有利的指标替代原先Accuracy判断。20个seed是同一图上的支持集重复采样；配对bootstrap区间未校正多重比较。三底座经前轮筛选，统一100更新不等于各方法收敛或等FLOPs。所有结果包括负格均保留。','','[14方法完整多指标表及全部配对区间](统一协议多指标对比.md) · [CSV](统一协议多指标对比.csv) · [机制与协议](机制与协议说明.md)。','', '传输完成与本地审计以state.json、archive_transport_verified.json、transport_audit.json、extension2026/transport_audit.json、metrics_transport_verified.json为准。']
(R/'结论与完成状态.md').write_text('\n'.join(lines)+'\n');print('wrote conclusion')
