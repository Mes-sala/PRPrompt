from pathlib import Path
import csv,json,statistics as st,hashlib
ROOT=Path('/Users/shq/Research/GP');OUT=Path(__file__).resolve().parent;R=OUT.parent
FAMILIES=['DGI','Edgepred_Gprompt','Edgepred_GPPT']
STAGES=['stage2_confirmation','stage2b_gppt_confirmation','stage2c_dgi_confirmation']
cells={};sources=[]
def source(p):
 sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for s in STAGES:
 p=R/s/'confirmation_pairs.json';source(p)
 for c in json.loads(p.read_text())['cells']:cells[c['dataset'],c['checkpoint']]=c
selected=[];agg={};skipped={};identity=[]
for dataset in ['Cora','CiteSeer']:
 p=ROOT/f'rep/ProG/results/{dataset.lower()}_5shot_full_matrix_seedgrid.csv';source(p);rows=list(csv.DictReader(p.open()))
 for family in FAMILIES:
  rr=[r for r in rows if r['pretrain']==family];selected+=rr
  for method in sorted({r['prompt'] for r in rr}):
   group=[r for r in rr if r['prompt']==method];key=(dataset,family,method)
   assert len(group)==9 and len({(r['split_seed'],r['train_seed']) for r in group})==9
   ok=[r for r in group if r['status']=='complete']
   if ok:
    assert len(ok)==9
    acc=[100*float(r['accuracy_mean']) for r in ok];f1=[100*float(r['f1_mean']) for r in ok]
    agg[key]={'accuracy_mean':st.mean(acc),'accuracy_sd_seedpairs':st.stdev(acc),'macro_f1_mean':st.mean(f1),'macro_f1_sd_seedpairs':st.stdev(f1),'n_seed_pairs':9,'input_mode':sorted({r['input_mode'] for r in ok}),'variant':sorted({r['variant'] for r in ok}),'artifact_paths_local_available':sum(Path(r['artifact_directory']).exists() for r in ok)}
   else:skipped[key]='skipped'
  ck=ROOT/f'rep/ProG/Experiment/pre_trained_model/{dataset}/{family}.GCN.128hidden_dim.pth';source(ck)
  h=hashlib.sha256(ck.read_bytes()).hexdigest();ch=cells[dataset,family]['sha256'];identity.append({'dataset':dataset,'family':family,'current_local_sha256':h,'confirmation_sha256':ch,'current_file_matches_confirmation':h==ch,'seedgrid_run_checkpoint_hash_verified':False})
# Preserve selected source records verbatim in JSON, including skipped rows and run IDs.
(OUT/'extracted_seedgrid_records.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
(OUT/'aggregated_comparison.json').write_text(json.dumps({'groups':[dict(dataset=k[0],family=k[1],method=k[2],**v) for k,v in agg.items()],'checkpoint_identity':identity,'sources':sources},ensure_ascii=False,indent=2))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def delta(c,key):
 x=c['contrasts'][key];return f"{x['mean_pp']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
def score(v):return f"{v['mean']:.2f} ± {v['std']:.2f}"
prompt=['GPF','GPF-plus','Gprompt','All-in-one','GPPT','SUPT-soft','SUPT-hard','GraphTOP']
base=['None','Frozen','FrozenNodePrototype','FrozenIdentityPrototype','FrozenGraph','FrozenGraphPrototype','GraphTOP-Frozen']
parts=['# 三模块方案与本地ProG seedgrid对照（2026-09-24）',
'''## 方案说明

方案用同一冻结两层GCN（隐藏维128）和一组可训练类别token完成5-shot节点分类。第一层激活前注入教师类别门控加权的token，经第二层GCN得到表示，最后分类。只有token训练；教师用支持标签与未标注节点特征构造，KD损失关闭，查询标签不进入训练。

1. **M1门控锐化**：教师类别分配温度从0.2降到0.1，使节点组合类别token时更集中。raw锐化也改变门控幅度，不能把全部增益归因为语义改善。
2. **M2尺度匹配优化**：用Shampoo矩阵预条件构造token更新方向，用同一梯度序列上的Adam更新范数进行匹配，替代普通Adam。优化器不是新增Prompt分支。
3. **M3支持约束概率传播**：在适配后表示上建立50近邻图，概率传播20轮，系数0.9，支持标签每轮固定。它属于末端读出，不增加训练。

共同基础包含普通温度教师门控、token Prompt、Adam和输出中心化。完整方案F111与F000比较衡量三个开关的整体增量；移除M1/M2/M3分别比较F111−F011/F101/F110。模块共享一个主干，但不是三个内部Prompt，也不是三个模型集成。''',
'## 哪些底座有明确正提升',
'这里“明确”只表示预定义配对差值的95% bootstrap区间下界大于0。每格20个支持采样seed，共用图与查询集，未校正多重比较，且候选底座是分阶段筛选；不能当作独立图显著性或普适结论。']
rows=[]
for (ds,fam),c in cells.items():
 rows.append([ds,fam,score(c['accuracy']['F111']),delta(c,'full_vs_base'),'是' if c['three_positive_ci_lower'] else '否', '、'.join('M'+str(i) for i in [1,2,3] if c['contrasts']['remove_M'+str(i)]['ci95'][0]<=0) or '无'])
parts.append(table(['数据集','预训练家族','完整准确率%','完整−共同基础pp [95%CI]','三项移除均明确正向','尚不明确的模块'],rows))
parts.append('**完整方案整体提升**：DGI、Edgepred_Gprompt、Edgepred_GPPT三个家族各自在Cora与CiteSeer上，共6个已确认格。**三个模块分别有清晰贡献**：DGI/Cora、DGI/CiteSeer、Edgepred_Gprompt/CiteSeer，共3个格、2个家族。Actor/GraphCL不成立。未将全checkpoint旧5-seed筛选格升级成20-seed确认结论。')
parts.append(table(['三模块成立范围','M1增量pp [95%CI]','M2增量pp [95%CI]','M3增量pp [95%CI]'],[[d+'/'+f,*[delta(c,'remove_M'+str(i)) for i in [1,2,3]]] for (d,f),c in cells.items() if c['three_positive_ci_lower']]))
parts += ['## 与历史seedgrid的可比范围', '''以下历史值来自本地原始CSV，按同名预训练家族检索，并非论文原文成绩。历史CSV没有逐次checkpoint哈希，所指向的远端预测目录不在本地；本次未对历史预测重新做审计。现存本地checkpoint只能用于辅助检查，不能证明它就是当时seedgrid使用的文件。

- Cora：三个现存本地checkpoint均与本次确认哈希相同，但历史运行身份仍未闭合。
- CiteSeer：三个现存本地checkpoint均与本次确认不同，因此尤其不能称严格同checkpoint比较。
- 历史为3个split seed（1729/17290/172900）×3个train seed（42/4242/424242），每行5 folds。本表先取每行fold均值，再对9行求均值及样本标准差；**不是45次独立重复的置信区间，也不是CSV内fold标准差的平均**。
- 本方案为各阶段20个新支持seed、100步、固定官方训练池/测试查询。历史v4同为5-shot、官方池、GCN两层128维，但默认200 support passes，且不同Prompt有不同子图、读出和优化设置（All-in-one另有交替预算）。两者无逐seed配对，不作跨协议显著性、相对增益或冠军声明。
- 支持池来源和NormalizeFeatures由现存运行代码核对；历史实际源代码/预测级一致性仍未复核。''']
for family in FAMILIES:
 parts.append('## '+family+'：历史Prompt与当前方案')
 rows=[]
 for m in prompt+base:
  v=[]
  for ds in ['Cora','CiteSeer']:
   a=agg[ds,family,m];v.append(f"{a['accuracy_mean']:.2f} ± {a['accuracy_sd_seedpairs']:.2f}")
  rows.append([m if m!='None' else 'None（全量微调）',*v,'历史seedgrid'])
 rows.append(['三模块F111',*[score(cells[ds,family]['accuracy']['F111']) for ds in ['Cora','CiteSeer']],'本次20支持seed；不同协议'])
 parts.append(table(['方法','Cora Accuracy %','CiteSeer Accuracy %','来源'],rows))
 parts.append('历史各方法n=9个seed pair，完整方案n=20个支持seed；两种标准差的统计单位不同。')
parts += ['## 同次实验的严格配对基线', '本表来自三模块确认的相同checkpoint、支持集、查询集与预算。graphprompt为本次轻量实现的GraphPrompt控制，不等于历史ProG Gprompt的子图训练实现。']
rows=[]
for (ds,f),c in cells.items():
 if f not in FAMILIES:continue
 rows.append([ds+'/'+f,*[score(c['accuracy'][k]) for k in ['F000','frozen','graphprompt','F111']]])
parts.append(table(['数据集/家族','共同基础F000','本次frozen','本次graphprompt','完整F111'],rows))
parts += ['## 对结果的判断', '''1. **三模块贡献最齐全的是DGI，但绝对性能偏低。** Cora/DGI完整50.605%，历史Gprompt49.482%（仅描述性并列），历史FrozenGraphPrototype58.291%；CiteSeer/DGI完整47.340%，历史Gprompt55.684%、FrozenGraphPrototype58.722%。DGI的内部消融提升不能写成对现有方法全面胜出。
2. **论文性能展示更值得关注CiteSeer/Edgepred_Gprompt。** 完整68.305%，三个模块移除增量均明确为正。不过本次中心化直接教师70.405%仍高2.100 pp，且历史CiteSeer checkpoint不一致，不能把与历史Prompt的大幅分差全归因于新模块。
3. **Cora的Edgepred_Gprompt与Edgepred_GPPT整体提升明确，但三模块叙事不完整。** 前者M1不明确，后者M2不明确；GPPT/CiteSeer的M2也跨零。不能把整体提升替代逐模块证据。
4. 若要制作论文主实验排名，需要在同一checkpoint、相同支持/查询、同一输入和预算下重跑ProG方法。本次完成历史数据提取和可比性检查，没有新训练。''',
'## 未支持项与覆盖',
'每个历史家族/数据集有15个complete方法、9个skipped方法，各9行。共提取1296行，其中810 complete、486 skipped。Skipped为Prodigy、GraphPrompter、EdgePrompt、EdgePromptplus、UniPrompt、SelfPro、ProNoG、PSP、DHBP；保留缺口，不填0。MultiGprompt是自带预训练的独立组合，不属于这三种指定底座。额外RSP seedgrid只有其自身已跑底座，未混入当前表。',
'## checkpoint身份辅助检查',
 table(['数据集/家族','当前实验SHA256','现存本地ProG SHA256','现存文件一致'],[[x['dataset']+'/'+x['family'],x['confirmation_sha256'],x['current_local_sha256'],'是' if x['current_file_matches_confirmation'] else '否'] for x in identity]),
'## 来源与复核',
'统计来源：\n\n'+'\n'.join(f"- [{Path(x['path']).name}]({x['path']})，SHA256 `{x['sha256']}`" for x in sources if not x['path'].endswith('.pth')),
'分析脚本、提取记录（保留run_id/status）与完整Accuracy/Macro-F1聚合值位于本目录。历史输入CSV未修改。']
(OUT/'方案与ProG_seedgrid对比.md').write_text('\n\n'.join(parts)+'\n')
assert len(selected)==1296 and len(agg)==90
assert all(v['artifact_paths_local_available']==0 for v in agg.values())
print('wrote report; complete_groups',len(agg),'selected rows',len(selected))
for f in FAMILIES:
 print(f, [(d,agg[d,f,'Gprompt']['accuracy_mean'],cells[d,f]['accuracy']['F111']['mean']) for d in ['Cora','CiteSeer']])
