from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
stages=['stage1_factorial','stage2_confirmation','stage2b_gppt_confirmation','stage2c_dgi_confirmation']
for stage in stages:
 p=R/stage
 for n in ['final_audit.json','artifact_transport_audit.json','transport_audit.json']:assert json.loads((p/n).read_text())['passed'],(stage,n)
 if stage=='stage1_factorial':assert json.loads((p/'reference_audit.json').read_text())['passed']
 else:assert json.loads((p/'pair_audit.json').read_text())['passed']
cells=json.loads((R/'chapter_assets_final'/'consolidated_results.json').read_text());strong=[c for c in cells if c['three_positive_ci_lower']]
assert len({c['dataset'] for c in strong})>=2,'Multiple clear conditional datasets still missing'
byfamily={}
for c in strong:byfamily.setdefault(c['checkpoint'],set()).add(c['dataset'])
assert any(len(v)>=2 for v in byfamily.values()),'No common backbone family with multi-dataset evidence'
summary='''# 三模块图Prompt方案说明与实验结论

## 结论和推荐定位

已形成以主线03为基础的三模块方案：**语义门控锐化、更新尺度匹配、支持约束概率传播**。三者分别作用于同一组Prompt的生成、优化和读出，共享一个冻结GCN。它可以作为“指定预训练底座上的少样本适配”研究点组织第三或第四章；不应写成通用最优方法，也不把三个已有原子技术声称为首创。

本研究按用户更新后的要求评价条件性收益，不要求所有数据集与底座有效。明确三项贡献的范围见下表，同时保留弱效应与失败格。表中增量都是完整方案减去只移除一个模块，单位pp；区间为20个支持采样seed的配对bootstrap95%区间，未作多重比较校正。

| 数据集/底座 | M1锐化 | M2尺度匹配优化 | M3传播 | 三项区间下界均正 |
|---|---|---|---|---|
'''
for c in cells:
 vals=[]
 for i in [1,2,3]:
  v=c['contrasts']['remove_M'+str(i)];vals.append(f"{v['mean_pp']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
 summary+='| '+c['dataset']+'/'+c['checkpoint']+' | '+' | '.join(vals)+' | '+('是' if c['three_positive_ci_lower'] else '否')+' |\n'
summary+='\n![三个模块的条件性贡献](chapter_assets_final/module_effects.png)\n\n蓝色表示配对区间下界大于0；灰色为正均值但区间跨0；红色为负均值。不是经过多重校正的独立图显著性检验。\n\n## 完整方法的收益与强基线差距\n\n| 数据集/底座 | 000共同基础% | 111完整% | 完整−基础pp | 完整−未中心化教师pp | 完整−中心化教师pp |\n|---|---:|---:|---:|---:|---:|\n'
for c in cells:
 a=c['accuracy'];v=c['contrasts'];summary+=f"| {c['dataset']}/{c['checkpoint']} | {a['F000']['mean']:.3f} | {a['F111']['mean']:.3f} | {v['full_vs_base']['mean_pp']:+.3f} | {v['full_minus_rect_uncentered']['mean_pp']:+.3f} | {v['full_minus_A2_center']['mean_pp']:+.3f} |\n"
summary+='''
000仍含条件Prompt、普通温度教师门控、Adam与共同中心化，不是没有任何适配的冻结模型。完整方案相对冻结/GraphPrompt的大增量不全部归因于三个新增开关。

CiteSeer/Edgepred_Gprompt提供清晰的三模块证据，但完整方法仍落后于中心化直接教师2.100 pp。Cora同底座的完整方法有较明确整体收益，M1却只有+0.025 pp、区间跨零；不能称其在Cora稳定有效。Actor/GraphCL未确认三模块成立。Edgepred_GPPT的优化项也未在两个数据集稳定成立。DGI的模块作用必须与它较弱的绝对性能同时解释，不能借较低起点夸大竞争力。

明确成立的同家族范围是本地ProG-adapted DGI的Cora与CiteSeer；另有CiteSeer/Edgepred_Gprompt证据。DGI完整准确率分别为50.605%和47.340%，相对未中心化直接教师仍低21.920和20.915 pp。因此，这是弱底座上的条件性适配收益，不是推荐用DGI替代强直接教师。

权重来源标记为ProG-adapted、source_parity=not_fully_verified；严格加载和哈希一致不等于已复现原论文。结论限定到报告中的实际SHA，不能代表所有DGI或GraphCL权重。

## 三个模块怎样构成一个研究点

1. **语义门控锐化**：用支持标签产生的修正原型教师及全图无标签特征构造节点条件门控，控制同一组类别token对每个节点的修正。温度0.2→0.1是本轮开关。不是训练另一个标签预测器后投票；教师仅提供门控，不使用query标签。
2. **更新尺度匹配优化**：对token梯度构造左右矩阵预条件方向，再匹配同轨迹Adam方向的范数。它是训练模块而非新增推理分支。必须与Adam比较；无匹配矩阵更新很差不等于完整矩阵本身必需，完整矩阵与对角对照的结果一并保留。
3. **支持约束概率传播**：在最终适配表示上构造50近邻，传播原型概率20步，并固定支持节点标签。它是末端读出；不能写成三种主干内Prompt相互作用。

M1改变输入到Prompt的条件，M2更新同一组参数，M3利用其输出。零初始化恢复冻结前向；主干始终冻结。当前不加入动态CE缩放和VAT，KD关闭但教师门控仍存在。输出中心化在因子实验中固定，不把共同设置充作第四个贡献。

## 试验过程与不可隐藏的选择

- 阶段1：8数据集、63份兼容checkpoint，补齐三个因子的8格；新增630读出、315次训练，锁定3860条历史参照。它是使用历史图与query的探索筛查。
- 阶段2：固定Cora/CiteSeer Edgepred_Gprompt及Actor GraphCL，20新seed24092500..19，1020结果、540次Prompt训练。
- 阶段2b：在前轮已知后，补验阶段1已记录的Cora/CiteSeer Edgepred_GPPT；固定20新seed24092600..19，680结果、360次训练。
- 阶段2c：继续固定阶段1已有候选Cora/CiteSeer DGI；20新seed24092700..19，680结果、360次训练。假设是底座对优化方向/尺度的适配需求不同，属于序贯探索扩展，不是首次盲测。

合计本目标新增3010条正式结果、1575次Prompt训练；引用不重复计入新增训练，smoke也不计入正式训练次数。每个确认轮完整保存8格、门控范数匹配/仅幅度、对角/无匹配优化、冻结/GraphPrompt及三个直接方法。各轮支持seed块不同，因此不同轮底座之间只作描述性比较，不能作同seed配对检验。

## 论文可写和不可写的主张

可写：该三阶段适配在明确数据集与指定SHA底座上有配对模块贡献；主干冻结且参数量小；实际收益具有预训练底座依赖。不同模块对应生成、优化和读出三个不同环节。

不可写：首创温度锐化/Shampoo/grafting/标签传播；三个模块普遍协同；所有图预训练模型均提升；所有数据集显著提升；整体超过全部强基线；新支持seed就是新图泛化。完整矩阵的不可替代性尚未获得普遍支持，raw锐化包含幅度变化，序贯候选选择及未校正区间必须披露。

参考两篇毕业论文的总体结构与消融组织，本研究定位为**任务适配、组件组织与条件性实证**。论文成果等级和学位审查不由这些表格自动决定。

## 文档与复现入口

- [论文章节初稿](论文章节初稿.md)：问题、公式、统一流程、实验、结果、消融和边界。
- [阶段1全因子分析](stage1_factorial/阶段1结果.md)、[全部条件性候选](stage1_factorial/条件性三模块评估.md)。
- [Edgepred_Gprompt/GraphCL确认](stage2_confirmation/确认结果.md)、[GPPT确认](stage2b_gppt_confirmation/确认结果.md)、[DGI确认](stage2c_dgi_confirmation/确认结果.md)。
- [实际成本](chapter_assets_final/成本表.md)、[梯度结构诊断](chapter_assets_final/梯度诊断.md)。条件数仅为实际轨迹的描述，不能单独证明底座类型的因果作用。
- 每阶段plan.json列出SHA、seed、100步、5-shot及源码哈希，results下保留result.json和sealed.pt；final_audit.json为服务器逐预测核验，transport_audit.json和artifact_transport_audit.json为本地核验。pair_audit.json核对共享训练，stage1另有reference_audit.json和shared_training_audit.json。

复现环境：服务器ProG Python 3.12.3、PyTorch 2.12.1+cu130、PyG 2.8.0.post1、CUDA13.0；设备报告NVIDIA GeForce RTX4080 SUPER。每次实验torch CPU线程数4，冻结模型eval，TF32关闭。按源代码和锁定计划复现，不通过改变预训练checkpoint名称代替SHA对应。
'''
(R/'三模块方案说明与实验结论.md').write_text(summary)
methods=(R/'方法章节草稿_待实验定稿.md').read_text();methods=methods[:methods.index('## 7 论文章节后续填充项')];methods=methods.replace('**研究草稿。** 三个模块的最终保留与适用数据集尚待实验确认。本稿对应当前实现，不能将“设计目的”写成“已经证明”。可作为论文第三章或第四章的技术部分底稿。','本章给出一个以主线03为基础、适用范围明确的三模块研究方案。实验与归档已完成，结论按数据集和底座分别陈述。模块采用已有原子机制的任务适配，不主张首创。').replace('确认阶段应包含等范数匹配或幅度对照。','本章确认阶段已包含等范数匹配及仅幅度对照。')
chapter=methods+'''\n## 7 统一训练与推理算法

```text
输入：图属性/结构，支持节点及其标签，冻结两层GCN，固定seed
构造两跳属性、修正原型教师和温度门控Q
初始化类别token P=0；冻结主干并保持eval
重复100步：
    在第一层预激活加入10*(Q@tanh(P))*std(U)
    前向得到Z；计算支持留一原型CE与token正则
    只对P反向求梯度
    按开关使用Adam或方向/尺度匹配更新
对最终Z作共同中心化
按开关直接原型分类，或构建50近邻并进行20步支持钳制概率传播
封存支持索引、query索引、分数、预测、token与诊断
预测封存后，才使用query标签计算准确率
```

## 8 实验协议与可复现性

使用Cora、CiteSeer的官方训练节点池/测试节点，每类抽取5个支持标签；Actor使用官方fold0。原8数据集筛查还包括PubMed、Computers、Photo、Chameleon、Cornell，不能把它们都声称为成功场景。Planetoid输入使用NormalizeFeatures；Actor主干输入保留原特征，教师使用规范化特征。查询集合固定，输入特征与结构可转导访问，查询标签不进入拟合。详细节点数、类别数、query规模、池划分及权重路径以各stage的plan.json为准。

主对比采用同seed、同checkpoint配对，另保留不依赖checkpoint的直接教师。教师每数据集每seed只计算一次，不因为在多个checkpoint比较就增加独立样本数。因子开关Fabc的a/b/c依次代表锐化、匹配优化、传播。每个模型的可训练token数为类别数×128；GraphPrompt基线为本地适配实现，不能把其数字直接宣称为原论文最佳复现。Edgepred_Gprompt是预训练family名称，GraphPrompt是下游适配基线，两者不能混淆。

20个新支持seed的配对区间使用固定随机数20260924和10000次有放回重采样；描述同一图、固定query集合下的支持采样波动。候选经过历史探索且未作多重比较校正，因此不用这些区间宣称无选择偏差的独立图显著性。

## 9 完整方法、组件消融与适用边界

'''+summary[summary.index('## 结论和推荐定位'):summary.index('## 三个模块怎样构成一个研究点')]
chapter+='\n### 9.1 八格因子消融\n\nFabc的三个二进制位依次是M1、M2、M3。表内为准确率均值±支持seed标准差，不能据跨轮不同seed块对底座作配对排名。\n'
for family in ['Edgepred_Gprompt','Edgepred_GPPT','DGI','GraphCL']:
 group=[c for c in cells if c['checkpoint']==family]
 chapter+='\n#### '+family+'\n\n| 因子组合 | '+' | '.join(c['dataset'] for c in group)+' |\n|---|'+'---:|'*len(group)+'\n'
 for key in ['F000','F001','F010','F011','F100','F101','F110','F111']:
  chapter+='| '+key+' | '+' | '.join(f"{c['accuracy'][key]['mean']:.3f} ± {c['accuracy'][key]['std']:.3f}" for c in group)+' |\n'
chapter+='\n### 9.2 机制对照\n\n以下为完整方法减对应机制替代，单位pp。完整区间、逐seed差和全部二/三阶交互可在各确认报告复核，不能只看本表正值。\n\n| 数据集/底座 | 对等范数门控 | 对仅幅度门控 | 对对角更新 | 对无匹配矩阵更新 |\n|---|---:|---:|---:|---:|\n'
for c in cells:
 chapter+='| '+c['dataset']+'/'+c['checkpoint']+' | '+' | '.join(f"{c['contrasts'][k]['mean_pp']:+.3f}" for k in ['full_minus_M1_normmatched','full_minus_M1_amplitude_only','full_minus_M2_diagonal','full_minus_M2_ungrafted'])+' |\n'
chapter+='\nDGI两个数据集的完整矩阵优于对角对照，Cora/GPPT则对角更优；不能把矩阵复杂性描述为始终必要。Cora/Edgepred_Gprompt的仅幅度门控均值更高，说明锐化语义解释依赖具体场景。二阶及三阶交互有负值，完整系统三个移除差为正不等于超加性协同。\n'
for stage in stages[1:]:chapter+=f'\n- [{stage}：完整对比、配对区间和全部交互]({stage}/确认结果.md)\n'
chapter+='\n## 10 计算成本与梯度诊断\n\n'+(R/'chapter_assets_final'/'成本表.md').read_text()+'\n'+(R/'chapter_assets_final'/'梯度诊断.md').read_text()
chapter+='''
## 11 与参考论文的组织对应及贡献边界

聂雄论文第三章按总体框架、三个模块、算法步骤、主对比、跨数据集与模块消融展开，PDF45页表3-5包含完整8种模块组合。本章借鉴该消融组织，但不照搬其对协同作用的解释。向苡霄论文第三章图3-1（PDF28页）强调模块在统一流程中的作用，PDF40页表3-4/3-5将改进与DAFormer基础比较；本章同样区分共享基础与新增开关，而不把冻结主干到最终系统的所有差值归因于三模块。两篇论文的指标是作者报告，未在本项目复现。

技术来源包括GraphPrompt、MixMatch中的温度锐化、Shampoo、方向/学习率分离，以及基于标签传播的Correct and Smooth。具体引用见研究设计与章节框架.md。这里的贡献应限定为图小样本Prompt场景的组件组织、接口设计和条件性实验；已有算法名称不能改名后宣称首创。

## 12 本章小结

本章围绕单一冻结GCN上的条件Prompt，分别从语义门控、更新方向与尺度、预测结构约束三个环节构建方法。三模块贡献在指定底座与数据集上成立，适用范围由完整因子筛查和固定支持采样确认界定；Cora部分设置的微弱/负贡献、Actor失败以及强直接教师优势同样属于结论。整体任务收益、单模块贡献、复杂性必要性与机制原创性是不同证据层级。本章不将有正均值的个别配置扩展为普适有效方法。

参考资料与完整实验入口：见三模块方案说明与实验结论.md，以及研究设计与章节框架.md。
'''
chapter+='\n梯度诊断补充：DGI的右侧累积矩阵条件数并不高于所有边预测底座，因此数据不支持“DGI条件数更高就导致优化增益更大”的简单解释。较低有效秩及训练轨迹差异是描述现象，不是已证明的因果机制。\n'
(R/'论文章节初稿.md').write_text(chapter)
(R/'README.md').write_text('# 主线03硕士论文三模块研究\n\n[方案说明与实验结论](三模块方案说明与实验结论.md) · [论文章节初稿](论文章节初稿.md)\n\n完整保留探索、三轮固定范围确认、机制与成本分析及全部失败边界。强直接教师差距不可省略。原始结果、源码与审计按stage分目录保存。\n')
print('chapter_documents_created',len(chapter),len(summary))
