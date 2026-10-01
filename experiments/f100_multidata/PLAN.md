# 未修正 Edgepred＋历史 F100 的多数据集验证

冻结于首次拟合前；原 Cora/CiteSeer 结果和源码不改。新增 PubMed、Photo、Computers、Actor、Chameleon，各用既有 Edgepred_Gprompt / Edgepred_GPPT 的一份原始 GCN-128 checkpoint，不重新预训练、已有 sidecar 标为 ProG-adapted / not_fully_verified；没有 sidecar 的标为 legacy_unverified。各权重预训练种子未知。

## 目标与范围


方法：FrozenLinear、FrozenPrototype、FrozenPrototypeCentered、FullFineTune、GPF、GPFPlus、GraphPromptDiag、v2 PRPrompt、v1 F100。FrozenPrototypeCentered 在原冻结 GCN 输出上使用与 F100 完全相同的全图中心化与余弦原型读出，作为无训练的图表示对照。它不是属性分类器。用户排除 AttributeRidge、AttributePrototype；不添加从零模型、M3、shot 曲线或方法搜索。GPF / GPFPlus / GraphPromptDiag 是本项目统一 GCN 与全图节点协议下的适配实现，不声称精确复现作者原始协议。

F100 直接加载未修改历史函数 AST：普通 Adam，lr=.01，eps=1e-8，weight_decay=0；M1=.1，幅度10，sample std，提示正则 .001，LOO 原型损失，最终中心化原型读出，无 M2/M3/KD/分类头。保持100步，不依据查询结果改配方。其余可训练方法复用 prprompt.train.fit_predict 与支持内三折选参：lr [.001,.01] × wd [0,.0001]，每折100步，最终100步；FrozenPrototype 两种版本免选参。F100 固定配方与其他方法支持内选参的差异明确披露。

PubMed 官方 train/test；Photo / Computers 按种子20260930每类20节点建立固定候选支持池，其余节点作为query；Actor / Chameleon 官方 fold0 train/test。每次从候选池抽每类5个标签。标签只用于固定划分、支持拟合和封存后的评分。全图无标签特征、结构可见，中心化和路由可见全图；属于同图传导式节点分类，不能解释为新类/跨图泛化。异配图保留加载器原方向，全部方法共享同一图、池、支持与query。

## 工程与验收

原数据、权重及 sidecar 保留；复制权重、保存张量化图/划分、冻结源码和协议 SHA256。每个任务分3个互斥支持抽样分片，单进程独立随机状态。每卡最多6个任务，并依据 smoke 峰值 reserved/allocated 显存加安全余量限流；不使用 AMP/TF32、不改科学预算以提高利用率。GPU smoke：每份checkpoint每方法1次、2步，独立目录，不是正式成绩。

拟合API只收到支持标签；先存状态、选参、预测与seal，后评分query。保存每个最终分数及状态，验证冻结模型和提示梯度。完成后逐单元检查 coverage、重复、支持query隔离、五项指标、预测/状态/选参SHA和配对划分；使用独立NumPy混淆矩阵、含ties的ROC/AP重算五指标。

数值验收在本轮首次拟合前声明：保存分数重放概率及argmax须完全一致；F100 从保存表示重建原型读出误差 <2e-6 且argmax相同；独立重建teacher/gate误差 <2e-6；独立重建固定gate前向采用 abs(error)/(2e-6+1e-6*abs(saved))<1，记录原始误差。完整新CUDA前向→中心化→归一化→预测，以及其他方法的独立状态前向误差/类别变化作为单独诊断完整公开；不把保存状态重放通过写成完整新前向逐位通过。这些边界来自前轮实际CUDA稀疏归约差异，已在本轮前预声明，不据新指标改阈值。

训练重复先在每个支持抽样内平均，再汇总20个支持抽样的均值和标准差；F100 对所有8个参考方法的五指标均输出配对差值及10000次bootstrap 95%描述性区间，随机种子20261001。区间仅针对固定图/权重下支持抽样，未做多重比较修正；不能把三次训练重复当作60个独立支持集或独立预训练。

回传全部源码、权重、环境、图/划分快照、状态、预测、日志、结果与逐文件SHA清单。本地独立重算指标和所有汇总/配对区间，写 local_delivery_status.json。报告完整90行方法×checkpoint五指标和80组Accuracy/Macro-F1配对，保留负结果与强微调差距。没有结果前不承诺普遍提升。
