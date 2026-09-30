# R133：Cora 教师门控固定锐化

日期：2026-09-24。目标是准确率提升，机制原创性不作筛选条件。

借鉴 DINO 教师概率锐化（https://arxiv.org/abs/2104.14294），迁移对象是冻结图编码器中的教师门控，而非复制自蒸馏训练。固定二次幂，不搜索温度。保留 Cora 已有强方案：冻结双层 GCN、后层注入、896 个 token 参数、ridge LOOCV 教师、两次 plain 平滑、动态 scale2 CE、原教师全节点 KD、Adam 0.01、100 步、最终不中心化。

设平滑教师为 $ Q $，原门控 $ G=\hat A(Q-\bar Q) $。锐化概率 $ Q'_i=Q_i^2/\sum_c Q_{ic}^2 $，候选门控 $ G'=\hat A(Q'-\bar Q') $。raw 使用 $ G' $；matched 使用 $ G'\|G\|_F/\|G'\|_F $；amplitude 使用 $ G\|G'\|_F/\|G\|_F $。三者 KD 目标均保持 $ Q $。纯幅度收益不归因于锐化方向，但若准确率达标也可进入独立确认。

scale2 是每步 focal 导数系数的均值乘普通 CE，不是 CE 常数乘二。保留原实现：$ c_i=(1-p_i)^2+2p_i(1-p_i)CE_i $，监督损失为 $ \operatorname{stopgrad}(\bar c)\overline{CE} $。

GraphMAE/GraphMAE2；官方训练池每类5个支持节点；开发查询由固定20260923打乱后的官方测试集前500节点组成。新支持种子198100–198104。相同图和查询的重复支持种子不是独立数据集证据。

方法：独立 Direct ridge+plain 教师（每种子一条）；每checkpoint含 frozen、graphprompt、base/gp k50、原token和scale2的mean/k50、raw/matched/amplitude的mean/k50，共29条/种子，145条。冒烟198199、3步，加power1分支，共33条。k50使用余弦近邻、温度0.2、alpha0.9、20步支持标签钳制。

进入确认门槛：候选k50在5种子均值上同时超过本轮跨checkpoint最强scale2 k50及Direct至少0.20个百分点。按开发均值选择，若并列按raw、matched、amplitude再GraphMAE、GraphMAE2的固定顺序。允许控制方案被选中，但明确对应幅度效应。确认使用独立支持种子和后500查询；不得只凭开发收益宣布成功。

审计：封存源码hash；逐条重建support/query/graph/预测hash；CPU重建teacher、门控、后层forward、原型和k50邻居权重/传播；全部梯度轨迹用NumPy Adam回放；首末步动态CE和KD梯度重建；显式145/33键集合和40/10训练轨迹覆盖。原legacy与power1三步CPU单线程token、embedding、k50位精确一致。GPU独立运行不要求近邻传播分数位精确，每条均独立几何审计。

只有全部冒烟审计通过才自动启动正式任务。任何失败保留现场，后续修复另存版本。严格checkpoint加载仅说明本地可用，不代表上游源码一致性已验证。
