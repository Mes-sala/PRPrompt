# 第99轮：已有配方跨预训练权重验证

已严格加载CiteSeer的DGI/GraphCL/Bandana/S2GAE，均匹配现有2层GCN128；SHA/sidecar保存round99_checkpoint_audit.json。权重标注与源实现一致性仅据sidecar，不新增严格上游复现声明。

原有跨领域经验迁移：中心化、温度调整、Shampoo方向与Adam步长匹配，详见round63/79/83设计。此次不新增训练算子，直接调用封存的run_round98_dcl.py中已有分支；不使用其中DCL损失。

固定共享两层GCN且完全冻结，单768参数提示在第一层预激活注入。中心化特征3步教师，零token、100步、支持LOO CE+.001token²均值，无KD。Adam.01是所有新权重的默认根；graftedShampoo.01是待比较候选，不能把GraphMAE2上的结论预先推广。

17方法：Direct、Frozen、GraphPrompt；Frozen/GraphPrompt后中心化的mean/k50；原温度.2提示tr2、.1提示tr1、.1等范数tm1、.1仅幅度ta1及.1步长匹配gsh，各mean/k50。最终k50参数保持alpha.9、20步、支持钳制。对照两种固定读出防止把只换读出归为提示训练。既有温度/幅度不能区分时按准确率选择简单可复现方案。

156199三步预检65条，四项审计（评分封存、教师身份、完整CPU几何及邻居、保存梯度的优化器重放）后正式156100..104五组325条；Direct仅每种支持记一次。模型eval/frozen检查及参数SHA每条保存；读取查询标签在sealed写盘之后。新训练损失未引入，所以不复制第98轮DCL梯度审计。tr/tm/ta继承原温度实现，优化器重放审计覆盖gsh，不宣称覆盖全部Adam轨迹。

主配对：tr1对tr2、gsh对tr1；其余幅度对照与postbase/postgp/Direct都报告。先看完整四权重开发，不挑单个seed；有绝对性能及匹配增益的候选固定20新支持确认。固定图/重复查询使开发有选择偏差，确认区间只能反映支持抽样变化；不能声称跨图泛化。
