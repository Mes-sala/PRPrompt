# 第126轮：AdaCos支持LOO训练尺度

来源与更新公式见已封原语及round126_design_draft.md，参考[原文第4节式13–15](https://arxiv.org/pdf/1905.00292)。固定使用偶数样本中间两角平均的中位数，t0尺度sqrt(2)log(C−1)，从t1更新，统计路径no_grad；logsumexp避免B指数溢出。所有模式只改训练CE，查询/teacher/门控温度不改。非正或非有限尺度显式失败，不隐藏裁剪。

CiteSeer5shot，GMAE/GraphMAE2，191100..104，100步lr.01。候选根分别Adam/graftedShampoo，冻结两层GCN及共享768 Prompt参数不变。完整动态acd、只partition动态acp、只角度动态aca、固定初始尺度acf；额外原tr1/gsh/Direct/Frozen/GraphPrompt，mean/k50，正式145条。冒烟191199三步33条，增加原尺度5的aco恒等恢复。

主要候选acd/acp/aca的k50须平均超过同批tr1、gsh、acf最强者至少.20个百分点才进入20新seed确认；相等优先acd、acp、aca。若仅部分动态赢，按对应统计归因，不宣称完整AdaCos效果。固定acf如有正向仅列诊断，另立后续设计才可选择。mean为诊断。

保存每步原始LOO logits、CE、previous_scale、scale、logB、median；保存首末步tokens/梯度。audit126_scale独立NumPy递推、稳定CE及CPU真实图首末步梯度，优化器使用全部实际梯度回放；原教师/几何/近邻/最终argmax审计保留。源码/config/设计/审计manifest封存；先原语再全部smoke审计通过后才正式运行。与前置原语区分真实图证据。

当前6类和支持LOO原型不满足原文海量身份类别及训练权向量设定，只做经验迁移。同图support相关、query跨轮复用限制保留。准确率评分在封存预测后进行。
