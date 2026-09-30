# 第106轮：互补熵跨预训练权重适用范围

沿用[Complement Objective Training](https://arxiv.org/html/1903.01182v2)启发与loss104.py已核验目标，不修改系数1/5、训练步数或损失公式。本轮是跨权重迁移，不是假称新机制。GraphMAE2首组+.13[.04,.23]、第二组+.05[−.07,.19]，按预注册不升级；检验新的checkpoint适用范围，不重跑同一条件追逐显著性。

仅CiteSeer DGI、GraphCL、S2GAE。三权重当前SHA与round99已strict_load/sidecar核对完全相同，未重新预训练；来源仍标注ProG-adapted、source_parity未完全核验。当前根按此前保留协议：DGI嫁接Shampoo.01，GraphCL/S2GAE Adam.01；cpr/cpa/cpp/cps/cpi各自使用匹配当前根，其他共享早层6×128token、零初始化、100步、中心化、单中心锐化门控/k50不变。保留另一优化器根以识别弱基线改善，不新增教师融合。

方法：互补熵、全类熵、错配排除、标签平滑0.1、两个优化器根、Frozen、GraphPrompt、Direct，mean/k50。169100..169104五支持，215条；169199三步49条预检含原CE恢复。支持/开发query协议沿用，标签只用于support训练与封存后计分。

数值审计预声明：直接复用第99轮跨权重geometry_v2的固定阈值，分数atol1e-3、邻居权重atol1e-4、rtol1e-5、topk门限1e-5，同时CPU重建argmax必须与保存prediction完全相同。较宽数值阈值来自先前DGI等权重诊断，不因本轮结果调整；若argmax失败仍不判通过。loss首末梯度、优化器重放、教师/引用/计分审计沿用104阈值。任何失败停批并保留失败记录，必要时单列作用域，不隐去或偷偷改阈值。

主要比较各自当前根，其他损失及Direct同步。若局部增益则固定新20支持确认；由于历史checkpoint弱于Direct，报告绝对表现与同权重增量，不能把相对差称为整体最好。开发只筛选，无创新要求。
