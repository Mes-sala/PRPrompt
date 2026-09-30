# 第110轮草案：提示连接扰动的双视图一致性

已读[R-Drop，NeurIPS2021第2.1节](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf)及[作者README示例](https://github.com/dropreg/R-Drop)。论文写两项NLL之和，示例代码用两CE平均；拟采用示例的平均CE，并将对称KL按支持数取平均、类别求和、系数1。两端分布都参与反传，不detach。这里改变的是提示连接掩码而非原文网络各层dropout，不移用原文理论保证。

共享同一冻结2层GCN与一个768token。借鉴97的有效矩阵DropConnect，每步两个独立保留率.9的掩码作用于tanh(T)，其余教师门控与主干保持不变。两个图前向分别构造支持LOO分数，并对相同支持节点的类别分布加双向KL；不是两个独立Prompt，也不是终端预测融合。推断用完整token单前向。

对照必须包括同两前向/同两掩码的平均CE无KL（计算预算与采样对照）、KL节点对应置换（两CE仍真标签）、单掩码CE（97式参照）、原无掩码强根及Direct。恒等同掩码KL=0只用于预检。单前向根的算力成本不同，不能宣称等计算预算收益；双CE无KL是主要机制参照。

首末步保存token、两个掩码及完整分数/梯度；独立CPU按LOO逐类重构目标及双端梯度，另NumPy重放全部优化器状态。原97损失需重写为返回完整LOO分数，避免KL误用含自身原型或query标签。使用独立CPU随机流，正式源码及种子尚未封存。loss110/check110目前仅损失原语，真实图接入未实现，不计准确率验证。


## 封存实现
真实图驱动run_round110_rdrop.py复用97骨架；drop110双掩码固定CPU seed+1100000，KL置换独立seed+1100001，所有分支两流预生成一致。rdr双CE+对称KL、rdc双CE、rdp真标签双CE+节点错配KL、rds单CE、rdi无掩码恢复根仅预检。每步保存梯度，首末步保存两个LOO分数和更新前token、全部掩码与置换SHA；CPU审计独立逐类LOO和logsumexp重算双端梯度，未调用训练损失辅助函数。原阈值梯度atol3e-5/rtol2e-4，分数8e-5/1e-5，失败不静默放宽。

固定176100..104，双checkpoint145开发；176199三步33预检，五审计全部通过才跑正式。根GraphMAE Adam.01、GraphMAE2gsh.01；100步，原根不混入此前未确认的调参。主候选rdr须超过当前根且超过同双前向rdc，开发增量至少.20pp才固定新20支持确认；错配及单CE只作解释，不以其下降证明主方法更好。
