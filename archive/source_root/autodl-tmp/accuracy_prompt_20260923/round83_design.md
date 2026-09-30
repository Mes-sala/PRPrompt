# 第83轮草案：借鉴Grafting匹配更新幅度

来源：[Disentangling Adaptive Gradient Methods from Learning Rates](https://arxiv.org/abs/2002.11803)，及[Google Research Shampoo实现](https://raw.githubusercontent.com/google-research/google-research/master/scalable_shampoo/pytorch/shampoo.py)的方向与范数分离。官方实现示范SGD/AdaGrad范数，本轮拟用既有强Adam的局部更新范数；属于适配，不称完整复现。原论文强调优化方向与隐式步长调度可能混杂。

第82轮独立诊断round82_step_diagnostic.json显示，GraphMAE Adam.01平均参数步长范数0.281，Shampoo.01仅0.00431、.03仅0.01473；最后支持CE约0.572对1.778/1.727。GraphMAE2对应0.257、0.00450、0.01526，CE0.429对1.765/1.667。这支持小步长可能造成训练不足的解释，但不能独立证明Shampoo方向正确；不能将支持CE直接当query收益。

固定第79轮softmax.1门控、768零token、100步、原CE+L2、最终中心化+k50。每一训练分支用自己的梯度维护Adam影子一二阶矩，计算含偏差修正的方向A；Shampoo方向D继承第82轮epsilon1e-4双精度L/R。更新为 $ -0.01D\|A\|_F/\|D\|_F $，不另加动量。对角L/R方向及原梯度方向也分别匹配各自Adam影子范数，强根仍真正Adam；加入影子Adam方向恢复对照。零方向需明确no-op或断言处理，不能静默制造非有限更新。

重要边界：不同分支训练轨迹不同，影子Adam在候选自身梯度上计算，不能宣称每一步数值等于独立Adam基线的更新范数。审计逐步检查同分支D/A范数及NumPy重放；保存梯度与必要状态诊断。预期新开发支持127100–127104，固定学习率0.01，不继续盲目搜Shampoo学习率。此文件仅设计，未实现、未运行。


## 封存实现

145正式/29预检：原Adam.01、原Shampoo.01、grafted Shampoo、grafted diagonal、grafted gradient、影子Adam恢复，均两读出另Frozen/GP/Direct。预检127199三步，正式127100–127104一百步。零方向明确no-op并记录影子Adam是否仍有动量；Adam恢复分支则保留自身非零动量。新优化器不另加动量。NumPy独立重放每步方向和范数，最终token误差限3e-5；预检另验证影子Adam恢复原根分数及预测。参数梯度包含原L2，训练源码已封存，预检与四审计通过后自动正式。
