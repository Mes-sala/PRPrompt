# R152 固定胜者确认：GraphMAE2 LoRA-r1 + Prompt

开发285条及110轨迹/220快照/140近邻均独立通过。预封存分析器选择GraphMAE2 l1 k50：76.20%，与l4精度持平但固定候选顺序l1先出现；最强旧根GraphMAE2 scale2 75.24%，+.96百分点，配对4胜1负；Direct74.64%。FourierFT最佳75.56%，不是本轮胜者。因此按用户准确率目标确认控制方法，不宣称FourierFT有效性升级。

固定原训练源码、方法矩阵和checkpoint；仅支持种子换222200..222219，query换固定confirm后500节点，共1140条/440轨迹/880首末快照/560近邻。锁定GraphMAE2 l1（rank1，Adam A/B/Prompt各.01），确认内不重选l4或Fourier。保留全部28种checkpoint方法与Direct。所有预测封存后评分。

运行完整独立CPU重建与全部梯度Adam回放之后，分析10000次支持级paired bootstrap（seed152000）。主候选与两个checkpoint的旧CE/scale2 k50以及Direct比较，全部95%下界>0才判强根升级。l4/Fourier/置换参考仅作相对解释；同图及重复confirm query的区间不作为独立图泛化。原型读出均值同时报告，即使高于k50也不在确认阶段换读出。
