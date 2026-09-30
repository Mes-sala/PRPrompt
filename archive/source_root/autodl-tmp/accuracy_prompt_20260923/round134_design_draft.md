# R134草案：Cora中的类别/样本相关性蒸馏

来源为NeurIPS2022视觉蒸馏DIST：[论文](https://arxiv.org/abs/2205.10536)、[官方repo](https://github.com/hunto/DIST_KD)。已读取官方README指向的classification loss，而非根据摘要猜测；DIST_KD固定commit `2418f18b1fedd7c2646a7b1b9a27d5c674c2170b`，classification子模块固定commit `d9662f7df68fe46b973c4580b7c9b896cedcd301`，快照为 `dist_classification_pinned.py`。第一次试探的根目录loss路径404、GitHubAPI403，无证据丢失；成功浅克隆后按README/子模块路径取得固定版本。

本地历史检索：R25已经做DKD，R65/R66是特征距离/角度RKD。本轮只借鉴预测概率的Pearson关系项，不重复前述目标，不宣称新机制或教师能力更强。现有Cora原教师仅由支持标签构造，独立Direct准确率可弱于提示学生；本文标题的stronger teacher条件不能直接外推。

保持Cora原强方案的冻结双层GCN、896token、late注入、原门控、dynamic scale2 CE、原始表示k50不变。令学生全节点概率 $ P=\operatorname{softmax}(S) $，原平滑岭教师概率 $ Q $。固定tau=1（S本身已含原cosine温度0.2），不把Q再softmax，不再次锐化Q。

$ L_{inter}=1-\frac1N\sum_i \rho(P_{i,:},Q_{i,:}) $，$ L_{intra}=1-\frac1C\sum_c \rho(P_{:,c},Q_{:,c}) $。相关系数先各向量减均值，再以范数乘积加1e-8作分母，与固定官方实现一致。query标签不参与任何相关项；全图预测/结构参与属于transductive协议。跨节点相关不是每个节点独立统计。

拟比较原KL、去KD、KL×2、inter-only、intra-only、inter+intra；CE权重与优化器不动。相关项替换KL而非同时叠加。去KD和KL×2都是已知控制，若朴素控制有效同样可保留准确率方案，不将其收益归于相关性。正式种子和矩阵待初始梯度诊断后封存。

必须先做：官方loss/gradient对齐、NumPy独立标量重算、double gradcheck、均匀概率退化情形、Cora初始原KL/各相关项的token梯度范数及方向（不读查询标签），明确不同项的梯度尺度。完整训练继续保存全部梯度/首末快照，CPU重建新的损失梯度；旧R133审计不能直接视为覆盖新KD。

状态：原语已编写，未接入训练；不得把草案当作准确率结果。R133已STOP，两个实验的结论分开记录。


## 原语与无查询初始梯度诊断（2026-09-24）

`dist134_primitive_check.json`：固定官方代码loss误差0、gradient误差6.94e-18、NumPy标量误差0、double gradcheck通过。原测试曾要求均匀概率梯度位精确为0，实际4.0998e-43，与官方实现逐元素完全一致；保留原测试，新v2改为官方精确对齐及有限性，不改变算子。

`dist134_initial_probe.json`：seed199199，仅支持标签、不评准确率。GraphMAE inter/intra/both的token梯度相对KL为1.44/1.54/2.66倍，both与KL余弦0.558；GraphMAE2为9.75/7.46/15.73倍，both与KL余弦0.603。原始系数1在两个checkpoint上尺度差异明显，不能将总梯度增强误写为关系信息收益。

训练矩阵封存前拟补入固定初始梯度范数匹配：每个run在T=0上算原KL和候选关系项的token梯度，将候选系数固定为两范数之比，之后不再更新；不得使用查询标签或准确率调整。both另保留raw系数1，去KD和KL×2作为朴素控制。inter/intra-only也各自匹配原KL初始范数。需保存原始梯度、固定系数、两项值及实际总梯度。该诊断只说明算子/数值可行，不说明准确率有效；正式实验仍未启动。
