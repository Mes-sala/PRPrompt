# 后续候选：句向量白化借鉴（未实现）

来源：[Whitening Sentence Representations for Better Semantics and Faster Retrieval，Su等，2021](https://arxiv.org/abs/2103.15316)。原文用白化改善预训练句向量几何。当前早期round2的sphered实际是逐维标准差平方根缩放，并非完整协方差白化（run_round2.py:51–56），所以可在新图上检查完整相关性修正。

候选：在每个固定主干/图结构的无提示输出上估计全节点无标签均值和128维协方差；收缩 $ C_\alpha=(1-\alpha)C+\alpha\operatorname{tr}(C)I/128 $，alpha预设0.1/0.5；矩阵逆平方根固定，Prompt训练不重新估计；作用于最终表示和支持原型计算。该操作属于固定度量校正，不单独命名为Prompt。

比较原几何、中心化、对角缩放、完整收缩白化，各自无提示与同几何GraphPrompt/内部token。避免只把白化组合与原几何弱基线比较。白化统计可用转导图全部输入但不使用查询标签；最终参数、协方差谱、源hash留档。先验证固定矩阵的实际协方差和有限性，再小矩阵实验。等第31轮结果后决定基础结构，暂未运行。
