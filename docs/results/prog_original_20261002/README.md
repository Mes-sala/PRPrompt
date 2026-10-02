# 原 ProG 划分复算（运行中）

本轮按用户要求复用之前的checkpoint，在八个数据集上重新计算。正文仍展示Cora、CiteSeer、PubMed，Photo、Computers、Actor、Chameleon、ogbn-arxiv放附录。

协议与执行范围见[实验计划](../../../experiments/prog_split_20261002/PLAN.md)。共5760个正式单元；GPU smoke的96个单元不属于正式结果。正式表格待完整运行结束后生成，当前论文仍是上一轮结果。

[划分一致性记录](split_parity.json)：480个任务划分均与[原ProG函数](https://github.com/sheldonresearch/ProG/blob/087aab45e25c336936d90d37540f140527567ebe/prompt_graph/data/load4data.py#L18)的支持、测试索引完全一致。原实现有支持／测试重叠；ogbn-arxiv有20个任务的候选池不足名义shot。结果将保留并注明这些情况，不把它解释为严格未标注节点泛化，也不据此推断其他论文的实际划分。

只对同一数据集、shot和方法的任务／训练seed求统计量，不进行跨设置平均。旧实验保留独立目录；权重、图快照与模型状态留在服务器，本地仅保存代码、实验结果、表格、文档和日志。
