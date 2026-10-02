# 原 ProG 划分复算：已完成

复用旧 checkpoint，八数据集 × 1/3/5 名义 shot × 20 任务 seed × 3 训练 seed × 4 方法，共 **5760/5760** 个正式单元。96 个 smoke 单元未计入。正文仍展示 Cora、CiteSeer、PubMed，Photo、Computers、Actor、Chameleon、ogbn-arxiv 放附录。

- [中文结果总结](结果总结.md)：逐数据集 Accuracy、路由配对增益和负结果。
- [五指标完整 CSV](comparison.csv)、[JSON](comparison.json)、[配对差值与区间](paired.csv)。
- [覆盖检查](coverage.json)、[本地交付检查](delivery_status.json)、[实际划分与每类数量](split_inventory.json)。
- [更新后的双语论文](../../paper/PRPrompt_LaTeX_双栏_20261002/README.md)。

协议见[实验计划](../../../experiments/prog_split_20261002/PLAN.md)。[480 个任务的原函数一致性记录](split_parity.json)全部通过；本地另核对了全部 5760 个预测档案的配对支持／测试索引、有限指标、同条件均值／标准差和配对 bootstrap。

原 ProG 两次独立随机排列保留支持／测试重叠，因此不是严格未标注节点泛化评估；ogbn-arxiv 有 20/60 个任务少数类不足名义 shot，不补抽样。不能据此推断 MTG 文献的实际划分也重叠。原文外部基线数字保持不变，预训练、选参、重复次数与具体索引仍不完全匹配。

先平均同一任务的三个训练 seed，再统计同数据集、shot、方法的 20 个任务，不跨设置平均。旧实验保留独立目录；权重、图快照与模型状态留在服务器，本地只接收代码、结果、预测、划分、表格、文档与日志。不执行重复的全文件哈希或逐位 CUDA 前向审计。
