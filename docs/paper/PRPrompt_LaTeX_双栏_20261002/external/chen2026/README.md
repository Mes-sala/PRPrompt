# ICML 2026 节点分类数据记录

来源：[Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](https://proceedings.mlr.press/v306/chen26cz.html)，Chen 等，ICML 2026，PMLR 306:16063–16104。数据从作者 [arXiv v1 PDF](https://arxiv.org/pdf/2606.03290v1) 第 37–38 页表 4–6 的节点分类部分提取，并检查表格页面。出版信息由 PMLR 核验；出版方 PDF 下载未成功，因此不宣称提取文件与出版方 PDF 字节相同。

[论文用数值表](manuscript_values.md)集中列出三个数据集的 63 个单元。

## 文件与范围

- [JSON 记录](node_classification.json)：完整字段、原始小数字符串、表号、页码与来源。
- [CSV 记录](node_classification.csv)：便于筛选和导入表格软件。
- [原表行记录](source_rows.json)：保留被提取的 21 行，不含 MTG 行。
- [来源与协议清单](source_manifest.json)：PDF SHA256、版本及统计口径。

记录覆盖表 4–6 中 Cora、CiteSeer、PubMed、Wisconsin、Texas、Actor、ogbn-arxiv 的 1/3/5-shot，共 7 种方法 × 7 个数据集 × 3 个 shot = 147 个 Accuracy 均值／标准差单元。方法为 Supervised、Fine-tuning、GPPT、Gprompt、All-in-one、GPF、GPF-plus。排除 MTG 及全部图分类数据；不导入本文方法的消融或其他深度、骨干、预训练策略展开表。

论文仅采用 Cora、CiteSeer、PubMed 的 63 个单元。其余节点数据作为原表记录留存，不进入论文比较，也不扩展本地实验范围。

## 协议及使用方式

| 项目 | 文献记录 | PRPrompt 本地记录 |
|---|---|---|
| 测试划分 | 节点分类 90% 测试集 | 官方测试掩码 |
| 重复与标准差 | 五次独立任务抽样的均值、标准差 | 每次支持抽样先平均三个训练 seed，再对二十次支持抽样计算均值、样本标准差 |
| 预训练 | 原文所报告的跨预训练策略最佳结果；未逐单元给出选中策略 | 每个数据集固定边预测预训练权重 |
| 选参 | 原文随机搜索 | 固定配方或支持内选参，按方法定义 |
| 基线结果归属 | ProG 结果与作者复现的组合；未逐单元分配来源 | 本地封存运行 |
| 可用原始预测 | 本次导入没有 | 本地配对档案中保留 |

文献行明确标注文献来源，保留原始数值，不重算或跨设置平均。论文的 Accuracy 表分为文献报告与本地测量两个面板；粗体和下划线只在各自面板内排名。不同来源不能做配对检验。原文这三张表未报告 Macro-F1、Micro-F1、ROC-AUC 或 AP，因此不为这些指标填入外部数值。

引用外部基线后，本地历史 GPF、GPF-plus、GraphPrompt 适配数值及其配对区间不再列入当前论文表格，原始 JSON 和 Git 历史完整保留。本地测量的冻结预训练、全量微调与原型路由消融继续展示；原型路由的价值由同协议消融支持。
