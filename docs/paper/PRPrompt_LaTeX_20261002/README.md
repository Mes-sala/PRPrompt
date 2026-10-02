# PRPrompt 中英文 LaTeX 论文

以已验收的三数据集叙事和数据为依据，形成完整中英文工作稿。当前采用通用论文版式，未指定投稿会议，未套用或声称符合某一会议的页数限制。已按后续要求在独立 Conda 环境中编译并逐页核对；主架构图仍为 LaTeX 原生占位框。

## 入口

- [英文论文](main_en.tex)，正文在 [paper_en.tex](sections/paper_en.tex)。
- [中文论文](main_zh.tex)，正文在 [paper_zh.tex](sections/paper_zh.tex)。
- [英文附录](sections/appendix_en.tex)与[中文附录](sections/appendix_zh.tex)。
- [主架构图占位符](architecture_placeholder.tex)。
- [参考文献](references.bib)、[文献核验记录](bibliography/verification.json)及[结构参考说明](叙事与结构参考.md)。

正文包含摘要、引言、相关工作、问题定义、方法、实验、局限性、结论及潜在影响。附录包含数据维度、原型计算与优化细节、算法流程、基线选参、统计公式、来源验收、五指标结果及配对区间。作者信息留空，等待论文署名确定。

## 不变的数据范围

固定方案为 PRPrompt；仅 Cora、CiteSeer、PubMed 的 1/3/5-shot。主比较为冻结预训练、全量微调、GPF、GPF-plus、统一适配的 GraphPrompt 与 PRPrompt；消融为无提示、共享提示和完整原型路由。不含分类器对照或其他方案探索。

所有均值、标准差与配对区间均在固定数据集、shot、方法及权重的条件内计算。先平均每个支持抽样的三个训练 seed，再统计 20 次支持抽样的均值与样本标准差；不跨数据集或实验设置求平均。

中英文使用相同表格文件及相同封存数据。`data/` 保留 54 行主比较、27 行消融、270 项五指标配对记录及来源/数值审计，不包含模型权重。正文展示 Accuracy 和 Macro-F1；附录补充其余主比较指标以及两项主要指标的逐条件配对区间。

## 源码检查与维护

在本目录运行：

```bash
python3 scripts/build_tables.py
python3 scripts/validate_sources.py
```

第一个命令只从 JSON 生成 LaTeX 表格，第二个命令检查数值、引用、文件依赖、环境配对和双语一致性。两个命令都不启动训练、不重做 bootstrap，也不运行 TeX 编译器。[静态检查结果](static_validation.json)不代表排版或编译通过。

## 编译环境与 PDF

独立 Conda 环境为 `prprompt-latex`，使用 Tectonic 0.17.0（XeTeX / BibTeX 自动构建）和 Poppler 26.09.0。中文使用 `ctexart` 的 Fandol 字体；英文使用 `article` 和 `fontspec`。首次编译会下载宏包与字体缓存。环境与缓存位于仓库外，不影响 ProG 实验环境。

从本目录执行：

```bash
# 仅首次创建环境时执行；环境已存在时直接编译
conda env create -f environment.yml
python3 scripts/compile_papers.py
```

中间文件与日志写入仓库的 `experiments/local/latex_build/`；最终 PDF 写入 `output/pdf/PRPrompt_20261002/`。源码目录只保留可编辑文件。

- [英文 PDF](../../../output/pdf/PRPrompt_20261002/PRPrompt_en.pdf)
- [中文 PDF](../../../output/pdf/PRPrompt_20261002/PRPrompt_zh.pdf)
- [编译与逐页核对记录](compilation_review.json)

本次修正英文命令后的词间距、消融表缩放、表格浮动与附录分页、参考文献分页，并核对公式、中文字体、引用及表格数据。编译检查不替代实验复现，也不代表已满足特定会议模板要求。

## 研究边界

正文保留三项关键限定：基线是统一协议实现；范围在实验完成后确定；冻结权重的预训练 seed 未记录。共享提示消融也未完全排除有效容量差异。没有将这些结果表述为全部图提示方法的最优成绩。图分类、未见类别、效率优势和原型校正各步骤的独立贡献均未得到当前实验验证。

[写作检查表](写作检查表.md)记录已覆盖的 ML 论文要素和投稿前仍需处理的事项。已有结果和方案不因写作发生改动。
