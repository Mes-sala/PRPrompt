# PRPrompt 双栏中英文论文

本稿在已验收的三数据集稿件基础上，阅读图 Prompt 论文原文后重构引言、相关工作和实验行文，并改为通用 A4、10pt 双栏。研究重点仍为支持集原型组织节点提示分配。未指定投稿会议，因此不宣称符合任何会议的正式模板或页数限制。

## 论文入口

- [中文源文件](main_zh.tex)与[英文源文件](main_en.tex)。
- [中文 PDF](../../../output/pdf/PRPrompt_双栏_20261002/PRPrompt_zh.pdf)与[英文 PDF](../../../output/pdf/PRPrompt_双栏_20261002/PRPrompt_en.pdf)（本地编译输出）。
- [结构与行文分析](叙事与结构参考.md)：代表论文的阅读范围、论证方式及本稿采用的改动。
- [22 篇参考文献](references.bib)、[核验索引](参考文献核验索引.md)、[文献核验记录](bibliography/verification.json)、[原文阅读清单](bibliography/reading_manifest.json)。
- [编译与视觉检查](compilation_review.json)、[静态核验](static_validation.json)、[写作检查表](写作检查表.md)。
- [上一版单栏源码](../PRPrompt_LaTeX_20261002/README.md)独立保留。

引言依次说明适配需求、提示分配问题、原型路由设计和两组实验问题。相关工作按图表示与预训练、任务对齐、自适应分配、类别原型与评估组织。实验包含 RQ1 完整适配比较和 RQ2 共享提示消融。分类头仅在正文简述为固定原型读出，精确定义位于附录。主架构图按既有要求保留占位框。

两版均已编译为 9 页（含参考文献与附录），并完成全部页面核对；无缺字、溢出框或未定义引用。中文在 Poppler、PDFium 和 pypdf 中均提取到 6035 个汉字。

## 不变的数据与统计

仅 Cora、CiteSeer、PubMed 的 1/3/5-shot。主比较包含冻结预训练、全量微调、GPF、GPF-plus、统一协议的 GraphPrompt 和 PRPrompt；消融包含无提示、共享提示和完整原型路由。不增加分类器对照或其他方案探索。

中英文使用同一组数据和表格。每个数据集、shot、方法与权重构成独立条件；先平均每个支持抽样的三个训练 seed，再统计 20 次支持抽样的均值与样本标准差。不跨数据集、shot 或实验设置求平均。432 个表格数值单元、54 行主比较、27 行消融以及270项配对统计沿用封存数据。没有新增训练、重做 bootstrap 或改动方法。

主比较采用跨栏三线表，方法按行、数据集与 shot 按分组列排列；消融合为两个指标面板的一张跨栏表，配对区间按指标合为两张跨栏表。最佳均值加粗、次优均值加下划线，标准差使用较小灰色字，PRPrompt 行用浅灰底强调。设计依据与逐项不变性检查见[表格设计说明](表格设计说明.md)。正文展示 Accuracy 和 Macro-F1，附录保留其余主比较指标及主要指标配对区间。相对增益的图依赖、事后确定范围、未知预训练 seed、基线实现限制与消融有效容量差异仍在正文说明。

## 维护与编译

```bash
python3 scripts/build_tables.py
python3 scripts/validate_sources.py
# 已存在 prprompt-latex 环境时直接编译
python3 scripts/compile_papers.py
# 仅首次创建环境需要：conda env create -f environment.yml
```

独立 Conda 环境使用 Tectonic 0.17.0 与 Poppler 26.09.0，不改动实验使用的 ProG 环境。中间产物位于仓库 `experiments/local/latex_twocolumn_build/`，PDF 位于 `output/pdf/PRPrompt_双栏_20261002/`；源码目录不包含 PDF、编译缓存或 checkpoint。

中文沿用已修复的 Noto Serif CJK SC 2.003 字体及嵌入 Unicode 映射。字体下载、固定 SHA256 与缓存由 `scripts/setup_fonts.py` 管理，文件保存在仓库外。`compile_papers.py` 自动检查中文 CID 字体、ToUnicode、Poppler 提取和中文内容，检查通过后才复制交付文件。视觉核验另用 Poppler 逐页渲染，并与 PDFium/pypdf 文本提取交叉检查；编译成功本身不作为排版通过的依据。

静态检查验证全部数值来源、双语公式与标签、参考文献完整性和引用覆盖。书目原始记录保留在 `bibliography/`，缺失的来源字段不会凭印象补写。22 篇研究文献均在两版正文实际引用，未用无关条目凑数。

## 写作工具与待定项

本轮使用 ml-paper-writing、citation-management 和 PDF 检查技能。按引用技能要求，[Scientific Agent Skills](https://arxiv.org/abs/2609.00065) 的工具来源署名记在[结构说明](叙事与结构参考.md#工具来源)，不计入22篇研究文献。

作者署名、目标会议和最终架构插图尚未确定；这些不妨碍当前双栏工作稿的编译，但仍需在投稿前完成。
