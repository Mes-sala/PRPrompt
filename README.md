# PRPrompt：面向少样本节点分类的原型路由图提示

**Prototype-Routed Graph Prompting for Few-Shot Node Classification**

当前版本 **v2**：用支持集原型关系固定节点的提示路由，学习少量类别提示向量，在冻结 GCN 的隐藏计算中完成任务适配。使用普通 Adam 与线性分类头。

当前使用原始 Edgepred 权重。7 个数据集的原始 checkpoint 按固定 SHA256 加载。

## 先读这些

- [Prompt 核心叙事与方法](docs/v2/02_Prompt核心叙事.md)：摘要、引言、方法公式和主张边界。
- [本地论文阅读与叙事判断](docs/v2/01_论文阅读与叙事判断.md)：9 篇原文的“问题—设计—证据”比较。
- [实验计划与执行命令](docs/v2/03_实验计划与执行.md)：完整方案对预训练冻结读出、微调及已有提示算子的比较。
- [版本变化](docs/v2/04_版本裁剪与证据迁移.md)：具体删改及历史结果的使用边界。
- [程序验证记录](docs/validation/v2_validation.md)。
- [Cora/CiteSeer 的原始权重 F100 结果](docs/validation/未修正Edgepred_F100验证_20261001.md)。
- [五个新增数据集的 F100 结果](docs/validation/五数据集_F100验证_20261001.md)。
- [F100、v2 与历史后续实验完整汇总](docs/results/后续实验汇总_20261002/后续实验汇总.md)：10,320个已完成正式单元，14个Edgepred条件，完整五指标、配对差值及负结果。
- [进行中的软分配、分类头与 1/3/5-shot 实验](experiments/boundary_20261002/PLAN.md)：仅原 Edgepred_Gprompt，七数据集完整配对；完成后再决定论文方案。
- [原F100配方的配对路由控制](experiments/f100_boundary_20261002/PLAN.md)：独立源码与计划，等待主实验完整审计后使用GPU；[执行记录](docs/validation/F100路由边界_20261002_运行记录.md)。

## 当前实现

```text
prprompt/                    提示、模型、支持内选参与预测封存
experiments/configs/         主比较、shot 曲线、提示机制诊断
experiments/prepare_original_edgepred.py  原始权重哈希/架构校验与标准路径解析
experiments/run_benchmark.py     CPU/CUDA 实验运行与续跑
experiments/summarize.py         覆盖审计、完整方法表、配对统计
experiments/tests/              冻结、标签隔离、梯度、协议与完整性测试
docs/v2/                       当前论文叙事与实验计划
```

复用已有 **ProG conda 环境**，依赖范围见 `pyproject.toml`。在仓库根目录先运行：

```bash
python experiments/tests/test_protocol.py
python experiments/tests/test_pipeline.py
```

然后按照[执行文档](docs/v2/03_实验计划与执行.md)准备本机 checkpoint 清单，执行 `--dry-run` 和 `--smoke`，最后运行正式配置。完整数据、权重、预测和本机路径清单均不进入 Git。

目前支持严格加载两层 ProG GCN checkpoint。当前主表有 7 个完整方法，聚焦预训练冻结读出、完整微调和 Prompt，采用同一支持/查询划分；1-shot 不使用额外验证标签，其余条件仅在支持集内选参。模型拟合只接收支持标签，查询概率封存之后才评分。GPF/GPFPlus 与 GraphPromptDiag 是明确标注的统一节点协议实现，不能把表格描述成全部原论文最佳配置复现。

## 历史与来源

`runtime/`、`archive/`、`local/` 保留先前的代码/文档来源快照，默认新入口是上面的 `prprompt/` 与 `experiments/`。[历史归档导航](docs/历史归档导航.md)保留原资料入口。历史数据检查仍可执行 `python tools/verify_repository.py`。

第三方源代码沿用原版权和许可证，例如 [ProG 许可证](archive/source_root/ProG/MIT-LICENSE.txt)。本仓库没有为混合来源历史内容重新指定统一许可证。
