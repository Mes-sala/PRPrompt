# 三底座统一协议对比与2026方法扩展

本轮为Cora、CiteSeer × DGI、Edgepred_Gprompt、Edgepred_GPPT，20个共同支持seed、每类5-shot、100次更新。主矩阵12方法1440结果，2026扩展2方法240结果，共1680结果、1560次拟合。F110和F111只改变读出，共享训练。

- 结果表：`统一协议多指标对比.md`、`统一协议多指标对比.csv`。
- 每seed原始指标：`all_metrics.json`；配对统计：`all_metric_pairs.json`。
- 机制、超参数与统一协议的适配边界：`机制与协议说明.md`。
- MTG、DAPrompt-fixed发表出处及复现边界：`extension2026/模型选择与复现边界.md`。
- 运行结果以两个`pipeline_status.json`、`final_audit.json`和`metrics_pipeline_status.json`为准；文档存在不代表所有结果已经完成。

## 指标

Accuracy、Macro-F1、Micro-F1、macro OVR ROC-AUC、macro AP。ROC/AP使用封存分数统一softmax；AP不是梯形PR面积。单标签多类任务中Micro-F1=Accuracy。指标在训练和预测封存后计算，未用于选择epoch、种子或超参数。

## 复核

服务器解释器为`/root/autodl-tmp/prog_env/bin/python`，实验目录为`/root/autodl-tmp/unified_three_backbone_20260924`。正式运行自动完成后：

```bash
python audit.py
(cd extension2026 && python audit.py)
python metrics.py
python analyze_metrics.py
```

本地复制后，在安装torch、numpy、scikit-learn的环境进行不依赖远端绝对路径的复核：

```bash
python archive_manifest.py --verify
python audit.py --transport
(cd extension2026 && python audit.py --transport)
python metrics.py --transport
```

`reproduction_assets/`保存六份底座权重、两份Planetoid数据和环境清单；`checkpoint_index.json`提供原路径与复制路径映射。迁移到新服务器复跑训练时，需显式映射`plan_draft.json`中的权重路径和`grid.py`的数据根目录，重新生成源码清单与plan；新运行置于独立目录，不覆盖本次封存结果。`upstream_commits.json`及扩展阶段的相应文件记录作者代码版本。`history/`保留工程修复前版本和日志，不能混入正式计数。

统计区间针对固定图、固定查询集下支持集抽样，并未把20seed当作20份独立图。三底座经过前轮筛选；统一100次更新不保证等FLOPs、等收敛程度或等超参数调优预算。这是冻结checkpoint上的统一协议对比，不能解释成各论文原表的完整复刻，尤其DAPrompt-fixed不包含其专用结构预训练及主干更新。

## 数值复核

每条结果另存规范float64概率`evaluation_probabilities.npy`及SHA；本地先核查其与重新Softmax的机器精度差异，再用同一份概率重算排序指标，以避免跨平台末位舍入改变并列关系。原1680条指标逐值不变，见[数值复核说明](数值复核说明.md)。
