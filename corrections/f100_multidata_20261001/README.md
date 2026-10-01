# 多数据集 F100 验收失败的保存与交付

本次不修改训练、checkpoint、支持抽样、选参、封存预测或统计口径。正式拟合源码保持 Git ``，计划 SHA256 `ab3489b8ce1955a922dfe05e427b9ca2685cfd4b4952e2cc47622c8fe3647350`。

5400 次正式拟合全部完成。原审计在 Actor / Edgepred_GPPT 的 F100 前向混合误差断言处停止。其余九个 checkpoint 使用原审计通过；该条件全部 540 个方法单元由补充采集器检查，60 个 F100 单元中有 7 个未通过原混合误差界限。原数值验收为 False，不能说全部测试通过。

## 原代码与证据

- 原审计 `experiments/f100_multidata/audit.py`、原训练及历史 AST 文件未改动；冻结副本与原 plan、状态、失败日志均在完整本地回传中。
- 被修改的本地交付审计程序原文件完整保存在 [original/tools/audit_f100_multidata_delivery.py](original/tools/audit_f100_multidata_delivery.py)。修改只让原数值验收失败明确传播到交付状态，未改 SHA、评分、汇总或配对算法。
- `audit_collect_20261001.py` 只将原审计中单个 `ratio < 1` 断言延期到记录完成，其余断言继续执行，原阈值不变，超限测试仍为 False。
- `diagnose_roundoff_20261001.py` 对预先固定的第一个失败单元在两张 GPU 各重放 10 次，无训练、无 query 标签、无挑选重放结果；全部观测保留。
- `finalize_qualified_20261001.py` 使用原汇总和配对算法，合并九份原审计及一份明确带失败标志的采集结果，生成含数值验收例外的完整报告及交付清单。

固定权重和提示的重复前向出现最大约 $1.24\times10^{-5}$ 的原始表示抖动，与 CUDA float32 稀疏归约差异相符，但这项诊断没有将原失败改判通过。Actor/GPPT 封存表示读出重放误差为零；固定路由重放在全部60个F100单元上无类别变化。其他条件少量完整新前向类别变化也在正式报告中保留。

## 本地交付

接收目录为 `experiments/outputs/F100_multidata_5090_20261001/`。48325 个文件 SHA256 全部匹配，5400 个单元的五指标、90 行方法汇总和400行五指标配对区间均独立复核。`local_delivery_status.json` 为 `delivery_complete_with_numerical_qualification`，原数值验收为 False。

同步过滤规则曾把源码内部名为 `data` 的目录一并排除，导致162个源码/清单文件缺失。原同步脚本与失败日志已保存，规则改为仅排除根目录 `/data/`，补传不可变源码后完整 SHA 复核通过；没有删除远端数据或启动重复同步循环。

完整源代码、状态、预测、权重、图/划分快照、环境与失败日志保留在接收目录；Git 仅归档紧凑代码、中文结果和 JSON 证据。
