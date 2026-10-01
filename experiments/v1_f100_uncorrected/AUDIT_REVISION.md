# 前向误差审计修订及原未通过项

240 次拟合全部完成后，原 `summarize.py` 遇到原始表示绝对误差 2.384185791015625e-6，超过事前 2e-6 阈值。原计划、拟合/审计源码、失败日志、全部状态和封存预测完整保留，没有重新拟合、改参数、改数据划分、改评分或统计规则。

独立 `audit_relative_forward_20261001.py` 仍按同一方程重建前向，补充标准混合误差检查：绝对误差除以 `2e-6 + 1e-6*abs(saved_embedding)`，要求比例 <1。这是执行后发生的尺度相关审计修订，不能包装成事前绝对阈值通过。

实际完整审计中，6/240 个单元超过原绝对阈值，最大误差 3.814697265625e-6；混合误差最大比例 0.375658。完整重建概率最大误差 2.1782777996637304e-6，也稍超原 2e-6 阈值，但两条前向重建路径均没有任何预测类别变化。保存状态的原型读出概率零误差重放、类别严格一致；五指标独立重算最大误差 3.33e-16。

`original_raw_forward_absolute_tolerance_passed`、`original_full_chain_replay_passed`、`original_all_numerical_checks_passed` 均保留为 false。超限明细在 `absolute_forward_failures.json`；原失败日志和状态在 `preserved_audit/`。不得称原严格数值验收全部通过。

源码独立执行时，用 `PRPROMPT_F100_SOURCE_ROOT` 指向被原计划冻结的源码根。原计划中的所有源码、checkpoint 和参考清单哈希依然必须通过。该限制随完整回传清单和中文结果总结交付。
