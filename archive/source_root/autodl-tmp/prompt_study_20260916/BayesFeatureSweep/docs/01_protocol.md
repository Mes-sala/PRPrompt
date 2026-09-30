# 第五轮：词项 Prompt 有限超参与 checkpoint 扩展

在查看本轮 query 结果前锁定。此前 EBFP 四组均小幅正向，但统计修正不胜简化版，因此按用户最新要求作探索性重开，保留第三轮停止结论。

选 DGI、GraphMAE、GraphCL、GraphMAE2，覆盖不同预训练目标，不按本轮结果选 checkpoint。Cora/CiteSeer，各四个5-shot support，种子16092401–104。仍是历史图/query开发证据，不是独立确认。

完整经验贝叶斯方差扣除与不扣除噪声的简化版本，各自使用相同五点配置 (kappa, gain amplitude)：(5,.25)、(5,.5)、(5,1)、(1,.5)、(20,.5)。以类别平衡五折 support MSE 联合选择配置与 ridge lambda；每折重建词项权重，禁止查询标签选参。并列时先较小振幅，再较小kappa，再较大lambda。保存全部配置以及两个选参输出，不能以query最优替代选参输出。

控制：相同搜索预算的标签置乱和特征置乱（固定seed11），原图fixed、frozen、raw、diffused_raw、dual_raw、noop。标签仅从 support 进入 Prompt；全体预测封存后才评分。原骨干冻结，原文件不修改，记录新checkpoint源哈希与副本。

主判断：每个数据集平均 selected_full-fixed >=0.5pp，四checkpoint至少三个正向；selected_full 对 selected_simple、selected_label_shuffle、selected_feature_permute 均需正向。全部通过才支持完整机制继续；只胜fixed而不胜simple仅支持简化方向。报告逐checkpoint、逐种子与全部配置，置信区间仅描述support变化，不能解释为独立图泛化。不继续扩展本轮网格。
