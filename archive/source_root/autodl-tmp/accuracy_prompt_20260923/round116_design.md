# 第116轮预注册确认：Entmax原始门控

第115轮GraphMAE2原始entmax开发71.56%，同批grafted根71.36%，恰达固定0.20个百分点筛选线；等范数71.52，幅度71.28，Adam70.72，Direct71.24。这些是开发结果，不是已确认提升。

本确认需第115轮传输、几何v2、优化器审计全部通过后才能启动。保留原几何失败与posthoc修正规则；修正限于内部近并列概率，实际概率与门控重建容差不变，并新增最终查询argmax完全一致。

使用不变115训练驱动和source manifest，GraphMAE2，182100–182119，100步，13方法260条，phase=confirm。主比较原始Entmax k50减grafted Shampoo k50；同时报告等范数、仅幅度、Adam及Direct。固定10000次bootstrap seed20260923，分位数(.025,.975)，匹配同图/checkpoint/source/split及support/query。主比较下界>0才启动第二20支持种子；否则停止，不更改根方案。确认查询仍是长期复用同图集，不能消除跨轮选择偏差。
