# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|GraphMAE2|frozen|20|40.110|+0.000|-0.450|
|CiteSeer|GraphMAE2|graphprompt|20|40.560|+0.450|+0.000|
|CiteSeer|GraphMAE2|rawprop2|20|59.800|+19.690|+19.240|
|CiteSeer|GraphMAE2|ridge_teacher|20|60.460|+20.350|+19.900|
|CiteSeer|GraphMAE2|token_kd10|20|54.270|+14.160|+13.710|
|CiteSeer|GraphMAE2|token_random_ridge10|20|56.080|+15.970|+15.520|
|CiteSeer|GraphMAE2|token_ridge10|20|58.380|+18.270|+17.820|
|CiteSeer|GraphMAE2|token_ridge_kd10|20|59.240|+19.130|+18.680|
|CiteSeer|GraphMAE2|token_ridge_kd10perm|20|24.840|-15.270|-15.720|
