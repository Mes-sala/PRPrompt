# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|GraphMAE|frozen|20|34.110|+0.000|+0.240|
|CiteSeer|GraphMAE|graphprompt|20|33.870|-0.240|+0.000|
|CiteSeer|GraphMAE|rawprop2|20|61.840|+27.730|+27.970|
|CiteSeer|GraphMAE|token3|20|44.080|+9.970|+10.210|
|CiteSeer|GraphMAE|token_kd3|20|44.240|+10.130|+10.370|
|CiteSeer|GraphMAE|token_kd3perm|20|32.710|-1.400|-1.160|
|CiteSeer|GraphMAE|token_random3|20|35.410|+1.300|+1.540|
