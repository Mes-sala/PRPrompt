# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.330|+0.000|-0.130|
|Cora|GraphMAE|graphprompt|20|64.460|+0.130|+0.000|
|Cora|GraphMAE|rawprop2|20|68.660|+4.330|+4.200|
|Cora|GraphMAE|token3|20|67.430|+3.100|+2.970|
|Cora|GraphMAE|token_kd3|20|67.920|+3.590|+3.460|
|Cora|GraphMAE|token_kd3perm|20|63.540|-0.790|-0.920|
|Cora|GraphMAE|token_random3|20|65.230|+0.900|+0.770|
