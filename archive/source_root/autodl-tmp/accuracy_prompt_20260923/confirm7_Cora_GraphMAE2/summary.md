# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE2|frozen|20|60.050|+0.000|+1.960|
|Cora|GraphMAE2|graphprompt|20|58.090|-1.960|+0.000|
|Cora|GraphMAE2|rawprop2|20|68.670|+8.620|+10.580|
|Cora|GraphMAE2|ridge_teacher|20|69.920|+9.870|+11.830|
|Cora|GraphMAE2|token_kd10|20|68.280|+8.230|+10.190|
|Cora|GraphMAE2|token_random_ridge10|20|69.260|+9.210|+11.170|
|Cora|GraphMAE2|token_ridge10|20|70.950|+10.900|+12.860|
|Cora|GraphMAE2|token_ridge_kd10|20|71.930|+11.880|+13.840|
|Cora|GraphMAE2|token_ridge_kd10perm|20|30.460|-29.590|-27.630|
