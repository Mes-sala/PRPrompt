# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.340|+0.000|-0.100|
|Cora|GraphMAE|graphprompt|20|64.440|+0.100|+0.000|
|Cora|GraphMAE|ridge_teacher|20|70.940|+6.600|+6.500|
|Cora|GraphMAE|ridge_teacher_plain|20|73.340|+9.000|+8.900|
|Cora|GraphMAE|token_ridge_kd10|20|73.910|+9.570|+9.470|
|Cora|GraphMAE|token_ridge_kd10late|20|75.170|+10.830|+10.730|
|Cora|GraphMAE|token_ridge_kd10plain|20|74.590|+10.250|+10.150|
|Cora|GraphMAE|token_ridge_kd10plainlate|20|75.780|+11.440|+11.340|
|Cora|GraphMAE2|frozen|20|61.210|+0.000|+1.850|
|Cora|GraphMAE2|graphprompt|20|59.360|-1.850|+0.000|
|Cora|GraphMAE2|ridge_teacher|20|70.940|+9.730|+11.580|
|Cora|GraphMAE2|ridge_teacher_plain|20|73.340|+12.130|+13.980|
|Cora|GraphMAE2|token_ridge_kd10|20|73.300|+12.090|+13.940|
|Cora|GraphMAE2|token_ridge_kd10late|20|73.610|+12.400|+14.250|
|Cora|GraphMAE2|token_ridge_kd10plain|20|73.830|+12.620|+14.470|
|Cora|GraphMAE2|token_ridge_kd10plainlate|20|73.820|+12.610|+14.460|
