# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.350|+0.000|+0.220|
|Cora|GraphMAE|graphprompt|20|64.130|-0.220|+0.000|
|Cora|GraphMAE|rawprop2|20|68.670|+4.320|+4.540|
|Cora|GraphMAE|ridge_teacher|20|69.920|+5.570|+5.790|
|Cora|GraphMAE|token_kd10|20|71.240|+6.890|+7.110|
|Cora|GraphMAE|token_random_ridge10|20|71.070|+6.720|+6.940|
|Cora|GraphMAE|token_ridge10|20|72.670|+8.320|+8.540|
|Cora|GraphMAE|token_ridge_kd10|20|72.960|+8.610|+8.830|
|Cora|GraphMAE|token_ridge_kd10perm|20|51.270|-13.080|-12.860|
