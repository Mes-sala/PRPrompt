# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.390|+0.000|+0.060|
|Cora|GraphMAE|graphprompt|20|64.330|-0.060|+0.000|
|Cora|GraphMAE|ridge_teacher|20|70.740|+6.350|+6.410|
|Cora|GraphMAE|ridge_teacher_bilateral|20|72.430|+8.040|+8.100|
|Cora|GraphMAE|ridge_teacher_plain|20|72.780|+8.390|+8.450|
|Cora|GraphMAE|token_ridge_kd10|20|73.890|+9.500|+9.560|
|Cora|GraphMAE|token_ridge_kd10bilateral|20|74.360|+9.970|+10.030|
|Cora|GraphMAE|token_ridge_kd10plain|20|74.420|+10.030|+10.090|
