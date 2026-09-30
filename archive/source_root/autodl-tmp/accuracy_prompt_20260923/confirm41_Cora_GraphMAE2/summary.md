# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE2|frozen|20|59.270|+0.000|+2.170|
|Cora|GraphMAE2|graphprompt|20|57.100|-2.170|+0.000|
|Cora|GraphMAE2|head_base_knn-0.9-clamp|20|53.860|-5.410|-3.240|
|Cora|GraphMAE2|head_token_knn-0.5-clamp|20|72.740|+13.470|+15.640|
|Cora|GraphMAE2|head_token_knn-0.9-clamp|20|73.100|+13.830|+16.000|
|Cora|GraphMAE2|head_token_knn-0.9-free|20|73.080|+13.810|+15.980|
|Cora|GraphMAE2|head_token_mean|20|72.670|+13.400|+15.570|
|Cora|GraphMAE2|head_token_orig-0.5-clamp|20|72.660|+13.390|+15.560|
|Cora|GraphMAE2|head_token_random-0.9-clamp|20|54.270|-5.000|-2.830|
|Cora|GraphMAE2|ridge_teacher_plain|20|71.820|+12.550|+14.720|
