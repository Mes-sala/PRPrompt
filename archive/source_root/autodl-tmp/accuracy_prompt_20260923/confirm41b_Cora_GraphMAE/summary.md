# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.480|+0.000|+0.210|
|Cora|GraphMAE|graphprompt|20|64.270|-0.210|+0.000|
|Cora|GraphMAE|head_base_knn-0.9-clamp|20|67.370|+2.890|+3.100|
|Cora|GraphMAE|head_token_knn-0.5-clamp|20|74.010|+9.530|+9.740|
|Cora|GraphMAE|head_token_knn-0.9-clamp|20|74.740|+10.260|+10.470|
|Cora|GraphMAE|head_token_knn-0.9-free|20|74.720|+10.240|+10.450|
|Cora|GraphMAE|head_token_mean|20|73.760|+9.280|+9.490|
|Cora|GraphMAE|head_token_orig-0.5-clamp|20|74.150|+9.670|+9.880|
|Cora|GraphMAE|head_token_random-0.9-clamp|20|54.470|-10.010|-9.800|
|Cora|GraphMAE|ridge_teacher_plain|20|71.500|+7.020|+7.230|
