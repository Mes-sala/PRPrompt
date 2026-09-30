# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.100|+0.000|-0.000|
|Cora|GraphMAE|graphprompt|20|64.100|+0.000|+0.000|
|Cora|GraphMAE|head_token_knn-0.9-clamp|20|75.650|+11.550|+11.550|
|Cora|GraphMAE|head_token_mean|20|74.680|+10.580|+10.580|
|Cora|GraphMAE|head_token_qefixed-50|20|75.340|+11.240|+11.240|
|Cora|GraphMAE|head_token_qeprob-50|20|75.490|+11.390|+11.390|
|Cora|GraphMAE|head_token_qeps-50|20|76.250|+12.150|+12.150|
|Cora|GraphMAE|head_token_qepu-50|20|76.210|+12.110|+12.110|
|Cora|GraphMAE|head_token_qeuniform-50|20|75.620|+11.520|+11.520|
|Cora|GraphMAE2|frozen|20|58.990|+0.000|+2.110|
|Cora|GraphMAE2|graphprompt|20|56.880|-2.110|+0.000|
|Cora|GraphMAE2|head_token_knn-0.9-clamp|20|73.500|+14.510|+16.620|
|Cora|GraphMAE2|head_token_mean|20|72.310|+13.320|+15.430|
|Cora|GraphMAE2|head_token_qefixed-50|20|73.390|+14.400|+16.510|
|Cora|GraphMAE2|head_token_qeprob-50|20|73.420|+14.430|+16.540|
|Cora|GraphMAE2|head_token_qeps-50|20|74.080|+15.090|+17.200|
|Cora|GraphMAE2|head_token_qepu-50|20|74.060|+15.070|+17.180|
|Cora|GraphMAE2|head_token_qeuniform-50|20|73.870|+14.880|+16.990|
