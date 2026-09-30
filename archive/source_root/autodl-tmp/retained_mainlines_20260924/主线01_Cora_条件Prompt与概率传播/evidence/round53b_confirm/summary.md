# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|64.130|+0.000|+0.180|
|Cora|GraphMAE|graphprompt|20|63.950|-0.180|+0.000|
|Cora|GraphMAE|head_focal2_mean|20|75.650|+11.520|+11.700|
|Cora|GraphMAE|head_focal2_qeps-50|20|76.530|+12.400|+12.580|
|Cora|GraphMAE|head_perm2_mean|20|75.260|+11.130|+11.310|
|Cora|GraphMAE|head_perm2_qeps-50|20|76.530|+12.400|+12.580|
|Cora|GraphMAE|head_scale2_mean|20|75.310|+11.180|+11.360|
|Cora|GraphMAE|head_scale2_qeps-50|20|76.480|+12.350|+12.530|
|Cora|GraphMAE|head_token_mean|20|74.610|+10.480|+10.660|
|Cora|GraphMAE|head_token_qeps-50|20|75.520|+11.390|+11.570|
