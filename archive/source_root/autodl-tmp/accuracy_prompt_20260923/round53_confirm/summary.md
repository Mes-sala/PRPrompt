# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|63.910|+0.000|-0.230|
|Cora|GraphMAE|graphprompt|20|64.140|+0.230|+0.000|
|Cora|GraphMAE|head_focal2_mean|20|74.020|+10.110|+9.880|
|Cora|GraphMAE|head_focal2_qeps-50|20|75.670|+11.760|+11.530|
|Cora|GraphMAE|head_perm2_mean|20|74.010|+10.100|+9.870|
|Cora|GraphMAE|head_perm2_qeps-50|20|75.630|+11.720|+11.490|
|Cora|GraphMAE|head_scale2_mean|20|74.040|+10.130|+9.900|
|Cora|GraphMAE|head_scale2_qeps-50|20|75.680|+11.770|+11.540|
|Cora|GraphMAE|head_token_mean|20|73.460|+9.550|+9.320|
|Cora|GraphMAE|head_token_qeps-50|20|75.000|+11.090|+10.860|
|Cora|GraphMAE2|frozen|20|59.530|+0.000|+2.200|
|Cora|GraphMAE2|graphprompt|20|57.330|-2.200|+0.000|
|Cora|GraphMAE2|head_focal2_mean|20|71.190|+11.660|+13.860|
|Cora|GraphMAE2|head_focal2_qeps-50|20|72.930|+13.400|+15.600|
|Cora|GraphMAE2|head_perm2_mean|20|71.090|+11.560|+13.760|
|Cora|GraphMAE2|head_perm2_qeps-50|20|73.010|+13.480|+15.680|
|Cora|GraphMAE2|head_scale2_mean|20|71.140|+11.610|+13.810|
|Cora|GraphMAE2|head_scale2_qeps-50|20|73.010|+13.480|+15.680|
|Cora|GraphMAE2|head_token_mean|20|70.770|+11.240|+13.440|
|Cora|GraphMAE2|head_token_qeps-50|20|72.810|+13.280|+15.480|
