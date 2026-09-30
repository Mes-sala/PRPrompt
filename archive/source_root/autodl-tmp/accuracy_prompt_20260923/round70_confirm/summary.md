# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|63.480|+0.000|+0.000|
|Cora|GraphMAE|graphprompt|20|63.480|-0.000|+0.000|
|Cora|GraphMAE|head_base_qeps-50|20|65.390|+1.910|+1.910|
|Cora|GraphMAE|head_postbase_mean|20|61.560|-1.920|-1.920|
|Cora|GraphMAE|head_postbase_qeps-50|20|60.740|-2.740|-2.740|
|Cora|GraphMAE|head_postscale2_mean|20|73.320|+9.840|+9.840|
|Cora|GraphMAE|head_postscale2_qeps-50|20|74.650|+11.170|+11.170|
|Cora|GraphMAE|head_posttoken_mean|20|72.980|+9.500|+9.500|
|Cora|GraphMAE|head_posttoken_qeps-50|20|74.310|+10.830|+10.830|
|Cora|GraphMAE|head_scale2_mean|20|75.020|+11.540|+11.540|
|Cora|GraphMAE|head_scale2_qeps-50|20|75.910|+12.430|+12.430|
|Cora|GraphMAE|head_token_mean|20|74.440|+10.960|+10.960|
|Cora|GraphMAE|head_token_qeps-50|20|75.770|+12.290|+12.290|
|Cora|GraphMAE2|frozen|20|60.320|+0.000|+1.950|
|Cora|GraphMAE2|graphprompt|20|58.370|-1.950|+0.000|
|Cora|GraphMAE2|head_base_qeps-50|20|41.140|-19.180|-17.230|
|Cora|GraphMAE2|head_postbase_mean|20|47.670|-12.650|-10.700|
|Cora|GraphMAE2|head_postbase_qeps-50|20|34.400|-25.920|-23.970|
|Cora|GraphMAE2|head_postscale2_mean|20|72.690|+12.370|+14.320|
|Cora|GraphMAE2|head_postscale2_qeps-50|20|73.810|+13.490|+15.440|
|Cora|GraphMAE2|head_posttoken_mean|20|72.540|+12.220|+14.170|
|Cora|GraphMAE2|head_posttoken_qeps-50|20|73.610|+13.290|+15.240|
|Cora|GraphMAE2|head_scale2_mean|20|73.350|+13.030|+14.980|
|Cora|GraphMAE2|head_scale2_qeps-50|20|74.920|+14.600|+16.550|
|Cora|GraphMAE2|head_token_mean|20|72.910|+12.590|+14.540|
|Cora|GraphMAE2|head_token_qeps-50|20|74.630|+14.310|+16.260|
