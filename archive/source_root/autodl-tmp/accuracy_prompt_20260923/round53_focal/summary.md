# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|5|61.480|+0.000|+0.040|
|Cora|GraphMAE|graphprompt|5|61.440|-0.040|+0.000|
|Cora|GraphMAE|head_focal1_mean|5|75.800|+14.320|+14.360|
|Cora|GraphMAE|head_focal1_qeps-50|5|75.440|+13.960|+14.000|
|Cora|GraphMAE|head_focal2_mean|5|75.920|+14.440|+14.480|
|Cora|GraphMAE|head_focal2_qeps-50|5|75.400|+13.920|+13.960|
|Cora|GraphMAE|head_perm1_mean|5|75.840|+14.360|+14.400|
|Cora|GraphMAE|head_perm1_qeps-50|5|75.240|+13.760|+13.800|
|Cora|GraphMAE|head_perm2_mean|5|75.960|+14.480|+14.520|
|Cora|GraphMAE|head_perm2_qeps-50|5|75.440|+13.960|+14.000|
|Cora|GraphMAE|head_scale1_mean|5|75.840|+14.360|+14.400|
|Cora|GraphMAE|head_scale1_qeps-50|5|75.160|+13.680|+13.720|
|Cora|GraphMAE|head_scale2_mean|5|76.000|+14.520|+14.560|
|Cora|GraphMAE|head_scale2_qeps-50|5|75.480|+14.000|+14.040|
|Cora|GraphMAE|head_token_mean|5|75.480|+14.000|+14.040|
|Cora|GraphMAE|head_token_qeps-50|5|75.400|+13.920|+13.960|
|Cora|GraphMAE2|frozen|5|57.960|+0.000|+2.320|
|Cora|GraphMAE2|graphprompt|5|55.640|-2.320|+0.000|
|Cora|GraphMAE2|head_focal1_mean|5|73.760|+15.800|+18.120|
|Cora|GraphMAE2|head_focal1_qeps-50|5|74.480|+16.520|+18.840|
|Cora|GraphMAE2|head_focal2_mean|5|73.960|+16.000|+18.320|
|Cora|GraphMAE2|head_focal2_qeps-50|5|74.520|+16.560|+18.880|
|Cora|GraphMAE2|head_perm1_mean|5|73.640|+15.680|+18.000|
|Cora|GraphMAE2|head_perm1_qeps-50|5|74.280|+16.320|+18.640|
|Cora|GraphMAE2|head_perm2_mean|5|73.920|+15.960|+18.280|
|Cora|GraphMAE2|head_perm2_qeps-50|5|74.600|+16.640|+18.960|
|Cora|GraphMAE2|head_scale1_mean|5|73.560|+15.600|+17.920|
|Cora|GraphMAE2|head_scale1_qeps-50|5|74.400|+16.440|+18.760|
|Cora|GraphMAE2|head_scale2_mean|5|73.840|+15.880|+18.200|
|Cora|GraphMAE2|head_scale2_qeps-50|5|74.600|+16.640|+18.960|
|Cora|GraphMAE2|head_token_mean|5|73.280|+15.320|+17.640|
|Cora|GraphMAE2|head_token_qeps-50|5|74.400|+16.440|+18.760|
