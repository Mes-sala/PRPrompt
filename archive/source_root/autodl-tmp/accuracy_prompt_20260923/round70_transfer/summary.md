# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|5|61.200|+0.000|-0.240|
|Cora|GraphMAE|graphprompt|5|61.440|+0.240|+0.000|
|Cora|GraphMAE|head_base_qeps-50|5|62.480|+1.280|+1.040|
|Cora|GraphMAE|head_postbase_mean|5|58.880|-2.320|-2.560|
|Cora|GraphMAE|head_postbase_qeps-50|5|59.560|-1.640|-1.880|
|Cora|GraphMAE|head_postscale2_mean|5|73.640|+12.440|+12.200|
|Cora|GraphMAE|head_postscale2_qeps-50|5|75.120|+13.920|+13.680|
|Cora|GraphMAE|head_posttoken_mean|5|73.640|+12.440|+12.200|
|Cora|GraphMAE|head_posttoken_qeps-50|5|74.960|+13.760|+13.520|
|Cora|GraphMAE|head_scale2_mean|5|74.040|+12.840|+12.600|
|Cora|GraphMAE|head_scale2_qeps-50|5|74.360|+13.160|+12.920|
|Cora|GraphMAE|head_token_mean|5|73.560|+12.360|+12.120|
|Cora|GraphMAE|head_token_qeps-50|5|74.160|+12.960|+12.720|
|Cora|GraphMAE2|frozen|5|55.760|+0.000|+2.240|
|Cora|GraphMAE2|graphprompt|5|53.520|-2.240|+0.000|
|Cora|GraphMAE2|head_base_qeps-50|5|41.200|-14.560|-12.320|
|Cora|GraphMAE2|head_postbase_mean|5|41.840|-13.920|-11.680|
|Cora|GraphMAE2|head_postbase_qeps-50|5|28.480|-27.280|-25.040|
|Cora|GraphMAE2|head_postscale2_mean|5|72.160|+16.400|+18.640|
|Cora|GraphMAE2|head_postscale2_qeps-50|5|72.560|+16.800|+19.040|
|Cora|GraphMAE2|head_posttoken_mean|5|72.360|+16.600|+18.840|
|Cora|GraphMAE2|head_posttoken_qeps-50|5|73.040|+17.280|+19.520|
|Cora|GraphMAE2|head_scale2_mean|5|72.560|+16.800|+19.040|
|Cora|GraphMAE2|head_scale2_qeps-50|5|72.160|+16.400|+18.640|
|Cora|GraphMAE2|head_token_mean|5|71.960|+16.200|+18.440|
|Cora|GraphMAE2|head_token_qeps-50|5|72.200|+16.440|+18.680|
