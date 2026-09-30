# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|5|68.880|+nan|+nan|
|CiteSeer|Direct|directrect_mean|5|67.080|+nan|+nan|
|CiteSeer|GraphMAE|frozen|5|32.880|+0.000|-0.480|
|CiteSeer|GraphMAE|graphprompt|5|33.360|+0.480|+0.000|
|CiteSeer|GraphMAE|head_early_mean|5|56.880|+24.000|+23.520|
|CiteSeer|GraphMAE|head_early_qeps-50|5|58.840|+25.960|+25.480|
|CiteSeer|GraphMAE|head_tdboth_mean|5|63.880|+31.000|+30.520|
|CiteSeer|GraphMAE|head_tdboth_qeps-50|5|65.120|+32.240|+31.760|
|CiteSeer|GraphMAE|head_tdgate_mean|5|63.840|+30.960|+30.480|
|CiteSeer|GraphMAE|head_tdgate_qeps-50|5|64.600|+31.720|+31.240|
|CiteSeer|GraphMAE|head_tdkd_mean|5|57.560|+24.680|+24.200|
|CiteSeer|GraphMAE|head_tdkd_qeps-50|5|58.640|+25.760|+25.280|
|CiteSeer|GraphMAE|head_tdnone_mean|5|63.760|+30.880|+30.400|
|CiteSeer|GraphMAE|head_tdnone_qeps-50|5|64.120|+31.240|+30.760|
|CiteSeer|GraphMAE|head_tdrect_mean|5|53.760|+20.880|+20.400|
|CiteSeer|GraphMAE|head_tdrect_qeps-50|5|53.280|+20.400|+19.920|
|CiteSeer|GraphMAE2|frozen|5|40.440|+0.000|+0.280|
|CiteSeer|GraphMAE2|graphprompt|5|40.160|-0.280|+0.000|
|CiteSeer|GraphMAE2|head_early_mean|5|57.760|+17.320|+17.600|
|CiteSeer|GraphMAE2|head_early_qeps-50|5|60.120|+19.680|+19.960|
|CiteSeer|GraphMAE2|head_tdboth_mean|5|67.400|+26.960|+27.240|
|CiteSeer|GraphMAE2|head_tdboth_qeps-50|5|68.880|+28.440|+28.720|
|CiteSeer|GraphMAE2|head_tdgate_mean|5|67.200|+26.760|+27.040|
|CiteSeer|GraphMAE2|head_tdgate_qeps-50|5|68.800|+28.360|+28.640|
|CiteSeer|GraphMAE2|head_tdkd_mean|5|57.920|+17.480|+17.760|
|CiteSeer|GraphMAE2|head_tdkd_qeps-50|5|59.960|+19.520|+19.800|
|CiteSeer|GraphMAE2|head_tdnone_mean|5|67.200|+26.760|+27.040|
|CiteSeer|GraphMAE2|head_tdnone_qeps-50|5|69.080|+28.640|+28.920|
|CiteSeer|GraphMAE2|head_tdrect_mean|5|60.600|+20.160|+20.440|
|CiteSeer|GraphMAE2|head_tdrect_qeps-50|5|61.680|+21.240|+21.520|
