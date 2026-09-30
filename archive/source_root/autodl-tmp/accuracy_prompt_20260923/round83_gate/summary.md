# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|5|69.480|+nan|+nan|
|CiteSeer|GraphMAE|frozen|5|33.960|+0.000|+2.280|
|CiteSeer|GraphMAE|graphprompt|5|31.680|-2.280|+0.000|
|CiteSeer|GraphMAE|head_gad_mean|5|68.600|+34.640|+36.920|
|CiteSeer|GraphMAE|head_gad_qeps-50|5|69.240|+35.280|+37.560|
|CiteSeer|GraphMAE|head_gdi_mean|5|68.120|+34.160|+36.440|
|CiteSeer|GraphMAE|head_gdi_qeps-50|5|69.120|+35.160|+37.440|
|CiteSeer|GraphMAE|head_gsg_mean|5|68.000|+34.040|+36.320|
|CiteSeer|GraphMAE|head_gsg_qeps-50|5|69.120|+35.160|+37.440|
|CiteSeer|GraphMAE|head_gsh_mean|5|67.920|+33.960|+36.240|
|CiteSeer|GraphMAE|head_gsh_qeps-50|5|68.960|+35.000|+37.280|
|CiteSeer|GraphMAE|head_osh1_mean|5|41.240|+7.280|+9.560|
|CiteSeer|GraphMAE|head_osh1_qeps-50|5|24.520|-9.440|-7.160|
|CiteSeer|GraphMAE|head_tr1_mean|5|68.600|+34.640|+36.920|
|CiteSeer|GraphMAE|head_tr1_qeps-50|5|69.240|+35.280|+37.560|
|CiteSeer|GraphMAE2|frozen|5|40.360|+0.000|+1.400|
|CiteSeer|GraphMAE2|graphprompt|5|38.960|-1.400|+0.000|
|CiteSeer|GraphMAE2|head_gad_mean|5|68.040|+27.680|+29.080|
|CiteSeer|GraphMAE2|head_gad_qeps-50|5|68.840|+28.480|+29.880|
|CiteSeer|GraphMAE2|head_gdi_mean|5|68.680|+28.320|+29.720|
|CiteSeer|GraphMAE2|head_gdi_qeps-50|5|69.400|+29.040|+30.440|
|CiteSeer|GraphMAE2|head_gsg_mean|5|68.560|+28.200|+29.600|
|CiteSeer|GraphMAE2|head_gsg_qeps-50|5|69.400|+29.040|+30.440|
|CiteSeer|GraphMAE2|head_gsh_mean|5|68.720|+28.360|+29.760|
|CiteSeer|GraphMAE2|head_gsh_qeps-50|5|69.480|+29.120|+30.520|
|CiteSeer|GraphMAE2|head_osh1_mean|5|41.280|+0.920|+2.320|
|CiteSeer|GraphMAE2|head_osh1_qeps-50|5|24.640|-15.720|-14.320|
|CiteSeer|GraphMAE2|head_tr1_mean|5|68.040|+27.680|+29.080|
|CiteSeer|GraphMAE2|head_tr1_qeps-50|5|68.840|+28.480|+29.880|
