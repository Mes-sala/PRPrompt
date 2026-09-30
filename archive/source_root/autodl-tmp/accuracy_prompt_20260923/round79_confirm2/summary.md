# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|20|69.150|+nan|+nan|
|CiteSeer|GraphMAE|frozen|20|33.610|+0.000|+0.540|
|CiteSeer|GraphMAE|graphprompt|20|33.070|-0.540|+0.000|
|CiteSeer|GraphMAE|head_inr10_mean|20|66.610|+33.000|+33.540|
|CiteSeer|GraphMAE|head_inr10_qeps-50|20|67.990|+34.380|+34.920|
|CiteSeer|GraphMAE|head_posttdboth_mean|20|66.270|+32.660|+33.200|
|CiteSeer|GraphMAE|head_posttdboth_qeps-50|20|68.110|+34.500|+35.040|
|CiteSeer|GraphMAE|head_posttdnone_mean|20|66.350|+32.740|+33.280|
|CiteSeer|GraphMAE|head_posttdnone_qeps-50|20|68.040|+34.430|+34.970|
|CiteSeer|GraphMAE|head_ta1_mean|20|67.670|+34.060|+34.600|
|CiteSeer|GraphMAE|head_ta1_qeps-50|20|68.770|+35.160|+35.700|
|CiteSeer|GraphMAE|head_tdboth_mean|20|61.270|+27.660|+28.200|
|CiteSeer|GraphMAE|head_tdboth_qeps-50|20|62.780|+29.170|+29.710|
|CiteSeer|GraphMAE|head_tdnone_mean|20|61.050|+27.440|+27.980|
|CiteSeer|GraphMAE|head_tdnone_qeps-50|20|62.420|+28.810|+29.350|
|CiteSeer|GraphMAE|head_tm1_mean|20|68.000|+34.390|+34.930|
|CiteSeer|GraphMAE|head_tm1_qeps-50|20|68.790|+35.180|+35.720|
|CiteSeer|GraphMAE|head_tr1_mean|20|68.380|+34.770|+35.310|
|CiteSeer|GraphMAE|head_tr1_qeps-50|20|69.180|+35.570|+36.110|
|CiteSeer|GraphMAE2|frozen|20|39.590|+0.000|+0.450|
|CiteSeer|GraphMAE2|graphprompt|20|39.140|-0.450|+0.000|
|CiteSeer|GraphMAE2|head_inr10_mean|20|67.320|+27.730|+28.180|
|CiteSeer|GraphMAE2|head_inr10_qeps-50|20|68.330|+28.740|+29.190|
|CiteSeer|GraphMAE2|head_posttdboth_mean|20|67.590|+28.000|+28.450|
|CiteSeer|GraphMAE2|head_posttdboth_qeps-50|20|68.580|+28.990|+29.440|
|CiteSeer|GraphMAE2|head_posttdnone_mean|20|67.440|+27.850|+28.300|
|CiteSeer|GraphMAE2|head_posttdnone_qeps-50|20|68.560|+28.970|+29.420|
|CiteSeer|GraphMAE2|head_ta1_mean|20|67.740|+28.150|+28.600|
|CiteSeer|GraphMAE2|head_ta1_qeps-50|20|68.810|+29.220|+29.670|
|CiteSeer|GraphMAE2|head_tdboth_mean|20|65.680|+26.090|+26.540|
|CiteSeer|GraphMAE2|head_tdboth_qeps-50|20|67.550|+27.960|+28.410|
|CiteSeer|GraphMAE2|head_tdnone_mean|20|65.260|+25.670|+26.120|
|CiteSeer|GraphMAE2|head_tdnone_qeps-50|20|66.850|+27.260|+27.710|
|CiteSeer|GraphMAE2|head_tm1_mean|20|68.230|+28.640|+29.090|
|CiteSeer|GraphMAE2|head_tm1_qeps-50|20|69.080|+29.490|+29.940|
|CiteSeer|GraphMAE2|head_tr1_mean|20|68.330|+28.740|+29.190|
|CiteSeer|GraphMAE2|head_tr1_qeps-50|20|69.180|+29.590|+30.040|
