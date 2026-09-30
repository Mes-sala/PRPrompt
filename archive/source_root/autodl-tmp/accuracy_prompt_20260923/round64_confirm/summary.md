# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|20|69.100|+nan|+nan|
|CiteSeer|Direct|directrect_mean|20|67.990|+nan|+nan|
|CiteSeer|GraphMAE|frozen|20|34.200|+0.000|-0.440|
|CiteSeer|GraphMAE|graphprompt|20|34.640|+0.440|+0.000|
|CiteSeer|GraphMAE|head_early_mean|20|57.500|+23.300|+22.860|
|CiteSeer|GraphMAE|head_early_qeps-50|20|57.990|+23.790|+23.350|
|CiteSeer|GraphMAE|head_tdboth_mean|20|63.440|+29.240|+28.800|
|CiteSeer|GraphMAE|head_tdboth_qeps-50|20|65.490|+31.290|+30.850|
|CiteSeer|GraphMAE|head_tdgate_mean|20|63.210|+29.010|+28.570|
|CiteSeer|GraphMAE|head_tdgate_qeps-50|20|65.580|+31.380|+30.940|
|CiteSeer|GraphMAE|head_tdkd_mean|20|57.920|+23.720|+23.280|
|CiteSeer|GraphMAE|head_tdkd_qeps-50|20|57.900|+23.700|+23.260|
|CiteSeer|GraphMAE|head_tdnone_mean|20|63.280|+29.080|+28.640|
|CiteSeer|GraphMAE|head_tdnone_qeps-50|20|65.180|+30.980|+30.540|
|CiteSeer|GraphMAE|head_tdrect_mean|20|54.030|+19.830|+19.390|
|CiteSeer|GraphMAE|head_tdrect_qeps-50|20|54.100|+19.900|+19.460|
|CiteSeer|GraphMAE2|frozen|20|40.710|+0.000|+0.620|
|CiteSeer|GraphMAE2|graphprompt|20|40.090|-0.620|+0.000|
|CiteSeer|GraphMAE2|head_early_mean|20|59.620|+18.910|+19.530|
|CiteSeer|GraphMAE2|head_early_qeps-50|20|61.060|+20.350|+20.970|
|CiteSeer|GraphMAE2|head_tdboth_mean|20|65.950|+25.240|+25.860|
|CiteSeer|GraphMAE2|head_tdboth_qeps-50|20|67.270|+26.560|+27.180|
|CiteSeer|GraphMAE2|head_tdgate_mean|20|66.060|+25.350|+25.970|
|CiteSeer|GraphMAE2|head_tdgate_qeps-50|20|67.330|+26.620|+27.240|
|CiteSeer|GraphMAE2|head_tdkd_mean|20|59.800|+19.090|+19.710|
|CiteSeer|GraphMAE2|head_tdkd_qeps-50|20|61.090|+20.380|+21.000|
|CiteSeer|GraphMAE2|head_tdnone_mean|20|65.990|+25.280|+25.900|
|CiteSeer|GraphMAE2|head_tdnone_qeps-50|20|67.490|+26.780|+27.400|
|CiteSeer|GraphMAE2|head_tdrect_mean|20|61.290|+20.580|+21.200|
|CiteSeer|GraphMAE2|head_tdrect_qeps-50|20|63.140|+22.430|+23.050|
