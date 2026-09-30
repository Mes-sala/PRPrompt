# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|20|69.490|+nan|+nan|
|CiteSeer|GraphMAE|frozen|20|32.760|+0.000|+0.480|
|CiteSeer|GraphMAE|graphprompt|20|32.280|-0.480|+0.000|
|CiteSeer|GraphMAE|head_inr10_mean|20|67.520|+34.760|+35.240|
|CiteSeer|GraphMAE|head_inr10_qeps-50|20|68.590|+35.830|+36.310|
|CiteSeer|GraphMAE|head_posttdboth_mean|20|66.750|+33.990|+34.470|
|CiteSeer|GraphMAE|head_posttdboth_qeps-50|20|68.540|+35.780|+36.260|
|CiteSeer|GraphMAE|head_posttdnone_mean|20|66.890|+34.130|+34.610|
|CiteSeer|GraphMAE|head_posttdnone_qeps-50|20|68.740|+35.980|+36.460|
|CiteSeer|GraphMAE|head_ta1_mean|20|68.190|+35.430|+35.910|
|CiteSeer|GraphMAE|head_ta1_qeps-50|20|69.690|+36.930|+37.410|
|CiteSeer|GraphMAE|head_tdboth_mean|20|64.600|+31.840|+32.320|
|CiteSeer|GraphMAE|head_tdboth_qeps-50|20|66.010|+33.250|+33.730|
|CiteSeer|GraphMAE|head_tdnone_mean|20|64.290|+31.530|+32.010|
|CiteSeer|GraphMAE|head_tdnone_qeps-50|20|65.610|+32.850|+33.330|
|CiteSeer|GraphMAE|head_tm1_mean|20|68.130|+35.370|+35.850|
|CiteSeer|GraphMAE|head_tm1_qeps-50|20|69.510|+36.750|+37.230|
|CiteSeer|GraphMAE|head_tr1_mean|20|69.040|+36.280|+36.760|
|CiteSeer|GraphMAE|head_tr1_qeps-50|20|69.840|+37.080|+37.560|
|CiteSeer|GraphMAE2|frozen|20|38.900|+0.000|-0.040|
|CiteSeer|GraphMAE2|graphprompt|20|38.940|+0.040|+0.000|
|CiteSeer|GraphMAE2|head_inr10_mean|20|68.060|+29.160|+29.120|
|CiteSeer|GraphMAE2|head_inr10_qeps-50|20|69.250|+30.350|+30.310|
|CiteSeer|GraphMAE2|head_posttdboth_mean|20|67.650|+28.750|+28.710|
|CiteSeer|GraphMAE2|head_posttdboth_qeps-50|20|69.040|+30.140|+30.100|
|CiteSeer|GraphMAE2|head_posttdnone_mean|20|67.670|+28.770|+28.730|
|CiteSeer|GraphMAE2|head_posttdnone_qeps-50|20|68.920|+30.020|+29.980|
|CiteSeer|GraphMAE2|head_ta1_mean|20|67.790|+28.890|+28.850|
|CiteSeer|GraphMAE2|head_ta1_qeps-50|20|68.990|+30.090|+30.050|
|CiteSeer|GraphMAE2|head_tdboth_mean|20|66.680|+27.780|+27.740|
|CiteSeer|GraphMAE2|head_tdboth_qeps-50|20|68.080|+29.180|+29.140|
|CiteSeer|GraphMAE2|head_tdnone_mean|20|66.310|+27.410|+27.370|
|CiteSeer|GraphMAE2|head_tdnone_qeps-50|20|67.640|+28.740|+28.700|
|CiteSeer|GraphMAE2|head_tm1_mean|20|68.590|+29.690|+29.650|
|CiteSeer|GraphMAE2|head_tm1_qeps-50|20|69.280|+30.380|+30.340|
|CiteSeer|GraphMAE2|head_tr1_mean|20|68.710|+29.810|+29.770|
|CiteSeer|GraphMAE2|head_tr1_qeps-50|20|69.400|+30.500|+30.460|
