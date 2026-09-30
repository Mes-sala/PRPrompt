# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|20|68.910|+nan|+nan|
|CiteSeer|GraphMAE|frozen|20|34.240|+0.000|+0.850|
|CiteSeer|GraphMAE|graphprompt|20|33.390|-0.850|+0.000|
|CiteSeer|GraphMAE|head_posttdboth_mean|20|66.600|+32.360|+33.210|
|CiteSeer|GraphMAE|head_posttdboth_qeps-50|20|68.170|+33.930|+34.780|
|CiteSeer|GraphMAE|head_posttdnone_mean|20|66.690|+32.450|+33.300|
|CiteSeer|GraphMAE|head_posttdnone_qeps-50|20|68.260|+34.020|+34.870|
|CiteSeer|GraphMAE|head_tdboth_mean|20|63.990|+29.750|+30.600|
|CiteSeer|GraphMAE|head_tdboth_qeps-50|20|65.130|+30.890|+31.740|
|CiteSeer|GraphMAE|head_tdgate_mean|20|63.430|+29.190|+30.040|
|CiteSeer|GraphMAE|head_tdgate_qeps-50|20|64.860|+30.620|+31.470|
|CiteSeer|GraphMAE|head_tdnone_mean|20|63.420|+29.180|+30.030|
|CiteSeer|GraphMAE|head_tdnone_qeps-50|20|64.600|+30.360|+31.210|
|CiteSeer|GraphMAE2|frozen|20|40.170|+0.000|+1.150|
|CiteSeer|GraphMAE2|graphprompt|20|39.020|-1.150|+0.000|
|CiteSeer|GraphMAE2|head_posttdboth_mean|20|67.210|+27.040|+28.190|
|CiteSeer|GraphMAE2|head_posttdboth_qeps-50|20|68.410|+28.240|+29.390|
|CiteSeer|GraphMAE2|head_posttdnone_mean|20|67.230|+27.060|+28.210|
|CiteSeer|GraphMAE2|head_posttdnone_qeps-50|20|68.520|+28.350|+29.500|
|CiteSeer|GraphMAE2|head_tdboth_mean|20|66.320|+26.150|+27.300|
|CiteSeer|GraphMAE2|head_tdboth_qeps-50|20|67.020|+26.850|+28.000|
|CiteSeer|GraphMAE2|head_tdgate_mean|20|66.320|+26.150|+27.300|
|CiteSeer|GraphMAE2|head_tdgate_qeps-50|20|67.060|+26.890|+28.040|
|CiteSeer|GraphMAE2|head_tdnone_mean|20|66.260|+26.090|+27.240|
|CiteSeer|GraphMAE2|head_tdnone_qeps-50|20|67.240|+27.070|+28.220|
