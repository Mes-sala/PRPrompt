# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|Direct|directcenter_mean|20|69.650|+nan|+nan|
|CiteSeer|GraphMAE2|frozen|20|40.010|+0.000|+0.720|
|CiteSeer|GraphMAE2|graphprompt|20|39.290|-0.720|+0.000|
|CiteSeer|GraphMAE2|head_gad_mean|20|68.470|+28.460|+29.180|
|CiteSeer|GraphMAE2|head_gad_qeps-50|20|69.180|+29.170|+29.890|
|CiteSeer|GraphMAE2|head_gdi_mean|20|68.910|+28.900|+29.620|
|CiteSeer|GraphMAE2|head_gdi_qeps-50|20|69.330|+29.320|+30.040|
|CiteSeer|GraphMAE2|head_gsg_mean|20|68.830|+28.820|+29.540|
|CiteSeer|GraphMAE2|head_gsg_qeps-50|20|69.360|+29.350|+30.070|
|CiteSeer|GraphMAE2|head_gsh_mean|20|68.980|+28.970|+29.690|
|CiteSeer|GraphMAE2|head_gsh_qeps-50|20|69.520|+29.510|+30.230|
|CiteSeer|GraphMAE2|head_osh1_mean|20|40.520|+0.510|+1.230|
|CiteSeer|GraphMAE2|head_osh1_qeps-50|20|26.770|-13.240|-12.520|
|CiteSeer|GraphMAE2|head_tr1_mean|20|68.470|+28.460|+29.180|
|CiteSeer|GraphMAE2|head_tr1_qeps-50|20|69.180|+29.170|+29.890|
