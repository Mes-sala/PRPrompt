# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cora|GraphMAE|frozen|20|63.910|+0.000|-0.060|
|Cora|GraphMAE|graphprompt|20|63.970|+0.060|+0.000|
|Cora|GraphMAE|ridge_teacher|20|69.510|+5.600|+5.540|
|Cora|GraphMAE|token_ridge_kd10|20|72.350|+8.440|+8.380|
|Cora|GraphMAE|token_ridge_kd10deep|20|72.860|+8.950|+8.890|
|Cora|GraphMAE|token_ridge_kd10late|20|73.490|+9.580|+9.520|
|Cora|GraphMAE|token_ridge_kd10output|20|72.560|+8.650|+8.590|
|Cora|GraphMAE2|frozen|20|59.580|+0.000|+0.960|
|Cora|GraphMAE2|graphprompt|20|58.620|-0.960|+0.000|
|Cora|GraphMAE2|ridge_teacher|20|69.510|+9.930|+10.890|
|Cora|GraphMAE2|token_ridge_kd10|20|70.540|+10.960|+11.920|
|Cora|GraphMAE2|token_ridge_kd10deep|20|71.050|+11.470|+12.430|
|Cora|GraphMAE2|token_ridge_kd10late|20|71.440|+11.860|+12.820|
|Cora|GraphMAE2|token_ridge_kd10output|20|71.070|+11.490|+12.450|
|Cora|Random_s20260923|frozen|20|55.070|+0.000|+1.010|
|Cora|Random_s20260923|graphprompt|20|54.060|-1.010|+0.000|
|Cora|Random_s20260923|ridge_teacher|20|69.510|+14.440|+15.450|
|Cora|Random_s20260923|token_ridge_kd10|20|68.930|+13.860|+14.870|
|Cora|Random_s20260923|token_ridge_kd10deep|20|68.640|+13.570|+14.580|
|Cora|Random_s20260923|token_ridge_kd10late|20|69.120|+14.050|+15.060|
|Cora|Random_s20260923|token_ridge_kd10output|20|69.010|+13.940|+14.950|
|Cora|Random_s20260924|frozen|20|53.830|+0.000|+1.210|
|Cora|Random_s20260924|graphprompt|20|52.620|-1.210|+0.000|
|Cora|Random_s20260924|ridge_teacher|20|69.510|+15.680|+16.890|
|Cora|Random_s20260924|token_ridge_kd10|20|69.200|+15.370|+16.580|
|Cora|Random_s20260924|token_ridge_kd10deep|20|68.920|+15.090|+16.300|
|Cora|Random_s20260924|token_ridge_kd10late|20|69.250|+15.420|+16.630|
|Cora|Random_s20260924|token_ridge_kd10output|20|69.140|+15.310|+16.520|
|Cora|Random_s20260925|frozen|20|51.900|+0.000|+1.090|
|Cora|Random_s20260925|graphprompt|20|50.810|-1.090|+0.000|
|Cora|Random_s20260925|ridge_teacher|20|69.510|+17.610|+18.700|
|Cora|Random_s20260925|token_ridge_kd10|20|68.870|+16.970|+18.060|
|Cora|Random_s20260925|token_ridge_kd10deep|20|68.870|+16.970|+18.060|
|Cora|Random_s20260925|token_ridge_kd10late|20|68.720|+16.820|+17.910|
|Cora|Random_s20260925|token_ridge_kd10output|20|68.910|+17.010|+18.100|
