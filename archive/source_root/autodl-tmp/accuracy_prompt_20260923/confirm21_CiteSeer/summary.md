# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|GraphMAE2|frozen|20|39.920|+0.000|+1.130|
|CiteSeer|GraphMAE2|graphprompt|20|38.790|-1.130|+0.000|
|CiteSeer|GraphMAE2|rawprop2|20|60.620|+20.700|+21.830|
|CiteSeer|GraphMAE2|rect_teacher_ref25|20|63.660|+23.740|+24.870|
|CiteSeer|GraphMAE2|rect_teacher_ref50|20|66.990|+27.070|+28.200|
|CiteSeer|GraphMAE2|rect_teacher_ref50shuffle|20|62.350|+22.430|+23.560|
|CiteSeer|GraphMAE2|ridge_teacher|20|61.800|+21.880|+23.010|
|CiteSeer|GraphMAE2|ridge_teacher_plain|20|62.530|+22.610|+23.740|
|CiteSeer|GraphMAE2|token_kd3|20|48.730|+8.810|+9.940|
|CiteSeer|GraphMAE2|token_kd3_ref25|20|50.920|+11.000|+12.130|
|CiteSeer|GraphMAE2|token_kd3_ref50|20|53.610|+13.690|+14.820|
|CiteSeer|GraphMAE2|token_kd3_ref50shuffle|20|47.640|+7.720|+8.850|
|CiteSeer|GraphMAE2|token_ridge_kd10plainlate|20|59.910|+19.990|+21.120|
