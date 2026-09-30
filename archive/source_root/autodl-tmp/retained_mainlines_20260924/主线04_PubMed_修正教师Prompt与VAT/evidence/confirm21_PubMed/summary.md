# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|PubMed|GraphMAE|frozen|20|71.500|+0.000|-2.100|
|PubMed|GraphMAE|graphprompt|20|73.600|+2.100|+0.000|
|PubMed|GraphMAE|rawprop2|20|69.570|-1.930|-4.030|
|PubMed|GraphMAE|rect_teacher_ref25|20|70.300|-1.200|-3.300|
|PubMed|GraphMAE|rect_teacher_ref50|20|70.530|-0.970|-3.070|
|PubMed|GraphMAE|rect_teacher_ref50shuffle|20|69.090|-2.410|-4.510|
|PubMed|GraphMAE|token_kd3|20|72.430|+0.930|-1.170|
|PubMed|GraphMAE|token_kd3_ref25|20|73.350|+1.850|-0.250|
|PubMed|GraphMAE|token_kd3_ref50|20|74.550|+3.050|+0.950|
|PubMed|GraphMAE|token_kd3_ref50shuffle|20|72.820|+1.320|-0.780|
|PubMed|GraphMAE|token_ridge_kd10plainlate|20|71.000|-0.500|-2.600|
|PubMed|GraphMAE2|frozen|20|68.070|+0.000|-0.850|
|PubMed|GraphMAE2|graphprompt|20|68.920|+0.850|+0.000|
|PubMed|GraphMAE2|rawprop2|20|69.570|+1.500|+0.650|
|PubMed|GraphMAE2|rect_teacher_ref25|20|70.300|+2.230|+1.380|
|PubMed|GraphMAE2|rect_teacher_ref50|20|70.530|+2.460|+1.610|
|PubMed|GraphMAE2|rect_teacher_ref50shuffle|20|69.090|+1.020|+0.170|
|PubMed|GraphMAE2|token_kd3|20|74.050|+5.980|+5.130|
|PubMed|GraphMAE2|token_kd3_ref25|20|74.320|+6.250|+5.400|
|PubMed|GraphMAE2|token_kd3_ref50|20|74.480|+6.410|+5.560|
|PubMed|GraphMAE2|token_kd3_ref50shuffle|20|73.810|+5.740|+4.890|
|PubMed|GraphMAE2|token_ridge_kd10plainlate|20|70.290|+2.220|+1.370|
