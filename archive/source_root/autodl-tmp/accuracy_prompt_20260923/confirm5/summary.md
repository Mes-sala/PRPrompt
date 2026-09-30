# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|CiteSeer|GraphMAE2|frozen|20|40.320|+0.000|+1.230|
|CiteSeer|GraphMAE2|graphprompt|20|39.090|-1.230|+0.000|
|CiteSeer|GraphMAE2|rawprop2|20|60.450|+20.130|+21.360|
|CiteSeer|GraphMAE2|token3|20|50.880|+10.560|+11.790|
|CiteSeer|GraphMAE2|token_kd3|20|50.870|+10.550|+11.780|
|CiteSeer|GraphMAE2|token_kd3perm|20|36.730|-3.590|-2.360|
|CiteSeer|GraphMAE2|token_random3|20|40.870|+0.550|+1.780|
|Cora|GraphMAE2|frozen|20|60.810|+0.000|+2.160|
|Cora|GraphMAE2|graphprompt|20|58.650|-2.160|+0.000|
|Cora|GraphMAE2|rawprop2|20|67.860|+7.050|+9.210|
|Cora|GraphMAE2|token3|20|67.860|+7.050|+9.210|
|Cora|GraphMAE2|token_kd3|20|67.900|+7.090|+9.250|
|Cora|GraphMAE2|token_kd3perm|20|57.030|-3.780|-1.620|
|Cora|GraphMAE2|token_random3|20|62.070|+1.260|+3.420|
|PubMed|GraphMAE2|frozen|20|68.760|+0.000|-0.840|
|PubMed|GraphMAE2|graphprompt|20|69.600|+0.840|+0.000|
|PubMed|GraphMAE2|rawprop2|20|70.840|+2.080|+1.240|
|PubMed|GraphMAE2|token3|20|72.190|+3.430|+2.590|
|PubMed|GraphMAE2|token_kd3|20|73.880|+5.120|+4.280|
|PubMed|GraphMAE2|token_kd3perm|20|68.590|-0.170|-1.010|
|PubMed|GraphMAE2|token_random3|20|69.090|+0.330|-0.510|
