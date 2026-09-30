# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|PubMed|GraphMAE|frozen|20|72.070|+0.000|-1.540|
|PubMed|GraphMAE|graphprompt|20|73.610|+1.540|+0.000|
|PubMed|GraphMAE|rawprop2|20|70.550|-1.520|-3.060|
|PubMed|GraphMAE|token|20|73.160|+1.090|-0.450|
|PubMed|GraphMAE|token_kd|20|73.990|+1.920|+0.380|
|PubMed|GraphMAE|token_kd_perm|20|72.060|-0.010|-1.550|
|PubMed|GraphMAE|token_random|20|72.100|+0.030|-1.510|
