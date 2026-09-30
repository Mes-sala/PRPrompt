# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|PubMed|GraphMAE|cons_none_0|20|75.630|+2.220|+0.070|
|PubMed|GraphMAE|cons_rpt_0.3|20|75.610|+2.200|+0.050|
|PubMed|GraphMAE|cons_vat_0.3|20|75.990|+2.580|+0.430|
|PubMed|GraphMAE|frozen|20|73.410|+0.000|-2.150|
|PubMed|GraphMAE|graphprompt|20|75.560|+2.150|+0.000|
|PubMed|GraphMAE|rect_teacher_ref50|20|72.220|-1.190|-3.340|
