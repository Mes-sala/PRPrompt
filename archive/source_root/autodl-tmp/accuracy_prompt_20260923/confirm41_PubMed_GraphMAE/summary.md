# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|PubMed|GraphMAE|frozen|20|71.170|+0.000|-1.130|
|PubMed|GraphMAE|graphprompt|20|72.300|+1.130|+0.000|
|PubMed|GraphMAE|head_base_knn-0.9-clamp|20|70.730|-0.440|-1.570|
|PubMed|GraphMAE|head_token_knn-0.5-clamp|20|73.370|+2.200|+1.070|
|PubMed|GraphMAE|head_token_knn-0.9-clamp|20|73.290|+2.120|+0.990|
|PubMed|GraphMAE|head_token_knn-0.9-free|20|73.280|+2.110|+0.980|
|PubMed|GraphMAE|head_token_mean|20|73.290|+2.120|+0.990|
|PubMed|GraphMAE|head_token_orig-0.5-clamp|20|73.380|+2.210|+1.080|
|PubMed|GraphMAE|head_token_random-0.9-clamp|20|61.930|-9.240|-10.370|
|PubMed|GraphMAE|rect_teacher_ref50|20|69.980|-1.190|-2.320|
