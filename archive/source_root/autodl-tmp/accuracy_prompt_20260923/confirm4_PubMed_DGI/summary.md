# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|PubMed|DGI|frozen|20|38.440|+0.000|-1.140|
|PubMed|DGI|graphprompt|20|39.580|+1.140|+0.000|
|PubMed|DGI|rawprop2|20|70.550|+32.110|+30.970|
|PubMed|DGI|token3|20|58.680|+20.240|+19.100|
|PubMed|DGI|token_kd3|20|60.430|+21.990|+20.850|
|PubMed|DGI|token_kd3perm|20|33.970|-4.470|-5.610|
|PubMed|DGI|token_random3|20|43.310|+4.870|+3.730|
