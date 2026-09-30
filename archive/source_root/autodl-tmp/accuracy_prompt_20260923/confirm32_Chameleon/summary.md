# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Chameleon|GraphCL|frozen|20|29.550|+0.000|+0.099|
|Chameleon|GraphCL|geom_center_base|20|30.592|+1.042|+1.140|
|Chameleon|GraphCL|geom_full10_base|20|33.805|+4.254|+4.353|
|Chameleon|GraphCL|geom_full10_gp|20|33.629|+4.079|+4.178|
|Chameleon|GraphCL|geom_full50_base|20|32.237|+2.686|+2.785|
|Chameleon|GraphCL|geom_original_base|20|29.737|+0.186|+0.285|
|Chameleon|GraphCL|graphprompt|20|29.452|-0.099|+0.000|
|Chameleon|GraphCL|raw|20|31.941|+2.390|+2.489|
|Chameleon|GraphCL|rawridge|20|32.314|+2.763|+2.862|
|Chameleon|GraphMAE|frozen|20|28.454|+0.000|-0.011|
|Chameleon|GraphMAE|geom_center_base|20|33.322|+4.868|+4.857|
|Chameleon|GraphMAE|geom_full10_base|20|32.961|+4.507|+4.496|
|Chameleon|GraphMAE|geom_full10_gp|20|33.191|+4.737|+4.726|
|Chameleon|GraphMAE|geom_full50_base|20|32.467|+4.013|+4.002|
|Chameleon|GraphMAE|geom_original_base|20|28.794|+0.340|+0.329|
|Chameleon|GraphMAE|graphprompt|20|28.465|+0.011|+0.000|
|Chameleon|GraphMAE|raw|20|31.941|+3.487|+3.476|
|Chameleon|GraphMAE|rawridge|20|32.314|+3.860|+3.849|
