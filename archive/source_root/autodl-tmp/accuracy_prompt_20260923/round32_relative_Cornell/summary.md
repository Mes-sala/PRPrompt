# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Cornell|GraphCL|frozen|5|41.695|+0.000|+4.746|
|Cornell|GraphCL|geom_center_base|5|48.814|+7.119|+11.864|
|Cornell|GraphCL|geom_center_ce|5|63.051|+21.356|+26.102|
|Cornell|GraphCL|geom_center_gp|5|43.051|+1.356|+6.102|
|Cornell|GraphCL|geom_diag10_base|5|48.136|+6.441|+11.186|
|Cornell|GraphCL|geom_diag10_ce|5|64.068|+22.373|+27.119|
|Cornell|GraphCL|geom_diag10_gp|5|41.695|+0.000|+4.746|
|Cornell|GraphCL|geom_full10_base|5|31.525|-10.169|-5.424|
|Cornell|GraphCL|geom_full10_ce|5|67.119|+25.424|+30.169|
|Cornell|GraphCL|geom_full10_gp|5|29.153|-12.542|-7.797|
|Cornell|GraphCL|geom_full50_base|5|38.305|-3.390|+1.356|
|Cornell|GraphCL|geom_full50_ce|5|64.407|+22.712|+27.458|
|Cornell|GraphCL|geom_full50_gp|5|31.864|-9.831|-5.085|
|Cornell|GraphCL|geom_original_base|5|42.373|+0.678|+5.424|
|Cornell|GraphCL|geom_original_ce|5|67.797|+26.102|+30.847|
|Cornell|GraphCL|geom_original_gp|5|42.034|+0.339|+5.085|
|Cornell|GraphCL|graphprompt|5|36.949|-4.746|+0.000|
|Cornell|GraphCL|raw|5|68.814|+27.119|+31.864|
|Cornell|GraphCL|rawridge|5|71.525|+29.831|+34.576|
