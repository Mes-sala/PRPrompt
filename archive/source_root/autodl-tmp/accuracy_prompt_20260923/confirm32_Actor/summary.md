# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Actor|GraphCL|frozen|20|21.661|+0.000|+0.010|
|Actor|GraphCL|geom_center_ce|20|24.230|+2.569|+2.579|
|Actor|GraphCL|geom_diag10_base|20|24.974|+3.312|+3.322|
|Actor|GraphCL|geom_diag10_ce|20|24.188|+2.526|+2.536|
|Actor|GraphCL|geom_diag10_gp|20|25.385|+3.724|+3.734|
|Actor|GraphCL|geom_diag10_random|20|24.859|+3.197|+3.207|
|Actor|GraphCL|geom_original_base|20|23.641|+1.980|+1.990|
|Actor|GraphCL|geom_original_ce|20|23.309|+1.648|+1.658|
|Actor|GraphCL|graphprompt|20|21.651|-0.010|+0.000|
|Actor|GraphCL|raw|20|21.339|-0.322|-0.312|
|Actor|GraphCL|rawridge|20|20.803|-0.859|-0.849|
|Actor|GraphMAE|frozen|20|20.562|+0.000|-0.148|
|Actor|GraphMAE|geom_center_ce|20|27.273|+6.711|+6.563|
|Actor|GraphMAE|geom_diag10_base|20|26.993|+6.431|+6.283|
|Actor|GraphMAE|geom_diag10_ce|20|26.954|+6.391|+6.243|
|Actor|GraphMAE|geom_diag10_gp|20|27.240|+6.678|+6.530|
|Actor|GraphMAE|geom_diag10_random|20|26.898|+6.336|+6.188|
|Actor|GraphMAE|geom_original_base|20|22.625|+2.063|+1.914|
|Actor|GraphMAE|geom_original_ce|20|24.316|+3.753|+3.605|
|Actor|GraphMAE|graphprompt|20|20.711|+0.148|+0.000|
|Actor|GraphMAE|raw|20|21.339|+0.776|+0.628|
|Actor|GraphMAE|rawridge|20|20.803|+0.240|+0.092|
