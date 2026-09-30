# Actor fold1–9复核

同一张图，测试集合重叠；两层重采样区间仅描述划分/支持敏感性。

|权重|方法|45次均值 %|
|---|---|---:|
|GraphCL|geom_center_base|23.944|
|GraphCL|geom_diag10_base|23.914|
|GraphMAE|frozen|20.548|
|GraphMAE|geom_center_base|25.773|
|GraphMAE|geom_diag10_base|25.971|
|GraphMAE|geom_diag10_ce|26.408|
|GraphMAE|geom_diag10_gp|26.215|
|GraphMAE|geom_diag10_random|25.991|
|GraphMAE|geom_original_base|21.743|
|GraphMAE|rawridge|20.168|

|候选/对照|增量 pp|正向fold/9|两层描述区间|
|---|---:|---:|---|
|GraphMAE:geom_diag10_base / GraphMAE:geom_original_base|+4.228|9|[3.01608174542586, 5.42701039297713]|
|GraphMAE:geom_diag10_base / GraphMAE:geom_center_base|+0.197|6|[-0.1637426730659273, 0.6418494251039288]|
|GraphMAE:geom_diag10_base / GraphMAE:rawridge|+5.803|9|[4.304056480526924, 7.156469218432903]|
|GraphMAE:geom_diag10_base / GraphMAE:frozen|+5.423|9|[3.7499996672074003, 6.934247399369875]|
|GraphMAE:geom_diag10_base / GraphCL:geom_diag10_base|+2.057|7|[0.4969297548135123, 3.7923975189526877]|
|GraphMAE:geom_diag10_ce / GraphMAE:geom_diag10_base|+0.437|6|[-0.12865491128630108, 1.0497079326046839]|
|GraphMAE:geom_diag10_ce / GraphMAE:geom_diag10_gp|+0.193|4|[-0.21783613661925, 0.6871342385808626]|
|GraphMAE:geom_diag10_ce / GraphMAE:geom_diag10_random|+0.417|6|[-0.10964920288986629, 1.0131576706965764]|
