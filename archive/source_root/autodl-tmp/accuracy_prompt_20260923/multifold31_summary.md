# Actor fold1–9复核

同一张图，测试集合重叠；两层重采样区间仅描述划分/支持敏感性。

|权重|方法|45次均值 %|
|---|---|---:|
|GraphCL|mixonly_base|23.048|
|GraphMAE|frozen|20.409|
|GraphMAE|graphprompt|20.534|
|GraphMAE|mixonly_base|22.015|
|GraphMAE|mixonly_ce|22.155|
|GraphMAE|mixonly_gp|22.123|
|GraphMAE|mixonly_kd|22.063|
|GraphMAE|mixonly_random|21.768|
|GraphMAE|raw|20.705|
|GraphMAE|rawridge|20.594|

|对照|增量 pp|正向fold/9|两层描述区间|
|---|---:|---:|---|
|GraphMAE:mixonly_base|+0.140|4|[-0.44301932470666033, 0.7690057175027]|
|GraphMAE:mixonly_kd|+0.092|5|[-0.07163739866680569, 0.2529239348239369]|
|GraphMAE:mixonly_random|+0.387|5|[-0.3699199242724312, 1.220760180718369]|
|GraphMAE:mixonly_gp|+0.032|5|[-0.4956509148081143, 0.6038377558191612]|
|GraphMAE:raw|+1.450|8|[0.5803727267516984, 2.333333263794581]|
|GraphMAE:rawridge|+1.561|8|[0.4224780549605688, 2.690131487117872]|
|GraphMAE:frozen|+1.746|9|[0.5087351600329082, 2.9811037662956434]|
|GraphMAE:graphprompt|+1.621|8|[0.40201031996144193, 2.8406796058019]|
|GraphCL:mixonly_base|-0.893|3|[-2.3757676821615963, 0.48391805837551755]|
