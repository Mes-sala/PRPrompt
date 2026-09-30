# 配对结果汇总

种子区间仅反映同一张图上支持采样变化，不是独立图置信区间。

|数据集|权重|方法|n|准确率 %|对 Frozen pp|对 GraphPrompt pp|
|---|---|---|---:|---:|---:|---:|
|Actor|GraphCL|frozen|20|20.951|+0.000|-0.401|
|Actor|GraphCL|graphprompt|20|21.352|+0.401|+0.000|
|Actor|GraphCL|raw|20|20.941|-0.010|-0.411|
|Actor|GraphCL|rawridge|20|20.605|-0.345|-0.747|
|Actor|GraphCL|self10|20|22.345|+1.395|+0.993|
|Actor|GraphCL|self100|20|23.303|+2.352|+1.951|
|Actor|GraphCL|selflearn_kd|20|21.187|+0.237|-0.164|
|Actor|GraphCL|selfonly|20|23.257|+2.306|+1.905|
|Actor|GraphMAE|frozen|20|20.461|+0.000|+0.151|
|Actor|GraphMAE|graphprompt|20|20.309|-0.151|+0.000|
|Actor|GraphMAE|raw|20|20.941|+0.480|+0.632|
|Actor|GraphMAE|rawridge|20|20.605|+0.145|+0.296|
|Actor|GraphMAE|self10|20|21.816|+1.355|+1.507|
|Actor|GraphMAE|self100|20|22.010|+1.549|+1.701|
|Actor|GraphMAE|selflearn_kd|20|21.204|+0.743|+0.895|
|Actor|GraphMAE|selfonly|20|22.158|+1.697|+1.849|
