# R139封存设计：Schedule-Free Prompt优化

来源/官方commit/原语见round139_design_draft.md；六条无query真实梯度见schedulefree139_graph_probe.json，最大状态误差9.99201e-16。训练source另存run_round139_schedulefree_v2.py，不修改已运行probe版本。

共享冻结GraphMAE/GraphMAE2的2层GCN、Cora原late token896参数、原岭teacher/gate/KD/动态scale2及.001T²、原未中心化k50。仅改token训练更新。sf0=beta1 .9无warmup、sfw=.9且10步warmup、sfz=beta1 0无warmup；三者均使用Reference，二阶beta2 .999、lr .01、weight_decay0、inner_momentum0，梯度在float32舍入y，CPUdouble状态x/y/z/v；最后评估float32舍入x。sfy直接复用sf0完整训练轨迹但用其y最终点读出，不重复训练。awm=标准Adam同样10步线性warmup；原scale2/普通CE保留。

没有解冻checkpoint或额外分类器。sfz仍平均x，分离插值求梯度效应；sfy分离最终参数点；awm分离warmup效应。sf0不是普通Adam的SWA，因为求梯度位置也随平均状态变化。所有配置按无query诊断和来源固定，没有依据本轮查询调参。

Cora两ck、5新支持207100..207104，官方train每类5shot，dev500，100调用。每seed Direct1条，每ck基础4+token/scale2/sf0/sfw/sfz/sfy/awm七来源双读出14，共185。冒烟207199、12步37条；覆盖warmup结束。必须通过identity与smoke才能正式。

完整审计60来源轨迹（含10条sfy复用，独立训练仅50条）/120首末快照/40SF状态（含10复用）/90近邻。smoke12/24/8/18。每步独立NumPy x/y/z/v、train点y及final点x/y重建，Adamwarmup单独公式；sfy与sf0的梯度和四状态逐项精确一致。图/teacher/支持/查询/源码/预测SHA、实际loss与梯度、k50、准确率继续全面核验。

扩大：候选sf0/sfw/sfz/sfy的k50最佳，须同时超过两ck token/scale2/awm最高者及Direct≥.20pp。平手顺序sf0/sfw/sfz/sfy再GraphMAE/GraphMAE2。控制胜出仍允许追求准确率，但归因须明确。通过后固定20新支持确认，不根据确认结果换候选；同图重复查询限制保留。
