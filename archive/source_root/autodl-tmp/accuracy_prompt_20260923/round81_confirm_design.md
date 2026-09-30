# 第81轮稀疏门控固定确认

开发145条：GraphMAE normmatched68.00%对当前softmax.1强根67.44%，+0.56pp；幅度67.56%，raw67.80%，Direct68.28%。GraphMAE2 matched68.12%对强根67.32%，+0.80；幅度67.20%，raw68.00%。固定matched为主候选，GraphMAE主权重，GraphMAE2次要；raw只作预定参照，不因确认结果重新挑主方法。

新支持123100–123119、固定confirm500、20支持×29=580条。源文件不变，只切换支持及查询phase。全矩阵保留原型/传播、旧温度/新温度强根、raw/matched/amplitude、Frozen/GP/Direct；主比较matched+k50对tr1+k50，另外幅度及Direct必须报告。bootstrap10000只描述同图支持变化，查询重复使用不代表独立泛化。开发全部审计通过后启动；不再次搜索温度或稀疏程度。
