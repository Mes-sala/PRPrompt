# R156 固定确认

开发285行/110轨迹/220快照/140邻域审计通过。预设顺序选择GraphMAE/head_f16_qeps-50 75.00%，对scale2 74.48%为+.52pp（3胜2负），Direct72.56%。与低LR LoRA l4 75.00%持平，不能归因FA优势。

锁定完整Gram预条件冻结A rank16联合Prompt、checkpoint与k50读出。支持种子226200..226219，20新支持；仍是同图复用确认query后500。完整1140/440轨迹/880快照/560邻域审计后分析；bootstrap156000、10000次。主比较Direct及两checkpoint旧CE/scale2，所有95%下界大于0才通过；机制比较l1/l4/无预条件n16/对角d16另列。保留原算法、legacy Adam及CPUdouble伪逆，不切换winner。
