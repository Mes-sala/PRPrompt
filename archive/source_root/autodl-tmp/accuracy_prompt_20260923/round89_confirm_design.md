# 第89轮固定确认

开发125：GraphMAE abm70.76 vs当前Adam70.56（+.20pp），gsh70.84；GraphMAE2 abr71.00 vs当前gsh70.64（+.36pp），abm70.84、幅度70.76；Direct71.12。固定主比较GraphMAE2 abr vs gsh，次比较GraphMAE abm vs Adam。不会按确认结果切换候选或超参。

新支持137100..119，官方test预定后500查询，全部500条；原Direct、Frozen、GraphPrompt与tr1/gsh/abr/abm/aba均保留。训练源不改，门控/损失/读出不改。复用四审计，配对10000次bootstrap种子20260923，来源/支持/查询/权重与预测SHA核验。保持同图重复查询限制。稳定超过各自根才新20支持复核，不把开发微小增益当已确认。
