# 第91轮GraphMAE2固定确认

开发主候选clm等范数CLR门控在GraphMAE2达到70.28 vs当前gsh68.92（+1.36pp），原CLR68.28、纯幅度68.16、Direct69.32。GraphMAE的clm62.96 vsAdam69.56明显失败，不扩大。固定GraphMAE2 clm，不改原温度.1和范数匹配规则。

新20支持141100..119，phase confirm预定后500查询，260条。tr1/gsh/clr/clm/cla的mean/k50、Frozen、GraphPrompt、Direct均保留；匹配当前gsh为主比较，原CLR/纯幅度/Direct为关键对照。源不改，100步、单冻结GCN/768token。四审计与配对bootstrap10000次seed20260923后决定是否第二20支持确认；同图重复查询，不当独立数据集泛化。
