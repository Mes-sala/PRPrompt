# 第90轮GraphMAE固定确认

开发125全矩阵中GraphMAE通道70.24、随机组70.32、L2匹配70.12、当前Adam70.08、gsh70.12；Direct70.44。GraphMAE2通道70.12低于gsh70.32，不扩大。按准确率而非结构原创性，选择GraphMAE随机组rgr（+.24pp）作为固定主候选，通道rgc仍留作对照，不预设通道语义有效。

新20支持139100..119，phase confirm预定后500查询；同一随机分组规则seed+900000，固定lambda=.1，全部260条。原Direct、Frozen、GraphPrompt和tr1/gsh/rgc/rgr/rgl的mean/k50保留；源不改。四审计通过后配对bootstrap10000次，seed20260923。若主候选未稳定胜当前根则停止扩大，不按确认结果换其他配置；同图重复查询边界不变。
