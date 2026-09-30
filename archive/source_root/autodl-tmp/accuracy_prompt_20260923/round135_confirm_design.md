# R135固定确认：Cora GraphMAE grafted Shampoo

开发165条、50轨迹/100快照全审计通过。固定候选GraphMAE gsh+k50为75.96%，原scale2 75.64，+0.32pp；Direct73.44。满足原封存开发门槛，因此进入确认。注意旧普通CE token+k50为75.88，候选仅+0.08，不能称已超越最强替代；确认必须把它作为主要比较。GraphMAE2普通梯度graft约75.96，对角75.92，也不能预先将收益归为完整矩阵方向。

保持run_round135_coraopt.py与全部优化器/审计源码不变。新支持seed201100–201119，官方训练池每类5个；query固定20260923打乱后的官方测试后500。该半区在过去轮次使用过，不称独立外部测试，同图20次支持样本相关。

两ck保留frozen/GP、base/GP k50、token/scale2/gsh/gdi/ggd各mean/k50与每seed一次Direct；删除已明显弱的osh对照，不影响候选训练。共29条/seed=580；160条轨迹/320快照/280邻域，全矩阵键集合审计。使用已通过的37条smoke及legacy/影子Adam恢复，不重新改算子或调参。

主候选只固定GraphMAE gsh+k50。以20个支持seed为配对单位，10000次固定seed135000 bootstrap，报告均值、95%区间、胜负平。成功须对两个checkpoint各自的token/scale2+k50及Direct的差异区间下界全部>0；简单gdi/ggd作为机制归因参照。小数tie按每次查询正确个数判断，不以float噪声算胜负。未通过则停止当前候选，不在确认结果后换选GraphMAE2梯度方向。可保留次要观察，但后续必须新设计/新支持检验。
