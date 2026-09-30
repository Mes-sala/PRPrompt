# R131：固定原始类中心DR候选确认

R130恢复265全审计和额外覆盖检查通过，预选GraphMAE2/head_ftc1_qeps-50 70.60%，同权重gsh70.12%、同批最强GMAE gsh70.20%、Direct70.08%。aligned ETF70.48%，故选择的是原始中心辅助DR，不声称ETF结构收益。GMAE随机目标诊断70.80%虽更高，依预先规则不在此轮更换主候选；随机/错位在本批仍作对照，若另有价值须后续固定新验证。

固定196100..196119，confirm500，同图重复查询边界不变。两权重，原/中心化Frozen/GraphPrompt、tr1/gsh各原型和k50、四目标weight1各两读出、Direct去重，740条。原driver130/frame130不改，无weight10/0。共享768token、100步、GMAE Adam/GMAE2 gsh.01，其他契约沿用R130。

唯一主候选GraphMAE2 ftc1+k50。双主要对照预固定：GraphMAE2 gsh（同模型）与GraphMAE gsh（开发全局最强），两者支持配对bootstrap95%区间下界均>0才进入第二组20支持；另报告Direct，只有其区间下界亦>0才称超越强直接教师。报告tr1、aligned、错位、随机，不用某个辅助显著替代主要条件；确认中不得重选checkpoint/权重/目标。

分析前必须核对740完整预设键、来源/图/拆分/预测SHA与argmax、每条frame100步、160轨迹/320梯度快照、geometry640/update200，全部通过后配对。每比较10000次支持种子重采样，随机种子20260923，区间.025/.975；不是独立图泛化显著性。GPU邻域敏感性见R130诊断，CPU零权重恒等已验证，正式每条独立几何审计不变。
