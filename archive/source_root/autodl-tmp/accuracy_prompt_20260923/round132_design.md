# R132：单视图方差与协方差辅助Prompt

来源[VICReg4.1及作者实现](https://arxiv.org/html/2105.04906)，详见variance_covariance_design_draft.md及probe_note。本轮只借鉴逐维标准差下限与非对角协方差平方，不含双视图、invariance或expander，25V+C是单视图适配。与R32/33固定白化读出不同，不改变查询几何操作，只通过最终节点表示的辅助损失训练原768token；两层GCN完全冻结。

初始无提示输出的无偏逐维方差平均开方为固定$r_0$，断言$r_0>10^{-12}$，没有随步自适应。$q=(z-\bar z)/r_0$，$C=q^Tq/(N-1)$，$V=\mathrm{mean}\max(0,1-\sqrt{\mathrm{diag}(C)+10^{-4}})$，$R=\sum_{i\ne j}C_{ij}^2/128$。三模式25V+R、25V、R；lambda .001/.01，CE+lambda辅助+.001参数L2。零权重跳过加法。不读取query标签，统计用全图无标签特征；保留原gate.1、post-center+k50、100步和两权重强根。

尺度选择来自先前单一195199支持的真实初始梯度诊断：联合辅助/CE范数约107/119，因此不用早期草案.1/1，现.001/.01对应约.1/1倍初始幅度。不是查询调参，也不保证其他种子/后续轨迹同样幅度。保留之前草案，未将诊断当成准确率。

CiteSeer GMAE Adam.01/GMAE2 gsh.01；197100..104共225正式，基线含tr1/gsh、原与中心化Frozen/GraphPrompt、Direct去重。六变体各原型与k50。197199三步49预检加vcb0恢复。全部通过后才正式。12个checkpoint×模式×lambda的k50候选，必须超过同批两权重四强Prompt与Direct最高均值至少.20pp，取最高者进入20新支持确认；相等按checkpoint顺序、lambda.001先、联合/方差/协方差顺序。单项胜出可按用户效果优先确认，但不称联合机制有价值。

每步保存128×128协方差、三个辅助标量、基础logits/CE、实际总梯度，首末token快照。独立NumPy标量复算；CPU全图初始尺度atol1e-9/rtol1e-5，快照协方差atol2e-4/rtol2e-4、总梯度atol3e-6/rtol1e-3，读出/优化器阈值沿用。真实CPU单线程完整原根与lambda0三步参数、表示、独立k50分数逐位恢复；GPU独立轨迹按R130已记录近邻敏感性，仅保持参数小容差、原型小容差和所有预测一致，各自传播结果仍须独立几何审计。

入口显式列出正确审计脚本；每阶段检查完整预设键、prediction SHA/argmax、source SHA、轨迹长度及审计计数。smoke geometry44/update16/VC14与28快照；正式geometry200/update70/VC60与120快照。禁止空集合passed替代覆盖。保持同图重复query和相关支持种子的统计边界。当前R131固定中心DR确认失败，未叠加该失败项。
