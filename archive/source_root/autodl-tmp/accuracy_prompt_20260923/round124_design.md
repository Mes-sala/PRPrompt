# R124：AdaMax最大梯度尺度的Prompt优化

来源：[Kingma与Ba，Adam论文及AdaMax扩展](https://arxiv.org/abs/1412.6980)；[PyTorch官方Adamax定义](https://docs.pytorch.org/docs/main/generated/torch.optim.Adamax.html)。采用官方数值epsilon放置，不宣称严格无epsilon论文式复现。来源覆盖视觉等优化场景，迁移到冻结GCN的共享768参数Prompt，不改变Prompt位置或做多模型融合。

更新：$ m_t=.9m_{t-1}+.1g_t $，$ u_t=\max(.999u_{t-1},|g_t|+10^{-8}) $，$ h_t=m_t/((1-.9^t)u_t) $；u不作偏差校正。状态CPU float64，实际tokens float32。保留零梯度时u的衰减，不能沿用Yogi零梯度二阶矩不变的断言。

同一当前梯度流计算基线方向d：GraphMAE为Adam，GraphMAE2为graftedShampoo；比较原式h、范数匹配h||d||/||h||、仅幅度d||h||/||d||；若分母为0保留0，不凭空构造方向。原式在GraphMAE2也为AdaMax，gsh只提供匹配/幅度对照。所有lr=.01，未调完整学习率网格，因此不泛化为优化器整体优劣。

CiteSeer5-shot，GraphMAE/GraphMAE2，189100..104开发支持种子，100步，既有门控/LOO CE/正则/全图中心化与k50不变。原始Frozen/GraphPrompt/Direct与tr1/gsh基线保留。125条正式，189199三步smoke29条（额外恒等根恢复）。三候选k50须超过同批tr1/gsh最强者至少.20个百分点才进入固定20新种子确认，按原式/匹配/幅度顺序打破相等。仅幅度胜出按步长变化归因。mean作诊断；query标签封存预测后才评分。

原语先检验独立NumPy递推、torch.optim.Adamax双精度100步一致性、零/缺失梯度与范数控制、恒等Adam/gsh恢复；smoke需通过预测SHA、教师、CPU几何/邻居/传播、完整优化器轨迹审计后再正式运行。源hash包含optimizer124。新结果不能凭继承Yogi字段判断，应逐个核对amax命名和递推替换。同图支持bootstrap与跨轮query适应性边界保留。
