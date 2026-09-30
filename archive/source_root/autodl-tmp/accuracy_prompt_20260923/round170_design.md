# R170正弦权重更新

借鉴[ICLR2025 Efficient Learning With Sine-Activated Low-rank Matrices](https://arxiv.org/abs/2403.19243)，全文sinelora170_paper.md；旧项目链接404后定位[作者当前项目页](https://yipingji.github.io/sine_activated_PEL/)，作者repo固定6dd03a0fe187960b098049b04cd9f286cbb25f6a。实际src/peft/tuners/sinlora.py封存为sinelora170_official.py，使用未合并weight-sine路径。

基础W冻结，第二GCN层有效权重为 $ W+\sin(\omega BA)/\sqrt{128} $，沿用类条件Prompt、teacher和k50。不是节点输出先线性后sin；作者网页伪码与源码需分开，固定以实际src权重前向为准。取alpha/r=1，A kaiming、B零，初始函数不变；作者s参数最终被sqrt(in_features)覆盖，本地明确使用sqrt128。

候选八新配置：频率5/20×rank1/4正弦，以及相应 $ \omega BA/\sqrt{128} $ 线性更新控制；所有因子Adam.0025、Prompt.01、完全同初始化。另保留旧l1/l4。四个配对的初始函数和梯度相同，区分非线性曲率与线性幅值；实际物理更新正则相同。sin配置不增加参数，训练状态仍保存每步梯度，独立CPU前向/梯度及NumPy Adam回放。

作者实际未合并forward四组double检查通过，最大梯度差7.1054e−15；与错误节点输出sin差异非零，初始匹配线性梯度相同。尚未接入真实GCN，不声称图准确率或秩提升必然改善预测。下一步20条三步GCN轨迹，继而支持240100..104、smoke240199、100步及既有预定门槛。


正式封存：真实GCN20条三步轨迹已通过，最大参数差7.4505806e-09、梯度差2.6077032e-08。先61条12步smoke再正式285，完整110轨迹220快照140邻域审计。候选顺序s51/s54/s21/s24/c51/c54/c21/c24/l1/l4，各先GraphMAE，整数正确数解平局。对两checkpoint旧CE/scale2最强根与Direct均至少+.20才固定20新支持确认。旧l4复用R155两次确认，不因候选为线性对照而改选。
