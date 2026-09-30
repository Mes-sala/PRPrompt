# R136固定设计：Muon式矩阵方向迁移至Cora

来源、官方适用边界、精度适配与NS非精确正交性质见 `round136_design_draft.md`。只借鉴语言/视觉模型中的矩阵更新变换，不主张官方推荐token用法。固定CPU double五次quintic（3.4445,-4.7750,2.0315），beta=.95、Nesterov先动量后变换；7×128下官方形状因子恰为1。原梯度不原位修改。

原强方案冻结2层GCN、late 896token、平滑岭教师门控、原KL、动态scale2 CE+.001T²、Adam影子beta .9/.999/eps1e-8、lr.01/100步和未中心化k50均固定。只改实际更新方向/幅度。

每分支自身梯度同时维护Adam影子方向A以及.95动量/Nesterov输入B。nmr使用NS5(B)原幅度；nmm方向NS5(B)、范数匹配A；nmg方向NS5(g)、范数匹配A（无momentum）；nmc方向B、范数匹配A（不做NS）；nma方向A、范数匹配NS5(B)（纯幅度）；nmi恒等影子Adam仅冒烟。零候选方向明确no-op、记录是否零；不将候选自身的影子Adam范数说成独立基线同一步数值。

无query诊断202199：两ck原NS范数2.304/2.122，Adam29.929/29.932；NS与Adam方向余弦.411/.580。实际梯度末奇异值约3.34e-8/1.58e-8，被NS提升至约3.24e-5/3.19e-5；class-sum方向占比由约7e-8变为约1.5e-5。仅合成精确零方向能严格保持，真实浮点梯度不宣称严格零空间保持。未依据准确率调节NS步数/精度/阈值。

Cora两ck GraphMAE/GraphMAE2；新5支持种子202100–202104；固定dev500查询。Direct每seed一次；每ck四基础frozen/GP/base-k50/GP-k50，token/scale2/gsh/nmr/nmm/nmg/nmc/nma各mean/k50，总205。冒烟202199/3步加nmi两读出共45。正式70轨迹/140首末梯度快照，冒烟16/32。Cora强根仍原Adam scale2，R135确认失败，gsh仅作累计矩阵方向参照。

每条封存预测、源码/checkpoint/graph/support/query/邻居SHA；CPU独立teacher/gate/late-forward/原型/k50；每步NumPy动量/NS/影子Adam/范数和参数回放；首末loss梯度重建；无零覆盖审计通过才启动正式。CPU legacy根精确恢复、GPU nmi均值分数5e-6与参数2e-6且预测一致，自己的k50各自重构。不要求不同运行近邻集位精确。

扩大门槛：候选k50开发均值同时超过两ck token/scale2/gsh中的最高者与Direct至少0.20pp。若纯幅度或无NS对照胜出，同样可作为准确率候选，不归因NS。选最大均值，平手顺序nmm/nmg/nmc/nmr/nma、GraphMAE/GraphMAE2。确认必须固定候选，带完整强根，不事后换选。所有统计仅为重复同图支持采样证据。
