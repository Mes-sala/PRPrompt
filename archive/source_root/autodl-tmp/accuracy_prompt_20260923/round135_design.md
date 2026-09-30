# R135：Cora强Prompt训练的方向与步幅分离

借鉴[Disentangling Adaptive Gradient Methods from Learning Rates](https://arxiv.org/abs/2002.11803)与[Google Shampoo官方实现](https://raw.githubusercontent.com/google-research/google-research/master/scalable_shampoo/pytorch/shampoo.py)。本轮重新核对官方graft说明；实际复用本地已审计 `optimizer82.py`/`optimizer83.py`，采用Adam影子范数而不是官方SGD/AdaGrad，未声称完整官方复现。R83在CiteSeer已有实验，本轮是Cora迁移，旧图结果不作为新图有效证据。

保持Cora原强根：冻结2层GCN，后层896token、平滑岭门控和原概率KL、动态scale2 CE+.001T²、100步、lr.01、原表示k50；只改变参数优化更新。每条轨迹用自身梯度维护Adam一二阶矩，标准beta .9/.999与eps1e-8，得到方向 $ A_t $。累积 $ L=10^{-4}I+\sum gg^T $，$ R=10^{-4}I+\sum g^Tg $；CPU double eig每步计算 $ D=L^{-1/4}gR^{-1/4} $。

gsh使用 $ -0.01D\|A_t\|_F/\|D\|_F $；gdi只取L/R对角；ggd用原梯度作方向；osh使用未graft矩阵方向。全部候选自身Adam范数不同，不宣称等于独立Adam基线每步范数。零候选方向明确no-op，影子Adam若保留动量仍记录；不静默截断负特征值。gsh/gdi/ggd均可按准确率被选中，osh为步幅对照也保留数据。

checkpoint GraphMAE/GraphMAE2，support每类5个、seed200100–200104，官方测试固定20260923打乱前500 dev查询。Direct每seed一次；各ck含Frozen/GP、base/GP k50、token/scale2/gsh/gdi/ggd/osh各mean/k50，165条。冒烟200199/3步加gad影子Adam恢复，37条。不同目标/种子的试验不能混为同一配对；重复同图查询限制保留。

原语零梯度/剩余动量处理；legacy CPU根精确恢复；smoke gad与原Adam参数2e-6、mean分数5e-6且预测相同，自己的k50独立重构，不要求GPU近邻集位精确。正式50更新轨迹/100梯度快照，smoke12/24。完整SHA/键集合/teacher/gate/late forward/原型/k50审计，NumPy逐步矩阵特征分解与影子范数/参数回放；原CE/KD梯度审计保持。

开发候选k50均值须同时比两ck最强scale2和Direct高≥0.20pp才扩大。候选平手顺序gsh、gdi、ggd、osh再GraphMAE、GraphMAE2。只根据实际准确率归因，不把PSD矩阵或loss下降当效果。先全部冒烟审计后正式；失败保留且修复另存版本。
