# R142 后备：分类概率的散度蒸馏

来源：[f-Divergence Minimization for Sequence-Level Knowledge Distillation, ACL 2023](https://aclanthology.org/2023.acl-long.605/)，已读原文方法与官方chat实现；仓库references_FDistill，commit 8e2ff7cdcea99db8cbbb87c4ebe80413e948969a。原方法包含序列采样/分解；这里仅借用类别散度，不能称完整f-DISTILL复现。原文报告的是文本任务效果，图上是否有收益待测。

官方chat/kd.py的calc_ce_loss使用KLDivLoss（定义在distillation.py），交换两个参数保留学生熵项；不是只对教师log概率求学生加权均值。calc_js_loss是单侧到混合分布的KL，由上层教师/学生采样两分支组合。图节点无需序列采样，可精确计算两侧。

设固定教师Q、学生P：原始KL(Q||P)；RKL为sum P(log P-log Q)；JS为0.5KL(P||M)+0.5KL(Q||M)，M=(P+Q)/2。P及M保留梯度；Q固定。先验证独立解析softmax链式梯度、finite-difference gradcheck、相等分布为0、JS对称，以及漏学生熵的错误反例。

拟保持R53强根：同checkpoint/支持query/教师/late896tokens/scale2监督CE/Adam100/k50，只替换KD。同时记录原幅度与初始化token梯度范数匹配版本，防止小损失尺度被误称散度形状收益；匹配常数仅初始化无query计算并固定。两ck的旧CE、scale2、Direct保留。具体正式矩阵/seed/初始化图诊断尚未封存，未启动图实验。先完成R141确认，再决定下一次GPU工作。

## 正式封存补充

无query真实图诊断通过。GMAE/GMAE2的RKL初始梯度范数为原KL的0.8183/0.7776，JS仅0.2028/0.1855；方向余弦分别0.9568/0.9928、0.9949/0.9983。因此保留原幅度和初始token梯度范数匹配两种（固定系数，训练不再重算）。旧根3步CPU tokens/embedding/k50完全一致。

Cora两ck，seed211100..104，开发query500固定；rkr/rkm/jsr/jsm四候选，加无KD、KL2、原scale2、旧CE、Direct、frozen/GraphPrompt/head对照。合计205；45冒烟含identity派生，12步；70全Adam轨迹140梯度快照100近邻正式审计。每条全轨迹独立numpy回放，初始全部散度梯度和匹配权重CPU重建，首末梯度含完整监督/蒸馏目标重建。

仅当最佳候选同时超过两ck的CE/scale2最强根及Direct至少0.20pp，才固定候选扩20支持；若只有KL2/无KD胜出，记为权重调整收益而非散度收益。同一Cora图重复query限制保留。R141确认失败（76.40<76.81）不更改本轮Adam强根。
