# 第129轮：支持LOO原型的加性角度间隔

来源[ArcFace原文III-A公式3](https://arxiv.org/html/1801.07698v4)：正确类logit采用归一化余弦角度加margin，其他类保留；借鉴人脸识别，不训练视觉分类权重。当前[官方PyTorch实现](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py)使用no_grad原位变换，本轮明确实现论文公式的完整链式梯度，并以停止间隔梯度对照分开。不是严格复现官方代码；不要求原创。不同于R77固定余弦减常数、R101 Circle独立权重、R126动态尺度。

冻结同一个两层GCN、同一768参数Prompt，门控/注入位置/teacher不变；仅支持LOO训练损失改变。对目标余弦$c$使用$\phi_m(c)=c\cos m-\sqrt{1-c^2}\sin m$；尺度固定5，推理无margin，post-center与k50保留。两档$m=0.1,0.3$弧度，各三种：angular完整梯度；detached采用$c-\mathrm{stopgrad}(c-\phi_m(c))$；mean采用$c-\mathrm{mean}(\mathrm{stopgrad}(c-\phi_m(c)))$。后两者分别匹配同状态逐样本、平均logit扣减，轨迹自然不同，不能声称全过程固定相同值。mean不是固定CosFace超参。

cosine范围容差1e-6，目标截断到[-1+1e-7,1-1e-7]并统计；若$c\le\cos(\pi-m)$直接失败并留证，不静默换分段损失。margin0走原CE精确恒等。保存全部100步基础logits/CE/间隔诊断和梯度，首末步token快照；独立NumPy解析值/梯度、CPU实际图梯度、优化器回放、读出几何/封存预测审计。

CiteSeer两权重GMAE/2，194100..104开发100步，GMAE Adam.01、GMAE2 gsh.01。保留同批Adam/gsh、原Frozen/GraphPrompt、中心化同读出Frozen/GP与Direct，共225条。194199三步49条smoke含ari0恒等，先全部审计才启动正式。

六种variant的k50允许按准确率晋级（用户效果优先），须超过同批两权重四个强Prompt和Direct的最高均值至少.20pp。取最高均值，相等按checkpoint顺序、margin.1先、angular/detached/mean顺序。入选20新支持固定确认；需相对本权重最强root的配对区间下界>0，同时报告全局最强参照和Direct。完整angular若未胜同margin detached/mean，不归因角度梯度；控制胜出可保留工程收益但不包装成角度机制。

此轮不更换查询分区：长期复用同图500开发/500确认，支持种子bootstrap仍非独立图泛化。禁止从确认重选候选。R128三迁移checkpoint全部落后于同批强参照，停止扩大。
