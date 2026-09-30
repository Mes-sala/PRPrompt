# R145 后备：Prompt与DoRA内部权重适配

来源：[DoRA, ICML2024](https://proceedings.mlr.press/v235/liu24bn.html)，已核对[原文4.1–4.3](https://arxiv.org/html/2402.09353v3)与官方NVlabs/DoRA，commit7e2f10abbe8efe212c8fca1d983ae1d04ef13a18。代码commonsense_reasoning/peft/src/peft/tuners/dora.py采用Linear的输出行范数；原文列范数需结合矩阵布局转置，不能直接按源码dim0套用。

固定第二层W0，令V=W0+BA，m=m0+delta_m，W_eff=m*V/norm(V)。rank1、alpha/rank=1，A Kaiming、B与delta_m零初始化；原W0和bias不更新。官方dora_simple对分母detach，仍每步重新计算数值；原文4.3所有后续实验用该节省内存版本。保留full可微分母对照，不能混用两者梯度结论。

拟在Cora当前强Prompt进入第二层的同一个hidden上接有效权重适配。它是Prompt+参数高效权重适配，不是纯Prompt创新；原checkpoint冻结但有效主干映射改变。必须用仅Prompt、仅adapter、Prompt+adapter做2×2，普通LoRA和仅幅度也配对，保留当前CE/scale2和Direct。R36/37已有普通低秩失败记录，新增的是幅度/方向分解而非重新命名旧LoRA。

Prompt896参数；rank1 LoRA256；DoRA256+128=384；magnitude128，所以联合分别1152/1280/1024。教师/门控、监督CE/KD、readout保持；拟所有adapter按实际DeltaW的平方均值正则，避免将A/B尺度等价性与正则混淆。具体正式矩阵和梯度审计仍待实现。对detach版本，正则也遵循同一个surrogate梯度，必须明确记录。

公式原语已经独立解析梯度与官方展开forward形式核验；full做finite difference，detach只检查指定surrogate梯度，不能拿普通数值微分错误否定detach。零初始化必须完全恢复W和GCN输出；GCN bias独立保留，不能无依据把行幅度说成带bias输出的通道缩放。尚无新图训练/准确率结果。

## 已验证并封存的正式协议

v1 CPU短轨迹发现initial.to(cpu)与Parameter共享存储，保存初值被更新；无query实验失败，不计正式。joint145_v2显式clone后，两ck×八模式共16条三步轨迹全部通过，初值独立再生、CPU全目标梯度与NumPy Adam重放一致（梯度最大2.24e-8，更新7.45e-9）。原文件保留，正式只调用v2。

Cora seed214100..104，checkpoint GraphMAE/GraphMAE2；Prompt+DoRA(dp)/full(fp)/LoRA(lp)/幅度(mp)，以及仅adapter(da/fa/la/ma)，加原scale2、旧CE、Direct、frozen/GraphPrompt/head基线、mean及k50双读出。245正式，90训练轨迹180梯度快照120近邻；53条12步冒烟额外identity，共20轨迹40快照26近邻。仅适配器不训练token，其参数量384/384/256/128；联合参数1280/1280/1152/1024。

所有trainable张量无损展平为一个Adam参数向量，逐坐标Adam与分张量等价；保存布局/种子再生的初值/全部梯度/首末快照，独立CPU手写有效W及稀疏传播+bias、梯度与NumPy Adam重建。正则为.001*(Prompt均方+实际DeltaW均方)，不按A/B各自大小惩罚。teacher/gate固定、Query标签不进入训练或选参。原checkpoint模型参数保持不变；有效W改变明确是PEFT。

门槛依旧：最佳配置超过两ck旧CE/scale2最强根及Direct均至少.20pp才固定20支持确认。若单adapter或LoRA/幅度控制胜出，按其实际类型记录，不能归功于DoRA或Prompt互补。即便联合提升，也须报告与仅Prompt、仅adapter及等rank LoRA差值。重复同图query边界保留，代码/config/分析/设计SHA封存后不改。
