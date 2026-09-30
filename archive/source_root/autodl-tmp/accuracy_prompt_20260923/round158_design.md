# R158 LoRA-Pro预注册开发

固定Cora5shot、GraphMAE/GraphMAE2两checkpoint、支持228100..228104、dev前500、100步；旧CE/scale2/Direct及frozen/GraphPrompt保留。A kaiming B0；Prompt0；gamma1，物理DeltaW=BA。A/B外层SGD lr.0025，Prompt普通Adam.01，wd0。LoRA-Pro在CPUdouble计算Gram伪逆/权重空间Adam/对称eigh Sylvester，更新后转回float32模型参数；明确非官方bf16训练逐位复现。

p1/p4：rank1/4完整Pro，double二阶矩；q1/q4：同配置，但第二步起二阶矩乘积先bf16取整，保留作者精度路径；z1/z4：同p但gauge X=0；a1/a4：完整Pro单独适配，无Prompt；l1/l4：原rank1因子Adam lr.01/.0025对照。共享同一冻结原始backbone，改第二GCN有效权重并联合其中间Prompt输入，不是终端预测融合。联合可训参数1152/1920，单独256/1024；额外权重空间矩状态32768double元素，不能声称普通LoRA内存规模。

旧scale2损失/固定teacher/KD/物理delta正则保持，k50读出相同。矩阵285正式、110轨迹220快照140邻域；先恒等与61条12步smoke审计，再正式。20条3步CPU真实GCN已通过独立首末原梯度与NumPy/SciPy全步更新回放。原版闭包错误保留在初始失败记录。

开发按预设候选顺序p1,p4,q1,q4,z1,z4,a1,a4,l1,l4和GraphMAE优先选最大k50均值。超过两checkpoint全部旧CE/scale2最强根及Direct至少.20pp才进入固定20新支持确认。没有提升就停止，不因机制来自论文降低门槛。新精度分支或旧LoRA胜出分别归因，不称Pro机制胜出。
