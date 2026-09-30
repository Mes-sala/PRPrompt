# R168 RandLoRA随机基缩放

来源：[ICLR2025 RandLoRA](https://arxiv.org/abs/2502.00987)，全文randlora168_paper.md；作者PaulAlbert31/RandLoRA提交371d82f4ef713fc4c35d6cb581d37eef558dbb31，原始layer/model文件保留。已读第4节，论文逼近结论有前提，不推断满秩必然有效。

保持现有共享冻结GCN checkpoint与类条件Prompt，第二层有效权重增加随机基更新；不是末端预测融合。基础W冻结但有效W改变，明确为Prompt+权重适配。A/B固定，lambda/gamma训练，lambda零、gamma=1/128保证初始无变化。初始化沿作者Kaiming并全局标准差归一化，scale=2/sqrt(n)。作者B为out,n,r直接展平、A为r,n,in直接展平；本地严格保留此顺序，独立einsum把B重排视图为out,r,n验证，不能用未经重排的自然求和冒充源码。

取r=16/32/64、n=128/r，额外可训练1152/640/384参数，加896Prompt；固定基矩阵另计存储。r16/r32/r64学习率.0025，k16/k32/k64为.00025；t16/t32只用n=1为低秩控制，各144/160参数；旧l1/l4保留。Prompt始终.01；其余损失、teacher、k50、100步一致。此轮不测稀疏随机基、不测DoRA，不宣称完整论文复现。

作者真实前向与自定义反传通过独立double公式检查，三种rank的随机非零更新均秩128；初始gamma梯度零由lambda零导致，属预期。20条两checkpoint真实GCN三步回放通过，尚无开发准确率。

开发Cora5shot238100..104、前500query，smoke238199先61条12步，正式285，全部110训练轨迹220梯度快照140邻域独立审计。选序r16/r32/r64/k16/k32/k64/t16/t32/l1/l4，各先GraphMAE，整数正确数解平局；最高候选对两ck旧CE/scale2最强根和Direct均至少+.20才固定20组新支持确认，旧l4复用既有两次确认不重复追显著。query标签不进入训练/原型/超参选择。
