# R163 LoRA-SB来源与设计草案

来源：[论文2411.19557](https://arxiv.org/abs/2411.19557)、[作者仓库](https://github.com/CERT-Lab/lora-sb)，固定提交4feb81c243e6e762c64b736c968b232caf7b44c4。全文lorasb163_paper.md；初始化/梯度/latent/SVD源码保存在references_LoRASB163，原语检查round163_primitive.json。

作者把首步更新近似为$ -\eta sign(G) $，SVD后固定左右正交基，只训练中心$ R $；R初始化为非零奇异值矩阵，不减回checkpoint。因此不是no-op初始化。源码SVD作用于转置梯度，随后交换并转置左右因子，最终仍为原方向的截断更新。实际采用randomized float32 SVD再转bf16，正交性只是近似。拟图实现128维用CPU确定性完整float64 SVD再float32，明确不声称逐位作者训练复现。

拟共享第二GCN有效权重$ W+B R A $与既有中间Prompt，原teacher/scale2/KD/k50保持。初始全权重梯度来自同一支持loss及固定teacher KD，不用query标签，checkpoint其他参数冻结。adapter rank4/16，对应中心16/256参数（再加Prompt896）；分别比较sign梯度子空间+非零首步、同子空间R零、原始梯度子空间R零、随机正交子空间R零，另有l1/l4。初始步长和中心Adam学习率均.0025，gamma1，首步近似忽略Adam epsilon的差异需留存；Prompt .01。

物理DeltaW正则.001，不把非零初始更新忽略。原语证明：截断SVD残差等于舍弃奇异值能量；固定基下梯度是双侧正交投影；但中心Adam和全权重Adam再投影不相同，反例差范数0.948/3.063，故不宣称完整Adam轨迹最优逼近或保证精度。作者bf16基正交误差实测.001788。官方get_delta_weight与本地BRA完全相同，rank2/4有限差分通过。

尚未接入真实GCN，正式配置、初始化完整审计、轨迹回放和准确率实验未运行。下一步重点保存初始全梯度/子空间/初始有效权重，独立重算避免用低support loss代替准确率证据。无需原创性门槛，仍需要匹配基线验证收益。


## 正式预注册

20条真实GCN三步轨迹已通过，包括初始全梯度重算、NumPy独立SVD子空间/物理截断矩阵、首末梯度和全部Adam回放。代码固定CPU完整double SVD后float32基，随机QR种子seed+163000；新配置均额外一次初始loss权重梯度，非零signstep还预注入截断sign更新，明确100步之外的初始化工作。

候选顺序s4/s16=sign非零首步，z4/z16=sign子空间零中心，g4/g16=raw梯度子空间零中心，r4/r16=随机正交零中心，l1/l4=旧LoRA；每个GraphMAE优先。Cora5shot233100..104，dev前500，100步，k50固定。全285/110轨迹220快照140邻域审计后，最高候选对两ck旧CE/scale2最强根及Direct均至少+.20才确认。旧l4不追加第三批；新配置锁定20新支持确认。原型保存但不重选读出。先61条12步smoke。
