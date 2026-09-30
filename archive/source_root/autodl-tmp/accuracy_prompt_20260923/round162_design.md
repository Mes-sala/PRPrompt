# R162 GLoRA 固定子结构预注册开发

来源：[One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2306.07967)，原文保存在glora162_paper.md；作者仓库Arnav0400/ViT-Slim提交390467ddf529d747168709ce2f007a85a0399cca，GLoRA/layer.py保存glora162_author.py；PEFT提交116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa，glora/layer.py保存glora162_official.py。

## 代码确认与化简

作者和PEFT均实际计算逐元素乘法：
$$ W_{eff}=W+W\odot(A_dA_u)+B_dB_u $$
$$ b_{eff}=b+b\odot D+E+WC $$

论文公式无显式Hadamard符号，故实现以源码为准。作者初始化上因子Xavier、PEFT用Kaiming；本地拟采用PEFT初始化，不声称完整作者训练复现。固定子结构不复现论文的超网训练和进化搜索，不使用query选择结构。

$ W\odot(A_dA_u)=\sum_i diag(A_{d,:,i})Wdiag(A_{u,i,:}) $，通常不是普通同rank LoRA；即使普通矩阵rank更大也不预言准确率。C/D/E在E自由时函数空间等同一个自由偏置，保留全分支只为测试优化参数化，不称为独立Prompt机制。C为128向量已足够，低秩128×1不增加函数空间。

GCN适配位置：共享冻结第二GCN的lin.weight和聚合后bias，原中间class Prompt照常共同训练。C项作为聚合后的常量偏置，不能把它误换成聚合前常量输入，因为不规则图上归一化邻接一般不保持常数。原语反例最大差0.14983。

## 固定十个候选及对照

- g1/g4：rank1/4完整A、B、向量C/D/E + 原Prompt。
- w1/w4：仅A、B + 原Prompt，剥离偏置分支。
- e1/e4：A、B + 单自由偏置E + 原Prompt，匹配完整C/D/E函数空间，明确参数数不同。
- a1：仅乘法A rank1 + 原Prompt，定位权重缩放。
- b1：仅加法B rank1 + E + 原Prompt，检验普通LoRA加偏置。
- l1/l4：既有rank1 LoRA的两种学习率对照。

两checkpoint、Cora5shot支持232100..104，固定teacher/scale2/KD/k50及100步。新因子和偏置学习率.0025，Prompt.01；初始化A/B下因子零、上因子Kaiming、C/D/E零。原Prompt896参数，完整适配rank1为896（512+384）、rank4为2432（2048+384），不能称与原256参数LoRA同预算。权重正则沿用物理DeltaW均方，并对物理Deltabias均方加.001，避免把等效偏置分支按参数范数不同惩罚；比较w/e/g时明示偏置自由度与正则差别。

下一步接入实际GCN，独立重建全部参数、初始化、有效权重和偏置、损失、首末梯度及全部Adam轨迹。原语rank1/4作者前向逐位相同、有限差分、PEFT no-op和首步可学习分支、对角展开和偏置化简已通过round162_primitive.json。20条两checkpoint真实GCN三步轨迹已通过独立前向、首末梯度及全步Adam回放；尚无query准确率。

正式启动前封存配置/代码/预算/候选顺序。若同一l4再胜出，复用R155两批确认，不追第三批；新winner开发达门槛后才固定20新支持确认。以准确率为目标，不以是否已有工作排除候选。


正式封存：候选顺序g1,g4,w1,w4,e1,e4,a1,b1,l1,l4，每个候选GraphMAE优先。k50最高者对两checkpoint旧CE/scale2最强根和Direct均至少+.20百分点才晋级；同一l4复用R155，不改选runner-up。smoke61条12步之后正式285条100步，要求110轨迹220快照140邻域通过独立审计后分析。所有原型读出保存但不切换预设k50。
