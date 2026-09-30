# 第三十轮：自环强度结构提示

借鉴[ResNet视觉残差的信息保留原则](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)，但此处实际算子为图自环权重，不是ResNet复现。改变同一冻结两层GCN中的消息图：原始非自环有向边不变，各层所有自环共用一个正标量lambda，重新走原GCN归一化。每层线性权重/bias保持冻结，无末端分数融合。

固定对照lambda=1/10/100、仅自环（沿用预训练权重的节点MLP）；学习两个log(lambda)，初始0即lambda1，100步Adam学习率0.05，约束log(lambda)在[-4.605,4.605]即[.01,100]。学习目标支持LOO CE，加0.001均方log(lambda)；另有加原始岭教师KL的版本。固定lambda变化不归因于学习。lambda1应与原模型一致；仅自环应等价于两个预训练线性层及原激活，需实际验证。支持/query/主干原始特征/有向非自环输入契约同第28轮。

10方法：Frozen、GraphPrompt、raw、rawridge、固定自环1/10/100、仅自环、可学CE、可学CE+KD。3图×3权重×5支持seed44100..44104=450条官方fold0验证集开发。每图每权重1支持3步预检90条，检查数值、lambda梯度有限和主干冻结。先选开发候选再固定测试确认；Cornell/Chameleon测试此前未用，Actor已在29轮用过。训练与固定强对照的匹配增量均单独报告。
