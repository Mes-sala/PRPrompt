# 第三十一轮：消息路径与内部条件token组合

前30轮显示Actor固定减弱邻居消息有效、可学自环无增量。此轮将该结构与内部条件token结合，检验信息进入主干后是否因消息路径改变而更有效。沿用前文视觉残差信息保留及支持教师借鉴，不因原创性淘汰。

共享参数：同一个冻结两层GCN，非自环有向边固定，分别使用self1/self100/selfonly；在第一层激活之后、第二层图卷积之前加入gamma3类别token。门控为原始归一化特征岭教师q减列均值，无额外邻接传播；第一、二层都使用各自固定消息图。输出只有该主干的支持余弦原型，无终端多模型融合。token零初始化准确恢复对应结构基础。

每种结构五种方法：无token、token+支持CE+KD、token仅支持CE、随机token、该结构输出上的普通GraphPrompt。随机token固定N(0,.3)，无训练；token两种训练均100步Adam .01，正则.001均方参数。普通GraphPrompt在各结构的输出上独立训练，对照不能只用原图GraphPrompt。所有教师/支持/query固定。

另含原Frozen、原GraphPrompt、raw、rawridge，共19方法×3图×2权重(GraphCL/GraphMAE)×5支持seed46100..46104=570条官方fold0验证集开发。预检19×3×1权重=57条3步，检查zero-token恢复、模型冻结、同一结构无token与对应固定自环输出一致。新图测试状态：Actor已用，Cornell/Chameleon尚未用。只有超过强直接分类、对应无token及随机/GraphPrompt对照的组合，才在新支持测试确认。
