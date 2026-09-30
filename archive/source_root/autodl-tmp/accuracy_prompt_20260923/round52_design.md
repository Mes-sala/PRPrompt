# 第52轮固定设计：LaplacianShot式非线性推断

来源：[ICML2020论文](https://proceedings.mlr.press/v119/ziko20a/ziko20a.pdf)式12及[作者代码](https://github.com/imtiazziko/LaplacianShot/blob/5de261226f6904379cff4cc5a1d183e309934809/src/lshot_update.py)bound_update，源快照保存。本轮复用原100步提示、cosine/.2原型分数L与相同有向50NN行softmax图W，更新 $ P^{t+1}=\operatorname{softmax}(L+\lambda WP^t) $，每步支持行钳制one-hot，20步。训练共享参数、注入位置及损失全部不变，属于提示后推断读出。

正式矩阵Cora两权重×5支持seed78100..78104×10方法=100dev：Frozen、GP、原型、原k50线性传播、lambda1/3/10非线性、lambda3单步、lambda3/10随机图。随机图为节点置换共轭，保持边权、入出度分布、无自边；打破与嵌入/标签对应。系数无query拟合。超过同批强方法才扩大20新支持。dev500反复使用，非独立测试。

与原文差别：使用图提示嵌入/cosine一元项、全部节点参与、支持钳制、固定20步，图不对称也不保证半正定。仅借鉴更新，不能继承MM收敛保证；记录每步最大概率残差。不按残差挑选查询结果。使用相同图使与线性强基线的比较不混入图替换。

预检必须核对lambda0恢复非支持原型概率、支持钳制、密集独立循环参考、类别置换、共享表示/邻接SHA，以及随机共轭图绑定。只清理完成预检，正式保留。

## 固定确认矩阵（开发后、确认前）

GraphMAE2 lambda10开发72.32对线性71.36，GraphMAE73.24对73.16；确认固定lambda10、两权重20新支持seed79100..79119、8方法共320。新增同lambda10单步，保留lambda3为低强度诊断，不再选确认赢家重新宣称验证。核心对照原型/强线性/随机10，Frozen/GP补齐。确认query500曾反复使用，新支持不等于独立图。高lambda开发末步残差最大0.3226（真实图），固定20步不称收敛。预检合成最初double与实现float标签不兼容，测试统一生产float后通过，正式代码未变。
