# 第54轮草案：支持标签平滑

借鉴[Inception CVPR2016原文](https://arxiv.org/pdf/1512.00567)第7节，已核对均匀标签混合与交叉熵分解。仅支持留一目标改为 $ q'=(1-\epsilon)\operatorname{onehot}(y)+\epsilon/C $，epsilon固定候选0.1/0.2；原token位置、KD和训练步数不变。先Cora两权重，再按结果决定是否迁移。既有TIM改无标签熵，本方案改支持监督目标，不混称同一配置。

平滑损失等价于 $ (1-\epsilon)CE+\epsilon CE(U,p) $。带单独(1-epsilon)CE对照分离监督弱化和均匀正则；可另给不降低CE系数的CE+epsilon/(1-epsilon)CE(U,p)诊断，但其相对KD强度不同，不能视为严格等价训练。所有训练各自共享原型/k50读出；第53轮scale2+k50已在新20支持确认，必须加入该最新强根，不只对原CE比较。

预检epsilon0恢复、PyTorch平滑CE与显式软标签CE/梯度一致、查询标签隔离、冻结主干。候选不是为了创新，准确率超过同批强方案即可保留。草案尚未实现，正式seed/矩阵待固定；反复dev/confirm500限制持续披露。
