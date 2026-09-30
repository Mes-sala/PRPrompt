# 第54轮固定设计：支持标签平滑

借鉴[Inception CVPR2016原文](https://arxiv.org/pdf/1512.00567)第7节，已核对均匀标签混合与交叉熵分解。仅支持留一目标改为 $ q'=(1-\epsilon)\operatorname{onehot}(y)+\epsilon/C $，epsilon固定候选0.1/0.2；原token位置、KD和训练步数不变。先Cora两权重，再按结果决定是否迁移。既有TIM改无标签熵，本方案改支持监督目标，不混称同一配置。

平滑损失等价于 $ (1-\epsilon)CE+\epsilon CE(U,p) $。带单独(1-epsilon)CE对照分离监督弱化和均匀正则；可另给不降低CE系数的CE+epsilon/(1-epsilon)CE(U,p)诊断，但其相对KD强度不同，不能视为严格等价训练。所有训练各自共享原型/k50读出；第53轮scale2+k50已在新20支持确认，必须加入该最新强根，不只对原CE比较。

预检epsilon0恢复、PyTorch平滑CE与显式软标签CE/梯度一致、查询标签隔离、冻结主干。候选不是为了创新，准确率超过同批强方案即可保留。草案尚未实现，正式seed/矩阵待固定；反复dev/confirm500限制持续披露。

## 固定矩阵

Cora两权重×seed83100..83104×18方法=180dev。原提示、scale2、平滑0.1/0.2、仅CE缩小0.9/0.8、scale2乘平滑0.1/0.2共8组训练，各两读出，加Frozen/GP。mix的scale2系数来自当前hardCE的解析系数均值，stop-gradient后乘完整平滑CE；不能把其缩放当成平滑梯度的解析系数。保存supervision_scale_history区别于Focal的系数记录。

## 确认前固定

主候选GraphMAE2 smooth2+k50开发74.64，对scale2+k50 74.28、旧k50 73.92、weak2+k50 74.04；GraphMAE没有超新强根。固定两权重20支持seed84100..84119×10方法=400confirm，原/scale2/smooth2/weak2各原型与k50，加Frozen/GP。GraphMAE保留跨权重绝对强参照。仍是反复confirm500，新增支持不是独立图。
