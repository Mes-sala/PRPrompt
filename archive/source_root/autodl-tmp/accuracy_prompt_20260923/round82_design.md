# 第82轮草案：提示矩阵的Shampoo预条件优化

来源：[Gupta等，ICML2018](https://proceedings.mlr.press/v80/gupta18a.html)，已核对[原文算法1](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf)。借鉴矩阵两轴累计梯度二阶矩，改变提示参数更新；原论文的凸优化保证不直接适用于本图提示非凸训练。

回到CiteSeer已保留的softmax温度0.1门控与768参数零初始化token，冻结GCN、训练目标、100步和中心化+k50读出全部固定；不沿用第81轮未稳定确认的稀疏映射。令token梯度 $ G_t\in R^{6\times128} $，累计 $ L_t=\epsilon I+\sum_{i\le t}G_iG_i^T $、$ R_t=\epsilon I+\sum_{i\le t}G_i^TG_i $，方向 $ D_t=L_t^{-1/4}G_tR_t^{-1/4} $。保留完整损失含原L2；不用Adam再包装该方向，避免混称原Shampoo。

固定小学习率组0.01/0.03，与同学习率Adam及逐元素AdaGrad比较；两权重、开发新支持125100–125104。双精度CPU求6/128维对称特征分解，每步更新；epsilon先固定1e-4，需预检确定数值条件，不通过query选择。记录特征值范围、参数步长及耗时。另考虑对角化L/R控制，分离跨坐标预条件与尺度，不把其称逐元素AdaGrad。最终矩阵待封存。

先完成独立NumPy更新重放、对角/恒等矩阵原语、冻结主干和最终几何审计；保存每步token梯度足够重建，不保存大型主干副本。未实现、未运行；第81轮已停止，不为失败分支新增有效项目入口。


## 封存矩阵

205条正式：两权重5支持，每个权重原Adam.01根及Shampoo/对角L-R/AdaGrad/Adam各.01/.03，共9训练×2读出+Frozen/GP；Direct一次。smoke125199共41条3步。Adam.01重复分支用于恢复诊断。新优化器epsilon1e-4，AdaGrad初始化逐元素累计V=epsilon，分母sqrt(V)，明确不同于PyTorch默认外加epsilon。L2梯度也进入预条件。Shampoo及控制每步CPU双精度累计，更新方向转回参数float32；NumPy独立逐步重放最终token，atol3e-5，并核对门控/几何/预测SHA。先原语，再预检及四审计全部通过才正式运行。源码manifest封存后不改训练实现。
