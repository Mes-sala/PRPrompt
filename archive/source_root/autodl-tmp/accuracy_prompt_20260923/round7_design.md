# 第七轮：闭式岭回归教师驱动条件提示

来源：[Meta-learning with differentiable closed-form solvers, ICLR 2019](https://arxiv.org/abs/1805.08136)，借鉴小样本岭回归闭式适配；结合 [PromptKD](https://arxiv.org/abs/2403.02781) 的教师约束提示。此处没有元训练，也不称R2D2原论文复现。

当前rawprop2余弦原型忽略类间相关性。尝试在相同传播特征上拟合支持核岭回归教师：单位化 $ R=\hat A^2X $ 后，$ K=R_SR_S^T $，$ C_\lambda=K+\lambda I $，$ A=C_\lambda^{-1}Y $。仅用支持标签，在 $ \lambda\in\{0.01,0.1,1,10\} $ 中按真实解析LOO预测 $ \hat Y_i^{(-i)}=Y_i-A_i/(C_\lambda^{-1})_{ii} $ 的交叉熵选择lambda。用暴力逐行重拟合验证公式。

教师全图logits为 $ (RR_S^TA)/0.2 $，概率替换原型教师的门控及KL目标；其他第一层条件token计算、冻结主干和原型读出保持一致。预先固定gamma3与gamma10。所有lambda选择无查询标签；教师内部LOO无需重构特征，因为R不依赖标签。

对照：Frozen、GraphPrompt、rawprop2原型、直接ridge教师、原教师token_kd3、ridge教师token gamma3/gamma10。3数据集×GraphMAE/GraphMAE2×5新支持seed11100..11104×7方法=210最终指标，开发半区评分。只有在提示胜过直接教师时，才能说主干提示在该教师之上提供准确率增益；否则仍保留对Frozen的局部提升。
