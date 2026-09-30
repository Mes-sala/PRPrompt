# R136后备草案：Muon式更新矩阵变换

来源：[Muon作者说明](https://kellerjordan.github.io/posts/muon/)、[官方实现](https://github.com/KellerJordan/Muon)。已读定义、quintic Newton–Schulz、与Shampoo关系和官方使用边界；固定源码commit `f98f1cacc0263b04290753e32be8d498c1efc806`保存在 `references_Muon`。本地总进展与历史设计未找到Muon实验；R72是参数子空间适配，R83/R135是累计矩阵预条件，不能当作该更新算子的验证。

**适用边界**：官方明确主要用于隐藏权重，embedding/head/bias推荐AdamW。我们将它迁移到7×128条件token，属于实验性适配，不是作者推荐用法。当前小样本完整图梯度不同于语言模型训练梯度，不能外推效率/精度收益。

原Cora强根仍是Adam scale2、late+原门控+原KL+未中心化k50；R135固定确认已失败，不将gsh升级为新主根。R135对角/原梯度方向也是有用控制，可酌情保留有限矩阵。

拟单独改变更新方向。动量 $ M_t=0.95M_{t-1}+0.05g_t $，Nesterov输入 $ B_t=0.05g_t+0.95M_t $。变换前转CPU float64并按Frobenius范数加1e-7归一，若行多于列则转置；固定5次 $ X\leftarrow3.4445X+(-4.7750XX^T+2.0315(XX^T)^2)X $。这与官方bf16数值精度不同，明确不称原生Muon复现。先不采用官方矩阵形状学习率缩放，拟借用每条轨迹自身Adam影子范数来隔离方向；必须配同momentum/Nesterov而不做NS的方向、原Adam及纯幅度对照。最终训练矩阵尚未封存，不能只测唯一组合。

NS5并非精确polar或严格正交矩阵：官方实现也指出输出奇异值并非全为1。初始gate列和为零，T=0时梯度有秩亏结构；不能用任意SVD零奇异子空间填充制造额外方向。多项式自然保留精确零方向，零/秩亏必须先验证；不以矩阵满秩或loss下降解释准确率。

已实现 `muon136_draft.py`：输入不原位修改（官方Nesterov grad.lerp_会修改grad，本地必须保留原梯度用于审计）；CPU double五次quintic及momentum原语。`muon136_primitive_check.json`：9种7×128/转置/方形×随机/秩3/全零情况，NumPy独立重算最大9.26e-15，SVD奇异值多项式重建最大5.50e-14，转置等变、有限与零矩阵通过。

状态：只有原语，尚无实际Cora初始化梯度诊断、无完整Prompt训练、无准确率结果。下一步先用真实Cora初始梯度检查方向、范数、零空间，再决定实现固定对照；确认/失败记录继续写入总进展。
