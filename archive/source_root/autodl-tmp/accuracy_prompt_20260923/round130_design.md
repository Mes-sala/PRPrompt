# R130：固定特征目标的点积回归辅助Prompt

借鉴[NC-FSCIL第4.1/4.2节、公式1/4](https://arxiv.org/html/2302.03004)：固定等角目标、归一化特征点积回归。另读[视觉跨域固定分类器研究](https://arxiv.org/html/2402.18614)和[最近ETF研究](https://proceedings.nips.cc/paper_files/paper/2024/hash/3eb660055cdcdc9a545a0b16c1eff80d-Abstract-Conference.html)。这里只采用固定初始目标辅助损失；没有增量会话、视觉投影层或逐步隐式求解，不声称复现其方法/理论保证。

R24约束的是token的类Gram；本轮约束最终共享GCN的节点表示，单个768参数token经原有早层注入改变表示，无额外可训练分类器、无输出融合。保留原支持LOO CE、.001参数L2、冻结GCN与教师门控、post-center原型+k50。新增辅助项：$L=L_{CE}+\lambda L_{DR}+0.001\mathrm{mean}(T^2)$，$L_{DR}=\frac{1}{2|S|}\sum_{i\in S}(\langle\mathrm{norm}(z_i-\bar z),e_{y_i}\rangle-1)^2$。全图均值只用特征，无query标签。

初始冻结输出先全图中心化和行归一化，取每类5支持均值、再减类均值，得$A\in\mathbb R^{6\times128}$。双精度CPU截断SVD取前5非零方向，$E=\sqrt{6/5}U_5V_5^T$，Gram对角1/非对角−1/5。保存A/奇异值/目标。该式是固定尺度ETF对A的Frobenius最近解，独立Gram特征分解与迹上界核对；不是原文的逐训练步宣告式优化。

四种固定目标：aligned E；centers normalize(A)（无强制等角）；permuted将E类别循环错位1；random用seed+130001独立CPU高斯矩阵居中后同法造ETF。target建立后detach，随机源不改训练RNG。支持自身参与初始类均值，这是支持监督辅助目标，不冒充LOO。秩阈值s5>s1*1e-8，否则显式停批，不随结果改阈值。

权重1/10，各两读出，CiteSeer GMAE Adam.01/GMAE2 gsh.01，195100..104，100步。加原/中心化Frozen、GraphPrompt，tr1/gsh及Direct，共265条。195199三步57条smoke含权重0恢复。两种语义对应目标aligned/centers为主候选，必须超过同批两权重四个强Prompt与Direct最高均值至少.20pp才选最高者进入20新支持确认；并报告错位/随机控制。相等按checkpoint、weight1优先、aligned优先。若不胜centers，不归因ETF；若不胜随机/错位，限制语义解释。控制仅诊断，若另有工程价值需后续预先固定再确认。

原语独立NumPy梯度/有限差分、SVD与Gram特征分解、ETF Gram、迹上界、重跑确定性已通过。每步记录基础CE/logits、DR/cos与实际梯度，首末步保存token。独立CPU全图重建目标atol5e-5/rtol1e-4；真实总梯度atol3e-6/rtol1e-3，考虑中心化归一化的计算敏感性。优化器与读出原阈值不变。零权重跨运行恢复采用R129已诊断容差token1e-6/score5e-6、rtol0、预测必须精确相同；权重0代码直接跳过加法。先完整smoke审计后正式，任何失败留证。长期复用查询和相关支持种子的统计边界保留。
