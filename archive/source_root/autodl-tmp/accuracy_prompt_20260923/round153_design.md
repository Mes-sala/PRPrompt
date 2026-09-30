# R153：SVFT 固定奇异向量权重适配与共享层 Prompt

来源与差异详 `svft153_source_audit.md`；[NeurIPS2024 SVFT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html)，作者仓库commit8303115d45868712f952e6a847735bb59b1a9f18。

## 实际操作与对照

对冻结第二层128×128权重做完整CPU double SVD，规范左右配对符号，转float固定基底；学习稀疏系数 $M$，$W_e=W_0+UMV^T$，保留bias。Prompt在该层输入加入固定teacher gate乘可训练token，二者联合共享CE/KD目标。不是预测融合。

- d0/d1：checkpoint奇异基，纯对角128系数/带宽1共382系数，联合896参数Prompt。
- r0/r1：相同参数量与稀疏位置，固定seed153000的独立随机正交左右基，联合Prompt。
- a0/a1：奇异基单独适配；b0/b1：随机基单独适配。
- l1/l4：历史LoRA-r1，A/B .01或.0025，Prompt均.01。
- 同批旧CE、scale2、Frozen、GraphPrompt与Direct全保留。

本地采用零系数初始化、无可训练sigmoid gate，直接在原W上加增量以精确保留checkpoint输出；这是论文形式的本地恒等初始化版本，和当前作者随机系数×sigmoid(gate)代码不同。完整基的范数等距不代表精度保证；只训练对角/带状不具备任意完整矩阵表达能力。奇异值重数时方向不唯一，本轮固定CPU实现与规范符号并记录/重建实际基。随机基每个checkpoint/support相同，不由标签选择。

Adam .01/100步，正则.001 token均方+.001有效权重增量均方。冻结模型权重，原teacher/scale2/k50未中心化扩散与R152保持一致。两种联合SVFT参数1024/1278，单独128/382。独立审计通过显式rank-one基重建前向和首末梯度，保存所有梯度并NumPy Adam回放。

## 协议和预封存门槛

Cora每类5-shot；GraphMAE/GraphMAE2；开发support223100..223104；query固定dev500，正式285条/110轨迹/220首末快照/140近邻重建；smoke223199/12步61条。预测先封存再访问query标签。

10方法×2checkpoint的k50候选固定顺序中选最高均值；同时超过全部旧CE/scale2 k50中的最强者和Direct至少.20百分点才进入20新support确认；控制胜出按准确率目标保留，但不归因奇异基。确认阶段不换胜者或读出；完整报告原型准确率。多轮同图/query筛选属于探索，支持bootstrap不能证明独立图泛化。

## 验证记录

16/128维×SVD/随机基×对角/带状八个原语全部通过：作者稀疏索引一致、原W零增量精确恢复、完整基正交、系数范数等距、独立rank-one重建及梯度。原语首版float32范数归约误差导致断言失败，诊断double误差1.47e−8；check153_v2仅用double累加和更严格1e−6阈值审计，训练原语未改。真实GCN三步回放须通过后才能封存和启动query训练。

两checkpoint×十方法真实三步20轨迹已通过独立CPU回放，最大梯度差1.3411e−7，Adam回放差7.45e−9，无query标签访问。
