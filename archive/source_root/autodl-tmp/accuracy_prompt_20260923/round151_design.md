# R151：ETHER+ 有界权重适配与图 Prompt 联合训练

## 来源、机制与实现边界

借鉴 [ETHER / ETHER+（ICML 2024）](https://proceedings.mlr.press/v235/bini24a.html)，已读 [论文 §3.2–3.4](https://arxiv.org/html/2405.20271v1) 和官方 `ether-instruct/lit_gpt/ether.py`；本地 `references_ETHER` commit `ddc103879b471ea8d56c2acde41bb7ab7007dbe8`。官方代码快照 `ether151_official.py`。

ETHER 的单个 Householder 反射不能恒等初始化。本轮使用 ETHER+：$H=I-uu^T/(u^Tu)+vv^T/(v^Tv)$，初始化 $v=-u$。仅用一个全宽块（nb=1），避免把官方多块 reshape 当作其他分块定义。使用 torch 权重布局 $W\in\mathbb R^{out\times in}$，输入侧 $W_e=WH$；双侧 $W_e=H_2WH$。保留原 bias。每侧 256 个参数；联合 Prompt 1152/1408 个参数，单独适配 256/512 个。每个 H 的变化有界不等于双侧有效 W 变化同一界，更不保证准确率。

在同一个第二层 GCN 内，Prompt 改变输入 $h+10g\tanh(T)\odot std(h)$，ETHER+ 改变该层权重，二者共享监督/KD目标，非最终预测集成。固定原 checkpoint，只有适配向量和可选 Prompt 训练。官方 kaiming_uniform 非零向量及负副本初始化；用独立 CPU seed+151000，记录额外 rand draw。局部将两个秩一项先相减再加 I 以获得精确恒等，数学公式相同，浮点操作顺序不同；必须实测官方前向/梯度一致容差，不宣称完整上游任务复现。

## 固定矩阵与归因对照

- e1/e2：归一化 ETHER+ 输入侧/双侧 + Prompt。
- n1/n2：对应固定初始平方范数作为分母 + Prompt；初始输出相同，但梯度不完全相同，不声称匹配初始梯度。
- a1/a2：ETHER+ 单独适配；b1/b2：固定分母单独适配。
- l1/l4：历史 LoRA-r1，分别 .01/.01 和 .0025/.0025，Prompt 均 .01。
- 同批旧 CE、scale2、Frozen、GraphPrompt、Direct 全保留。

ETHER+ 及固定分母控制统一 Adam .01/100 步；正则 .001 Prompt 均方 + .001 有效权重变化均方。固定原 teacher、图、支持集、k50 扩散和 R150 损失。归一化可能放大方向梯度，固定范数控制仅诊断归一化影响。

Cora 5-shot/class；GraphMAE/GraphMAE2；开发支持种子 221100..221104，固定开发 query500，285 条预测。smoke221199/12步61条。正式预期110轨迹/220首末快照/140近邻重建。先检查 no-query 原语、实际3步20轨迹回放、旧根恒等，再 smoke 和全矩阵。每次预测封存后才访问 query 标签；独立 CPU 重建 teacher、有效权重、首末梯度和全部梯度 Adam 回放。

## 预封存决策

固定候选10方法×2checkpoint的 k50，选平均最高者；必须同时超过同批全部旧 CE/scale2 的最强者和 Direct 至少0.20百分点才能进入20个新支持种子的确认。控制胜出保留准确率结果，但不归因 ETHER+；不在确认结果里换胜者。单一 checkpoint 局部提升记录，不冒充全局升级。同一 Cora 和复用 query 的多次筛选有选择偏差，不作独立图泛化证明。

## 无 query 验证结果（正式封存前）

16/128维×单/双侧四个原语均通过官方代码前向/梯度核验、恒等和范数界；最大双精度前向差1.60e−14，梯度差7.28e−12。实际两个GCN×10方法的20条三步轨迹全部通过独立CPU重建；最大梯度差4.47e−8、Adam回放差1.49e−8。见 `round151_primitive.json` / `round151_trajectory_probe.json`。无query标签、无准确率筛选。
