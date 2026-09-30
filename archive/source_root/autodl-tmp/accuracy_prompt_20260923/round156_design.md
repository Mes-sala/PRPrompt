# R156：LoRA-FA 冻结投影与梯度预条件

来源：[LoRA-FA 2023论文](https://huggingface.co/papers/2308.03303)及固定PEFT commit116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa的optimizers/lorafa.py；源码快照lorafa156_official.py。与R36/R37仅冻结A区分，详lorafa156_design_draft.md。

冻结原第二层W/bias及随机投影A，训练B与可选Prompt；有效权重 W+B A，gamma=1（alpha=r），A使用kaiming_uniform(seed+146000)、B=0。Prompt改变相同GCN层的输入，联合同一监督与KD损失，非终端融合。

## 矩阵与更新

- f4/f16：rank4/16，完整Gram伪逆变换B梯度，联合Prompt。
- d4/d16：仅Gram对角逆变换，联合Prompt。
- n4/n16：不变换梯度，只冻结A，联合Prompt。
- a4/a16：完整预条件、无Prompt。
- l1/l4：旧rank1双因子LoRA，A/B学习率.01或.0025，Prompt均.01。
- 原CE/scale2/Frozen/GraphPrompt/Direct保留。

前三组在相同A、B、Prompt初始化上比较；rank4/16训练B为512/2048参数，联合1408/2944。完整预条件采用原梯度右乘 AA转置加1e-8单位矩阵的伪逆。因A固定，CPU double算一次Gram及伪逆后转float，训练时float32乘法；不启用PEFT可选bf16。这是明确精度实现差异，不宣称GPU位级一致。

前八模式及Prompt统一legacy AdamW-hf式更新：lr=.01，betas(.9,.999)，eps1e-8加在未偏差校正的二阶矩平方根，偏差校正放入步长；weight_decay=0。损失仍+.001 token均方+.001实际DeltaW均方。此更新eps位置与旧torch Adam不同；f/d/n在相同更新器内比较以隔离预条件。l1/l4保持旧实现作为强基线。不宣称内存/速度优势。

每步保存原始梯度及实际更新梯度。独立审计重建固定A、NumPy Gram/伪逆、完整梯度变换、首末原梯度及所有legacy Adam状态；最终图传播/teacher/k50照旧。没有梯度的A不纳入优化向量，checkpoint所有参数均不更新。

## 协议和门槛

Cora每类5shot，GraphMAE/GraphMAE2，support226100..226104；dev query500。正式285条、110轨迹、220快照、140近邻；smoke226199/12步61条。旧根恒等→smoke全审计→正式全审计之后才运行分析器；预测先封存再取query标签。固定10方法×2checkpoint的k50候选，最佳须同时高于全部旧CE/scale2 k50最强者和Direct .20百分点才20新support确认，不在确认中换赢家。控制胜出也保留准确率，但不归因完整预条件。所有原型读出报告，重复同图/query有选择偏差。

## 无query验证

官方CPU double优化器四例×四步与NumPy回放最大差8.14e-16。真实两GCN×十方法三步20轨迹通过：原始梯度重建最大2.98e-8，实际更新梯度逐步变换核验，全部Adam回放最大7.45e-9。另检查本地float32更新与官方关闭bf16的CPU路径，量化double预计算伪逆带来的差异。文件round156_optimizer_primitive.json、round156_trajectory_probe.json、round156_local_optimizer_check.json。均未查询准确率。

本地float32四步更新对官方CPU差：rank4最大3.54e-8、rank16最大1.98e-7；验证通过，不声称逐bit相同。
