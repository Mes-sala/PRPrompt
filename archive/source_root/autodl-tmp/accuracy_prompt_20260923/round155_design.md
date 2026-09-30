# R155：较高秩 LoRA 与 rsLoRA 缩放

## 借鉴、既有经验和边界

已读 [rsLoRA 论文 §2–3](https://huggingface.co/papers/2312.03732)及[作者说明](https://github.com/Damjan-Kalajdzievski/rsLoRA_Blog)，本地 `rslora155_paper.md` / `rslora155_author.md`。核对已固定PEFT commit116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa的 `peft146_layer_snapshot.py` 第278–281行：use_rslora用alpha/sqrt(r)，否则alpha/r。

本轮只改变低秩权重增量的rank及缩放，$W_e=W_0+\gamma_r BA$，固定alpha=1，rs为$\gamma_r=1/\sqrt r$，ordinary为$1/r$。保留原W/bias、teacher、Prompt第二层输入位置；联合损失中Prompt和LoRA共同更新，不是读出融合。rs与普通LoRA alpha换成sqrt(r)严格等价，不主张新算子；论文大秩渐近分析不自动证明本地Adam/rank4/16稳定或精度收益。

R36 Actor曾用隐藏通道残差rank1/4与1/r，但数据、标准差归一化、目标及读出不同；R146等近期强根主要测权重LoRA rank1/2。此次直接在当前Cora强根下测rank4/16与缩放，旧结论不能替代匹配实验。

## 固定矩阵

- r4/r16：rs缩放rank4/16 + Prompt；o4/o16：ordinary缩放同rank + Prompt。
- a4/a16：rs单独适配；b4/b16：ordinary单独适配。
- l1/l4：旧rank1 LoRA，A/B学习率.01/.0025，Prompt均.01。
- 原CE/scale2/Frozen/GraphPrompt/Direct全保留。

前8模式A为同generator(seed+146000)的kaiming_uniform、B=0；同rank rs/ordinary初始因子逐位相同。Adam .01/100步；.001 token均方+.001真实DeltaW均方。rank4/16适配1024/4096参数，联合1920/4992，不声称与rank1等参数或等训练成本；checkpoint物理权重冻结。rank1两缩放相同。支持集、teacher、门控、k50及训练步数固定。

Cora每类5-shot，GraphMAE/GraphMAE2；support225100..225104，dev query500，共285预测/110轨迹/220快照/140近邻。smoke225199/12步61条。预测先封存再评分，代码/config/设计/分析器先SHA封存。固定10候选×2checkpoint的k50；最高候选须同时超过全部旧CE/scale2 k50最强者和Direct .20百分点才能20新support确认。确认不换胜者；若ordinary/旧LoRA胜出按准确率保留，不归因rs缩放。多次复用同图/query的选择偏差持续披露。

## 无query验证

六组原语rank1/4/16×两缩放：PEFT缩放表达式一致，零B恢复、有限差分梯度通过；rs与普通alpha=sqrt(r)等价。该测试覆盖数值公式，非完整PEFT类/优化器复现。实际两GCN×10模式20条三步轨迹独立重建通过，最大梯度差2.2352e−8、Adam回放差1.49e−8。独立审计显式effectiveW及GCN/bias、首末梯度和全步NumPy Adam，不复用训练forward。
