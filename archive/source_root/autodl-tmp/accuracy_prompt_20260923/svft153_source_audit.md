# R153 后备：SVFT 来源与本地迁移边界

已读 [SVFT 论文 §3.1–3.2](https://huggingface.co/papers/2405.19597)、[NeurIPS2024正式出版](https://proceedings.neurips.cc/paper_files/paper/2024/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html)及作者实现 `references_SVFT/svft/svft_layers.py`（commit8303115d45868712f952e6a847735bb59b1a9f18）。不是将检索中的ICML workshop标签当最终发表记录。

论文形式 $W_e=W_0+UMV^T$：固定 checkpoint 的完整左右奇异向量，只学习M选定位置。对角改变奇异值，带状允许相邻奇异方向交互。不同于R146 PiSSA：后者训练低秩因子，本方案冻结基底。

作者当前代码须区分：

- banded 的实际循环为offsets=0..off_diag及负offset，off_diag=0才是纯对角。docstring写off_diag=1对应plain与实现不一致，以实际索引为准。
- self.s先置零后又kaiming_normal_，gate初始化0，经sigmoid后系数乘0.5；当前作者版本并非恒等初始化。
- random用randint可重复采样；若本地用无放回索引，应标为本地变体，不能声称作者完全复现。
- 作者经SVD重建完整原权重。为了固定原checkpoint的初始输出，本地拟用原W加零初始化增量，不用重建原W；这是明确的恒等初始化迁移版本。

拟先比较128维纯对角和带宽1（382系数）零初始化增量，与同维随机正交基对照、Prompt单独和LoRA。所有W/bias冻结；仅M与可选896维token训练，作用在同一第二层GCN。还未封存正式配置，不开展query训练。先核验完整SVD重建、正交性、系数范数到增量范数等距、零初始化、显式rank-one重建和梯度。
