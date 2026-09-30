# R152：FourierFT 频谱权重增量与图 Prompt

## 来源及真实操作

[ICML2024 FourierFT 论文](https://proceedings.mlr.press/v235/gao24o.html)，本地正文 `fourier152_paper.md`（arXiv v1 §3.1/4.5）；作者仓库将实现指向 [PEFT](https://github.com/Chaos96/fourierft)。代码快照 `fourier152_official.py` / `fourier152_config_official.py` 固定于 PEFT commit `116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa`。

少量实系数填入稀疏频谱，$\Delta W=\alpha\operatorname{Re}(\operatorname{IFFT2}(F))$。沿用 PEFT 默认 inverse FFT 归一化 $1/(d_{out}d_{in})$，有效第二层权重 $W_e=W_0+\Delta W$，bias 不变。Prompt 同时改变此层输入，联合同一 CE/KD 目标；不是最终输出融合。论文正文描述系数正态初始化，PEFT init_weights=True 则置零，本轮明确用 PEFT 零初始化以保留 checkpoint 原输出。频率索引 seed2024（论文采用值，PEFT 默认777），支持集不改变索引。

本轮两个 checkpoint 均128方阵。选128/256个系数，scaling64/128（局部适配选择，不是作者官方默认150）。随机频点经 real IFFT 会出现共轭重复：实测128系数有127个独立余弦方向，256有252个；不删除或重新抽样，不把系数数目当严格独立维数，不声称必然满秩。频谱是权重坐标的频谱，不是图拉普拉斯频谱。

## 固定矩阵

- f1/f2：128频点，scaling64/128，联合Prompt。
- f3/f4：256频点，scaling64/128，联合Prompt。
- s1/s2：128/256频点、scaling128，但将生成的增量16384个位置按固定seed2025置换，再加到权重；保留各系数基的Gram及任意系数增量的Frobenius范数，改变与权重坐标的对应。
- fa/sa：256、128尺度的原频谱/位置置换，均无Prompt。
- l1/l4：原LoRA-r1 .01/.01与.0025/.0025，Prompt均.01。

原CE/scale2/Frozen/GraphPrompt/Direct保留。Fourier系数/Prompt用Adam.01/100步；正则.001 token均方+.001真实权重变化均方。相同图、teacher、门控、support/query、未中心化k50扩散。

Cora每类5-shot；GraphMAE/GraphMAE2；开发支持222100..222104，共285预测，110轨迹/220首末快照/140近邻重建。smoke222199/12步61条。query500划分沿用固定开发划分，预测先封存后取标签。候选10方法×2ck的k50中最高必须同时高于同批所有旧CE/scale2的最强者与Direct至少.20百分点才进入20个新support确认。置换或LoRA胜出仍按准确率保留，不归因Fourier。复用同图/query的多轮搜索有选择偏差，支持重采样不等于独立图证据。

## 验证与工程记录

六组无query原语（16/128维、128/256频点、两尺度）与官方get_delta_weight前向逐bit一致。独立双精度余弦基重建前向与梯度通过，零初始化精确无变化。真实轨迹审计用显式余弦基而非复用FFT训练代码；旧根仍需恒等检查，之后smoke与全量CPU教师/梯度/Adam/近邻审计。

首次三步probe在读取元数据时因`adapter_n`和审计器`n`字段不一致停止；没有query训练。原文件保留，`joint152_v2.py`修正字段并经v2入口重新验证。训练数学未改，正式运行尚未开始时修复。

三步20轨迹v2预检全部通过：最大梯度差2.98e−8，Adam回放差7.45e−9；不访问query标签。
