# R167 ReLoRA顺序累积适配草案

来源：[ReLoRA，ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d6fdc308a3b0f017f16059e5888f3346-Abstract-Conference.html)，全文relora167_paper.md；作者Guitaricet/relora固定提交176f37633fe02019835387258ddabcf6d91e328d，relora167_official.py和relora167_training_official.py原样保存。

原论文是大模型训练路线，包含全参数warm start、顺序合并LoRA、部分优化器重置、重启学习率预热；本图实验从现有checkpoint开始，仅借鉴顺序累积和重置，不称完整预训练复现。checkpoint文件和原参数不改，但额外保存可累积的全尺寸DeltaW；不能仍宣称有效主干完全不变，也不能把存储成本只算活动rank1参数。共同训练原中间Prompt，teacher/readout不变。

源码事实：merge_and_reinit合并BA后A重新Kaiming/B零，前后函数相同。optimizer_reset默认标称清零，实际random pruning .999，且不重置Adam step。论文更常用magnitude pruning .99；quantile后严格大于阈值，等幅条目可能全部被剪。原语验证rank1/4合并no-op、B零、Adam step保留、等幅全剪；10万元素随机.999仍剩103，不当作严格清零。

拟100步rank1、每25步合并（25/50/75，不在最终100步追加无意义重启），累计项D更新D+=BA；A重新抽样/B零，Prompt及其优化器持续。物理正则用整个D+BA，避免重启后遗漏已有更新。累计最高rank4但不保证实际rank或精度，保留同预算活动rank1与最终rank4普通LoRA控制。

拟对照：完整merge+99%幅值动量清理+5步线性预热；merge+严格清零动量（步数保留）；merge+保留动量；merge+清理但无预热；不merge仅同时间清理+预热；不merge仅预热；持续rank4普通LoRA；rank4同预热；旧l1/l4。初始和每次重启seed固定并保存，因子Adam.0025、Prompt.01。严格清零是本地变体，不伪称官方默认。所有预热时间线和总训练步数需在正式配置前封存。

初步检查已通过round167_primitive.json；真实GCN尚未接入，独立审计必须覆盖每个合并前后函数、累积矩阵、重置RNG、动量清理/step语义、学习率时间线与最终读出。不能只回放最终A/B忽略历史合并。


## 正式预注册

20条真实GCN 31步轨迹通过，覆盖第一次25步边界、合并前后函数、全部动量与因子重置和预热。独立Adam与累积矩阵回放；幅值剪枝阈值的float32舍入仅允许边界1e−10绝对误差，掩码和所有事件保存，正式仍逐事件审计。累计D使用float32，训练和审计都相对完整D+BA正则。

m1=merge/prune.99/warm；z1=merge/严格零动量/warm；k1=merge/保持动量/warm；n1=merge/prune.99/常学习率；o1=不merge/prune.99/warm；s1=不merge/保持动量/warm；f4=rank4连续常率；v4=rank4仅warm；l1/l4旧LoRA。候选顺序如上、各GraphMAE优先，整数正确数解平局。新因子Adam.0025；第25/50/75次更新后发生事件；后续j=25..29/50..54/75..79的因子学习率乘0,.2,.4,.6,.8，第30/55/80步索引恢复1，初始前25步常率。Prompt始终Adam.01且不重置。Adam step始终保留。初始化seed+167000，每次重启加第几个周期，B零。新active参数rank1 256/rank4 1024加896Prompt；额外累计D为128×128存储。

Cora5shot237100..104、100步、dev前500、固定k50。先61条31步smoke覆盖真实合并，再285正式，完整110轨迹220首末快照140邻域及全部事件审计后分析。最高候选对两ck旧CE/scale2最强根及Direct均至少+.20才确认。旧l4引用R155不第三次追确认，新配置锁定20新支持。


## v2数值修订：替代原独立幅值剪枝

原正式尚未启动，原31步smoke因梯度CPU/GPU回放超出容差失败，原文件/数据封存。GraphMAE2 m1/n1独立幅值掩码留下一个一阶矩非零/二阶矩零的位置，参数幅度升至1312.78/7338.82；严格清零z1仅.341。单独无query诊断证实，详round167_mask_diagnostic.json。

v2的m1/n1/o1改为以exp_avg_sq的99%幅值阈值生成一张保留掩码，同时作用exp_avg和exp_avg_sq；步数依旧不重置。此为本地paired-moment变体，不冒充原论文独立剪枝。其余候选、种子、步数、学习率与读出保持已定配置，没有根据开发准确率改选。新版20条31步实际GCN回放通过，最大参数差1.4901e−7、梯度差1.4901e−8。新目录round167v2_smoke/gate独立保存，原失败不计科学正式行数。原文件不覆盖。
