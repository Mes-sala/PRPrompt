# 三模块方案与本地ProG seedgrid对照（2026-09-24）

## 方案说明

方案用同一冻结两层GCN（隐藏维128）和一组可训练类别token完成5-shot节点分类。第一层激活前注入教师类别门控加权的token，经第二层GCN得到表示，最后分类。只有token训练；教师用支持标签与未标注节点特征构造，KD损失关闭，查询标签不进入训练。

1. **M1门控锐化**：教师类别分配温度从0.2降到0.1，使节点组合类别token时更集中。raw锐化也改变门控幅度，不能把全部增益归因为语义改善。
2. **M2尺度匹配优化**：用Shampoo矩阵预条件构造token更新方向，用同一梯度序列上的Adam更新范数进行匹配，替代普通Adam。优化器不是新增Prompt分支。
3. **M3支持约束概率传播**：在适配后表示上建立50近邻图，概率传播20轮，系数0.9，支持标签每轮固定。它属于末端读出，不增加训练。

共同基础包含普通温度教师门控、token Prompt、Adam和输出中心化。完整方案F111与F000比较衡量三个开关的整体增量；移除M1/M2/M3分别比较F111−F011/F101/F110。模块共享一个主干，但不是三个内部Prompt，也不是三个模型集成。

## 哪些底座有明确正提升

这里“明确”只表示预定义配对差值的95% bootstrap区间下界大于0。每格20个支持采样seed，共用图与查询集，未校正多重比较，且候选底座是分阶段筛选；不能当作独立图显著性或普适结论。

| 数据集 | 预训练家族 | 完整准确率% | 完整−共同基础pp [95%CI] | 三项移除均明确正向 | 尚不明确的模块 |
|---|---|---|---|---|---|
| Cora | Edgepred_Gprompt | 73.92 ± 2.35 | +2.370 [+1.635, +3.330] | 否 | M1 |
| CiteSeer | Edgepred_Gprompt | 68.30 ± 2.10 | +4.570 [+3.860, +5.240] | 是 | 无 |
| Actor | GraphCL | 21.78 ± 1.60 | +0.211 [-0.250, +0.681] | 否 | M1、M2、M3 |
| Cora | Edgepred_GPPT | 71.57 ± 2.78 | +2.930 [+2.275, +3.595] | 否 | M2 |
| CiteSeer | Edgepred_GPPT | 65.61 ± 2.09 | +6.270 [+5.405, +7.260] | 否 | M2 |
| Cora | DGI | 50.60 ± 3.60 | +19.290 [+17.135, +21.490] | 是 | 无 |
| CiteSeer | DGI | 47.34 ± 5.29 | +14.805 [+12.045, +17.580] | 是 | 无 |

**完整方案整体提升**：DGI、Edgepred_Gprompt、Edgepred_GPPT三个家族各自在Cora与CiteSeer上，共6个已确认格。**三个模块分别有清晰贡献**：DGI/Cora、DGI/CiteSeer、Edgepred_Gprompt/CiteSeer，共3个格、2个家族。Actor/GraphCL不成立。未将全checkpoint旧5-seed筛选格升级成20-seed确认结论。

| 三模块成立范围 | M1增量pp [95%CI] | M2增量pp [95%CI] | M3增量pp [95%CI] |
|---|---|---|---|
| CiteSeer/Edgepred_Gprompt | +1.400 [+0.830, +1.935] | +0.895 [+0.210, +1.735] | +0.740 [+0.440, +1.030] |
| Cora/DGI | +9.025 [+7.330, +10.575] | +12.480 [+9.670, +15.455] | +2.895 [+1.775, +3.965] |
| CiteSeer/DGI | +8.270 [+5.755, +10.700] | +10.260 [+7.695, +13.010] | +3.780 [+2.875, +4.745] |

## 与历史seedgrid的可比范围

以下历史值来自本地原始CSV，按同名预训练家族检索，并非论文原文成绩。历史CSV没有逐次checkpoint哈希，所指向的远端预测目录不在本地；本次未对历史预测重新做审计。现存本地checkpoint只能用于辅助检查，不能证明它就是当时seedgrid使用的文件。

- Cora：三个现存本地checkpoint均与本次确认哈希相同，但历史运行身份仍未闭合。
- CiteSeer：三个现存本地checkpoint均与本次确认不同，因此尤其不能称严格同checkpoint比较。
- 历史为3个split seed（1729/17290/172900）×3个train seed（42/4242/424242），每行5 folds。本表先取每行fold均值，再对9行求均值及样本标准差；**不是45次独立重复的置信区间，也不是CSV内fold标准差的平均**。
- 本方案为各阶段20个新支持seed、100步、固定官方训练池/测试查询。历史v4同为5-shot、官方池、GCN两层128维，但默认200 support passes，且不同Prompt有不同子图、读出和优化设置（All-in-one另有交替预算）。两者无逐seed配对，不作跨协议显著性、相对增益或冠军声明。
- 支持池来源和NormalizeFeatures由现存运行代码核对；历史实际源代码/预测级一致性仍未复核。

## DGI：历史Prompt与当前方案

| 方法 | Cora Accuracy % | CiteSeer Accuracy % | 来源 |
|---|---|---|---|
| GPF | 39.55 ± 1.81 | 40.49 ± 2.08 | 历史seedgrid |
| GPF-plus | 46.83 ± 2.19 | 37.57 ± 3.11 | 历史seedgrid |
| Gprompt | 49.48 ± 1.09 | 55.68 ± 2.33 | 历史seedgrid |
| All-in-one | 20.79 ± 3.06 | 26.05 ± 3.55 | 历史seedgrid |
| GPPT | 24.65 ± 2.85 | 26.42 ± 6.86 | 历史seedgrid |
| SUPT-soft | 40.14 ± 3.60 | 39.16 ± 2.55 | 历史seedgrid |
| SUPT-hard | 40.54 ± 3.23 | 37.64 ± 2.55 | 历史seedgrid |
| GraphTOP | 21.60 ± 2.24 | 26.64 ± 2.00 | 历史seedgrid |
| None（全量微调） | 59.98 ± 2.75 | 41.66 ± 1.95 | 历史seedgrid |
| Frozen | 22.35 ± 0.98 | 24.94 ± 1.40 | 历史seedgrid |
| FrozenNodePrototype | 38.33 ± 2.82 | 41.50 ± 2.20 | 历史seedgrid |
| FrozenIdentityPrototype | 20.94 ± 0.76 | 26.07 ± 0.62 | 历史seedgrid |
| FrozenGraph | 28.49 ± 2.34 | 40.89 ± 2.32 | 历史seedgrid |
| FrozenGraphPrototype | 58.29 ± 1.95 | 58.72 ± 0.82 | 历史seedgrid |
| GraphTOP-Frozen | 23.87 ± 2.37 | 25.03 ± 2.59 | 历史seedgrid |
| 三模块F111 | 50.60 ± 3.60 | 47.34 ± 5.29 | 本次20支持seed；不同协议 |

历史各方法n=9个seed pair，完整方案n=20个支持seed；两种标准差的统计单位不同。

## Edgepred_Gprompt：历史Prompt与当前方案

| 方法 | Cora Accuracy % | CiteSeer Accuracy % | 来源 |
|---|---|---|---|
| GPF | 34.29 ± 4.23 | 28.54 ± 3.47 | 历史seedgrid |
| GPF-plus | 54.86 ± 2.45 | 45.08 ± 1.00 | 历史seedgrid |
| Gprompt | 53.23 ± 1.40 | 38.60 ± 1.68 | 历史seedgrid |
| All-in-one | 17.88 ± 3.01 | 21.17 ± 1.73 | 历史seedgrid |
| GPPT | 24.64 ± 2.30 | 26.20 ± 3.90 | 历史seedgrid |
| SUPT-soft | 29.49 ± 1.44 | 26.64 ± 1.36 | 历史seedgrid |
| SUPT-hard | 29.66 ± 2.31 | 23.02 ± 2.38 | 历史seedgrid |
| GraphTOP | 39.95 ± 0.89 | 32.70 ± 0.62 | 历史seedgrid |
| None（全量微调） | 62.50 ± 3.25 | 46.43 ± 3.18 | 历史seedgrid |
| Frozen | 31.90 ± 0.54 | 26.78 ± 1.17 | 历史seedgrid |
| FrozenNodePrototype | 31.67 ± 0.41 | 28.65 ± 1.11 | 历史seedgrid |
| FrozenIdentityPrototype | 18.32 ± 1.68 | 21.75 ± 0.55 | 历史seedgrid |
| FrozenGraph | 45.61 ± 2.09 | 41.13 ± 1.69 | 历史seedgrid |
| FrozenGraphPrototype | 53.31 ± 0.79 | 41.36 ± 1.04 | 历史seedgrid |
| GraphTOP-Frozen | 34.29 ± 0.45 | 27.89 ± 1.25 | 历史seedgrid |
| 三模块F111 | 73.92 ± 2.35 | 68.30 ± 2.10 | 本次20支持seed；不同协议 |

历史各方法n=9个seed pair，完整方案n=20个支持seed；两种标准差的统计单位不同。

## Edgepred_GPPT：历史Prompt与当前方案

| 方法 | Cora Accuracy % | CiteSeer Accuracy % | 来源 |
|---|---|---|---|
| GPF | 44.44 ± 2.97 | 27.59 ± 3.04 | 历史seedgrid |
| GPF-plus | 62.06 ± 1.27 | 44.33 ± 2.20 | 历史seedgrid |
| Gprompt | 54.75 ± 1.55 | 38.53 ± 2.15 | 历史seedgrid |
| All-in-one | 40.96 ± 2.78 | 31.94 ± 2.06 | 历史seedgrid |
| GPPT | 35.47 ± 0.68 | 26.73 ± 0.84 | 历史seedgrid |
| SUPT-soft | 45.38 ± 2.17 | 30.64 ± 2.66 | 历史seedgrid |
| SUPT-hard | 47.93 ± 1.18 | 32.91 ± 3.20 | 历史seedgrid |
| GraphTOP | 35.50 ± 1.37 | 28.32 ± 1.65 | 历史seedgrid |
| None（全量微调） | 57.31 ± 3.03 | 43.57 ± 1.77 | 历史seedgrid |
| Frozen | 29.54 ± 0.78 | 25.28 ± 0.51 | 历史seedgrid |
| FrozenNodePrototype | 28.91 ± 0.95 | 25.72 ± 0.74 | 历史seedgrid |
| FrozenIdentityPrototype | 19.26 ± 0.98 | 19.17 ± 0.96 | 历史seedgrid |
| FrozenGraph | 52.68 ± 2.25 | 38.78 ± 0.76 | 历史seedgrid |
| FrozenGraphPrototype | 53.93 ± 1.05 | 37.65 ± 2.22 | 历史seedgrid |
| GraphTOP-Frozen | 30.32 ± 0.64 | 26.00 ± 0.51 | 历史seedgrid |
| 三模块F111 | 71.57 ± 2.78 | 65.61 ± 2.09 | 本次20支持seed；不同协议 |

历史各方法n=9个seed pair，完整方案n=20个支持seed；两种标准差的统计单位不同。

## 同次实验的严格配对基线

本表来自三模块确认的相同checkpoint、支持集、查询集与预算。graphprompt为本次轻量实现的GraphPrompt控制，不等于历史ProG Gprompt的子图训练实现。

| 数据集/家族 | 共同基础F000 | 本次frozen | 本次graphprompt | 完整F111 |
|---|---|---|---|---|
| Cora/Edgepred_Gprompt | 71.55 ± 2.65 | 32.47 ± 4.39 | 32.35 ± 4.35 | 73.92 ± 2.35 |
| CiteSeer/Edgepred_Gprompt | 63.74 ± 2.34 | 28.88 ± 3.13 | 28.46 ± 3.00 | 68.30 ± 2.10 |
| Cora/Edgepred_GPPT | 68.64 ± 2.52 | 30.00 ± 2.47 | 29.79 ± 2.28 | 71.57 ± 2.78 |
| CiteSeer/Edgepred_GPPT | 59.33 ± 2.06 | 25.81 ± 2.69 | 25.69 ± 3.06 | 65.61 ± 2.09 |
| Cora/DGI | 31.31 ± 4.98 | 35.97 ± 3.73 | 36.28 ± 4.21 | 50.60 ± 3.60 |
| CiteSeer/DGI | 32.53 ± 2.74 | 45.70 ± 6.00 | 46.31 ± 6.03 | 47.34 ± 5.29 |

## 对结果的判断

1. **三模块贡献最齐全的是DGI，但绝对性能偏低。** Cora/DGI完整50.605%，历史Gprompt49.482%（仅描述性并列），历史FrozenGraphPrototype58.291%；CiteSeer/DGI完整47.340%，历史Gprompt55.684%、FrozenGraphPrototype58.722%。DGI的内部消融提升不能写成对现有方法全面胜出。
2. **论文性能展示更值得关注CiteSeer/Edgepred_Gprompt。** 完整68.305%，三个模块移除增量均明确为正。不过本次中心化直接教师70.405%仍高2.100 pp，且历史CiteSeer checkpoint不一致，不能把与历史Prompt的大幅分差全归因于新模块。
3. **Cora的Edgepred_Gprompt与Edgepred_GPPT整体提升明确，但三模块叙事不完整。** 前者M1不明确，后者M2不明确；GPPT/CiteSeer的M2也跨零。不能把整体提升替代逐模块证据。
4. 若要制作论文主实验排名，需要在同一checkpoint、相同支持/查询、同一输入和预算下重跑ProG方法。本次完成历史数据提取和可比性检查，没有新训练。

## 未支持项与覆盖

每个历史家族/数据集有15个complete方法、9个skipped方法，各9行。共提取1296行，其中810 complete、486 skipped。Skipped为Prodigy、GraphPrompter、EdgePrompt、EdgePromptplus、UniPrompt、SelfPro、ProNoG、PSP、DHBP；保留缺口，不填0。MultiGprompt是自带预训练的独立组合，不属于这三种指定底座。额外RSP seedgrid只有其自身已跑底座，未混入当前表。

## checkpoint身份辅助检查

| 数据集/家族 | 当前实验SHA256 | 现存本地ProG SHA256 | 现存文件一致 |
|---|---|---|---|
| Cora/DGI | fb2b361637cbcc95e5885fecaf40a7ad45c80001f554ec7e19c203a56d925de9 | fb2b361637cbcc95e5885fecaf40a7ad45c80001f554ec7e19c203a56d925de9 | 是 |
| Cora/Edgepred_Gprompt | 87bd546f7897218161bfb0876e792cf8f4682f3a42d6fc6e2227b9aa0f723493 | 87bd546f7897218161bfb0876e792cf8f4682f3a42d6fc6e2227b9aa0f723493 | 是 |
| Cora/Edgepred_GPPT | a279caf82ae2d2fdbfc280143494c82054738b4fa2166034380ad83c971d3f7b | a279caf82ae2d2fdbfc280143494c82054738b4fa2166034380ad83c971d3f7b | 是 |
| CiteSeer/DGI | 460f97d1cd973575e7d2d28b08a1b4532d50bd55e5edc0fc777c3820ce616e5f | 8f4dd5622b98873830af823ef05e637a78d7d6a9e3dcd37837467e5902842f6b | 否 |
| CiteSeer/Edgepred_Gprompt | 7bf67dc55e3a33541646a03e797ad32e1e16ce2f7a4be4c1fe2064d900de6e0b | 5b3b70a23a7853be9c376db8d4ef054399566d15cc51c1b7be32e62816fac522 | 否 |
| CiteSeer/Edgepred_GPPT | e5488f6ef53701dc38f4401933e1bce64cae52406ca37bd41a9a0ae53762243d | 71a6af7c13c889d55fe664bbfcffc318eb0782418e6eabbf21731a6d6b4b3c51 | 否 |

## 来源与复核

统计来源：

- [confirmation_pairs.json](/Users/shq/Research/GP/持续推进/主线03_硕士论文三模块_20260924/stage2_confirmation/confirmation_pairs.json)，SHA256 `1bed90edb85f88e656e39e64eada2d61179cd3b65435d6f463c150ffee74e80c`
- [confirmation_pairs.json](/Users/shq/Research/GP/持续推进/主线03_硕士论文三模块_20260924/stage2b_gppt_confirmation/confirmation_pairs.json)，SHA256 `615b97b81712d2203ab28dd5d26074d601d346e556aa3008a119a46dda578b9b`
- [confirmation_pairs.json](/Users/shq/Research/GP/持续推进/主线03_硕士论文三模块_20260924/stage2c_dgi_confirmation/confirmation_pairs.json)，SHA256 `74d45958a34e3f7948f653c7c8adea01055f9b4d1c7256432083bf9e29eab989`
- [cora_5shot_full_matrix_seedgrid.csv](/Users/shq/Research/GP/rep/ProG/results/cora_5shot_full_matrix_seedgrid.csv)，SHA256 `70c7f953f9b6ca61f63573e7801e6dea33cff77c1b04e0a252ba1d540d061728`
- [citeseer_5shot_full_matrix_seedgrid.csv](/Users/shq/Research/GP/rep/ProG/results/citeseer_5shot_full_matrix_seedgrid.csv)，SHA256 `3f2cd321a875a2abd4ba133115de4f63a1e532419c7275c4d84aa8c5618a4fb1`

分析脚本、提取记录（保留run_id/status）与完整Accuracy/Macro-F1聚合值位于本目录。历史输入CSV未修改。
