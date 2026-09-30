# 图小样本 Prompt 准确率持续探索

开始：2026-09-23。目标：持续借鉴其他领域方法并在真实预训练权重上验证准确率；不以查新或原创性淘汰方法。主服务器：`root@connect.westb.seetacloud.com:51542`。本目录保存代码、配置、逐次结果和预测；本文为总入口。

## 当前可保留结果

第64轮新增CiteSeer Prompt局部提升：GraphMAE2中心化教师门控无KD+k50为67.49%，旧提示61.06%，+6.43pp [4.06,8.92]；但低于直接教师69.10%。这是更强Prompt基线，非整体最高准确率。见[当前CiteSeer提示方案](当前CiteSeer提示方案.md)。

第63轮新增CiteSeer直接教师中心化：两批20支持分别69.07%对67.71%，+1.36pp [0.78,1.97]；68.97%对67.20%，+1.77pp [1.15,2.49]。不是Prompt提升，后续再验证接入提示训练。见[当前CiteSeer强教师](当前CiteSeer强教师.md)。

第53轮新增：Cora/GraphMAE动态整体CE缩放＋k50传播，在第二批20支持达到76.48%，对旧传播75.52%高0.96pp [0.54,1.51]；Focal/置换对照76.53%，差异不明确，采用简单整体缩放作为后续强根。见[当前可复核方案](当前Cora强方案.md)。同图重复查询边界不变。

第47轮匹配50邻居后：Cora/GraphMAE概率传播76.25%，原10邻居75.65%，+0.60pp [0.27,0.93]；GraphMAE2 74.08%对73.50%，+0.58 [0.10,1.06]。表示聚合未超过同规模概率传播，优先保留k50概率方案；同图重复查询限制不变。

第46轮：Cora强提示后的均匀50邻居表示聚合，GraphMAE 75.17%对原74.01%，+1.16pp [0.50,1.93]；GraphMAE2 74.03%对原72.14%，+1.89 [1.21,2.65]。仍为传导读出；与旧10邻居概率传播的比较混合邻域规模，第47轮已补齐：同规模概率传播更好，见下文。

第41轮传导读出：Cora/GraphMAE2 73.10%对原72.67%，+0.43pp [0.02,0.86]；同批更强GraphMAE传播75.92%对原75.22%，+0.70 [0.20,1.17]，后者为次要对照结果；现新增20支持集完整复核，GraphMAE 73.76%→74.74%，+0.98pp [0.53,1.44]，见第41轮补充。

最新局部改进：第39轮PubMed/GraphMAE的VAT一致性75.99%，对同批原提示75.63%高0.36pp [0.09,0.63]，对随机一致性高0.38pp；仍低于同批GraphMAE2原提示76.43%。详情及限制见第39轮确认。

截至第七十二轮及确认，共 **33965 条最终查询指标**已在服务器完成并核对预测，全部33965条本地预测及邻接SHA核验通过，另有600次支持内部拟合。第九轮证实Amazon输入归一化会显著压低GraphMAE的Frozen基线；后续Amazon实验采用与当前ProG一致的原始主干输入，教师预处理单独固定。下表含多轮各20支持种子的结果，不同轮支持种子不同，不用于不配对的权重排名；均是同图配对验证，不是新数据集独立测试。

|数据集 / 权重|提示方案|准确率|对同权重 Frozen|对普通 GraphPrompt|主要限制|
|---|---|---:|---:|---:|---|
|Cora / GraphMAE|第一层条件 token + KD，gamma3|67.92%|+3.59 pp|+3.46 pp|rawprop2 68.66%，略高于提示；提示比上一轮最佳 +1.59 pp|
|CiteSeer / GraphMAE|同上|44.24%|+10.13 pp|+10.37 pp|明显低于 rawprop2 61.84%|
|PubMed / DGI|同上|60.43%|+21.99 pp|+20.85 pp|明显低于 rawprop2 70.55%|
|PubMed / GraphMAE|第一层条件 token + KD，gamma1|73.99%|+1.92 pp|+0.38 pp|对 GraphPrompt 的区间跨零；高于 rawprop2 70.55%|
|PubMed / GraphMAE2|第一层条件 token + KD，gamma3|73.88%|+5.12 pp|+4.28 pp|同轮 rawprop2 70.84%，提示高3.04 pp|
|Cora / GraphMAE|岭教师条件token+KD，gamma10|72.96%|+8.61 pp|+8.83 pp|直接岭教师69.92%，提示高3.04 pp|
|Cora / GraphMAE2|岭教师条件token+KD，gamma10|71.93%|+11.88 pp|+13.84 pp|直接岭教师69.92%，提示高2.01 pp|
|CiteSeer / GraphMAE2|岭教师条件token+KD，gamma10|59.24%|+19.13 pp|+18.68 pp|直接岭教师60.46%，提示仍低1.22 pp|

|Cora / GraphMAE|后层岭教师token，gamma10|73.49%|+9.58 pp|+9.52 pp|同轮原位置72.35%；后层增量+1.14 pp，区间[0.43,1.84]|
|Cora / GraphMAE|普通平滑岭教师token，gamma10|74.42%|+10.03 pp|+10.09 pp|同轮原教师提示73.89%；增量+0.53 pp，区间[0.21,0.81]；平滑直接教师72.78%|

|Cora / GraphMAE|后层+普通平滑岭教师token|75.78%|+11.44 pp|+11.34 pp|同轮最好单项75.17%，组合高0.61 pp，[0.33,0.88]；直接平滑教师73.34%|

|PubMed / GraphMAE|修正原型教师beta0.5＋gamma3早层token|74.55%|+3.05 pp|+0.95 pp|对原提示72.43%高2.12 pp，[1.31,2.98]；对GraphPrompt区间跨零|

新增固定几何正向结果：Actor/GraphMAE仅自环基础+中心化及对角收缩26.99%，同轮未校正基础22.63%、原图Frozen20.56%、原始岭20.80%；对基础+4.37pp区间[3.58,5.18]。后补同支持中心化对照26.60%，对角缩放额外+0.39pp；内部token26.95%，没有新增训练收益。该项是固定表示/度量校正，不是Prompt学习创新，见第32轮。

新增固定结构正向结果：Actor/GraphCL将两层自环权重固定100，23.30%，同轮Frozen20.95%、原始特征原型20.94%；配对增量+2.35/+2.36pp区间均为正。但仅自环23.26%，二者差0.05pp区间跨零，收益不能归因于学习Prompt，主要保留为传播方式调整。详见第30轮。

另有独立教师分类器正向结果：CiteSeer修正原型教师66.99%，原rawprop2教师60.62%、平滑岭教师62.53%；这是非Prompt的直接分类改进，单列保留。

保留限定条件提升，继续寻找更强 checkpoint 和提示匹配。没有将“提升弱基线”描述为超过所有方法，也不以是否已有论文做过来淘汰这些方案。

## 运行资源与历史经验

- 已实测 SSH 私钥登录成功，GPU 空闲，显存 32760 MiB；复用 `/root/autodl-tmp/prog_env/bin/python`，torch 2.12.1+cu130、CUDA 可用，ProG 可导入。
- 权重在 `/root/ProG/Experiment/pre_trained_model`；Cora、CiteSeer 包含 DGI、GraphMAE、GraphCL 等多种 GCN 权重。DGI 是两层 128 维 GCN。权重 sidecar 标注 ProG-adapted、source_parity=not_fully_verified，不宣称上游严格复现。
- 参考 `outputs/多轮Prompt方案失败归因与研究反思_2026-09-10.md`：训练 loss 下降不等于查询提升；弱分类头增益需要强基线核对；分类头融合和真正改变主干内部计算的 Prompt 分别记录。保留此前失败信息，但不沿用此前新颖性门槛。

## 固定评估约定

1. 首先做节点 5-shot 分类：Cora、CiteSeer，DGI/GraphMAE/GraphCL，每类仅用官方训练池 5 个标签。主干 eval 且全冻结。
2. 每个方法用同一 checkpoint、支持节点、查询节点与原型分类规则。输出 Frozen 原型和普通 GraphPrompt 两个直接对照，另用原始特征原型作为诊断。
3. 官方 test 节点按无标签固定随机排列分为 500 开发节点和剩余确认节点；官方 validation 标签不使用。首轮仅评分开发查询。全图无标签特征/边允许使用，属于传导设定。
4. 首轮 seed=4100..4104；最终候选固定后用 seed=5100..5119 和预留查询确认。支持种子不是独立图，预留查询也不是全新数据集；历史研究可能已见官方 test，本轮仅保证当前搜索不访问预留标签。
5. 预测先保存并计算 SHA256，再评分；保存 checkpoint/source/graph/split 哈希。支持标签用于训练与原型，查询标签只在评分函数使用。逐任务落盘、可续跑。
6. 开发集探索允许比较配置，明确为开发选择；确认集结果不得用于修改同一轮候选并反复确认。局部提升也保留，不强求所有数据集提升。

## 第一轮：保持预训练预测的中间层提示

借鉴 [ProGrad, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.html)：少样本更新可能破坏已有预测知识。本文实现的是 KL 锚定变体，不是原文梯度投影复现。借鉴 [Manifold Mixup, ICML 2019](https://proceedings.mlr.press/v97/verma19a.html)：在支持表示空间插值，抑制尖锐分类边界。备用参考 [Tip-Adapter, ECCV 2022](https://arxiv.org/abs/2207.09519)：支持样本缓存检索可作为后续强读出对照。

冻结 GCN 第一层得到 $ H=\operatorname{LeakyReLU}(\hat A XW_1+b_1) $。中间层 Prompt 为 $ H'=H\odot(1+0.5\tanh a)+0.5\operatorname{std}(H)\odot\tanh b $，然后真实进入同一个冻结第二层 $ Z'=\hat A H'W_2+b_2 $。所有节点共享 256 个可学习参数，无末端多模型融合。零参数严格还原主干。

普通 GraphPrompt 对照为 $ Z'=Z\odot(1+0.5\tanh a) $，128 个参数，使用相同余弦原型与优化预算。这是受界幅度控制的 GraphPrompt 风格对照，不称完整原论文默认配置。

训练以支持集 leave-one-out 原型交叉熵为主，温度 0.2、Adam lr=0.01、100 步，固定参数 L2=0.001。比较：Frozen、GraphPrompt、HiddenPrompt、HiddenPrompt+KL、HiddenPrompt+Mixup、HiddenPrompt+KL+Mixup。KL 系数 1，对全图不读标签，以无提示支持原型预测为锚；Mixup 系数 0.5，固定混合权重 0.5，支持表示随机配对。KL 和 Mixup 的实际效果待实验确认。

工程预检：六种方法各 3 步 smoke 已完成，冻结参数检查、split 分离检查通过。首次 no-op 检查观察到 CUDA float32 scatter 累加差异（最大绝对值 1.14e-5）；核对同一 GCN forward 后，改用 atol=5e-5/rtol=1e-5 及单位化表示误差小于 1e-5 的双重数值检查。不是声称逐位完全一致。3 步 smoke 不作为准确率效果结论。

正式首轮：2 数据集 × 3 checkpoint × 5 seed × 6 方法，共 180 个任务，100 步预算，已完成。

### 第一轮开发结果（180/180 完成）

[全部结果](round1_dev/summary.md) · [复算审计](round1_dev/audit.json)。所有封存预测哈希、准确率复算、配对 split/checkpoint 身份通过。

- CiteSeer / GraphMAE：HiddenPrompt 35.08%，Frozen 33.24%，GraphPrompt 33.76%；分别 +1.84/+1.32 pp。KL 与不加 KL 持平，固定较简单的 HiddenPrompt 进入确认。
- Cora / GraphCL：HiddenPrompt+KL 47.92%，Frozen 45.80%，GraphPrompt 47.52%；分别 +2.12/+0.40 pp，进入确认。
- 其他条件无中间层提示的明确双对照优势；Mixup 本轮没有独立优点。停止这些具体配置，保留负结果，不将方案整体判死。
- 候选与确认协议在读取确认评分前保存为 [round1_selection.json](round1_selection.json)。每项 20 个新支持种子 × 3 方法，共 120 任务。确认结果不用于回头修改第一轮方案。

### 第一轮确认（120/120 完成，预测审计通过）

|条件|候选准确率|对 Frozen|对 GraphPrompt|处置|
|---|---:|---:|---:|---|
|CiteSeer / GraphMAE / HiddenPrompt|35.84%|+1.22 pp，种子区间 [0.68,1.80]，15胜1平4负|+0.25 pp，区间 [-0.80,1.28]，13胜7负|保留对 Frozen 的局部增益；尚不能明确胜过普通 GraphPrompt|
|Cora / GraphCL / HiddenPrompt+KL|45.08%|+0.57 pp，区间 [-0.15,1.26]，12胜8负|+0.71 pp，区间 [0.17,1.24]，15胜1平4负|保留对 GraphPrompt 的局部增益；对 Frozen 不确定|

两者未得到同时稳健超过双对照的证据，不夸大结论。用户目标允许任何准确率提升，因此作为限定条件的实测正向结果保留。不能宣称跨图泛化或最高性能。[Cora](confirm1_Cora/summary.md) · [CiteSeer](confirm1_CiteSeer/summary.md)。

## 第二轮：中心化几何上的提示

来源：[SimpleShot](https://arxiv.org/abs/1911.04623) 的中心化与 L2 归一化；[Channel Importance Matters, ICML 2022](https://proceedings.mlr.press/v162/luo22c.html) 的通道变换。此处借鉴原则，不宣称原文复现。

动机：第一轮绝对准确率有些偏低，提示可能只是在弱原型几何上改善。先用全图无标签均值 $ \mu=\operatorname{mean}(Z_0) $ 定义固定中心化 $ T(Z)=Z-\mu $；另有弱尺度校正 $ T_s(Z)=(Z-\mu)/\sqrt{\operatorname{std}(Z_0)} $，std 下限 0.001。几何统计在训练前固定，提示仍进入冻结第二层；不得把纯中心化收益归给 Prompt。

条件：Frozen、原 GraphPrompt、centered、centered GraphPrompt、centered HiddenPrompt、centered HiddenPrompt+KL、sphered、sphered HiddenPrompt+KL、raw 原始特征原型。相同 6 数据集/权重条件和 5 支持种子，共 270 任务；只读已用于开发的 500 查询节点。保留 round1 不变，独立源文件 run_round2.py。任何下一轮确认都明确披露预留查询已被第一轮确认使用，不再称完全未见查询。

### 第二轮开发结果（270/270，审计通过）

[结果表](round2_dev/summary.md)。中心化不是普适收益：DGI/Cora 从 37.32% 降到 14.76%，DGI/CiteSeer 从 43.56% 降到 24.28%。未将视觉经验直接当作图上的事实。

Cora/GraphMAE 是较强条件：sphered HiddenPrompt+KL 63.72%，Frozen 60.80%、原 GraphPrompt 61.76%、sphered 无提示 60.76%、centered GraphPrompt 63.24%。固定此候选，增加相同 sphered 几何的 GraphPrompt 控制，并直接迁移到尚未在本任务评分的 PubMed/GraphMAE。新支持 seed=6100..6119，2 数据集 × 6 方法 × 20=240 任务，固定清单见 [round2_selection.json](round2_selection.json)。

CiteSeer 原始特征原型 44.56%，高于 GraphMAE 系列所有提示，说明这些局部提示增益尚无实际最佳性。Cora 与 CiteSeer 的 DGI/GraphCL 中心化不扩展。

观察：跨轮复跑 CiteSeer/DGI GraphPrompt 差 0.04 pp，符合 float32 非确定累加与分类边界敏感性；所有比较以同轮配对为准，不把小于此量级差异归因于新机制。

### 第二轮新支持验证与 PubMed 迁移（240/240）

[汇总](confirm2/summary.md)与预测复算已完成。Cora/GraphMAE 的 sphered HiddenPrompt+KL 为 **66.74%**，Frozen 64.47%、原 GraphPrompt 64.27%、同几何 GraphPrompt 65.97%；分别 **+2.27/+2.47/+0.77 pp**。纯 sphered 基线只有 63.66%，因此改善不是纯无监督尺度校正造成。保留为当前训练型中间层 Prompt 正向方案，但需对照第三轮发现的强 rawprop2。

不调参迁移 PubMed/GraphMAE：候选 72.50%，Frozen 71.76%、原 GraphPrompt 72.72%、同几何 GraphPrompt 70.92%。对 Frozen +0.74 pp，但对原 GraphPrompt -0.22 pp，迁移没有显示最强基线优势，不声称通用胜出。

Cora 20 支持种子的配对 bootstrap：对 Frozen +2.27 pp，95%区间 [1.44,3.16]，18胜2负；对原 GraphPrompt +2.47，[1.71,3.28]，18胜1平1负；对同几何 GraphPrompt +0.77，[0.28,1.27]，14胜2平4负。区间反映同一图的支持采样，不是跨图证据。

## 第三轮：支持记忆检索生成输入提示

跨领域来源：[Modern Hopfield Networks and Attention for Immune Repertoire Classification, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/da4902cb0bc38210839714ebdcf0efc3-Abstract.html) 与 [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) 的关联记忆检索；结合 Tip-Adapter 的少样本缓存思想。这里只实现一次 softmax 记忆检索，不宣称完整 Hopfield 动力学或论文复现。

设计：无标签图传播 $ R=\hat A^2X $，支持标签形成每类键 $ K_c=\operatorname{mean}_{i\in S_c}R_i $ 和值 $ V_c=\operatorname{mean}_{i\in S_c}X_i $。检索 $ Q=\operatorname{softmax}(\operatorname{cos}(R,K)/0.2) $，输入提示为 $ P_\alpha=\alpha(QV-X) $。真正向同一冻结 GCN 输入 $ X+P_\alpha $，然后保持原余弦原型分类。提示由支持任务条件化，无额外梯度训练；与训练型提示单列。

预定 alpha=0.25/0.5/0.75。对照包含 Frozen、GraphPrompt、raw、rawprop2、直接在 $ X+P_{0.5} $ 上分类（绕过主干）、随机打乱支持样本的类别分组生成记忆、仅全局均值输入平滑。类别名称整体置换会严格抵消，不作为无效的“打乱”对照；实际打乱节点到类别的分组。三个数据集 Cora/CiteSeer/PubMed × DGI/GraphMAE/GraphCL × 5 支持 seed × 10 方法=450 任务。使用开发查询，PubMed 开发半区在本任务尚未评分，但结果仍计为开发探索。

### 第三轮开发结果（450/450，审计通过）

[汇总](round3_dev/summary.md)。PubMed/DGI input075 达 52.60%，Frozen 34.16%、GraphPrompt 35.80%，提升 +18.44/+16.80 pp；Cora/GraphCL input075 为 51.04%，对 Frozen/GraphPrompt +5.24/+3.52 pp。这是实际进入同一冻结 GCN 的任务条件输入提示，保留其准确率改善。

但它们低于直接 rawprop2：Cora 69.56%、CiteSeer 61.76%、PubMed 70.16%。多数 DGI/CiteSeer 输入提示严重退化；完整记录负结果。下一步固定 PubMed/DGI 与 Cora/GraphCL 的 input075，用新支持/后半查询检验，并补相同 alpha 的打乱与均值控制。开发探索中其他条件不扩大。

### 第三轮新支持验证（280/280，审计通过）

固定清单：[round3_selection.json](round3_selection.json)。每条件 20 支持种子，相同 alpha=0.75 的三种提示全部配对。

|条件|输入提示|Frozen|GraphPrompt|打乱支持类别分组|全局均值提示|直接提示特征分类|rawprop2|
|---|---:|---:|---:|---:|---:|---:|---:|
|PubMed / DGI|50.49%|37.28%|37.77%|47.29%|38.84%|70.67%|71.77%|
|Cora / GraphCL|47.05%|44.49%|45.32%|44.13%|43.75%|61.38%|68.92%|

PubMed/DGI 对 Frozen +13.21 pp，对普通 GraphPrompt +12.72；Cora/GraphCL 分别 +2.56/+1.73。真实分组好于打乱，但 PubMed 打乱本身也提升 10.01 pp，因此不能把全部提升归于正确类别记忆；大部分可能是特征分布改变，具体归因仍需控制。两者保留为局部 Prompt 提升，不推荐替代强 rawprop2。 [Cora 结果](confirm3_Cora/summary.md) · [PubMed 结果](confirm3_PubMed/summary.md)。

## 第四轮：把强教师信息蒸馏到第一层条件提示

来源：[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 和 [PromptKD, CVPR 2024](https://arxiv.org/abs/2403.02781)。第三轮直接 rawprop2 强于多个冻结权重，因此下一步把它作为无额外标签的教师，避免强行坚持较弱主干原预测。不是照搬 CLIP 双编码器架构。

令 $ Q=\operatorname{softmax}(\operatorname{ProtoCos}(\hat A^2X,S)/0.2) $，只依赖支持标签。固定门控 $ B=\hat A(Q-\operatorname{mean}(Q)) $；每类一个 128 维可训练 token $ T $，在冻结第一层激活前加入 $ P=\gamma BT_{\tanh}\odot\operatorname{std}(U_0) $，$ H'=\operatorname{LeakyReLU}(U_0+P) $，再进入原冻结第二层。主干参数不动，训练参数为类别数乘 128；原型头不另训练。初始 token=0 为 no-op。

训练：支持 LOO 交叉熵 + 教师 KL（有/无）+0.001 token L2，100 步 lr0.01。固定 gamma=1 或3，不据 query 早停。方法包括 Frozen、GraphPrompt、rawprop2、token、token_kd、token_kd3、token_kd_perm（随机打乱节点门控对应）、token_random（固定随机 token，正确门控）。3 数据集 × 3 权重 × 5 种子 × 8 方法=360 任务，均在开发查询评分。教师直接预测是必要强对照；超过 Frozen 而低于教师仍可记录准确率提升，但不是最优方案或蒸馏增益。

### 第四轮开发结果（360/360，审计通过）

[完整表](round4_dev/summary.md)。PubMed/GraphMAE 的 token_kd 为 **76.24%**，Frozen 71.64%、GraphPrompt 75.20%、rawprop2 教师70.16%；对 GraphPrompt +1.04 pp，对教师 +6.08 pp。这是当前值得继续确认的组合；没有宣称蒸馏普遍提高。

gamma=3 的其他局部正向：Cora/GraphMAE 66.32%（对 Frozen +5.52、GraphPrompt +4.56，但低于教师69.56%）；CiteSeer/GraphMAE 46.28%（+13.04/+12.52，低于教师61.76%）；PubMed/DGI 59.72%（+25.56/+23.92，低于教师70.16%）。DGI 的 Cora/CiteSeer 明显退化，停止这些条件。

固定上述4条件，以 seed8100..8119 进行560个新支持配对；对 gamma=3 补同幅度不蒸馏、打乱门控、固定随机 token 三个控制。原开发 gamma=1 控制不用于解释 gamma=3 的净贡献。候选清单：[round4_selection.json](round4_selection.json)。另对 Cora/GraphMAE 重跑此前 sphered HiddenPrompt+KL，支持与查询完全匹配，以免只比较较弱基线。

### 第四轮新支持验证（560+40/600 完成，审计通过）

结果见页首总表。[Cora](confirm4_Cora_GraphMAE/summary.md)、[CiteSeer](confirm4_CiteSeer_GraphMAE/summary.md)、[PubMed/DGI](confirm4_PubMed_DGI/summary.md)、[PubMed/GraphMAE](confirm4_PubMed_GraphMAE/summary.md)。

- Cora/GraphMAE 对 Frozen 的种子95%区间 [2.91,4.23] pp；对 GraphPrompt [2.66,4.24]。对旧 sphered HiddenPrompt+KL：67.92% 对66.33%，+1.59 pp，[0.58,2.62]，14胜1平5负；支持/query/checkpoint 身份均相同，见 [跨轮配对](round4_vs_round2.json)。
- CiteSeer/GraphMAE 对 Frozen +10.13，[8.69,11.64]；PubMed/DGI +21.99，[19.14,24.69]，两者20/20支持种子均优于 Frozen，但仍不及教师。
- PubMed/GraphMAE 对 Frozen +1.92，[0.65,3.15]；对 GraphPrompt +0.38，[-0.44,1.23]，只有点估计领先；对 rawprop2 +3.44，[1.77,5.20]，有跨支持稳定收益。
- 不蒸馏同幅度 token 对照：Cora/GraphMAE67.43%、CiteSeer/GraphMAE44.08%、PubMed/DGI58.68%、PubMed/GraphMAE73.16%。因此多数增益来自条件 token，KD 的额外提升是0.49/0.16/1.75/0.83 pp，不能把全部提升归给 KD。
- 对同幅度打乱门控：Cora63.54%、CiteSeer32.71%、PubMed/DGI33.97%；固定随机 token 为65.23/35.41/43.31%。正确门控与优化确有正向对照线索，具体机制结论仍限当前图和预训练条件。

## 第五轮：固定条件提示扩展已有预训练权重

已检查另外18份权重均是兼容的两层128维GCN：三个数据集各有 Edgepred_GPPT、Edgepred_Gprompt、GraphMAE2、S2GAE、SimGRACE、Bandana。无需重新预训练或复制数据，复用第四轮代码。

不根据第四轮确认结果调参，预先固定 token_kd 与 token_kd3 两配置，在开发半区比较 Frozen、GraphPrompt、rawprop2。3数据集×6新增权重×5支持seed×5方法=450任务。checkpoint sidecar/hash 仍逐项核验；这些是本地 ProG 适配权重，精确原论文实现与各预训练契约不能仅凭名称假定成立。

### 第五轮开发（450/450，审计通过）

[全结果](round5_checkpoint_dev/summary.md)。新增权重中，GraphMAE2 + token_kd3 在三个数据集的绝对准确率均最高：Cora65.40%、CiteSeer53.88%、PubMed73.92%；对各自 Frozen +9.28/+14.72/+6.16 pp，对普通 GraphPrompt +11.60/+14.48/+4.72 pp。PubMed 同时高于 rawprop2 教师3.76 pp；另两个数据集仍落后教师4.16/7.88 pp。

固定这三个条件，seed9100..9119，与 Frozen/GraphPrompt/rawprop2/同幅度不KD/打乱/随机token共7方法，420任务已启动。见 [round5_selection.json](round5_selection.json)。未确认前不把开发集最高值当作复现结果。

### 第五轮确认（420/420 完成，审计通过）

[完整结果](confirm5/summary.md)：Cora/GraphMAE2 67.90%，对 Frozen +7.09 pp、GraphPrompt +9.25 pp，教师67.86%基本持平；CiteSeer50.87%，分别+10.55/+11.78 pp，但教师60.45%明显更好；PubMed73.88%，分别+5.12/+4.28 pp，超过教师70.84%（+3.04 pp）。

相同 gamma3 不蒸馏为67.86/50.88/72.19%。因此Cora/CiteSeer的KD没有可见额外价值，PubMed额外+1.69 pp。正确门控比打乱门控更好，但不把条件token的整体增益全归给KD。预训练来源仍为ProG-adapted。

## 第六轮：支持内部选择提示强度

[预先设计](round6_design.md)。新增统计学交叉验证借鉴，五折4-shot拟合、1-shot内部验证，教师/门控每折完整重建；选择gamma1/3/10、gamma3无KD或no-op，然后完整5-shot重训。3数据集×2checkpoint×5新seed，共30个选择任务，600次内部拟合与240条最终查询指标分开计数。已通过3步预检，折间索引与teacher_label_indices检查通过，正式100步运行中。参数选择只看支持标签，查询仍为本项目已使用的池，不声称新图泛化。

### 第六轮结果（240/240，600次内部拟合；审计通过）

[结果](round6_cv/summary.md)与[额外审计](round6_7_additional_audit.json)：30个选择文件、150个互斥内折全部检查通过；选中配置与平均内部loss一致，选择文件哈希匹配。

Cora/CiteSeer的20个任务全部选gamma10，准确率分别为Cora GraphMAE69.48%、GraphMAE2 67.60%、CiteSeer GraphMAE50.80%、GraphMAE2 56.04%。这些提升来自较强提示，CV并未优于固定gamma10，暂不为这四条件保留昂贵CV。

PubMed GraphMAE的CV为71.00%，低于Frozen72.56%；GraphMAE2的CV69.52%，低于固定gamma3的72.04%。支持内部loss选择也会出错，少量标签不能保证可靠选参。保留gamma10的局部增益，停止当前CV方案的扩大；没有把训练/内部验证更优写成查询更优。

## 第七轮：闭式岭回归教师

[设计与来源](round7_design.md)。借鉴R2D2的闭式小样本适配，用支持LOO选择lambda的核岭回归替换原型教师，给条件token提供门控与KD目标；仍保持同一冻结GCN和原型读出。固定gamma3/10，3数据集×2权重×5新支持种子×7方法=210条最终指标。原型教师、直接ridge教师和普通提示均配对。

60次暴力留一重拟合验证解析LOO公式，最大误差1.59e-14；3步端到端预检完成，正式100步实验已启动。折内模型/特征不落盘，结果尚待分析。

### 第七轮开发（210/210，审计通过）

[全结果](round7_ridge/summary.md)。Cora/GraphMAE岭教师提示gamma10达到74.88%，Frozen60.84%、GraphPrompt60.56%、直接岭教师72.36%；Cora/GraphMAE2提示73.24%，也高于同岭教师72.36%。CiteSeer/GraphMAE2提示57.80%，虽明显高于Frozen40.68%，仍低于岭教师61.12%。PubMed没有较强收益，停止该教师替换在PubMed上的扩大。

固定以上三个条件，20新支持种子12100..12119，并补gamma10原教师、无KD、打乱门控、随机token四对照，540个任务进行中。[确认清单](round7_selection.json)。实际ridge函数与独立双精度参考分数最大误差6.31e-7，lambda和LOO损失核对通过。

### 第七轮确认（540/540，审计通过）

[Cora/GraphMAE](confirm7_Cora_GraphMAE/summary.md)、[Cora/GraphMAE2](confirm7_Cora_GraphMAE2/summary.md)、[CiteSeer/GraphMAE2](confirm7_CiteSeer_GraphMAE2/summary.md)。Cora两种权重的提示达到72.96/71.93%，同轮直接岭教师69.92%，分别+3.04/+2.01 pp；均保留为超过直接教师的真实中间层Prompt正向方案。CiteSeer提示59.24%，教师60.46%，保持低于教师的边界。

同幅度旧教师token为71.24/68.28/54.27%；固定随机岭token为71.07/69.26/56.08%；无KD为72.67/70.95/58.38%。因此Cora/GraphMAE的大部分提升可在无KD条件得到，KD额外只有0.29 pp；不以总提升替代KD独立贡献。打乱门控为51.27/30.46/24.84%，正确节点条件对应十分关键。

对直接岭教师的配对种子95%区间：Cora/GraphMAE +3.04 pp，[1.78,4.47]，18胜2负；Cora/GraphMAE2 +2.01 pp，[1.36,2.62]，19胜1负。CiteSeer -1.22 pp，[-2.95,0.29]，8胜12负。只对前两者记录超过教师的稳定支持采样证据。

## 第八轮：商品共购图直接迁移

[预注册协议](round8_transfer_design.md)。服务器已有Photo/Computers及三种权重，复用数据与环境；固定2000个无标签随机查询节点，其余节点作5-shot支持池。无需预训练或复制大数据。固定第七轮岭教师gamma10方案及8个对照，270条最终指标正在运行。Photo预检通过，不据预检查询结果改超参数。权重缺少sidecar时明确标legacy_unverified。

### 第八轮结果与输入契约限制（270/270，预测复算通过）

[完整结果](round8_transfer/summary.md)。归一化输入下，Photo/GraphMAE提示85.97%，Frozen32.33%、GraphPrompt31.08%、随机岭token84.32%、直接岭教师87.44%；Computers/GraphMAE提示63.71%，Frozen22.10%、GraphPrompt21.88%、随机岭token62.83%、直接岭教师66.83%。其余权重也有局部提升，但均没有超过各自直接岭教师。

发现重要预处理差异：当前ProG的Planetoid loader带NormalizeFeatures，而Amazon loader不带；当前DGI/GraphCL/GraphMAE预训练均调用该loader。第八轮统一NormalizeFeatures，可能与旧Amazon权重输入尺度不匹配。缺少sidecar无法追溯确定历史状态。上述巨大增幅暂限定为归一化输入条件，不据此声称匹配预训练协议的大幅改进；原结果保留，不删除。

## 第九轮：Amazon原始主干输入配对诊断

[诊断协议](round9_preprocessing_audit.md)。仅把主干输入换成原始特征，冻结第八轮教师、支持/query、模型与提示设置。150任务已启动；目标先确认过弱Frozen是否由预处理尺度引起。原始输入匹配当前ProG源码，但不补造历史checkpoint provenance。原Planetoid结论不受该Amazon差异影响。

### 第九轮诊断结果（150/150，审计通过）

[全结果](round9_rawinput/summary.md)。主干改为原始输入后，Photo/GraphMAE Frozen从32.33%恢复到63.74%，Computers/GraphMAE从22.10%恢复到52.25%；证实第八轮部分巨大增幅来自过弱输入基线，不能沿用旧基线夸大Prompt价值。

原始主干输入下，Photo/GraphMAE提示86.13%，Frozen63.74%、GraphPrompt64.53%、随机token85.70%、直接岭教师87.44%；Computers/GraphMAE提示64.95%，Frozen52.25%、GraphPrompt52.92%、随机token64.46%、教师66.83%。仍有真实大幅对Frozen提升，但训练相对随机token仅+0.43/+0.49 pp，而且均未超过教师；保留为任务条件提示的跨领域局部效果，不称训练增益强或超过直接方法。

150个对应记录的支持/query/checkpoint身份一致；30条教师记录的lambda与全部60000个预测标签相同，但浮点分数最大差1.85e-4，原定1e-5严格分数检查未通过，不声称逐位相同。[输入诊断配对审计](round9_pair_audit.json)。教师定义和准确率保持一致；观察到的大幅Frozen变化定位到主干输入尺度，而不是换了教师或split。权重历史来源仍不完全可验证。

## 第十轮：提示位置和随机主干

[设计与论文来源](round10_design.md)。第九轮已完成，实测GPU空闲后启动此轮，未重启历史任务。借鉴VPT深层提示思路，对同一条件门控比较激活前、激活后、双位置，以及末端适配控制。加入固定随机GCN，考察已有预训练权重是否优于无预训练。360个开发指标，3步16任务预检完成、所有零提示no-op及冻结检查通过；正式100步运行中。随机权重单独保存约数MB，不复制旧权重。

### 第十轮开发结果（360/360，审计通过）

[全结果](round10_depth/summary.md)。Cora/GraphMAE的late提示72.32%，原激活前71.36%、deep71.60%、末端69.76%、岭教师65.48%；固定late进入新支持验证。Cora/GraphMAE2的deep68.20%，原激活前67.08%、late67.88%；小幅正向，固定deep但披露参数翻倍。随机主干同配置最高66.96%，当前GraphMAE有优势，不能只凭一个随机初始化下结论。

CiteSeer各位置仍低于岭教师，PubMed新位置不如普通GraphPrompt或旧原型教师提示，停止这些位置分支的扩大。固定Cora两候选，再加3种随机初始化、全部位置对照，共5权重×20支持种子×7方法=700任务已启动。[固定清单](round10_selection.json)。

### 第十轮确认（700/700，审计通过）

[完整结果](confirm10/summary.md)。Cora/GraphMAE预选late为73.49%，同批早层72.35%、Frozen63.91%、GraphPrompt63.97%、直接岭教师69.51%。late对早层+1.14 pp，支持种子95%bootstrap区间[0.43,1.84]，15胜5负。GraphMAE2预选deep为71.05%，早层70.54%，+0.51 pp，[0.06,0.95]，12胜7负1平；其late71.44%是探索性观察，未冒充预选结果。

三种冻结随机主干的late均值69.12/69.25/68.72%，没有超过预训练GraphMAE的73.49%；表明本配置下预训练信息仍有用。末端适配GraphMAE72.56%，不归类为内部Prompt。[增量审计](round10_11_increment_audit.json)。

## 第十一轮：图像双边滤波借鉴

[设计](round11_design.md)。原始特征相似度控制图边权，普通均值邻域与双边权重分别迭代两次，修正岭教师概率，再用于条件token。直接滤波教师分类单列对照，防止把教师修正归给Prompt。算子行随机性、恒定场与孤立节点保持验证通过；240条开发指标已启动。不保存中间滤波矩阵。

### 第十一轮开发（240/240，审计通过）

[结果](round11_filter/summary.md)。Cora/GraphMAE原提示73.80%，普通平滑教师提示74.64%、双边提示74.44%；相应直接教师为69.48/71.56/71.68%。普通平滑已足够，双边权重未表现更优，不增加复杂度。固定普通平滑提示在Cora/GraphMAE做20新支持×8方法=160确认任务，见[清单](round11_selection.json)。

CiteSeer直接双边教师63.68%虽略高于原62.56%，提示仍低于教师；PubMed滤波教师和提示没有改善，停止扩大这些配置。不能把滤波和提示整体增幅都归因于双边权重。

### 第十一轮确认（160/160，审计通过）

[完整结果](confirm11/summary.md)。预选普通平滑提示74.42%，同轮原提示73.89%、双边提示74.36%、Frozen64.39%、GraphPrompt64.33%。普通平滑对原提示+0.53 pp，[0.21,0.81]，14胜4负2平；对直接普通平滑教师72.78%为+1.64 pp，[0.64,2.69]，14胜4负2平。保留普通平滑，小收益不归于双边权重。[增量审计](round10_11_increment_audit.json)。

## 第十二轮：置信伪标签和特征扰动

[设计与FixMatch来源](round12_design.md)。原提示、仅隐藏特征dropout、仅高置信硬伪标签、二者结合，210条指标已启动。伪标签排除支持节点；记录数量并确认支持重叠为0。没有读取未标注节点真标签，测试时关闭dropout。阈值0.7和系数0.5预先固定。

### 第十二轮结果（210/210，审计通过）

[全结果](round12_pseudo/summary.md)。Cora/GraphMAE原提示75.12%、仅扰动75.32%、仅硬标签74.88%、两者74.88%；小幅扰动增益未稳定出现。CiteSeer/GraphMAE2仅硬标签65.68%对原提示64.64%，但直接岭教师65.52%，优势仅0.16 pp；不据5个种子立即扩大。PubMed新方法低于普通GraphPrompt或没有明确提升。暂不扩大本轮方案，记录局部正向而不宣称无效于所有场景。

## 第十三轮：可微闭式分类头

[设计](round13_design.md)，[全结果](round13_head/summary.md)。210/210已审计。实际分类头解析LOO与60次暴力重拟合核对；有限差分方向导数误差4.76e-11，见[梯度审计](round13_gradient_audit.json)与[LOO审计](round13_loo_audit.json)。

Cora/GraphMAE Frozen原型61.20%→Frozen岭头70.00%；原条件提示73.20%→岭头提示71.88%。分类头对弱冻结读出有用，但不提升当前强条件Prompt。CiteSeer/GraphMAE提示53.20%→61.96%，仍低于直接岭教师63.40%。PubMed/GraphMAE2普通GraphPrompt68.64%→岭头GraphPrompt71.52%，但本轮未包含旧原型教师gamma3强方案，暂列局部正向，不声称最佳。后续若确认该分支，必须带上该强对照。

## 第十四轮：配对组合已有效的两项改进

[固定设计](round14_design.md)。同一冻结GCN的提示位置和教师平滑组成2×2因子，2权重×20新支持×8方法=320。直接教师和单项方案均同批比较，检测收益能否叠加；没有融合多个分类器。3步预检后正式启动。

### 第十四轮结果（320/320，审计通过）

[完整结果](confirm14/summary.md)，[增量审计](round14_increment_audit.json)。Cora/GraphMAE组合75.78%，原方案73.91%、仅late75.17%、仅plain74.59%、直接平滑教师73.34%、Frozen64.34%、GraphPrompt64.44%。组合对最好单项late+0.61 pp，[0.33,0.88]，17胜3负；对原方案+1.87 pp，[1.44,2.29]，19胜1负；对直接平滑教师+2.44 pp，[1.60,3.27]。保留为当前明确超过两个单项的组合结果。

GraphMAE2组合73.82%，仅plain73.83%、late73.61%、原方案73.30%。组合对plain -0.01 pp，[-0.31,0.29]；没有叠加证据，不把复杂度增加视为改进。所有区间仅表同图支持采样变化，历史已用过query，非未见数据集证据。

## 第十五轮：信息最大化损失

[设计与TIM原文来源](round15_design.md)。借鉴视觉传导小样本的条件熵/边际熵目标，比较单纯降熵、信息最大化、教师先验约束三个损失。继承平滑岭教师后层Prompt，支持节点排除于无标签熵集合；3图×2权重×5支持×8方法=240条，预检后启动。PubMed原型教师gamma3强参考保留，不用不同批种子替代配对。

### 第十五轮结果（240/240，审计通过）

[完整结果](round15_infomax/summary.md)，[无标签集合与损失记录](round15_loss_audit.json)。Cora/GraphMAE原组合74.80%，三种附加损失均74.88%，增益仅0.08 pp；GraphMAE2没有改善。CiteSeer两权重新增损失下降0.16–0.52 pp。PubMed/GraphMAE2信息最大化69.96%，虽高于同基础69.72%，仍低于原型教师gamma3提示72.08%。固定系数0.1下不扩大，没有根据查询继续调高系数来追逐增益。

所有记录的无标签集合与支持集合重叠为0；有限新增损失效果不等于一般信息最大化方法无效，当前只停止该具体接入方式。

## 第十六轮：提示参数邻域稳健训练

[设计与SAM来源](round16_design.md)。只扰动Prompt参数，冻结共享GCN；rho0.05/0.2、100次更新。增加Adam200步对照，使梯度计算量相当，和Adam100步区分。3图×2权重×5支持×8方法=240；预检已验证参数恢复与冻结，正式运行中。

### 第十六轮开发（240/240，审计通过）

[全结果](round16_sam/summary.md)。Cora/GraphMAE Adam100为75.20%、Adam200为75.92%、SAM0.05为75.12%、SAM0.2为75.60%；当前更长普通训练胜过SAM，不能把收益归给稳健扰动。GraphMAE2 SAM0.2为75.12%，原74.88%、长训练74.76%，增幅较小。CiteSeer SAM均不及长训练，而且长训练仍不及直接教师；PubMed各新方案不及原型教师gamma3或普通GraphPrompt。

固定Cora/GraphMAE长训练和GraphMAE2 SAM0.2，20新支持seed23100..23119，同批8方法共320确认。其余分支暂不扩大。[固定清单](round16_selection.json)。

### 第十六轮确认（320/320，审计通过）

[全结果](confirm16/summary.md)，[长训练增量](confirm16_long_audit.json)，[SAM增量](confirm16_sam_audit.json)。Cora/GraphMAE预选Adam200为73.42%，原100步74.97%，-1.55 pp，20支持bootstrap区间[-2.41,-0.77]，4胜15负1平。开发阶段的+0.72 pp没有保持，停止用长训练替换100步。

GraphMAE2预选SAM0.2为72.49%，原100步72.56%，-0.07 pp，[-0.33,0.18]，5胜11负4平。GraphMAE上SAM0.2为75.05%，只比原74.97%高0.08 pp，[-0.09,0.24]；属于非预选探索性结果，也没有稳健增益。确认不支持扩大SAM。教师与冻结基线不变的配对证据表明：训练更久/增加梯度计算并不自动带来更高准确率。

## 第十七轮：提示参数轨迹平均

[设计与来源](round17_design.md)。借鉴SWA与EMA权重平均概念，对200步训练后半程的提示参数求平均；只保留一个Prompt推断，不保存中间模型。100步、200步末步、200步均值、200步EMA同批比较，3图×2权重×5支持×8方法=240。3步预检通过，正式任务已实测运行。此设计在第十六轮确认结果取得前固定；确认后仍以100步作为强参照，不能仅把恢复200步的退化称为超过当前最佳。

### 第十七轮结果（240/240，审计通过）

[全结果](round17_average/summary.md)，[实际平均步数及冻结审计](round17_average_audit.json)。Cora/GraphMAE的100步77.16%，200步末步76.92%，均值77.16%、EMA77.08%；仅恢复而未超越100步。GraphMAE2均值75.80%，100步75.60%，只有+0.20 pp的5支持局部信号，暂不扩大。

CiteSeer均值/EMA都低于200步末步，且都低于直接教师62.88%；PubMed两权重参数平均均低于原型教师gamma3的71.72/70.52%。保留具体范围内失败，不推断所有参数平均方法无效。本轮没有训练损失或平坦性代替准确率结论。

## 第十八轮：九种预训练权重固定方案矩阵

[设计](round18_design.md)，[27个权重哈希与结构](round18_checkpoint_audit.json)。Cora/CiteSeer/PubMed各九种同结构GCN，固定8方法、5新支持，1080条正式指标；每模型先做3步预检。主干分别冻结，避免把多checkpoint预测融合当Prompt。MultiGprompt架构不同，另留后续专门适配。

### 第十八轮开发（1080/1080，审计通过）

[全结果](round18_backbones/summary.md)，[跨权重配对审计](round18_pair_audit.json)。1080条graph/split身份一致性检查、270对教师预测标签一致性检查通过；不同权重在同一支持/query条件下比较，直接教师完全共享。

Cora最高仍为GraphMAE平滑后层提示76.08%；新权重中Edgepred_Gprompt75.32%，未超过直接平滑教师75.36%。CiteSeer全部提示低于直接平滑教师61.08%；不为显著改善弱Frozen而扩大这些分支。Bandana/S2GAE在当前固定提示协议表现较弱，结论仅限这些已核验的本地适配权重，不外推预训练算法本身无效。

PubMed/Edgepred_Gprompt早层岭教师提示73.56%，Frozen66.08%、GraphPrompt66.84%、直接岭教师69.72%；同批GraphMAE普通GraphPrompt73.40%、原型教师提示73.00%，GraphMAE2原型教师提示72.36%。候选相对最强跨权重对照优势很小，必须确认。固定该候选并带上GraphMAE/GraphMAE2全部8方法，20支持seed27100..27119，共480任务已启动，见[固定清单](round18_selection.json)。

### 第十八轮PubMed确认（480/480，审计通过）

[全结果](confirm18_PubMed/summary.md)，[跨checkpoint配对区间](confirm18_cross_audit.json)。预选Edgepred_Gprompt早层岭提示69.43%，同权重Frozen66.54%、GraphPrompt66.58%、直接岭教师67.19%。分别+2.89 pp，[0.45,5.25]；+2.85 pp，[0.02,5.67]；+2.24 pp，[0.53,3.77]。保留为同权重有效改进，但区间较宽。

同批GraphMAE普通GraphPrompt73.13%、原型提示71.73%，GraphMAE2原型提示72.37%。新候选比三者分别-3.70/-2.30/-2.94 pp，各支持bootstrap区间均低于零；不能替换强搭配。开发阶段73.56%没有在新支持与确认query保持，报告确认69.43%，不挑开发值作最终成绩。支持/query身份跨checkpoint一致，仅checkpoint本身不同；仍不是新图独立验证。

## 第十九轮：FiLM式条件仿射提示

[设计与FiLM原文](round19_design.md)，[预检](round19_smoke_audit.json)。在同一冻结GCN第一层激活后增加教师条件缩放，配对原单平移、仅缩放、缩放+平移、双平移参数量对照。参数翻倍和双平移对照能力局限均已披露；梯度只更新提示。零初始化、参数量、主干冻结预检通过，240正式任务运行中。

第十八轮当前观察903/1080，第十九轮155/240；均为实测运行进程，不是仅由状态文件推断。第十九轮预检已清理，完成与审计保留在[清理清单](cleanup_006.json)。磁盘仍余31 GiB。结果未完成前不按部分数据选择候选。

### 第十九轮开发（240/240，审计通过）

[全结果](round19_film/summary.md)。Cora/GraphMAE原平移75.96%、FiLM76.00%、双平移76.00%，FiLM没有超出对照；GraphMAE2原73.48%、FiLM73.32%，下降。CiteSeer的FiLM比原提示+0.40/0.44 pp，但均低于双平移且明显低于直接教师62.88%。PubMed FiLM无收益，仅缩放GraphMAE73.08%也低于GraphPrompt73.32%。停止当前仿射调制扩大，保留原单平移更简单的方案。

## 第二十轮：置信度加权教师依赖

[设计与UPS来源](round20_design.md)。只借鉴可靠伪标签思想，以教师熵为代理，分别加权KD、内部提示门控和二者；增加打乱节点权重对照，并固定平均权重1防止仅削弱蒸馏强度。既检查Cora强岭教师后层方案，也保留PubMed原型教师gamma3分支。3图×2权重×5支持×12方法=360，3步预检通过、正式运行中。记录权重标准差，退化成常数时不夸大机制差异。

### 第二十轮开发（360/360，审计通过）

[完整结果](round20_confidence/summary.md)，[实际权重分布](round20_confidence_audit.json)。权重不是常数，不能把本轮失败归为未生效。原型教师同时加权在CiteSeer/GraphMAE2从52.84%降至29.32%，而打乱权重52.60%；按置信度对应节点并不自动更好。PubMed两权重原型提示也下降，岭门控加权多数退化，停止推广。

Cora/GraphMAE2同时加权71.68%，原71.32%、打乱71.08%、教师70.48%。只加门控71.88%更高，但缺少只打乱门控权重的精确对照；本轮预选同时加权，保持机制对照完整。20新支持29100..29119、8方法共160确认已启动，[清单](round20_selection.json)。CiteSeer仅KD的小幅+0.32/0.36 pp暂留开发信号，不直接升级为确认收益。

第七次清理删除完成的12条3步预检，审计保留：[预检核验](round20_smoke_audit.json)、[清理清单](cleanup_007.json)。环境、数据及正式预测保留。

### 第二十轮确认（160/160，审计通过）

[全结果](confirm20/summary.md)，[配对区间](confirm20_increment_audit.json)。预选同时加权70.55%，原71.41%、打乱71.70%、直接教师71.22%。对原方案-0.86 pp，支持bootstrap区间[-1.98,0.17]；对打乱-1.15 pp，[-2.06,-0.27]，6胜13负1平。没有保持开发增益，停止扩大当前熵置信度接入方式。

只门控71.56%是非预选探索结果，仍缺少完全匹配的只门控打乱对照，不将它追认为确认成功。结论限定：本固定教师和归一化熵权重下，置信度对应关系未帮助提示；不泛化为所有不确定性估计或伪标签选择均无效。

## 第二十一轮：教师原型修正

[设计与ECCV2020来源](round21_design.md)。以固定支持原型为锚，三次无标签软类别分配修正原型，再用于原gamma3内部Prompt。beta0.25/0.5与打乱分配、直接修正教师分列，保留平滑岭后层强方案。3图×2权重×5支持×11方法=330。beta0退化检验分数误差0，预检通过后正式运行；修正只改变教师，不改最终提示表示的支持原型头。

### 第二十一轮开发（330/330，审计通过）

[全结果](round21_rectify/summary.md)，[支持隔离核对](round21_support_audit.json)，[beta0退化核对](round21_operator_audit.json)。PubMed/GraphMAE beta0.5修正教师提示73.92%，原提示71.52%、GraphPrompt73.44%、直接修正教师68.24%；beta0.25提示72.76%，匹配打乱72.32%。预选beta0.5在两种GraphMAE权重上20新支持确认，并补beta0.5完全匹配打乱，440任务。

CiteSeer直接修正教师beta0.5为65.96%，原教师59.40%，而提示仍只有GraphMAE43.04%/GraphMAE2 50.08%，低于平滑岭教师提示53.76/59.48%。直接教师增益独立保留，不算内部Prompt增益。GraphMAE2条件下补直接岭/平滑岭强对照，20新支持共260任务确认。两批固定seed31100..31119，[清单](round21_selection.json)。

Cora直接教师虽改善，仍低于平滑岭后层提示；修正后的原型提示没有接近当前强方案，暂不扩大。第八次清理删除完成的11条3步预检，正式预测保留：[清单](cleanup_008.json)。

### 第二十一轮确认（700/700，审计通过）

[PubMed全结果](confirm21_PubMed/summary.md)、[增量](confirm21_PubMed_increment.json)。预选GraphMAE修正教师gamma3提示74.55%，原72.43%、相同beta0.5打乱72.82%、直接修正教师70.53%、GraphPrompt73.60%。对原提示+2.12 pp，[1.31,2.98]，16胜3负1平；对打乱+1.73 pp，[0.83,2.68]；对直接教师+4.02 pp，[2.82,5.21]。保留修正教师带来的真实内部Prompt增量。对GraphPrompt仅+0.95 pp，[-0.64,2.60]，不能声称稳定超过该强基线。

GraphMAE2同修正提示74.48%，原74.05%，+0.43 pp区间[-0.31,1.13]，不把探索性小幅提高当作明确增量。它对直接修正教师+3.95 pp，对普通GraphPrompt+5.56 pp，显示骨干和教师作用依赖具体搭配。

[CiteSeer全结果](confirm21_CiteSeer/summary.md)、[增量](confirm21_CiteSeer_increment.json)。直接修正教师66.99%，原教师60.62%、相同强度打乱62.35%、岭教师61.80%、平滑岭教师62.53%。分别+6.37/+4.64/+5.19/+4.46 pp，支持bootstrap区间均高于0；前3项20胜0负、最后19胜1负。保留为强直接教师，不归因于GCN Prompt。修正教师提示53.61%虽比原48.73%提高，仍明显低于直接教师。

## 第二十二轮：修正教师与更强内部提示结合

[固定设计](round22_design.md)。冻结beta0.5教师，比较gamma3、gamma10早层/后层、无KD、随机token，并保留旧强提示及直接教师。3图×2权重×5支持×11方法=330；预检后正式运行。不能用教师单独变好替代内部提示实际改善。

### 第二十二轮结果（330/330，审计通过）

[全结果](round22_strength/summary.md)。Cora修正教师gamma10仍低于已确认的平滑岭后层提示（GraphMAE76.00%、GraphMAE2 74.80%）。CiteSeer修正教师gamma10最高GraphMAE56.44%/GraphMAE2 61.20%，虽比各自gamma3更高，仍低于直接修正教师67.76%；不扩大。

PubMed/GraphMAE修正教师gamma3为75.76%，gamma10早层72.32%、后层67.88%、无KD74.68%、固定随机68.08%、普通GraphPrompt75.16%。保留gamma3，更强注入反而下降。GraphMAE2原教师gamma3 74.04%、修正教师gamma3 73.84%，其余更强变体也无优势。第五支持开发小批差异不改写前轮20支持确认结论。

随机token预检未出现训练记录，主干冻结全部通过；[预检审计](round22_smoke_audit.json)。第九次清理移除完成的11条预检，见[清单](cleanup_009.json)。正式预测全部保留。

## 第二十三轮：商品图迁移

[固定设计](round23_design.md)。将beta0.5修正教师和gamma3提示直接迁移Photo/Computers，保留原始主干输入、独立归一化教师，与第九轮入口一致；3种现存权重、5新支持、11方法共330指标。直接教师与随机token均单列，明确历史query已被使用。六种图/权重组合先3步预检，再正式100步；不复制数据、权重或环境。

### 第二十三轮结果（330/330，审计通过）

[全结果](round23_amazon/summary.md)。Photo直接修正教师85.11%，原教师83.56%，但打乱85.14%、岭教师86.29%；正确软类别对应没有表现优于打乱。Computers修正教师69.34%，原69.56%、打乱70.25%、岭74.28%，修正没有提高。

GraphMAE的Photo修正教师gamma3提示73.64%，Frozen77.22%、普通GraphPrompt77.11%、随机修正token77.37%、旧岭提示86.14%、平滑岭后层86.30%；Computers新提示62.97%，Frozen63.26%、随机63.50%、旧岭73.09%、平滑岭后层73.74%。修正教师gamma3未跨领域保持优势；其他两权重也没有超过直接岭教师。平滑岭提示Photo仅比岭教师高0.01 pp，Computers仍低于教师，不据开发小差扩大。

[输入审计](round23_input_audit.json)核对66条预检raw图哈希、checkpoint哈希与第九轮一致，主干raw、教师独立NormalizeFeatures；query2000和冻结检查通过。此轮Frozen均值与第九轮不同来自不同支持采样条件，不能当作预处理改进。历史权重来源仍legacy_unverified。

第十次清理移除已完成66条预检约7.4MB，正式预测保留：[清单](cleanup_010.json)。原型修正在CiteSeer的确认收益仍有效，但本轮表明它不宜无条件推广到商品图。

## 第二十四轮：类别token的单纯形正则

[设计与外域文献](round24_design.md)。从正交正则检索转为适合中心化类别token的等角Gram目标；两种已确认基础各比较系数0.01/0.1与范数对照。固定主干/教师/头，只改提示训练损失，前10步warmup，预检12步覆盖新损失。3图×2权重×5支持×12方法=360条，正在运行。几何惩罚下降不能替代准确率提升。

### 第二十四轮结果（360/360，审计通过）

[全结果](round24_simplex/summary.md)，[实际几何与范数记录](round24_geometry_audit.json)。CiteSeer弱原型修正提示在0.1约束下，GraphMAE41.52→43.68%、GraphMAE2 52.00→53.08%，但均远低于直接修正教师67.52%和既有岭提示57.24/60.20%。仅保留局部提高，不因改善弱方案扩大。

Cora强岭后层原方案75.20/74.96%，0.1等角约束75.08/75.00%，范数对照75.32/74.96%，没有明确新增优势。PubMed修正教师gamma3原70.96/71.08%，强约束70.88/71.00%，下降。停止当前单纯形约束扩大，不把几何更规则等同于准确率提高。

[预检](round24_smoke_audit.json)12步覆盖10步warmup之后的新项，训练中梯度有限性检查通过。第十一次清理删除12条完成预检，正式预测保留：[清单](cleanup_011.json)。

## 第二十五轮：解耦蒸馏目标

[设计与DKD来源](round25_design.md)。将固定教师首选类别与其他类别条件分布分开蒸馏，首选索引是教师argmax而非query标签。两种强基础各比较原KL、解耦、单一分量、双倍KL、无KD，共480正式指标。实际KL分解误差6.66e-16、梯度误差4.34e-18，[恒等审计](round25_identity_audit.json)；预检后正式运行。

### 第二十五轮开发结果（480/480，审计通过）

[完整结果](round25_dkd/summary.md)。解耦项没有超过强对照：Cora/GraphMAE强岭原76.08%、DKD76.08%，GraphMAE2原74.72%、DKD75.04%但无KD同为75.04%；PubMed修正原型原72.40/70.00%、DKD71.72/69.92%。CiteSeer局部提高仍远低于直接修正教师66.80%。停止解耦机制扩大。

简单双倍KL在Cora/GraphMAE强岭达到76.24%（+0.16pp），PubMed修正原型72.76/70.52%（+0.36/+0.52pp）。按准确率目标保留这个朴素候选，不宣称DKD收益。[预先选择](round25_selection.json)：新支持seed36100..36119，confirm查询半区，Cora一权重与PubMed两权重，原方法/双KL/无KD/Frozen/GraphPrompt/对应直接教师共360条。该查询半区已在过去轮次使用，并非全新独立测试。

### 第二十五轮确认（360/360）

[配对审计Cora](confirm25_Cora_increment.json)、[PubMed](confirm25_PubMed_increment.json)。双KL对原KL：Cora/GraphMAE74.87%对74.65%，+0.22pp，区间[-0.09,0.54]；PubMed/GraphMAE73.37%对73.25%，+0.12pp，[-0.82,1.06]；GraphMAE2 73.24%对72.82%，+0.42pp，[-0.15,1.03]。均不足以稳定替换原配置。原有提示相对直接教师/Frozen的优势保留，但不能归因于此次调权重。

## 第二十六轮：梯度冲突控制

[设计与论文来源](round26_design.md)。保护支持梯度、对称投影、教师梯度限幅，带原KL/双KL/无KD控制；共480条。[实际函数检查](round26_gradient_audit.json)原sum与直接求导误差4.44e-16，冲突、无冲突、零梯度检查通过；16条3步预检已完成，正式运行。Adam更新不保证满足原梯度投影的方向性质，结论只按配对准确率判断。

第十二次清理只删除第25/26轮已完成预检，共32条；保留配置和预测哈希核验记录：[清单](cleanup_012.json)。正式预测全部保留。

### 第二十六轮开发结果（480/480）

[完整结果](round26_grad/summary.md)、[实际冲突统计](round26_conflict_audit.json)。PubMed/GraphMAE2原型修正提示原68.72%、对称投影69.32%（+0.60pp）、单向保护67.32%，预先选择对称投影追加确认。GraphMAE原69.68%、对称69.24%；较弱岭提示限幅有所改善但未超过原型修正提示/GraphPrompt。Cora强岭方案变化仅约0.1pp，不扩大；弱原型对称改善但仍低于强岭。CiteSeer最高提示仍低于直接修正教师69.28%。

[确认选择](round26_selection.json)：PubMed两权重×20新支持seed39100..39119×7方法=280条，GraphMAE作为额外强权重对照。只将GraphMAE2对称方法视作预先选出的候选，不把另一个权重事后挑作确认成功。

## 第二十七轮：类别质量约束的原型修正

[设计和SwAV来源](round27_design.md)。以log域Sinkhorn替换原型修正中的softmax分配，比较均匀质量、混合初始预测比例、打乱分配及原方案；直接教师与内部token提示分别评估。新增方案不读取真实类别比例。[实际函数和36条预检](round27_operator_audit.json)通过，最大质量误差1.49e-7；360条正式任务排在第26轮确认之后串行执行。

### 第二十六轮确认结论（280/280）

[跨权重配对审计](confirm26_cross_audit.json)。PubMed/GraphMAE2对称投影73.56%，原73.58%，增量-0.02pp，区间[-0.67,0.70]；对双KL73.82%为-0.26pp，[-0.49,-0.03]。GraphMAE原73.76%也高于选中候选。停止本轮梯度投影扩大；“训练目标梯度冲突存在”不能解释为“消除冲突必然提高泛化”。

### 第二十七轮结果及确认（360+100条）

[开发结果](round27_sinkhorn/summary.md)。Cora强岭仍76.16/74.76%，新原型提示约66%；PubMed原修正提示73.68/72.00%，uniform73.24/71.32%，mixed73.76/71.68%，没有值得扩大Prompt收益。CiteSeer直接uniform教师65.00%对原64.48%，按[预先选择](round27_selection.json)做20新支持确认。

[确认审计](confirm27_increment.json)：uniform68.03%、原67.98%，+0.05pp，区间[-0.15,0.25]；mixed68.06%，与uniform无明确差异。虽然超过打乱教师62.90%和平滑岭61.72%，但原型修正本来已有优势，不能归因于新Sinkhorn约束。停止这一约束扩大。第十三次清理删除36条完成预检约1.06MB，[清单](cleanup_013.json)。

## 新数据资源准备：Cornell / Chameleon / Actor

已从5090的`/home/shq/ProG`选取三图数据与DGI/GraphCL/GraphMAE共9个checkpoint及其sidecar、当前数据加载和GCN源代码，共65文件、72,645,617字节；本地与主服务器任务目录均逐个验证[来源哈希清单](assets5090_manifest.json)。只复制本轮需要的小型资产，未复制环境或全部模型。

[主服务器加载审计](assets5090_load_audit.json)：Cornell183节点/1703维/298边，Actor7600/932/30019，Chameleon2277/2325/36101；三者原始边均有向。当前来源loader均使用原始主干特征，Chameleon为geom_gcn_preprocess=True；教师若需归一化，应单独进行。9权重维度与sidecar哈希一致；这些检查不是历史预训练全过程复现。

预备扩展用官方fold0训练池内每类5-shot，查询为官方fold0测试集；三个查询大小37/1520/456。Cornell单次准确率分辨率约2.70pp，多个支持种子不能增加查询节点数，后续须多fold或其他图补充，不能把支持重复当独立图证据。尚未在这三图运行正式新提示；下一轮先固定设计、原始输入和有向图契约，再执行。资源在`assets5090/`，实验仍使用主服务器既有ProG环境。

## 第二十八轮：异配图迁移结果（585/585）

[固定设计](round28_design.md)、[全结果](round28_hetero/summary.md)、[输入审计](round28_input_audit.json)。新三图正式开发只使用官方fold0验证集，从官方训练池采5-shot；测试集未用于本轮筛选。原始主干特征、有向边的图哈希与5090来源一致，9模型完整forward与分层forward比对通过；117预检中的36对直接分类预测跨权重一致。所有封存结果重新评分并核验支持/查询掩码。

|数据集|原始特征原型|原始特征岭分类|二跳原型|最高训练内部token（仅开发排名）|结论|
|---|---:|---:|---:|---:|---|
|Cornell|66.10%|70.17%|34.92%|37.63%（GraphCL修正教师）|提示远低于原始特征；不以超过弱Frozen作为成功|
|Actor|21.59%|21.09%|19.50%|21.04%（GraphCL修正教师）|未超过原始原型、GraphCL Frozen21.11%及GraphPrompt21.20%|
|Chameleon|31.47%|32.10%|27.57%|26.36%（DGI修正教师）|低于全部Frozen权重28.31–29.68%，停止直接迁移|

Cornell/GraphMAE修正提示33.90%对Frozen26.78%有局部提升，但距原始岭70.17%很远；GraphCL修正提示37.63%甚至低于对应随机token40.68%。这些均记录为迁移效果失败，无需消耗新图测试集做弱候选确认。二跳特征相对原始特征劣势是观察，不足以单独证明所有损失来自异配性或有向边。

下一轮针对教师信息路径：[支持缓存检索借鉴笔记](next_cache_retrieval_note.md)，来源Tip-Adapter视觉少样本缓存。拟比较原始特征缓存/原始岭教师、传播与不传播门控，继续带原始直接分类强基线；笔记尚未实现，不能记为结果。已完成第14次清理，仅删除117条已审计预检约5.32MB，[清单](cleanup_014.json)。

## 第二十九轮：支持缓存与原始教师路径

[固定设计](round29_design.md)借鉴Tip-Adapter的支持缓存指数余弦检索；不做最终分类分数融合。beta10/30缓存与原始岭教师各配传播/直接门控，全部固定后层gamma10。带直接教师、原始特征、Frozen/GraphPrompt/旧平滑岭提示及随机token，共675条。支持自匹配移除并计数校正，[实际函数检查](round29_operator_audit.json)通过；45条3步预检及评分审计通过，正式运行。官方fold0验证集用于开发，测试集不参与。

### 第二十九轮结果与确认（675+160条）

[开发结果](round29_cache/summary.md)、[执行契约审计](round29_contract_audit.json)：675行全部完整，所有405条token实际幅度均为10；缓存自匹配项为零、每类剩余key不少于4；120对直接分类跨权重预测一致。Cornell原始岭直接门控DGI提示50.17%（对应Frozen35.25%、旧提示32.20%），但原始岭直接分类69.49%，提示仍明显不足；其他权重类似。Chameleon缓存直接教师34.27%与原始岭34.24%接近，但最好训练提示29.71%，低于Frozen/直接分类，不扩大。

Actor/GraphCL缓存10直接门控开发21.70%，原始原型21.40%，所以按[预先选择](round29_selection.json)用20新支持seed43100..43119确认，并增加同beta10同门控同幅度随机token。开发用官方fold0验证集，确认首次使用本任务中的官方fold0测试集（1520节点）；仍是同一图，支持重复不是独立图证据。

[确认结果](confirm29_Actor/summary.md)、[配对审计](confirm29_increment.json)：候选21.148%，Frozen21.428%、GraphPrompt21.260%、原始原型21.194%、原始岭21.046%、缓存教师20.516%、对应随机token20.931%。候选对Frozen -0.280pp，区间[-0.895,0.306]；对原始原型 -0.046pp，[-1.605,1.438]；对随机token +0.217pp，[-0.711,1.142]。开发微小提高未复现，停止缓存教师/门控路径原样扩大。

第十五次清理移除45条完成预检约2.09MB，[清单](cleanup_015.json)。下一候选草案为[自身消息结构提示](next_self_message_note.md)：固定和可学习自环强度、仅自环强对照，借鉴视觉残差的信息保留原则；尚未实现，不能计为进展结果。Cornell/Chameleon测试集仍未用于选择；Actor测试集自本轮起已使用，后续需明确披露。

## 第三十轮：自环权重结构提示

[固定设计](round30_design.md)：冻结两层GCN权重，改变每层图自环权重，比较固定1/10/100、仅自环、学习两个层级标量CE及CE+原始岭KD。3图×3权重×5支持×10方法=450条官方fold0验证集开发。[实际算子审计](round30_operator_audit.json)90条预检完成，9个模型的lambda1预测和分数恢复原模型，CPU双精度有向图梯度有限差分误差6.08e-12，仅自环与冻结MLP误差0。正式运行中，尚无最终准确率结论。

### 第三十轮开发与确认（450+320条）

[开发结果](round30_selfloop/summary.md)、[实际学习自环范围](round30_lambda_audit.json)。Cornell固定/可学自环大幅超过原Frozen，但最高55.59%仍低于原始岭72.88%；Chameleon所有方案低于原始岭30.92%，不扩大。Actor固定self100/仅自环分别在GraphCL/GraphMAE达到23.17/23.44%，高于原始原型21.83%；按[预先选择](round30_selection.json)固定候选，20新支持seed45100..45119×2权重×8方法确认。Actor测试已在29轮使用，不能称全新测试集。

[确认完整结果](confirm30_Actor/summary.md)，[self100配对](confirm30_self100_audit.json)、[selfonly配对](confirm30_selfonly_audit.json)、[跨权重配对](confirm30_cross_audit.json)。GraphCL固定自环100为23.303%，Frozen20.951%、GraphPrompt21.352%、raw20.941%、rawridge20.605%。对应增量+2.352pp [1.240,3.474]、+1.951 [0.888,3.020]、+2.362 [0.918,3.898]、+2.697 [1.184,4.234]。这是同图支持采样区间，非独立图显著性。

GraphCL仅自环23.257%，与self100差0.046pp [-0.220,0.349]；可学KD21.187%，比固定self100低2.115pp。GraphMAE仅自环22.158%，对Frozen20.461%高1.697pp [0.615,2.747]，对raw高1.217pp [0.033,2.444]。两权重比较：GraphCL self100对GraphMAE selfonly高1.145pp，但区间[-0.095,2.493]跨零，不宣称权重排名稳定。

结论：保留Actor固定减弱邻居消息的准确率收益；没有学到两个标量优于固定传播方式的证据。仅自环已经能解释几乎全部GraphCL增益，不将其称作新Prompt学习成功。下一步以固定self100和selfonly为强基础，检验内部条件token与图消息路径是否存在组合增益，匹配无token、随机token、无KD与直接原始岭教师；必须比较两模块单独与组合，避免末端融合表述。

第十六次清理删除90条完成预检约3.90MB，[清单](cleanup_016.json)。正式预测与参数保留。

## 第三十一轮：图结构与内部提示的组合

[固定设计](round31_design.md)、[570条开发结果](round31_combine/summary.md)、[57条预检审计](round31_smoke_audit.json)。同一个冻结GCN内，self1/self100/selfonly与无token/KD/仅CE/随机token/同结构GraphPrompt配对，原始岭教师和原始特征作为强直接对照。token插入第一层激活后、第二层前，gamma3固定，无末端融合。最大零token归一化误差1.04e-7，原图无token恢复Frozen。第17次清理57条预检约2.57MB，[清单](cleanup_017.json)。

开发观察：Cornell/GraphMAE仅自环+KD66.10%，无token45.08%、随机55.59%，但原始岭69.49%，不扩大。Chameleon组合未超过原始岭32.59%。Actor/GraphMAE仅自环+CE23.31%，无token21.73%、随机21.67%、同结构GraphPrompt22.19%；GraphCL自环100+GraphPrompt24.86%，对应基础24.43%。按[预先选择](round31_selection.json)用20新支持×2权重×12方法=480条测试确认，Actor测试已使用过。

[开发配对交互差值](round31_combine/interaction_audit.json)记录“组合−结构单项”减去“原图相同适配−原Frozen”，只是准确率描述，不作为创新或独立图统计证据。下一候选[协方差校正笔记](next_covariance_note.md)借鉴NLP句向量白化；已核对早期sphered仅为逐维缩放，完整协方差尚未运行。

### 第三十一轮确认（480/480）

[确认完整结果](confirm31_Actor/summary.md)、[内部token配对审计](confirm31_internal_audit.json)、[GraphPrompt配对审计](confirm31_gp_audit.json)。GraphMAE仅自环+CE token23.336%，对应无token22.457%、随机token22.424%、同结构GraphPrompt22.336%、KD token23.132%、原图CE token21.155%。对同结构GraphPrompt+1.000pp [0.122,1.789]，随机+0.911 [0.141,1.622]，原始岭+3.217 [2.020,4.461]；对无token+0.878 [-0.026,1.724]，区间略跨零。保留为局部训练候选，不宣称已经稳定超过无提示结构基础。

同轮GraphCL仅自环无token23.437%，比GraphMAE组合23.336%略高；二者配对差-0.102pp [-1.171,1.033]。GraphCL self100+GraphPrompt23.289%对无适配23.286%，增量0.003pp [-0.247,0.280]，没有额外收益；停止这个组合扩大，仍保留固定结构收益。

下一步优先用Actor其他官方fold检验GraphMAE内部token局部增量，匹配无token、随机、同结构GraphPrompt、原始岭及GraphCL固定结构。多fold共享同一张图且查询节点可能重叠，必须以fold与支持两个层次报告，不能把全部运行当独立图；fold0测试已重复使用。完整协方差方案仍是未运行候选。

## 第31轮跨fold复核（450/450完成）

[预先设计](round31_multifold_design.md)：Actor官方fold1–9，各5新支持，GraphMAE九方法和GraphCL仅自环base，共450条。沿用原脚本/幅度/步骤/教师，不因fold调参。每fold每权重独立结果、配置和预测审计；最后报告fold均值与支持内差异，保存查询集合重叠。

下一轮[协方差设计草案](round32_design_draft.md)已核对NLP白化原文，暂未实现。它只是经验度量候选，不接受“各向同性是余弦有效性的必要条件”的理论说法，也不将固定白化效果当Prompt训练收益。

### 多fold复核结果

[汇总](multifold31_summary.md)、[逐fold配对和查询重叠审计](multifold31_audit.json)。GraphMAE内部CE token均值22.155%，仅自环base22.015%、同结构GraphPrompt22.123%、随机21.768%；对应增量+0.140、+0.032、+0.387pp，两层描述区间均跨零。相对base只有4/9个fold正向，单fold均值差从-0.803到+1.263pp，fold0的局部训练优势未稳定复现。

相对原Frozen20.409%和原GraphPrompt20.534%，组合仍高1.746/1.621pp，但固定结构已贡献主要收益。GraphCL仅自环23.048%，高于GraphMAE组合22.155%，配对差-0.893pp；描述区间[-2.376,0.484]，不宣称跨权重排名有独立图显著性。

9测试集各1520节点，合计只覆盖6596个唯一节点，两两Jaccard约0.102–0.119；不是9张独立图。18个目录全部完成掩码/5-shot/封存预测评分审计，450条已与本地哈希核对。停止第31轮内部token组合原样扩大，保留fold0局部结果及固定结构收益。下一步实施第32轮协方差草案，并带同几何无提示/GraphPrompt以区分固定度量与训练效果。

## 第三十二轮：协方差几何校正

[固定设计及数值修订](round32_design.md)、[570条开发结果](round32_geometry/summary.md)、[算子检查](round32_operator_audit.json)、[57条预检](round32_smoke_audit.json)。每种原始/中心/对角收缩/完整收缩几何匹配无适配、GraphPrompt、内部CE token。统计使用无标签基础表示，训练期间固定，共享保存矩阵。无收缩合成白化协方差误差1.31e-14，alpha1等价中心化统一缩放误差8.88e-16；零token恢复相应几何基础，主干冻结。

[数值审计](round32_geometry_audit.json)发现Cornell/GraphCL输出协方差trace仅4.44e-19；原绝对特征值下限1e-12使对角与完整收缩都退化为1000000I。原版本保留，不将这组相同结果解读为白化等效。相对下限max(1e-12×trace/d,1e-30)修订后的[尺度不变性及其他五组矩阵审计](round32_relative_audit.json)通过，其他组矩阵不受影响；[单独95条Cornell/GraphCL重跑](round32_relative_Cornell/summary.md)仍未超过原始岭71.53%，不扩大。这是数值实现边界修订，与效果失败分别记录。

开发候选：Actor/GraphMAE对角缩放+内部CE token26.41%，同几何base24.84%、GP25.20%，中心+CE26.16%；Chameleon/GraphMAE完整收缩.1的base36.52%、GP36.76%，原始岭33.88%、未校正仅自环base30.12%。按[预先选择](round32_selection.json)用20新支持seed50100..50119与两权重做800条确认，Actor440、Chameleon360。Actor加同几何随机token；确认源码只追加这一随机分支，保留所有旧分支及相对下限，3条新分支预检通过。Actor测试已用，Chameleon此次首次使用任务内官方fold0测试。

第18次清理删除57条已审计预检约3.57MB，[清单](cleanup_018.json)。正式结果及30份共享矩阵保留；随机分支3条预检待确认结束后清理。

### 第三十二轮确认及归因补充（800+40条）

[Actor结果](confirm32_Actor/summary.md)、[选定内部token配对](confirm32_Actor_audit.json)、[固定校正配对](confirm32_Actor_fixed_audit.json)。GraphMAE对角缩放无token26.993%，内部token26.954%、同几何GraphPrompt27.240%、随机token26.898%。内部token对无token-0.039pp [-0.628,0.477]，对随机+0.056 [-0.391,0.513]，无新增训练收益。

固定校正仍有清楚的同图增益：26.993%对未校正仅自环22.625%高4.368pp [3.576,5.184]，对原Frozen20.562%高6.431 [5.145,7.642]，对原始岭20.803%高6.191 [4.898,7.470]，对GraphCL同几何基础高2.020 [0.967,3.056]。这些区间只反映同图支持变化，不等于跨图泛化。

为拆开中心化与对角缩放，事后按完全相同支持/query补40条仅中心化base：[结果](confirm32_Actor_center/summary.md)、[归因审计](confirm32_center_attribution.json)。GraphMAE中心化26.599%，对角缩放额外+0.395pp [0.066,0.711]；GraphCL中心化25.016%，对角缩放差-0.043区间跨零。补充只作同一批数据的归因，不冒充新的确认集。主体收益来自固定校正，完整协方差并非Actor的赢家。

[Chameleon结果](confirm32_Chameleon/summary.md)、[固定完整校正配对](confirm32_Chameleon_base_audit.json)、[GraphPrompt增量](confirm32_Chameleon_gp_audit.json)。GraphMAE full.1基础32.961%，中心化33.322%、原始岭32.314%；对原始岭+0.647pp [-1.151,2.336]、对中心化-0.362 [-1.985,1.108]。加GP33.191%，对同几何基础+0.230 [-0.428,0.976]，不保留强基线之外的新优势。该批首次使用本任务的Chameleon官方fold0测试，后续需按已使用测试披露。

第19次清理删除3条完成随机分支预检，[清单](cleanup_019.json)。所有正式预测及共享矩阵校验保留。下一步以当前更强固定几何作对照，迁移至原有Cora平滑岭后层Prompt、PubMed修正教师早层Prompt，检验实际训练方案能否叠加收益；原教师门控保持不变，几何仅用于最终表示与原型损失。Actor的固定校正仍可另做多fold核验，当前不是新的独立图结论。

## 第三十三轮：原有强Prompt与几何校正（320/320）

[固定设计](round33_design.md)、[完整开发结果](round33_geometry_prompt/summary.md)、[旧脚本等价性检查](round33_equivalence_audit.json)。Cora与PubMed、GraphMAE/GraphMAE2、5新支持seed51100..51104；四种几何分别匹配base/GraphPrompt/强内部token并带直接教师。统计来自原图冻结GCN，教师与门控保持不变。未校正分支对未修改第21轮代码四种条件预测一致，分数最大误差1.43e-6。

|数据/权重|原强token|中心token|对角token|完整收缩token|
|---|---:|---:|---:|---:|
|Cora/GraphMAE|76.12|71.24|71.44|72.16|
|Cora/GraphMAE2|72.84|71.20|70.60|70.96|
|PubMed/GraphMAE|71.56|68.20|69.04|70.36|
|PubMed/GraphMAE2|72.84|70.00|71.16|68.48|

以上为准确率百分比。几何校正均没有叠加强提示收益，停止这组原样扩大。Cora/GraphMAE完整收缩base68.88虽超过Frozen60.68，但低于直接修正教师74.72和强token76.12；PubMed/GraphMAE对角base71.64仍低于普通GraphPrompt72.32。不把弱基础改善当成超过强对照。查询开发半区已重复使用。

可微闭式岭头候选回查发现第13轮已经实现并审计：Cora强条件提示73.20→71.88下降；不重复同样配置。第20次清理保留审计记录后删除80条本轮预检，正式预测与矩阵保留。

## 第三十四轮：Actor几何收益跨fold复核

[预先设计](round34_design.md)，沿用相对下限版本及固定超参数，fold1–9每fold5新支持，GraphMAE八方法、GraphCL两方法，共450条。重点检查固定对角缩放及中心化的收益范围，同时匹配同几何内部token、GraphPrompt与随机token。测试节点与第31轮重叠，不能称独立新测试。已启动；结果完成前不作跨fold收益判断。

### 第34轮跨fold结果（450/450）

[汇总](multifold34_summary.md)、[配对审计及查询重叠](multifold34_audit.json)。GraphMAE固定diag10基础25.971%，未校正仅自环21.743%、中心化25.773%、原始岭20.168%、原图Frozen20.548%，GraphCL同几何23.914%。对未校正+4.228pp [3.016,5.427]、对原始岭+5.803 [4.304,7.156]，均9/9fold正向。对原图Frozen+5.423 [3.750,6.934]；固定校正优势跨这些划分保留。

对中心化额外+0.197pp [-0.164,0.642]、6/9fold正向，故不能将主要收益归于对角缩放。内部CE token26.408%，对同几何base+0.437 [-0.129,1.050]，对同几何GraphPrompt26.215%高0.193 [-0.218,0.687]，对随机25.991%高0.417 [-0.110,1.013]；训练增量仍不稳定。区间是同一图上fold/支持两层描述重采样；查询集合重叠，不是跨图证据。全部450条正式预测及共享矩阵已本地哈希核验。

## 第35轮：共享仿射提示

[论文借鉴和固定设计](round35_design.md)：以SSF输入无关的共享缩放/平移替换教师条件门控，比较激活前后位置、只缩放/只平移/组合/随机。Actor两权重、5支持、13方法共130条，沿用固定diag10基础，带同几何GraphPrompt和内部条件token。26条3步预检及封存评分审计通过，正式运行中。它是不同于第19轮条件FiLM的受控迁移，不以论文原创性作筛选。

### 第35轮结果（130/130）

[完整结果](round35_affine/summary.md)。GraphMAE共享仿射最高为激活前scale+shift 26.283%，固定基础25.740%、同几何GraphPrompt26.275%、旧条件token26.965%；新方法未超过强训练对照，不扩大。GraphCL最好激活后shift25.140%，基础24.663%、GP24.868%、条件token25.041%，但仅略高于对应随机25.008%，且低于GraphMAE强基础/训练方法；不据五支持微差追加确认。

激活前后仅scale均为GraphMAE25.913%、GraphCL24.227%。本实现缩放系数始终为正，LeakyReLU逐通道正齐次，所以两个分支代数等价；不能当作两个独立改进。随机参数与训练结果区分。第21次清理删除26条预检，保存配置、结果及哈希验证清单[cleanup](cleanup_021.json)。正式130条评分审计及本地sealed/共享几何SHA256通过。当前总数20545，无运行任务。

下一步继续检索不同通道间交互的低秩残差适配，先核对其是否退化为已有逐通道/共享平移，再针对强几何基础与既有条件提示做同位置对照；这只是待设计方向，尚无实现或结果。

## 第36轮：低秩通道残差（180/180）

[设计及LoRA来源](round36_design.md)、[完整结果](round36_lowrank/summary.md)、[算子预检](round36_operator_audit.json)。第一层激活前后、rank1/4各配双因子训练/固定投影只训练B/随机残差，带现有强对照。36条预检的24个残差分支零初始化恢复误差为0；激活后残差折叠为第二层权重更新，合成双精度误差5.33e-15。主干冻结及固定A不变检查通过；不把激活后分支包装成复杂图交互。

GraphMAE训练低秩最高为激活前rank1固定A的27.146%，低于无提示diag10 28.076%、条件token28.553%、同几何GraphPrompt27.928%；相应随机28.363%。双因子训练最高25.715%，更低。GraphCL最高训练低秩24.975%，基础25.099%、相应随机25.041%。这些是5支持开发结果，不将偶然随机扰动正向视作稳定改进；停止当前Actor低秩配置扩大。

第22次清理删除36条完成预检，保留[清单](cleanup_022.json)，正式预测和共享矩阵完整。全任务20725条已本地哈希核验，盘余约31GiB。

后续转回Cora/PubMed：在已有效的后层平滑岭提示/早层修正教师提示上检查低秩残差是否与提示互补。必须用2×2（无提示、仅低秩、仅条件提示、组合）与固定随机投影对照，并保持相同教师KD，避免把蒸馏差异归为模块组合；rank先固定1、同图旧查询使用如实披露。此处是下一轮待实现设计方向，尚未运行，不新增效果结论。

## 第37轮：原图强提示与低秩残差组合

[预先设计](round37_design.md)、[56项预检与等价审计](round37_equivalence_audit.json)。Cora/PubMed×GraphMAE/GraphMAE2×5支持×12方法=240正式指标已启动。原提示所用教师、KD权重、位置与幅度保持不变；token无/有与低秩无/固定A训练B/双因子训练/固定随机残差构成配对。所有可训练分支相同CE+KD，原图GCN冻结，不引入Actor的固定几何。

四种图/权重条件下，无模块对Frozen及单提示对旧root共8对预测完全一致，分数最大误差1.43e-6；旧token、教师和logits函数AST与第21轮源码一致。零残差恢复、主干冻结、梯度有限检查通过。第23次清理删除56项预检并保留[清单](cleanup_023.json)。正式结果未完成前不作增量结论。

### 第37轮开发与确认完成（240+180条）

[开发完整结果](round37_combine/summary.md)。Cora/GraphMAE原提示74.20%，固定A组合73.84%、双因子组合73.52%；GraphMAE2原72.44%、双因子72.52%但随机组合72.72%，不扩大。PubMed/GraphMAE原75.80%，固定A组合74.88%下降；GraphMAE2固定A组合73.68%对原73.00%高0.68pp，随机72.44%。按[预先选择](round37_selection.json)锁定这一候选，20新支持seed56100..56119、同一已重复使用的confirm查询，并带GraphMAE更强权重对照，共180条。

[GraphMAE2确认](confirm37_GraphMAE2/summary.md)、[GraphMAE对照](confirm37_GraphMAE/summary.md)、[配对增量](confirm37_increment_audit.json)。候选73.93%，原强提示74.67%、随机残差+训练token74.66%、低秩单项71.79%、普通GP71.24%、修正教师71.16%，GraphMAE原强提示74.57%。对原提示-0.74pp [-1.80,0.13]，对随机-0.73 [-1.51,-0.06]；开发微增未复现。虽然超过低秩单项/普通GP/直接教师，已有原提示更好，停止本低秩组合原样扩大。查询重复使用，区间仅反映同图支持采样变化。

全任务21145条正式结果及共享矩阵已本地哈希核验。下一轮[类协方差读出草案](round38_design_draft.md)已读Simple CNAPS第4.2节；先区分同表示下的读出改善，再考虑训练提示，不将固定度量提升归为Prompt学习。草案未实现、未运行。Mixup初轮已试，不重复同样配置。

## 第38轮：类协方差读出（320/320开发）

[固定设计](round38_design.md)、[预检算子审计](round38_operator_audit.json)、[完整结果](round38_covhead/summary.md)。72条预检通过原余弦与旧root/Frozen预测一致、欧氏手算、共同缩放与支持顺序不变性；所有协方差Cholesky正定、分数有限。正式40个表示组每组六个读出共享完全相同表示SHA256，[核验](round38_embedding_audit.json)。只替换最终分类距离，不改Prompt训练。

Cora/GraphMAE类收缩beta1读出76.44%，原余弦76.04%、欧氏76.08%、共享beta1 75.80%；类beta.1降至72.12%。PubMed/GraphMAE共享beta.1 72.64%，原余弦72.32%、欧氏72.08%、共享beta1 72.12%，类beta1仅66.40%。GraphMAE2未发现值得扩大微增。部分协方差头改善弱Frozen，但仍低于强提示；不夸大固定读出效果。

按[预先选择](round38_selection.json)仅确认Cora类beta1与PubMed共享beta.1，每条件20新支持seed58100..58119、8方法，共320条，原脚本/训练/超参数不变。confirm查询已使用，非新独立测试。确认正在运行，不能据开发0.40/0.32pp增量作有效结论。第24次清理删除72条预检，保留[清单](cleanup_024.json)。全任务21465正式条目已本地哈希核验。

### 第38轮确认（320/320）

[Cora结果](confirm38_Cora/summary.md)、[配对审计](confirm38_Cora_increment.json)：类协方差读出74.06%，原余弦75.00%、欧氏75.08%、共享协方差74.09%。对原余弦-0.94pp [-2.13,0.18]，对欧氏-1.02 [-2.20,0.08]。比同头Frozen69.98%更高，但这是已有Prompt价值，不是新读出增益。

[PubMed结果](confirm38_PubMed/summary.md)、[配对审计](confirm38_PubMed_increment.json)：共享beta.1读出69.28%，原余弦72.93%、欧氏72.99%、共享beta1 71.09%、普通GraphPrompt73.21%。对原余弦-3.65pp [-6.48,-1.20]，对欧氏-3.71 [-6.62,-1.06]。停止本轮固定协方差读出扩大，不因开发小幅正向而继续调参。全任务21785条正式预测已本地核验；统计仍是同图支持变化。

## 第39轮：隐藏表示VAT

[固定设计及原文](round39_design.md)、[44条预检](round39_smoke_audit.json)。教师和支持CE不变，在提示后的隐藏表示比较VAT与相同幅度随机一致性，epsilon .1/.3，另带无一致性及强直接对照。VAT一轮扰动梯度，方向停止反传；原型和当前概率在一致性项内停止梯度。节点间存在图消息耦合，这是整图隐藏版本，不声称原输入VAT复现。

四种图/权重无一致性分支恢复旧提示预测，扰动RMS符合预定幅度、梯度有限、主干冻结。180条正式实验run39.py已启动（Cora/PubMed×两权重×5支持×9方法），seed59100..59104，旧dev查询。第25次清理保存[预检清单](cleanup_025.json)后删除44条预检；正式预测保留。

### 第39轮开发与确认（180+140条）

[开发结果](round39_vat/summary.md)。Cora两权重无VAT增量；PubMed/GraphMAE epsilon.3为73.56%，原提示与RPT均73.08%，+0.48pp但仅追平普通GP；GraphMAE2原提示74.76%更强。按[预先选择](round39_selection.json)锁定GraphMAE epsilon.3，20新支持seed60100..60119，确认候选/原提示/同幅度RPT/GP/Frozen/教师，并带GraphMAE2原提示，共140条。查询仍是过去用过的confirm500。

[GraphMAE确认](confirm39_GraphMAE/summary.md)、[GraphMAE2强对照](confirm39_GraphMAE2/summary.md)、[配对增量](confirm39_increment_audit.json)：VAT75.99%，原提示75.63%、随机一致性75.61%、GP75.56%、Frozen73.41%、直接修正教师72.22%。对原提示+0.36pp [0.09,0.63]，对随机+0.38 [0.10,0.65]，分别11/20、13/20支持正向；保留为PubMed/GraphMAE局部准确率改进。对GP+0.43 [-1.04,1.98]未形成明确优势，GraphMAE2原提示76.43%更高，跨权重差-0.44 [-1.55,0.75]。不宣称成为所有权重最佳或跨图泛化。VAT增加forward/扰动梯度计算，非等计算预算收益。

全任务22105条正式预测及矩阵本地哈希核验完成。下一轮[鲁棒原型草案](round40_design_draft.md)借鉴联邦聚合几何中位数；先测固定中心估计，不假设数据被攻击。草案尚未运行，后续PubMed/GraphMAE应带这次VAT强提示作匹配基础。

## 第40轮：鲁棒原型聚合

[固定设计](round40_design.md)、[预检算子审计](round40_operator_audit.json)。Cora/PubMed、GraphMAE/GraphMAE2，原Frozen与强提示表示各配mean/30步平滑几何中位数/等权折中/单步重加权/随机权重，并带直接教师与GP，共280条。PubMed/GraphMAE使用第39轮确认的VAT.3强提示，其余强root不变；每个支持条件只训练一次，五种中心共用表示。

68条预检完成，八对原均值与旧脚本/Frozen预测一致，分数最大差1.67e-6。刚体变换/样本重排/重复相同点误差约1e-16；每步平滑目标下降。合成四个零点加一个极端点时均值范数20、平滑中位数约0.00025，只证明该合成诊断，不说明真实图存在异常或必然增益。正式run40.py已启动，结果待审计。第26次清理删除68条完成预检，保留[清单](cleanup_026.json)。

### 第40轮开发完成（280/280）

[完整结果](round40_robust/summary.md)、[40组共享表示核验](round40_embedding_audit.json)。Cora/GraphMAE median74.96%，mean74.76%、one74.92%、blend74.88%、random74.52%，微小提高。GraphMAE2 median72.40%对mean72.32%，但低于随机72.68%及直接教师73.00%，不扩大。PubMed/GraphMAE（VAT基础）median75.00%等于mean，one75.52%、blend75.32%，仍低于GP75.96%，不选择该条件。

PubMed/GraphMAE2 median74.08%，mean73.28%、one74.04%、blend73.96%、random73.48%；重加权有局部优势，但30次迭代几乎没有超出单步。按[预先选择](round40_selection.json)确认Cora/GraphMAE与PubMed/GraphMAE2，每组8方法×20新支持seed62100..62119，并带PubMed/GraphMAE VAT均值与GP强对照40条，共360条。旧confirm查询已重复使用，不作独立测试声明。正式run_confirm40.py已启动。

本地正式预测哈希累计22385条通过；几何中位数聚合收益仍待确认，不把固定读出改动称为新Prompt训练。

### 第40轮确认完成（360/360）

[Cora结果](confirm40_Cora_GraphMAE/summary.md)、[PubMed/GraphMAE2结果](confirm40_PubMed_GraphMAE2/summary.md)、[强GraphMAE对照](confirm40_PubMed_GraphMAE/summary.md)、[配对审计](confirm40_increment_audit.json)。Cora中位数与均值均74.85%，对均值差约0.00pp [-0.25,0.25]，对单步74.81%仅+0.04 [-0.08,0.16]；开发小增未复现。

PubMed/GraphMAE2中位数73.71%、均值73.66%、单步73.61%、blend73.62%、随机73.35%。对均值+0.05pp [-0.29,0.40]，对单步+0.10 [-0.14,0.34]，对随机+0.36 [-0.38,1.18]均区间跨零。强GraphMAE VAT均值73.76%、普通GP73.40%；中位数并未增加稳健优势。停止本轮稳健聚合原样扩大，保留开发局部正向，不能由合成抗异常诊断推断实际图增益。

累计22745条正式预测及矩阵本地哈希审计通过；盘余31GiB，无运行任务。下一轮[表示近邻传播草案](round41_design_draft.md)借鉴视觉LaplacianShot的邻域一致性，以锚定概率传播配对原图/表示近邻/随机重连及支持钳制，不宣称原论文优化器复现。它区别于旧原图教师平滑，草案未实现、未运行。

## 第41轮：表示近邻预测传播（360/360开发）

[固定设计与LaplacianShot/Local-Global Consistency来源](round41_design.md)、[84条预检](round41_operator_audit.json)、[正式结果](round41_propagation/summary.md)。旧强提示训练保持不变，原图或10近邻上作锚定概率传播，alpha.5/.9、20步，匹配支持钳制开关和随机重连。不是原论文界优化器复现，收益先归于传导读出。8对原读出预测恢复，零传播、常数概率、精确近邻、概率行和与钳制检查通过。

[邻接与表示绑定审计](round41_neighbors_audit.json)：40组共享表示、80个稀疏邻接文件、160个结果引用，SHA256绑定、行和、无自边/重复邻居全部核验。只存稀疏索引/权重，无N平方磁盘缓存。第27次清理删除84条预检及其图缓存，[清单](cleanup_027.json)，盘余约30GiB。

Cora/GraphMAE2强提示knn.9钳制75.80%，原74.40%，+1.40pp；GraphMAE同方案75.84%对原75.64%，+0.20。PubMed/GraphMAE（VAT）knn.9为73.04%，原72.08%，+0.96，GP72.96%；GraphMAE2同方案70.72%对原70.36%。原图平滑未超过强近邻方案。弱Frozen的局部提高不作全方法最佳结论。

按[预先选择](round41_selection.json)锁定Cora/GraphMAE2和PubMed/GraphMAE的knn.9；补齐相同alpha.9的随机重连及无钳制对照，再配原提示、.5近邻、原图、同传播Frozen、GP/教师；另带各数据另一权重原提示与knn.9。20新支持seed64100..64119，共480条run_confirm41.py已启动。查询仍是反复使用的confirm500，开发增量尚未确认。全任务23105正式结果哈希及几何/邻接引用通过。

### 第41轮确认完成（480/480）

[Cora/GraphMAE2](confirm41_Cora_GraphMAE2/summary.md)、[PubMed/GraphMAE](confirm41_PubMed_GraphMAE/summary.md)、[完整配对与邻接绑定审计](confirm41_increment_audit.json)。Cora/GraphMAE2 knn.9钳制73.10%，原提示72.67%，+0.43pp [0.02,0.86]；无钳制73.08%，差+0.02 [-0.05,0.09]，钳制不是主要增量。随机重连54.27%明显更差，但其大幅劣化不等于新机制优势。保留同权重传导收益，仍低于同批GraphMAE原提示75.22%。

[Cora/GraphMAE对照](confirm41_Cora_GraphMAE/summary.md)传播后75.92%，原提示75.22%；[事后补算配对区间](confirm41_Cora_GraphMAE_secondary.json)为+0.70pp [0.20,1.17]、16胜3负1平。该权重作为预设强对照一起运行，但完整无钳制/随机/原图对照只在主候选GraphMAE2上，故保留为次要局部增益，不能声称已完成GraphMAE归因或独立复核。它是最终传导读出，不是提示训练新增。

PubMed/GraphMAE knn.9与原VAT提示均73.29%，增量约0 [-0.37,0.38]；.5近邻73.37%、原图.5 73.38%，无扩大价值。GraphMAE2原提示74.15%、传播74.00%也无额外收益。停止PubMed本传播配置扩大。

全部480条预测、邻接SHA256及表示绑定核验，累计23585正式条目本地通过。无运行任务，稀疏图保留，盘余约30GiB。下一轮[训练内邻域一致性草案](round42_design_draft.md)拟将近邻约束用于第二阶段提示优化；必须带相同200步和随机图对照及纯推理传播。草案未实现未运行。若优先巩固Cora/GraphMAE次要增益，应另做完整归因，不把现有两方法比较当完成证明。

## 第41轮补充：强GraphMAE新增20支持集复核（200/200）

[预先固定设计](round41b_design.md)、[配置](round41b_selection.json)、[结果](confirm41b_Cora_GraphMAE/summary.md)、[配对及邻接绑定审计](confirm41b_increment_audit.json)。seed65100..65119，旧源码、100步提示、k10/alpha.9/20步传播保持不变，10方法共200条。

Cora/GraphMAE强提示73.76%，近邻传播74.74%，配对+0.98pp [0.53,1.44]，20个支持集16个为正。无钳制74.72%，两者差+0.02 [-0.02,0.06]，支持钳制贡献很小。近邻alpha.5为74.01%，原图alpha.5为74.15%；候选分别高0.73 [0.41,1.04]、0.59 [0.10,1.08]。原图对照alpha不同，不能单独把差异全归因于邻接结构。随机重连alpha.9为54.47%，这只表明破坏邻域有害，不单独证明方法有效。

Frozen64.48%、同传播Frozen67.37%、普通GraphPrompt64.27%、岭平滑教师71.50%。保留强提示后的表示近邻传播为Cora读出改进；它不属于提示训练增益。两批均正向，但查询仍是同一反复使用的confirm500，支持种子区间不能代表独立图泛化。无钳制与钳制基本相当，后续保留原固定候选，避免按微小差别重新选择。

全部200条封存预测、邻接SHA及表示绑定通过；累计23785条本地哈希审计通过。无运行任务、盘余约30GiB，无需删正式结果。下一步优先在Cora实现第42轮训练内一致性，带100步、相同200步、真/随机图损失及本轮纯传播基线；PubMed尚无推理传播收益，不先扩大其训练矩阵。

## 第42轮：训练内近邻一致性（100/100开发）

[固定设计](round42_design.md)、[预检](round42_operator_audit.json)、[结果](round42_consistency/summary.md)、[阶段状态与固定图审计](round42_consistency_stage_audit.json)。Cora两个权重、各5支持种子，第一阶段100步后复用token和Adam再训练100步，加真实/随机邻居概率差平方惩罚1/10。各分支的初始token、Adam矩、图哈希一致；主干冻结、图固定。旧100/200步与零新增损失输出恢复误差小于1.7e-6。独立GPU重新训练会导致近乎并列邻居换位，预检传播概率存在小差异（最大0.00173），分类相同；未宣称位级复现，正式各分支共用同一图。

GraphMAE原100步72.20%、200步72.28%、近邻权重1/10为72.32/72.44%、纯预测传播72.88%。GraphMAE2对应69.40/70.40/70.44/70.48/70.88%。训练约束对匹配200步仅+0.04..0.16pp，而且两权重都低于纯传播，停止原配置扩大；GraphMAE2从100到200步的主要提高不能归于新损失。随机权重10明显劣化，不是候选有效性的充分证据。

保存100条正式封存预测；第28次清理移除26条完成预检及缓存约10MB，比较摘要保留，正式不删。累计23885条。下一步[跨语言检索密度校正草案](round43_design_draft.md)，已检索并读CSLS原文式6；TIM在第15轮做过，不重复。下一轮尚未实现运行。

## 第43轮：CSLS原型密度校正（160/160开发）

[固定设计](round43_design.md)、[预检](round43_operator_audit.json)、[训练代码未变审计](round43_source_audit.json)、[结果](round43_density/summary.md)、[校正与共享表示审计](round43_density_density_audit.json)。借鉴跨语言词检索CSLS，原型对非支持节点top10相似度作为类别偏置，alpha.25/.5，配常数偏置和非零循环置换。两图两权重、5种子，每组共用强提示表示；查询真值封存后评分。零校正完全恢复、常数偏置分类相同、类别置换误差0。

Cora/GraphMAE原提示及两种校正均75.88%，原10NN传播76.76%；GraphMAE2校正74.12%对原74.08%，传播75.28%。PubMed/GraphMAE原75.68%，校正75.64%，普通GP76.64%；GraphMAE2校正最高75.64%对原75.52%，仍低于同批更强参照。停止本配置扩大，不把0.04/0.12pp零星正向当稳定收益。

[诊断](round43_density_diagnostic.json)：四条件的类别偏置极差均值仅0.0077..0.0164，每500查询改变0–5个分类，证据支持该固定密度校正在当前表示上作用很小；不据此断言所有密度校正无效。第29次清理删36条已完成预检约4.75MB，正式160条保留。累计24045条本地封存哈希审计通过，盘余约30GiB，无运行任务。

下一步[互近邻传播草案](round44_design_draft.md)借鉴行人重识别论文，配每行等边数随机删边，检验已有效的Cora10NN传播能否进一步改进。草案未实现未运行。

## 第44轮：互近邻过滤与混合传播（90开发+200确认）

[固定设计](round44_design.md)、[图审计](round44_reciprocal_graph_audit.json)、[开发结果](round44_reciprocal/summary.md)。共享同一表示/10NN，互近邻保留后归一，空行回退原10NN；等每行边数随机删边、50/50混合、无钳制作对照。GraphMAE混合76.64%对原传播76.32%，开发+0.32pp；纯互近邻76.04%不优。GraphMAE2混合75.44%低于原传播75.80%。

按[预先选择](round44_selection.json)，主GraphMAE8方法加另一权重2方法，20新支持seed69100..69119共200条。v2仅补同配比50/50随机删边混合。确认[主结果](confirm44_Cora_GraphMAE/summary.md)、[配对区间](confirm44_increment_audit.json)：候选混合75.39%，原传播75.70%，差−0.31pp [−0.48,−0.15]；随机混合75.67%，差−0.28 [−0.48,−0.07]。相对原提示75.03%只有+0.36 [−0.03,0.80]，不足以越过强基线。纯互近邻74.81%。停止本互近邻配置，保留原10NN传播。

GraphMAE2对照原提示73.30%、原传播74.05%。本批查询仍反复使用confirm500，支持种子不是独立图。真实互惠/空行回退/随机边数/权重公式及同表示绑定已审计，确认120图文件保留。[主图审计](confirm44_Cora_GraphMAE_graph_audit.json)、[另一权重](confirm44_Cora_GraphMAE2_graph_audit.json)。第30次清理18条预检约3.96MB，正式不删。

累计24335条封存结果。下一步[支持集监督对比草案](round45_design_draft.md)回到提示训练，借鉴SupCon式2，匹配原100步与标签置换对照，尚未实现运行。

## 第45轮：支持集监督对比损失（90/90开发）

[固定设计](round45_design.md)、[预检](round45_operator_audit.json)、[正式结果](round45_supcon/summary.md)、[支持标签审计](round45_supcon_loss_audit.json)。借鉴SupCon式2，原100步强token的CE/KD保持，添加权重.1/1监督对比，每anchor4个同类正样本，自身排除分母。置换只影响新增对比标签，CE/KD标签与教师不变。逐项参考loss完全一致，零权重旧预测误差小于1.5e-6，梯度有限，类别/支持顺序重排不变；查询封存后评分。

Cora/GraphMAE原提示74.12%，SupCon.1为74.04%、1为73.48%；强传播75.60%。GraphMAE2原73.56%，.1为73.64%、1为72.80%，强传播74.16%。置换.1分别74.04/73.20%，更大权重置换均明显下降。小权重的0.08pp单点正向未超过强方法，停止本配置扩大，不扩PubMed；不能根据支持loss下降宣称泛化收益。

第31次清理删16条预检约0.60MB，审计保留。累计24425条正式封存结果本地哈希通过；无运行任务。下一步[检索式邻居表示聚合草案](round46_design_draft.md)借鉴alpha加权查询扩展，先聚合表示再重算支持原型，与现有概率传播做匹配比较，未实现运行。

## 第46轮：检索式邻居表示聚合（120开发+360确认）

[固定设计](round46_design.md)、[同步聚合/原型预检](round46_operator_audit.json)、[开发](round46_qe/summary.md)。强提示训练不变，归一化表示聚合10/50邻居及自身，alpha3相似度幂权重对照均匀AQE、相同边权随机邻居、原图特征聚合与原概率传播。一次同步更新后重新计算支持原型，非新提示参数。开发GraphMAE均匀50邻居77.12%、加权77.08%、原10邻居概率传播77.08%；GraphMAE2为75.52/75.48/74.80%，进入复核。

[确认配置](round46_selection.json)固定20新支持seed72100..72119，两权重9方法，共360条；v2仅补均匀随机邻居。结果：[GraphMAE](confirm46_Cora_GraphMAE/summary.md)、[GraphMAE2](confirm46_Cora_GraphMAE2/summary.md)、[配对审计](confirm46_increment_audit.json)。GraphMAE均匀50聚合75.17%，原提示74.01%，+1.16pp [0.50,1.93]；旧10邻居概率传播74.66%，差+0.51 [0.00,1.15]，区间触零，不能写成明确超过。加权50为75.10%，差+0.07 [−0.02,0.18]，没有相似度幂加权的优势。

GraphMAE2均匀50为74.03%，原72.14%，+1.89 [1.21,2.65]；旧概率传播72.91%，+1.12 [0.48,1.86]。加权73.84%，均匀高0.19 [0.00,0.49]，同样边界触零；仍低于同批GraphMAE均匀75.17%。原图特征聚合74.18/72.36%，但图度数/边权总量不匹配，不独立归因于近邻结构。随机邻居均匀27.90/31.12%的破坏结果只作对照，不单独证明新方案有效。

保留均匀50邻居表示聚合相对原提示的收益；源于传导后处理，查询仍反复使用confirm500。当前未匹配50邻居概率传播，不能把对旧k10的改善全部归于表示聚合或原型更新。[下一轮草案](round47_design_draft.md)补齐匹配k50、一次概率平均与原型重算开关。全部图SHA/表示绑定、随机边权及原型重算检查通过；确认200稀疏文件保留。第32次清理24条预检和单元输出约9.28MB，正式结果不删。

累计24905条正式结果本地哈希审计；无运行任务，下一轮尚未实现。

## 第47轮：匹配50邻居的表示与概率聚合（360/360）

[固定设计](round47_design.md)、[独立算子预检](round47_operator_audit.json)、[结果](round47_matched/summary.md)、[同图绑定审计](round47_matched_matched_audit.json)。Cora两权重、20新支持seed73100..73119，9方法；query仍旧confirm500。各50邻居方案共享原提示表示和top50列表，比较重算/固定原型的表示聚合、一次概率平均、均匀/softmax20步概率传播；保留旧10NN强对照。

[表示方案配对](round47_increment_audit.json)：GraphMAE均匀表示聚合75.62%，原提示74.68%，+0.94pp [0.43,1.50]，但与旧10NN75.65%无差异，低于50NNsoftmax概率传播76.25%，差−0.63 [−1.02,−0.25]。固定原型75.34%、一次概率平均75.49%，原型重算与聚合位置的增量区间均跨零。GraphMAE2表示聚合73.87%，原72.31%，+1.56 [0.80,2.44]；旧10NN73.50%、50NNsoftmax74.08%，表示方案没有超过匹配概率方案。

[概率方案配对](round47_probability_audit.json)：GraphMAE50NNsoftmax76.25%较旧10NN75.65%高0.60pp [0.27,0.93]，较原提示74.68%高1.57 [0.92,2.21]；GraphMAE2为74.08%较旧73.50%高0.58 [0.10,1.06]、较原72.31%高1.77 [1.08,2.48]。均匀50NN20步为76.21/74.06%，与softmax仅0.04/0.02之差，无明确权重优势。优先保留更高准确率的50NN概率传播，不再宣称表示聚合优于概率聚合；第46轮相对原提示的收益仍保留。

归因边界：大邻域的作用有同批对照，重算原型的独立优势未确立；概率迭代/钳制与单次平均仍含多因素。这是传导读出，非提示训练新增；不能因20新支持称独立测试。算子文件的self_weight字段是表示聚合中间量，对20步概率模式不适用，具体序列化解释见设计文档。稠密参考已核对其实际无自边行归一算子。

累计25265条本地封存/邻接SHA通过；第33次清理18条预检及合成输出约12.15MB，正式240邻接文件保留，无运行任务。下一步[固定配置迁移PubMed](round48_design_draft.md)，保留VAT与GraphMAE2强根，不重新扫参数，草案尚未运行。

## 第48轮：固定50邻居方案迁移PubMed（360/360）

[固定设计](round48_design.md)、[算子预检](round48_operator_audit.json)、[强提示恢复](round48_legacy_audit.json)、[结果](round48_pubmed/summary.md)、[共享表示/图审计](round48_pubmed_matched_audit.json)。沿用Cora配置，20新支持seed74100..74119、2权重9方法，不在PubMed扫k/alpha。GraphMAE保留VAT.3，GraphMAE2保留ref50教师根；旧输出最大误差小于1.5e-6，预检显存足够，正式冻结100步提示。

[配对审计](round48_increment_audit.json)：GraphMAE原提示74.37%，k50softmax概率传播73.82%，−0.55pp [−1.07,−0.02]；旧k10为74.06%、均匀表示聚合74.08%、普通GP73.93%。GraphMAE2原74.10%，k50概率73.74%，−0.36 [−0.75,0.08]；旧k10为73.91%、表示聚合73.81%。不保留PubMed传播改动，Cora收益不能泛化为所有图有效。固定原型聚合/一次概率平均仅74.41/74.38及74.09/74.13，微小变化不追参。

全部360条封存评分及240图绑定通过，本地同步哈希全部通过，累计25625。query仍反复使用confirm500。第34次清理20条已完成预检及单元输出约82.76MB，正式保留；服务器盘余28GiB，无运行任务。下一步[Poly-1提示损失草案](round49_design_draft.md)已读视觉分类/检测原文，匹配动态样本权重与常数CE缩放，尚未实现运行。

## 第49轮：Poly-1支持损失（120/120开发）

[固定设计](round49_design.md)、[解析梯度/零系数预检](round49_operator_audit.json)、[结果](round49_poly/summary.md)、[常数缩放及共享表示审计](round49_poly_poly_audit.json)。原100步token、KD不变，支持留一CE加epsilon±1的一阶多项式项，匹配由初始支持概率决定的固定CE缩放；各训练同时评估原读出与相同50NN概率读出。梯度公式和数值一致，零epsilon恢复误差小于1.5e-6，极端置信度有限；每个训练方法的两读出共用表示。

Cora/GraphMAE原读出72.44%，Poly负/正72.24/71.72%，常数负/正72.40/72.32%；传播后原71.88%，Poly负71.76%，未改善。本批50NN传播比原读出低0.56pp，提醒已有平均收益并非每组支持都改善。GraphMAE2原69.48%，Poly负69.68%；传播原69.80%、Poly负70.16%、常数负70.04%。局部+0.36pp、超匹配常数仅+0.12pp，且低于同批GraphMAE原72.44%，不扩大本Poly配置。保留小局部正向，不将其写成所有损失无效。

全部120条封存预测与50个邻接引用通过，本地累计25745条哈希审计。第35次清理26条预检约11.87MB，正式保留。服务器仍约28GiB，无运行任务。

下一步[九种权重迁移草案](round50_design_draft.md)：第18轮没有测后来有效的k50读出，复用现有九种兼容checkpoint，固定强提示/传播做匹配筛选。已核实服务器文件存在，尚未运行；不把弱Frozen的巨大相对增量当新最佳。

## 第50轮：强读出跨九权重迁移（315/315开发）

[固定设计](round50_design.md)、[九权重算子预检](round50_operator_audit.json)、[结果](round50_backbones/summary.md)、[45组共享表示/图审计](round50_backbones_matched_audit.json)。Cora九种兼容两层GCN128各5支持、7方法，强提示100步固定。63条预检完成严格加载、冻结、no-op和来源SHA；MultiGprompt结构不同不混入，不代表其无效。

GraphMAE原提示75.84%、k10概率77.24%、k50概率77.60%、表示AQE77.08%，是同批最高。GraphMAE2最好为AQE76.08%，k50概率75.64%。其余权重最好：Edgepred_Gprompt74.72%、GraphCL69.64%、SimGRACE68.48%、Edgepred_GPPT67.68%、DGI49.84%、S2GAE32.52%、Bandana31.20%。原始平滑岭教师统一75.12%，多个权重连直接教师都未超过。无新权重进入20支持确认；弱Frozen上的大幅相对增益不作新最佳，仍保留GraphMAE强方案。

本轮只检验这些已有checkpoint及固定适配方式，不否定相应预训练方法的全部实现。第18轮没有k50读出，本轮是新增迁移证据。dev500同图查询已反复使用，不将5支持筛选视为独立测试。全部315条封存、135邻接引用通过，本地累计26060条哈希审计。

第36次清理63条预检及单元输出约23.93MB，正式保留，盘余约28GiB。下一步[支持缓存终端组合草案](round51_design_draft.md)借鉴Tip-Adapter缓存与分类预测组合，在现有强方案上检验准确率互补；明确是读出融合，不作为新Prompt内部机制，尚未实现运行。

## 第51轮：提示表示支持缓存组合（140/140开发）

[固定设计](round51_design.md)、[算子预检](round51_operator_audit.json)、[结果](round51_cachemix/summary.md)、[共享表示及融合审计](round51_cachemix_cache_audit.json)。借鉴Tip-Adapter，保持强提示训练不变，将beta10/30支持缓存概率以0.25/0.5混入强传播；支持自身排除、每类剩余数量校正，带纯缓存与相同权重标签置换对照。合成独立实现、lambda0恢复和实际混合重算均通过。

Cora/GraphMAE强传播76.68%、原型76.04%，缓存组合最高76.56%（beta10、0.25），纯缓存76.08%，未超过强方法。GraphMAE2强传播73.92%、原型74.40%，真缓存组合最高74.08%，纯缓存最高74.24%；置换缓存0.25反而74.44%，没有可保留的真实缓存增量。不扩大本配置；不将这批结果推广为所有缓存组合无效。

5个支持seed77100..77104、dev500仍是反复使用的同图查询。140条封存及90邻接引用、10组共享嵌入核对通过；本地累计26200条预测SHA验证。第37次清理18条已完成预检约2.88MB，正式结果保留，服务器约28GiB可用，无训练任务。下一步[LaplacianShot草案](round52_design_draft.md)已读原文，尚未实现；先解决邻域算子的理论条件和匹配线性对照，再固定矩阵。

## 第52轮：LaplacianShot式非线性推断（100开发＋320确认）

[设计与作者代码来源](round52_design.md)、[算子预检](round52_operator_audit.json)、[开发](round52_laplace/summary.md)、[确认](round52_confirm/summary.md)、[共享图审计](round52_confirm_laplace_audit.json)。同一提示训练与同一50NN图，仅将线性锚定传播换为原型logits加邻域概率后softmax；lambda1/3/10开发，选lambda10确认，新增相同强度单步与共轭置换随机图。预检lambda0恢复、NumPy独立参考、类别置换通过。

开发GraphMAE2为72.32%对线性71.36%，GraphMAE73.24%对73.16%。固定20新支持seed79100..79119确认：[配对](round52_increment_audit.json)GraphMAE2为74.53%，相对原型72.33%高2.20pp [1.14,3.27]，相对单步73.08%高1.45 [0.54,2.39]；但相对强线性74.15%仅+0.38 [−0.21,1.04]，未确立增量，且低于同批GraphMAE线性75.59%。GraphMAE非线性75.18%对强线性75.59%，−0.41 [−0.86,0.08]，不替换当前最佳。lambda3为75.46/73.84%；确认不是重选lambda的开发集。GraphMAE2 lambda10对lambda3区间下界数值约0，视为触零。

[残差审计](round52_residual_audit.json)：真实图lambda10在GraphMAE20/20、GraphMAE2的17/20末步最大概率变化仍大于0.001，最大0.425/0.298。结果是固定20步工程变体，不称已收敛解，也不继承原文对称半正定条件下的证明。保留相对原型的局部效果；本配置对强基线未验证增量，后续阻尼/对称算子若再试须独立固定，不回改本结果。

query仍为反复使用的confirm500，新支持不是独立图。420条封存及270邻接引用核对通过，本地累计26620条SHA审计。第38次清理18条预检约5.10MB，正式保留，服务器约28GiB，无训练任务。下一步[Focal Loss支持训练草案](round53_design_draft.md)，与Poly-1区分并带梯度尺度/置换对照，尚未实现运行。

## 第53轮：Focal与动态CE缩放（160开发＋400确认＋200再确认）

[固定设计](round53_design.md)、[算子审计](round53_operator_audit.json)、[开发](round53_focal/summary.md)、[初次确认](round53_confirm/summary.md)、[再确认](round53b_confirm/summary.md)。借鉴目标检测Focal，完整梯度与解析系数核对通过；对照为每步梯度系数均值乘CE及系数置换乘CE，KD保持不变，所有训练的原型/k50读出共享表示。各对照按自身轨迹计算系数，不声称全程数值尺度严格一致。

开发GraphMAE原型：原75.48%、Focal2 75.92%、scale2 76.00%、perm2 75.96%；Focal未超对照，按准确率目标确认更简单scale2。初次20支持主候选scale2原型74.04%对原73.46%，+0.58pp [−0.02,1.11]，低于强传播75.00%；次要scale2+k50为75.68%，对原传播+0.68 [0.02,1.30]，Focal/perm传播75.67/75.63%，没有特异难样本优势。GraphMAE2 scale2原型71.14对70.77有+0.37 [0.21,0.53]，但低于旧传播72.81；scale2传播73.01增量区间跨零。见[原型配对](round53_increment_audit.json)、[传播配对](round53_secondary_audit.json)。

为避免确认后挑选，另固定GraphMAE新20支持seed82100..82119、主候选scale2+k50再确认：[配对](round53b_increment_audit.json)76.48%对旧传播75.52%，+0.96pp [0.54,1.51]，16胜2负2平；对原型74.61%高1.87 [1.15,2.57]。Focal2与perm2传播均76.53%，与scale2差异区间跨零。保留三者近似表现，默认选择无需逐样本难度对应的scale2，作为后续强训练根；不把0.05pp差异称确定优胜，不宣称Focal机制必要。

[系数审计](round53_coefficient_audit.json)：scale2初始平均系数约1.104–1.168，最终0.160–0.255，实际改变CE/KD平衡，不能笼统称增强监督。两次确认使用不同支持但同一反复confirm500，并非独立图/新测试集。760条正式结果、310邻接引用的服务器审计通过。第39次清理18条预检约2.88MB，正式保留。下一步[标签平滑草案](round54_design_draft.md)需加入新scale2强根，尚未实现。

## 第54轮：支持标签平滑（180开发＋400确认）

[固定设计](round54_design.md)、[软标签梯度及零强度恢复](round54_operator_audit.json)、[开发](round54_smooth/summary.md)、[确认](round54_confirm/summary.md)、[共享表示/图审计](round54_confirm_smooth_audit.json)。借鉴Inception标签平滑，仅改变支持留一CE目标，epsilon0.1/0.2；带仅减小CE系数和乘scale2的组合。显式软标签与PyTorch平滑CE梯度一致，smooth0恢复旧根、mix0恢复scale2。主干、原型和KD保持一致。

开发GraphMAE没有超过scale2+k50的76.12%；GraphMAE2 smooth2+k50为74.64%，高于scale2的74.28%，据此固定20支持seed84100..84119确认。结果[配对](round54_increment_audit.json)：主候选GraphMAE2平滑+k50为74.41%，对旧传播73.87%高0.54pp [0.08,1.05]，对仅弱化CE的73.95%高0.46 [0.07,0.87]；但对新强根scale2的74.34%仅+0.07 [−0.24,0.40]，没有明确增量。GraphMAE平滑76.59%，对旧75.68%高0.91 [0.52,1.29]，对weak75.80%高0.79 [0.43,1.14]；相对scale2的76.50%仅+0.09 [−0.22,0.41]。

保留标签平滑相对旧CE的效果及相对单纯弱化监督的证据，将其作为可用替代，不替换第53轮默认强根；不宣称与scale2叠加有效（开发组合未胜）。同批scale2仍超过旧传播，支持已有改进的稳定性，但不是新独立图测试。confirm500反复使用，支持种子重复采样的区间不代表跨图泛化。

580条服务器封存及240邻接引用审计通过；本地确认文件已同步完成，开发文件仍由session43913同步，完成后需重跑本地累计SHA审计，预期27960。第40次清理22条预检约3.04MB，正式保留，服务器约27GiB，无训练任务。下一步[PubMed固定缩放迁移草案](round55_design_draft.md)，保留VAT/原型强根且不传播，尚未实现运行。

## 第55轮：动态CE缩放固定迁移PubMed（280/280确认）

[固定设计](round55_design.md)、[VAT/非VAT零强度恢复和梯度](round55_operator_audit.json)、[结果](round55_transfer/summary.md)、[VAT保留审计](round55_transfer_transfer_audit.json)。两权重20支持seed85100..85119、7方法，直接固定gamma2规则，未搜索PubMed参数。GraphMAE保持VAT0.3与ref50，GraphMAE2保持ref50；支持损失接口之外不改训练，也不迁移Cora概率传播。

[配对](round55_increment_audit.json)：GraphMAE scale2 73.41%对原强方案73.95%，−0.54pp [−1.65,0.61]；Focal73.66%、置换73.48%、普通GP73.40%。GraphMAE2 scale2 74.33%对原74.29%，+0.04 [−1.16,1.27]；Focal74.53%、置换74.49%，相互差异均无明确证据。直接ref50教师70.56%，提示超过直接教师但这不是缩放新增收益。

保留PubMed原根，不叠加固定缩放；Cora的可重复改进没有在本PubMed矩阵迁移成功。不以同权重Frozen的增益替代强根增量，不将区间跨零解释为严格等效。confirm500仍反复使用，两权重源SHA/冻结/零强度恢复均通过，80条GraphMAE提示记录保持VAT的0.3扰动。

全部280条封存审计通过，本地累计28240条SHA验证通过；第41次清理10条预检约0.27MB，正式保留。前54轮最后同步已完成。下一步[DWA与简单监督退火草案](round56_design_draft.md)，借鉴视觉多任务损失下降速率调权，带新scale2强根及固定/退火对照；已读原文式7，尚未实现。需补完整无query轨迹，不能由旧首尾history推造中间loss。

## 第56轮：DWA与简单监督退火（120/120开发）

[固定设计](round56_design.md)、[无查询训练轨迹](round56_trace.json)、[算子恢复](round56_operator_audit.json)、[结果](round56_weight/summary.md)、[每步权重及图审计](round56_weight_weight_audit.json)。借鉴视觉多任务DWA，只按过去两个完整5步块的CE/KD均值比值调权，T2，前10步等权，完整记录100步权重。另测固定CE0.2、线性1.1到0.2，保留旧提示和新scale2各两读出。预检等权恢复、独立DWA计算通过；额外两个无query标签评分的100步轨迹不计正式查询条数。

GraphMAE旧原型75.04%、旧传播75.16%，DWA75.04/75.20%，固定低权重74.68/74.92%、线性74.88/74.68%，scale2为74.84/75.04%。GraphMAE2旧73.84/74.68%，DWA73.84/74.72%，固定73.56/74.64%、线性73.52/74.60%、scale2为73.60/74.56%。DWA仅多0.04pp，其他未胜，不扩大本配置。

[作用幅度](round56_dwa_diagnostic.json)：正式DWA CE权重范围约[0.970,1.004]/[0.982,1.021]；相对旧传播，每个支持集500查询仅改变0–3/0–1个预测。不是数学上严格等价，但实际改动很小。scale2在本批比旧传播低0.12pp，保留第53/54轮平均收益的同时明确不保证每批支持都提升；简单固定0.2或线性退火未解释为同等有效替代。

120条封存及50邻接引用审计通过，本地累计28360条SHA通过；第42次清理12条15步预检约2.68MB，正式结果和无查询完整轨迹保留。下一步[CLIP式训练温度草案](round57_design_draft.md)，保持查询读出/教师温度不变以隔离训练影响，已读原文，尚未实现。

## 第57轮：固定与可学习训练温度（180/180开发）

[固定设计](round57_design.md)、[结果](round57_temperature/summary.md)、[温度与邻接审计](round57_temperature_temperature_audit.json)、[温度诊断](round57_temperature_diagnostic.json)。借鉴CLIP可学习logit尺度，在支持LOO监督项使用固定温度0.1、0.4或从0.2开始学习的一个尺度参数，分别搭配普通CE和scale2。教师KD与查询读出温度始终为0.2，保持强提示及k50传播不变。Cora两权重、5支持种子87100–87104、18方法共180条。22条预检验证固定0.2恢复原CE/scale2、尺度梯度和保存参数正确；全部正式预测重算通过。

GraphMAE本批最强仍为旧CE+k50的74.88%；固定0.1/0.4及可学习温度的普通CE传播为74.60/74.52/74.72%，对应scale2传播74.28/74.60/74.00%，原scale2传播74.32%。GraphMAE2固定0.1或可学习温度加scale2传播均73.40%，比原scale2的73.16%高0.24pp，但低于同批GraphMAE强方案74.88%。不扩大本配置，保留GraphMAE2局部正向证据，不推断训练温度普遍无效。

学习温度最终约0.091–0.096（GraphMAE普通CE）、0.119–0.126（GraphMAE scale2）、0.096–0.101（GraphMAE2普通CE）、0.125–0.131（GraphMAE2 scale2），未触及尺度100上限。120条温度记录、80表示组和80邻接引用通过核验。本地累计28540条封存预测SHA通过，正式输出全部保留；[第43次清理](cleanup_043.json)删除22条已完成预检约3.05MB，磁盘剩余27GiB，无训练任务。

下一步[跨预训练权重概率集成草案](round58_design_draft.md)，借鉴Deep Ensembles分类概率平均，先固定成员和权重，再验证是否超过最强单模型及旧CE强对照。明确属于终端预测组合；目前仅完成原文阅读和设计草案，未实现或运行。若复用已有开发预测，需标明重复开发而非新独立证据；选择后使用新支持集确认。同图重复查询的局限持续保留。

## 第58轮：跨权重概率平均（20/20复用开发预测）

[固定设计](round58_design.md)、[配置](round58_config.json)、[结果](round58_ensemble/summary.md)、[审计](round58_ensemble/audit.json)。借鉴Deep Ensembles概率平均，两个成员为GraphMAE/GraphMAE2的scale2+k50；第二成员权重0.25、0.5、0.75，复用第57轮已封存五支持开发预测。此为终端预测组合，无共享主干；20条是新增混合查询指标，没有新增训练，四个单模型既有参照不重复计数。

三个合法混合准确率分别74.36%、74.08%、73.40%。0.25比GraphMAE scale2的74.32%仅高0.04pp，低于同批旧CE GraphMAE传播的74.88%，因此按预定标准不进入新支持集确认。GraphMAE2 scale2为73.16%，旧CE传播72.84%。0.5类别循环置换对照40.12%，说明正确类别对应重要，但不能据此证明合法集成优于强单模型。

独立审计核对两个不同checkpoint及预测SHA、图与split SHA、支持/查询顺序、重复成员混合恒等，并从成员概率重建每条混合分数；全部预测先封存再加载查询标签评分。全部20条通过，本地累计28560条SHA通过。没有新增邻接或权重副本，也无临时训练产物可清理，磁盘约27GiB可用。本轮不是新独立开发数据，不能用重复查询的结果给出独立泛化结论。

下一步[提示优化余弦退火草案](round59_design_draft.md)，借鉴SGDR式5，保留Adam和100步预算，带恒定学习率及累计量匹配对照，先无重启；尚未实现或运行。

## 第59轮：提示优化余弦学习率（180开发＋400确认）

[固定设计](round59_design.md)、[开发结果](round59_schedule/summary.md)、[确认设计](round59_confirm_design.md)、[确认结果](round59_confirm/summary.md)、[配对区间](round59_confirm_pairs.json)。借鉴SGDR式5，保持Adam100步和相同提示参数，仅改变学习率。普通CE与scale2分别比较恒定0.01、恒定0.005、余弦峰值0.01/0.02至0；后者累计学习率等于恒定0.01。20条预检的恒定恢复分数最大误差小于1.5e-6，端点、累计量、实际更新范数和冻结审计通过。

开发Cora五支持88100–88104：GraphMAE scale2+cosine0.02+k50为76.44%，对原scale2+k50的76.36%高0.08pp；GraphMAE2对应75.72对75.68%，+0.04pp。其他较低学习率配置未超过强对照。按预定标准，将GraphMAE此配置固定为主候选，使用89100–89119新20支持及原确认500查询验证，两权重共400条。

确认GraphMAE候选74.24%，当前scale2+k50为74.16%，增量+0.08pp，95%种子bootstrap区间[-0.22,0.43]，8胜8负4平；不足以替换现用强方案。对旧CE传播73.74%高0.50pp [0.03,0.98]，对普通CE同余弦传播73.68%高0.56pp [0.17,0.97]，但这不能将整体收益归给退火。GraphMAE2次要结果72.97%，对原scale2的72.83%高0.14pp [-0.06,0.35]，同样不明确。保留小幅正向均值，不宣称所有调度无效，也不继续原样扩大本配置。

[开发调度审计](round59_schedule_schedule_audit.json)、[确认调度审计](round59_confirm_schedule_audit.json)核对完整每步学习率、末步零更新、896个提示参数、共享表示和邻接SHA；580条正式预测重算通过，本地累计29140条预测及邻接SHA核验通过。仍为同图重复开发/确认，不是独立图泛化证据。第44次清理20条已完成预检约0.80MB，正式结果保留。下一步[Lookahead草案](round60_design_draft.md)，已读原文算法1及内部状态说明，尚未实现；第17轮终端EMA平均不等同于训练中同步快慢提示。

## 第60轮：Lookahead与恒定学习率对照（220开发＋480确认）

[固定设计](round60_design.md)、[开发结果](round60_lookahead/summary.md)、[恒等预检](round60_operator_audit.json)。借鉴Lookahead算法1，每5次Adam更新后以0.5向快提示插值并回写，保留Adam状态，100步/20次同步。普通CE、scale2分别带恒定0.005/0.01/0.02及Lookahead内lr0.01/0.02，各原型/k50读出。28条10步预检验证alpha1恢复Adam、k1/alpha0.5等于半学习率，误差均小于1.5e-6。正式设计纠正了初始草案关于k1不等价的错误判断，未将恒等控制重复计入正式矩阵。

五支持90100–90104开发：GraphMAE的恒定0.02 scale2+k50达到76.68%，原scale2 75.28%、旧CE75.32%，Lookahead内lr0.02为75.24%；普通CE恒定0.02为76.32%。因此Lookahead没有独立优势，但按准确率优先目标，将更简单的高学习率固定为确认主候选。GraphMAE2高学习率为74.76%，原scale2 74.28%，仍低于GraphMAE主候选。

[确认设计](round60_confirm_design.md)、[确认结果](round60_confirm/summary.md)、[配对区间](round60_confirm_pairs.json)：新20支持91100–91119，GraphMAE高学习率scale2传播76.21%，原scale2传播76.48%，-0.27pp [-0.82,0.29]，6胜12负2平；对旧CE76.15%仅+0.06pp且区间跨零。Lookahead76.35%、普通CE高学习率76.25%，均未超过原scale2。GraphMAE2高学习率74.83%，原scale2 74.67%，+0.16pp [-0.28,0.62]，同样不明确。开发收益未复现，不替换当前0.01强方案，也不原样扩大。

高学习率GraphMAE原型73.82%，原scale2原型75.34%，传播后恢复到76.21%，提示不要把读出恢复误当训练本身进步。700条正式预测均重算通过，本地累计29840条预测及邻接SHA通过，调度审计验证896个参数、Adam状态步数持续递增及共享表示/邻接哈希。[第45次清理](cleanup_045.json)删除28条已完成预检约1.14MB，正式输出保留。同图重复查询限制不变。

下一步[CiteSeer迁移草案](round61_design_draft.md)，对已有强修正教师和预训练提示分别应用固定k50传播，先保持当前Adam0.01，带未传播强教师参照；目前仅草案，尚未运行。

## 第61轮：CiteSeer教师与提示概率传播迁移（110/110开发）

[固定设计](round61_design.md)、[结果](round61_transfer/summary.md)、[独立重算审计](round61_transfer_transfer_audit.json)。固定Cora的k50/alpha0.9/20步设置，五个新支持92100–92104。直接教师仅计算一次，原始/修正原型分别不传播或传播；两GCN权重保留Frozen、GraphPrompt及原位置/后层CE/后层scale2提示两读出。直接教师不依赖checkpoint，记录为Direct/null SHA并保留独立特征图来源；不把它冒充Prompt。

修正教师66.24%，传播54.12%，下降12.12pp；普通教师57.88%→49.32%。GraphMAE最好提示为原位置提示传播57.36%，GraphMAE2为58.76%，均低于未传播修正教师。原位置提示本身57.04/57.80%，后层scale2传播55.80/58.36%；因此不能将Cora后层或scale2配置直接称为CiteSeer强方案。Frozen传播也下降（32.96→27.60、41.76→26.32）。停止扩大本固定迁移配置。

独立审计从特征重建修正教师，最大误差1.91e-6；核对原始概率锚、支持标签、alpha0、无自边/邻居唯一/行和，并以索引加权求和重算传播。20条直接教师记录、50邻接引用与40表示组通过。110条正式封存预测重算通过，本地累计29950条预测及邻接SHA通过，22条预检已完成。[第46次清理](cleanup_046.json)删除预检临时输出，正式预测和必要稀疏图保留，磁盘剩余26GiB。

[无查询标签图诊断](round61_graph_diagnostic.json)：教师空间top50图最大入度818、平均50，12节点入度0；支持92102的500查询某类预测从84变为261，另一个类别42变2。只记录集中现象，不将其直接归因为hub，也不以置信度上升证明变好。同图反复查询限制持续保留。

下一步[局部尺度亲和核草案](round62_design_draft.md)，已读视觉Self-Tuning Spectral Clustering第2节式1/2：第7邻居确定尺度；比较同邻居改权重和局部距离重排，并带全局核同强度传播对照。尚未实现或运行。

## 第62轮：局部尺度核与传播强度（40开发＋120确认）

[固定设计](round62_design.md)、[开发结果](round62_local/summary.md)、[算子审计](round62_local/operator_audit.json)。借鉴视觉Self-Tuning Spectral Clustering局部尺度式1/2，比较全局余弦核、同top50局部核、局部距离重排top50；每种alpha0.5/0.9，固定修正教师概率起点。新五支持93100–93104：未传播教师67.16%，全局0.5为67.28%，局部/重排0.5均67.12%；alpha0.9分别57.52/55.52/58.60%。局部尺度没有开发优势，固定简单全局0.5为主确认候选。

[确认设计](round62_confirm_design.md)、[结果](round62_confirm/summary.md)、[配对区间](round62_confirm_pairs.json)。20新支持94100–94119，全局0.5为68.51%，原强教师68.24%，+0.27pp [-0.40,0.89]，12胜6负2平；不替换未传播强教师。局部0.5为68.74%、重排0.5为68.52%，只是次要结果，不按确认表现更换主候选。全局0.9为57.52%，减弱传播主要恢复过强传播的损失，尚未证明超过强原方案。

局部重排最大入度818→238，但准确率未明确增加；同邻居局部核的平均有效邻居数由15.79变35.04，重排43.06。只能说明图分布改变，不能将图指标当作性能结论。真实CiteSeer表示无零向量、无尺度下限触发，尺度范围约0.190–1.311。独立CPU核权重/尺度/topk门限及传播重算、alpha0恒等、额外零向量/重复向量预检均通过。160条正式预测重算通过，本地累计30110条预测与邻接SHA通过，稀疏图每批三份共享。[第47次清理](cleanup_047.json)删除8条预检约4.26MB，正式结果保留。

下一步[SIF式特征加权与公共方向消除草案](round63_design_draft.md)，已检索作者资料并读官方权重/未中心化PC实现；原论文PDF受浏览器验证阻挡，阅读边界明确。先改独立强教师，不改冻结预训练主干输入；尚未实现或运行。同图重复开发/确认限制持续保留。

## 第63轮：SIF式加权与公共方向处理（40开发＋80确认＋80复核）

[固定设计](round63_design.md)、[开发结果](round63_sif/summary.md)。已读SIF官方权重/PC源码，原论文PDF受验证限制的边界已记录。五支持95100–95104：原强教师68.68%，中心化70.20%、去PC70.00%；频率加权a1e-3/1e-4为58.20/50.20%，加权去PC60.76/50.96%，明显退化。收益来自公共方向处理的简单控制，而不是SIF加权组合。

[第一批确认设计](round63_confirm_design.md)、[结果](round63_confirm/summary.md)、[配对](round63_confirm_pairs.json)：固定center为主候选，新20支持96100–96119，center69.07%对base67.71%，+1.36pp [0.78,1.97]，18胜2负；PC69.26%，相对center高0.19pp，属于次要对照。

[第二批固定复核](round63b_design.md)、[结果](round63b_confirm/summary.md)、[配对](round63b_pairs.json)：新20支持97100–97119，center68.97%对base67.20%，+1.77pp [1.15,2.49]，19胜1负。PC69.18%，比center高0.21pp，但差异区间跨零。保留简单中心化强教师及PC替代，不夸大PC额外优势。[可复核方案](当前CiteSeer强教师.md)已单独记录。两批仍用同图confirm500查询，不是独立图证据。

预检验证二元词袋、频率双精度计算、常数权重恢复、去PC与合成数据SVD一致；实际幂迭代9/10/34步，特征残差约1e-7。只保存均值、权重、PC和表示哈希；200条正式封存预测重算通过，本地累计30310条预测及几何/邻接SHA核验通过。[第48次清理](cleanup_048.json)删除8条预检约0.35MB，正式结果保留。

这是不使用GCN权重的直接教师提升，尚非Prompt提升。下一步[门控/KD分离接入草案](round64_design_draft.md)，固定冻结主干输入、token位置和训练预算，分别替换中心化教师门控/蒸馏，并保留强直接教师参照；尚未实现或运行。

## 第64轮：中心化教师门控与KD分解（150开发＋600确认）

[固定设计](round64_design.md)、[开发结果](round64_teacher/summary.md)、[恒等恢复](round64_operator_audit.json)。CiteSeer原位置token、冻结两种GCN、100步Adam0.01保持，分别替换中心化教师门控/KD/两者，带未中心化修正教师及无KD控制。旧教师恢复误差<1.5e-6；实际门控/KD概率由各自SHA验证。直接教师独立来源，不伪造checkpoint。

开发五支持98100–98104：GraphMAE2中心门控无KD传播69.08%，旧提示60.12%、直接中心化教师68.88%；因此按预定标准固定无KD方案为主候选进入20新支持确认。门控+KD68.88%、仅门控68.80%、仅KD59.96%，主要改善见于门控。

[确认设计](round64_confirm_design.md)、[确认结果](round64_confirm/summary.md)、[跨方法配对](round64_confirm_pairs.json)：支持99100–99119，GraphMAE2主候选67.49%，旧提示61.06%，+6.43pp [4.06,8.92]，18胜2负；但直接中心化教师69.10%，主候选低1.61pp [-2.61,-0.66]。门控+新KD67.27%、门控+旧KD67.33%，无KD的差异区间均跨零；仅改KD61.09%，未形成有用增量。不把改善归给概率KD，也不宣称提示超过强教师。

GraphMAE次要主配方65.18%，旧57.99%，+7.19pp [3.45,11.15]；仍低于直接教师3.92pp [-5.85,-2.31]，其中仅门控+旧KD65.58%略高于无KD。保留[更强CiteSeer Prompt训练基线](当前CiteSeer提示方案.md)，整体最高准确率参照仍为强直接教师。不原样扩大该配置，下一步针对教师与提示的差距设计。

750条正式预测重算通过，本地累计31060条预测及几何/邻接SHA通过，门控/KD分支、768个参数、冻结与表示/邻接绑定审计通过；跨Direct比较严格核对图/支持/查询，不要求不存在的共同checkpoint。[第49次清理](cleanup_049.json)删除16条完成预检约0.73MB，正式预测保留。同图重复查询限制不变。

下一步[关系蒸馏草案](round65_design_draft.md)，已读CVPR2019 RKD第3.2.1节式5–8；以归一化中心化教师关系约束提示表示，固定门控无KD根并带概率KD/置换控制及直接教师。尚未实现或运行。

## 第65轮：距离关系蒸馏（270/270开发）

[固定设计](round65_design.md)、[结果](round65_relation/summary.md)、[算子恢复](round65_operator_audit.json)、[审计](round65_relation_transfer_audit.json)。借鉴视觉RKD式5–8，以支持+128无标签锚点的平均距离归一化Huber损失约束学生；教师先中心化再L2，学生L2，权重0.1/1/10各带置换和未中心化对照。主干及768个Prompt参数、100步预算保持。

CiteSeer新五支持100100–100104：GraphMAE2正确中心关系三权重传播66.16/66.24/66.60%，无KD根66.08%、门控旧KD66.56%、新KD66.40%，直接中心化教师67.72%。最高关系项仅比最强已有Prompt高0.04pp，仍低强教师1.12pp，不扩大。权重1置换传播66.32%还略高正确关系66.24%；权重10正确关系66.60%高于置换62.04%、未中心化61.84%，只保留局部对应信号，不等于超过强基线。

GraphMAE正确关系原型最高62.20%，传播最高62.16%，原无KD传播58.88%、新KD60.76%，仍明显低于直接教师。不能把较弱权重上的恢复作为综合突破。此前第64轮新Prompt收益保留，但本批支持造成较大波动，不用跨批绝对数值代替配对比较。

14条预检通过λ0恢复（最大误差1.44e-6）、独立样本对距离、Huber值与梯度检查；关系梯度非零。正式核对锚点唯一/支持包含、12403个距离目标及SHA、冻结和两读出表示绑定。[关系作用诊断](round65_relation_diagnostic.json)记录目标离散程度及首末损失/梯度；不能以训练损失降低替代准确率。270条封存预测重算通过，本地累计31330条预测与几何/邻接SHA核验通过。[第50次清理](cleanup_050.json)删除14条已完成预检约1.13MB，正式输出保留。

下一步[角度关系蒸馏草案](round66_design_draft.md)，来源同文第3.2.2节式9/10，固定稀疏三元组避免三阶全图存储；尚未实现或运行。强直接教师仍为必须超过的参照，同图重复查询限制保持。

## 第66轮：角度关系蒸馏（230/230有效开发，首版工程失效另存）

[固定设计](round66_design.md)、[有效v2结果](round66_angle_v2/summary.md)、[审计](round66_angle_v2_transfer_audit.json)。借鉴RKD第3.2.2节式9–10，支持与128个无标签锚点中固定2048个非退化三元组，以角度Huber损失约束提示表示；权重0.1/1/10均带教师置换控制，另保留距离权重10、无KD和概率KD强根。冻结主干、768个提示参数及100步训练预算保持。

首版230条训练虽完成，但独立CPU目标审计发现一个教师边在CPU长度为0、GPU约1.10e-7，产生约0.69角度差异。该版全部判为工程失效，不计有效实验。没有放宽审计容差：另建v2，把教师非退化边筛选阈值从1e-8改为1e-5，固定相同支持种子重跑230条。[修复说明](round66_v2_repair.md)、[失效取证](failure66/failure.json)、[CPU/GPU诊断](failure66/edge_diagnostic.json)保留。重跑不是新的独立支持证据。

有效v2支持101100–101104：GraphMAE2正确角度三权重传播68.68/68.76/63.44%，无KD68.60%、新概率KD69.08%、旧KD68.88%、距离权重10为69.00%；直接中心化教师69.60%。角度最高68.76%未超过最强Prompt或直接教师，不扩大。置换权重0.1/1为68.72/68.84%，也不支持正确角度对应带来明确增量。GraphMAE角度最高传播67.04%，距离67.28%，仍低直接教师；不把弱主干上的恢复当作整体突破。

v2预检通过零权重恢复（最大误差9.54e-7）、独立角度值与梯度核对；正式通过唯一锚点、非退化三元组、跨候选固定采样SHA、冻结、门控/KD与读出表示绑定审计。独立教师重构最大误差4.53e-6，230条封存预测重算通过，本地累计31560条正式预测及几何/邻接SHA核验通过。[第51次清理](cleanup_051.json)删除32条完成预检和230条失效首版的大输出，约153.67MB；有效v2及失效取证保留，磁盘剩余26GiB。

下一步[九预训练权重筛选草案](round67_design_draft.md)：CiteSeer中心化门控Prompt跨Bandana、DGI、Edgepred_GPPT、Edgepred_Gprompt、GraphCL、GraphMAE、GraphMAE2、S2GAE、SimGRACE比较。已确认文件存在，严格加载与正式训练尚未开始。同图重复查询限制保持。

## 第67轮：九预训练权重的中心化门控（455/455开发）

[固定设计](round67_design.md)、[配置](round67_config.json)、[权重审计](round67_checkpoint_audit.json)、[结果](round67_checkpoints/summary.md)、[审计](round67_checkpoints_transfer_audit.json)。九权重均通过两层GCN128严格加载和已有sidecar哈希，来源一致性边界按sidecar保留。新支持102100–102104，同一dev500查询。

中心化门控Prompt各权重最佳（含原型/传播选择，均为探索性）：Bandana24.44%、DGI38.60%、Edgepred_GPPT61.24%、Edgepred_Gprompt62.24%、GraphCL44.20%、GraphMAE66.52%、GraphMAE2 67.88%、S2GAE49.00%、SimGRACE55.56%。最高仍GraphMAE2旧KD门控传播67.88%；同批新KD67.76%、无KD67.48%，直接中心化教师69.16%。其他七权重没有超过现有强Prompt或直接教师，不扩大。

不能只看相对旧弱提示增益：DGI新提示最高38.60%仍低Frozen43.12%；Bandana新提示24.44%还低旧提示传播24.84%。Edgepred_Gprompt新提示62.24%虽高Frozen27.96%，仍比最佳Prompt低5.64pp。按本轮预定规则，无候选进入20支持确认。

455条正式预测及180个邻接引用、180组共享表示通过服务器审计，独立教师重构最大误差2.86e-6。全部正式输出同步本地后累计32015条；[第52次清理](cleanup_052.json)移除91条已完成预检约51.29MB，保留正式输出，服务器余25GiB。

下一步[新提示表示中心化草案](round68_design_draft.md)，借鉴SimpleShot第2.1节并保留第32轮已有几何经验；区分只读几何、训练内动态中心与冻结中心，不把已有操作称新机制。重新检索的DKD已由第25轮验证，不重复运行。同图反复查询限制不变。

## 第68轮：新提示表示中心化（265开发＋500确认完成）

[封存设计](round68_design.md)、[配置](round68_config.json)。新支持103100–103104，两预训练权重，无KD/新KD各比较原、只读中心化、训练内动态中心化和冻结基础中心；带旧KD强提示和直接教师，正式265条。57条短预检带零变换恢复，批处理只有通过预测/门控/几何三项审计后才自动进入正式。

只读分支复用根提示参数，中心统计不使用标签。动态中心保留均值梯度，冻结中心取零提示基础表示；变换后仍按原余弦原型和k50读出。结果中的training_readout=cosine_unchanged表示余弦公式保持，实际几何变化由geometry、center及哈希字段区分，不表示输入表示未改变。审计独立重建中心与原型分数，并核对只读分支参数相同。

57条预检已全部通过，几何独立重构、均值梯度和平移不变性、零变换恢复及参数复用核验通过，见[几何审计](smoke68_geometry_audit.json)和[分支审计](smoke68_transfer_audit.json)。批处理PID99753已进入正式265条训练（子进程100365）；尚无正式结论，不计入32015条有效累计。

开发支持103100–103104：[完整结果](round68_center/summary.md)、[独立几何审计](round68_center_geometry_audit.json)。GraphMAE2无KD只读中心化+k50为70.04%，原无KD67.44%、新KD67.16%、旧KD67.28%，直接中心化教师69.88%。只读中心化+新KD为69.80%；训练内动态/冻结中心无KD仅65.48/65.76%，带KD56.20/56.36%，不能把只读收益归于训练中心化。

GraphMAE无KD只读中心化传播69.24%，原65.28%，但低直接教师69.88%；新KD只读中心68.88%，原64.60%；训练中心最高66.60%。Frozen/普通GraphPrompt的中心化并不普遍有效，例如GraphMAE2 Frozen中心原型28.04%低原40.28%，说明作用依赖提示后的表示，不是一切嵌入通用增益。

265条预测和130个邻接引用通过服务器审计，160条几何记录独立重构最大误差1.02e-5；只读分支参数逐项完全相同。全部输出同步后，本地累计32280条SHA通过。[第53次清理](cleanup_053.json)删除57条完成预检约36.79MB，保留正式结果，服务器余25GiB。

按开发前规则预选GraphMAE2无KD只读中心化+k50，领先教师仅0.16pp，尚不称确定提升。[预选记录](round68_selection.json)、[20支持确认设计](round68_confirm_design.md)已封存，支持104100–104119，confirm500查询，两权重共500条；新/旧KD及直接教师保持。正式确认PID101132（父101131）正在运行，未计入有效累计；当前查询半区已在过去使用，不是新图独立测试。

第68轮确认完成：[结果](round68_confirm/summary.md)、[配对](round68_confirm_pairs.json)、[几何审计](round68_confirm_geometry_audit.json)。预选GraphMAE2无KD只读中心化+k50为68.52%，同批原67.24%，+1.28pp [0.68,1.90]，15胜4负1平；对新KD原67.02%高1.50pp [1.01,2.02]、旧KD67.06%高1.46pp [0.95,2.01]。直接教师68.91%，差值−0.39pp [−0.99,0.21]，开发时微弱领先未复现，不能声称超过教师。新KD只读中心68.41%，无KD高0.11pp区间跨零。

GraphMAE次要只读方案68.26%，原64.60%，+3.66pp [2.72,4.54]，18胜1负1平；低于直接教师0.65pp [−1.40,0.07]。据此保留[更新后的CiteSeer提示读出](当前CiteSeer提示方案.md)，不采用训练内中心化。收益属于固定读出改善，同图重复查询边界保持。

500条确认正在同步本地（rsync session88111；无需重启）。500条确认及200个邻接引用已通过服务器审计，160条几何记录独立重构最大误差6.44e-6；只读参数一致性核对通过。下一步[中心化强度草案](round69_design_draft.md)，固定已有效的b1与b0控制，并匹配直接教师；尚未实现或运行。

## 第69轮：只读中心化强度（批处理已启动）

[固定设计](round69_design.md)、[配置](round69_config.json)。新支持105100–105104，两个预训练权重，无KD根训练后共享参数；中心化强度0/0.5/0.75/1/1.25/1.5两读出，并带新KD原/中心化和直接教师。正式185条，预检45条。

预检第一次因恢复分数阈值1e-7停下，正式未启动。[数值诊断](round69_recovery_diagnostic.json)证实b0/b1与根实现的嵌入SHA、参数、邻居索引和权重、预测均一致，仅GPU传播分数最大差4.77e-7。没有修改训练或读出；审计增加精确一致性检查，并将分数容差改为1e-6。批处理103012复用已完成预检重审已通过，[几何审计](smoke69_geometry_audit.json)最大重构误差9.66e-6。185条正式训练已完成，当前正在正式审计；尚未计数。不将该审计误报计为算法失败。

第68轮确认文件仍由rsync session88111同步邻接，已观察179/200份，未重启传输；全部完成后再执行本地累计32780条审计。

第69轮185条正式结果已通过服务器预测、分支及几何审计，独立重构最大误差6.44e-6。GraphMAE2强度0/0.5/0.75/1/1.25/1.5传播分别68.36/68.12/69.16/69.36/66.40/64.08%；GraphMAE分别66.08/64.84/66.28/68.40/61.92/57.20%。没有新强度超过完整中心化，保留b1、不扩大。

本批直接教师69.36%；GraphMAE2中心化新KD69.52%是已有替代对照，第68轮确认与无KD差异不明确，不将本批0.16pp解释为新强度突破。第68轮本地同步和32780条累计SHA已完成；第69轮由rsync session16436和审计文件scp session54930同步，服务器累计32965条，全部同步后再执行本地累计审计。45条smoke69尚未清理，下一次清理编号054。

第69轮全部正式文件及审计文件已同步，本地32965条预测/几何/邻接SHA重新核验通过。[第54次清理](cleanup_054.json)移除45条已完成预检约28.42MB，正式结果和数值诊断保留，盘余25GiB。

## 第70轮：Cora强提示只读中心化迁移（130开发＋520确认完成）

[固定设计](round70_design.md)、[配置](round70_config.json)。固定新支持106100–106104、两预训练权重；原后层token+KD和强scale2训练根均保持，只比较最终表示减均值与否，带原型/k50匹配及Frozen几何控制。26条预检通过预测/损失轨迹/独立表示重构后才进入130条正式开发，批处理PID105180。

不迁移CiteSeer教师，保留Cora旧平滑岭门控，896提示参数、100步预算、k50传播均固定；只读分支必须复用原训练参数。本轮尚无正式结论，未计入32965条有效累计。首轮独立CPU审计遗漏checkpoint的map_location，遇旧CUDA1保存索引停止；补全CPU映射后复用26条已封存预检重审，训练与权重不变。

第70轮开发130条全部通过预测、损失缩放及独立几何审计（重构最大误差5.48e-6）：GraphMAE主候选中心化scale2+k50为75.12%，原scale2 74.36%，原token74.16%、中心化token74.96%；但中心化scale2原型73.64%低原74.04%。GraphMAE2中心化scale2传播72.56%对原72.16%，中心化token73.04%对原72.20%，均为次要参照。Frozen中心化不改善，收益不能归为通用嵌入变换。

[预选记录](round70_selection.json)固定GraphMAE中心化scale2+k50。[确认设计](round70_confirm_design.md)采用新支持107100–107119、confirm500，完整13方法×2权重×20支持=520条已启动（父进程106492）。不按确认结果更换主候选，同图重复查询边界保留。

[第55次清理](cleanup_055.json)移除26条已完成预检约14.00MB，正式结果保留。130条开发正在rsync session67212同步，本地累计审计需待同步完成，预期33095；服务器已有33095条通过审计，520条确认暂不计数。

第70轮确认：[完整结果](round70_confirm/summary.md)、[预选配对](round70_confirm_pairs.json)。主候选GraphMAE中心化scale2+k50为74.65%，原scale2 75.91%，−1.26pp [−1.98,−0.42]，2胜17负1平；对旧token75.77%低1.12pp [−1.82,−0.27]。开发正向未复现，停止Cora中心化迁移，保留[原Cora强方案](当前Cora强方案.md)。

GraphMAE2次要候选73.81%对原74.92%，−1.11pp [−1.81,−0.34]。中心化scale2虽高于中心化旧token（GraphMAE+0.34、GraphMAE2+0.20），但都低于未中心化强参照，不能据较弱比较宣称成功。GraphMAE中心化原型73.32%低原75.02%；传播带来恢复而不是超过强根。

520条确认预测、240邻接引用和240条独立几何记录通过服务器审计，几何重构最大误差9.78e-6。服务器累计33615条；本地开发累计33095条SHA已通过，确认输出由rsync session62303、审计文件scp session27718同步，结束后需累计审计33615。

下一步[PubMed固定迁移草案](round71_design_draft.md)，保持其修正教师/VAT强根，只测固定中心化原型，不叠加已失败的k50或scale2；尚未实现。

第70轮确认同步完成，本地33615条预测及几何/邻接SHA重新核验通过。

## 第71轮：PubMed固定强提示中心化迁移（65/65完成）

[封存设计](round71_design.md)、[配置](round71_config.json)。新支持108100–108104，两权重原强提示及只读中心化，GraphMAE保留VAT0.3并带无VAT控制；独立修正教师Direct每支持仅计算一次。13条预检，65条正式，只做原型读出，不叠加k50/scale2。

首版按权重两次调用同一目录触发配置保护，GraphMAE的8条预检完成后停止，正式未开始；v2改为一次CLI顺序运行两权重，GraphMAE2跳过重复novat。训练算法及矩阵不变，不绕过配置保护，原日志及smoke71保留，重审目录smoke71_v2。批处理PID107815，只有通过预测、VAT/冻结和独立教师/中心重构审计才进入正式。v2的13条预检已通过，[几何审计](smoke71_v2_geometry_audit.json)和[教师/表示审计](smoke71_v2_transfer_audit.json)已保留；65条正式训练子进程108411正在运行。本轮尚无结论，未计入33615条有效累计。

第71轮[完整结果](round71_pubmed/summary.md)、[审计](round71_pubmed_transfer_audit.json)、[独立几何核对](round71_pubmed_geometry_audit.json)：GraphMAE VAT强根74.16%，只读中心化72.32%；无VAT原74.12%、中心化72.32%。GraphMAE2原72.12%、中心化70.88%。直接修正教师66.72%，中心化虽仍高于教师却低于强提示，不扩大、不替换PubMed原方案。Frozen中心化也不普遍改善（GraphMAE+0.32pp，GraphMAE2−1.52pp）。

65条正式预测/教师/冻结/VAT审计通过，25条中心化记录独立重构最大误差4.53e-6；文件同步后本地累计33680条SHA核验通过。[第56次清理](cleanup_056.json)删除首版8条及v2共13条预检约1.00MB，保留失败编排日志和所有正式结果，盘余25GiB。

下一步[正交子空间干预草案](round72_design_draft.md)。新读ReFT第3.2节及附录E，并对照第36/37轮已失败的普通低秩残差；不声称子空间公式本身具有新表达能力。拟对当前CiteSeer强提示做正交参数化/固定子空间/普通低秩的匹配比较，官方实现与梯度还需核对，尚未实现运行。SSF检索与第19轮FiLM重叠，不另行重复。

## 第72轮：正交子空间干预（285/285开发完成）

[实现与官方代码差异](round72_implementation.md)、[算子审计](round72_operator_audit.json)、[真实GCN原语审计](round72_primitive_audit.json)。已核对官方正交参数化，完成orth/fixed/plain/residual四个rank4实现及CiteSeer门控后、激活前的组合。这里是表示适配器与Prompt的共享隐藏层交互，不是预测融合。

四种模式均恒等初始化；正交补不变、显式低秩仿射展开、输入梯度、固定投影不变及数值有限检查通过。实际新增可训练参数1028（固定子空间516），带token共1796/1284；正交参数化还需16896缓冲元素。明确同低秩仿射表示类，检验的是约束/初始化/优化轨迹，保留第36/37轮失败边界。

真实CiteSeer/GraphMAE2四模式×token有无共8项3步检查均通过，未读取查询标签，不计入33680条正式累计。正式封存预测、联合源码哈希、独立干预前向/读出审计及两权重完整矩阵仍待接入；当前没有训练任务或准确率结论。

第72轮[正式设计](round72_design.md)、[矩阵](round72_config.json)已封存：两权重、新支持109100–109104、每权重28方法及独立Direct，共285条。四模式×token有无及两读出为16项，其他均是同批强/基础对照。57项3步预检启动，批处理PID109674；只有封存预测、教师分支和独立仿射展开/传播分数审计全部通过才进入正式。

已接入驱动及两个辅助模块联合源码SHA；保存最终旋转矩阵、映射、偏置、提示和中心，核对参数量、关闭token、冻结及非零梯度。操作位置保持同一共享GCN第一层卷积后/激活前，最终中心化保持只读。57条预检已全部通过，[独立重构](smoke72_geometry_audit.json)最大误差1.11e-5，[分支审计](smoke72_transfer_audit.json)通过。285条正式训练已启动，子进程110272；当前仍无准确率结论，33680条累计不变。

第72轮[完整结果](round72_subspace/summary.md)、[120组开发配对](round72_development_pairs.json)、[独立重构审计](round72_subspace_geometry_audit.json)：GraphMAE2正交+token传播65.24%，固定子空间66.00%、无约束差分60.04%、普通残差59.88%；但原强中心化无KD提示68.56%、新KD68.52%、直接教师68.88%。正交胜部分弱适配器不等于超过强提示，不扩大。

GraphMAE正交组合63.92%，固定子空间65.56%、无约束差分60.56%、普通残差62.80%，原强提示66.60%。无token适配器最高GraphMAE2原型45.48%、GraphMAE传播45.12%，虽较Frozen恢复，仍远低教师条件提示。没有组合超过原强根，不增加rank或直接扩大该配置。

285条封存预测、130个邻接引用和220条独立几何记录通过服务器审计，重构最大误差1.06e-5；联合源码SHA、参数量、正交约束、token关闭及梯度核对通过。正式文件同步后，本地累计33965条预测与几何/邻接SHA核验通过。[第57次清理](cleanup_057.json)删除57条完成预检约36.88MB，正式结果和无query原语审计保留。

下一步[支持类别初始化草案](round73_design_draft.md)，已读Lester等EMNLP2021第2.1/3.2节，借鉴类别相关初始化，但图场景改用支持隐藏均值；保持原768参数，不新增适配器，带零/置换/匹配幅度随机对照。初始化幅度规则与正式矩阵仍待封存，尚未实现。

## 第73轮：支持类别初始化（已启动）

[固定设计](round73_design.md)、[配置](round73_config.json)。新支持110100–110104，语义/置换/随机方向×0.01/0.1两幅度；对照匹配经过教师门控后的实际隐藏扰动RMS。保持768参数、100步、原强中心化门控及读出，带零初始化、新KD和Direct强参照，正式225条。

有效token在tanh空间构造再atanh初始化，原始初值并不等于草案中的直接倍乘。置换/随机只匹配实际总扰动，不声称逐类参数范数仍相同；若有效token越界或类别方向退化则停审，不静默截断。49条预检包含强度0恢复、独立CPU类别均值与初值重构、扰动匹配、冻结和两读出共享参数检查。首次审计因两次零初始化重训传播分数差异停下；[数值诊断](round73_recovery_diagnostic.json)显示token最大差8.08e-7、原型1.19e-6且预测一致，近并列邻居替换导致GraphMAE传播分数差0.00874。修订审计保留参数/原型恢复，并增加每条中心化记录的独立CPU近邻门限、权重及传播重算；不声称两次重训图/分数严格一致。算法未改，批处理PID112206复用预检重审，重审已通过，[几何/近邻传播审计](smoke73_geometry_audit.json)最大重构误差7.33e-6；正式225条子进程112801已启动，本轮尚无准确率结论。

## 继续入口

1. 前72轮及确认共33965条服务器及本地SHA通过，第73轮批处理运行中；第65/66轮距离与角度关系未超过强教师，不扩大；第67轮九权重未超过GraphMAE2，第68轮只读中心化增益确认，下一步中心化强度，保持新Prompt与直接教师双重强参照。Cora近邻传播保留局部读出收益，PubMed无增量。训练内一致性第42轮未超过纯传播，第43轮CSLS校正作用很小，第44轮互近邻确认未超过原传播，第45轮监督对比未超过强方法，第46轮表示聚合确认增益，第47轮匹配后k50概率传播更好，第48轮PubMed迁移失败保留原提示，第49轮Poly-1未超过强基线，第50轮九权重未超过GraphMAE，第51轮支持缓存未超过强方法，第52轮非线性推断确认相对原型有效但未超过强线性，第53轮整体CE缩放+k50复核增益成立，第54轮标签平滑相对旧CE有效但未超过新强根，第55轮PubMed缩放迁移未改善强根；Cora/GraphMAE增益已在新20支持集复现，完整配对见41轮补充。Cornell测试仍未用，Chameleon/Actor测试已用。
2. Cora/GraphMAE后层+平滑组合已超过最好单项，GraphMAE2未叠加；PubMed继续保留原型教师gamma3强对照。
3. 继续借鉴其他领域，不因已有工作淘汰。新设计先记录共享参数、提示位置、强对照与固定配置，再运行；同图反复查询须披露。
4. 所有收益保留范围、区间和强基线边界；失败按效果/工程区分。Amazon原始主干输入与独立教师预处理固定，避免弱基线夸大。
5. 目标active。第六轮昂贵CV未超过固定强度，不原样重复。环境/数据/checkpoint复用，只清理本任务已完成预检与可重建缓存。

## 存储执行记录

按用户追加要求，每轮检查磁盘空间；2026-09-23 开始数据盘 75 GB、已用 44 GB、剩余 32 GB。本任务目录初始 240 KB。复用已有环境、数据和权重，不复制大 checkpoint；仅保存最终提示参数、500 节点预测、索引、JSON 结果。失败预检保留简短记录，可重建缓存及时删除；不清理其他研究任务的文件。低于 5 GB 时先清理本任务可重建临时文件，再决定后续批次规模。

已执行第一次清理：删除本任务 smoke/smoke_v2 可重建预检输出（约 187 KB），正式结果、最终参数、预测保留；[清理清单](cleanup_001.json)。第三轮确认后约 80 MB。环境/主干源文件 SHA256 和实际 GPU 信息见 [资源审计](resource_audit.json)。

第二次清理已删除完成的第6/7/8轮3步预检输出和Python缓存，合计约1.43 MB，正式记录全部保留；[清单](cleanup_002.json)。磁盘仍余约31 GiB，尚无容量压力，不清理其他任务资产。

第三次清理移除第10–14轮已完成3步预检与Python缓存约2.4 MB；[清单](cleanup_003.json)。正式预测、随机主干对照权重均保留，任务目录约292 MB，盘余约31 GiB。

第四次清理移除第15/16轮已完成预检约0.5 MB，见[清单](cleanup_004.json)。确认任务运行中目录约311 MB，正式预测保留。[优化步数审计](round16_optimizer_audit.json)核对了Adam100/200与SAM梯度计算数量和主干冻结。

第十八轮27模型×8方法=216条3步预检全部完成，正式1080矩阵已实测PID19535运行。第五次清理移除完成的第17/18轮预检输出，保存完成清单：[cleanup005](cleanup_005.json)。正式预测及原有权重保留，数据盘仍约31 GiB可用。

第十八/十九轮及追加确认完成后，全部9820条正式sealed预测均重新校验本地SHA256。新权重候选只保留同权重增益，跨权重强基线未超过；后续可针对教师误差与提示依赖程度设计，继续带上GraphMAE普通GraphPrompt及GraphMAE2原型教师强对照。


### 第73轮开发完成与随机初始化确认

[225条完整结果](round73_initial/summary.md)、[教师审计](round73_initial_transfer_audit.json)、[独立初值/几何/传播审计](round73_initial_geometry_audit.json)均通过，160条重构最大误差6.68e-6。GraphMAE2语义0.01/0.1传播68.32%/69.40%，零初始化68.36%、新KD68.72%、直接教师70.48%；类别语义初始化没有超过直接教师。置换0.1为69.36%，随机0.1达到70.56%，比零初始化+2.20pp、比新KD+1.84pp，但比Direct仅+0.08pp。GraphMAE随机0.1为69.76%，亦高于零初始化67.92%，但低于Direct。当前不能将收益归因于类别语义。

按准确率目标保留随机配置进入[固定确认](round73_confirm_design.md)：GraphMAE2、新支持111100–111119、确认query后500节点，7提示来源×2读出及Frozen/GraphPrompt/Direct共340条。只确认随机0.1+k50主候选，保留语义/置换0.1及两强根对照，训练源码不改。批处理113631、训练113632已启动；本轮确认尚无结论，不计入完成数。同图和确认query被重复使用，不称独立图验证。

225条开发已同步，本地累计34190条预测及2115个几何/邻接引用SHA通过。[第58次清理](cleanup_058.json)移除49条已完成预检约31.53MB，正式结果保留，盘余24GiB。


### 第73轮首组20支持确认完成

[340条确认结果](round73_confirm/summary.md)、[配对区间](round73_confirm_pairs.json)、[独立几何审计](round73_confirm_geometry_audit.json)均通过。GraphMAE2随机0.1+k50达到69.31%，原强零初始化68.79%，+0.52pp，95%支持采样bootstrap区间[0.13,0.95]，11胜8负1平；新KD68.73%，+0.58[0.20,0.95]。语义68.82%、置换68.75%，随机分别+0.49[0.13,0.82]、+0.56[0.21,0.95]。相比其原型67.84%，传播+1.47[1.02,1.94]。

直接教师69.11%，随机提示+0.20pp区间[-0.37,0.76]，尚不能确认超过教师。当前证据支持小幅初始化收益，不支持类别语义机制，且随机幅度仍由支持类别方向的总扰动RMS确定，不能称完全不依赖支持类别。200条独立几何/传播重构最大误差5.66e-6，140邻接引用通过。同步后本地累计34530条SHA通过。

为验证小增量，[第二组固定复核](round73_confirm2_design.md)已启动：只换新支持112100–112119，其他矩阵/源码/确认query不变，共340条，批处理114470。仍属同图重复查询，第二组尚无结论；结束后执行paired73b.py并同步审计。目标继续active。


### 第73轮第二组20支持复核完成

[340条结果](round73_confirm2/summary.md)、[配对区间](round73_confirm2_pairs.json)、[独立几何审计](round73_confirm2_geometry_audit.json)通过，200条重构最大误差5.96e-6，140邻接引用通过。随机69.76%，零初始化69.62%，+0.14pp [-0.43,0.66]，10胜8负2平；新KD69.65%，差+0.11区间跨0；语义69.77%，随机差-0.01；置换69.49%，随机差+0.27区间跨0。Direct70.16%，随机差-0.40 [-0.94,0.13]。

两组均值方向为正，但第二组没有确认第一组的明确优势，不替换默认零初始化，不继续搜索初始化幅度。保留首组局部增益与第二组不确定性，不能称稳定超过语义/直接教师。正式同步后本地累计34870条SHA通过，本轮无新增预检需清理，原58次清理记录保持。

## 第74轮：符号动量优化器原语

[设计草案](round74_design_draft.md)，借鉴Lion视觉/语言优化研究，实际核对官方PyTorch更新顺序。保持现有768参数及共享GCN，隔离优化器；Lion/Adam/无动量sign匹配三个学习率，保留原强根及随机候选，避免把学习率调小误归因Lion。

[6项算子核验](round74_operator_audit.json)已在ProG环境通过：三学习率×双动量/无动量、20步逐步对照官方实现及独立张量递推，检查动量和缺失梯度不更新。源码生成首尝试因字符串转义语法错误未写入文件，修复后才执行，未产生失效训练结果。真实GCN训练接入、正式矩阵/轨迹审计和准确率实验尚未运行。当前无训练任务，下一步实现完整驱动并预检。


### 第74轮完整矩阵与真实训练预检通过

[固定设计](round74_design.md)、[配置](round74_config.json)。Lion/Adam/无动量sign各0.001/0.003/0.01，同768参数、100步、原LOO CE及token正则；零初始化用于隔离优化器，随机初始化Adam保留为候选参照。两权重、新支持113100–113104，共305条正式，61条3步预检。

61条预检全部通过：[教师/冻结审计](smoke74_transfer_audit.json)、[48条独立CPU前向/读出审计](smoke74_geometry_audit.json)、[36条更新回放](smoke74_update_audit.json)。重构最大8.94e-6，官方Lion/CPU Adam逐步梯度回放误差5.59e-9；Adam0.01恢复原强提示参数核对通过。回放证明更新规则实现一致，不独立证明所有梯度正确。联合源码SHA包含驱动、初始化及优化器辅助模块；所有正式提示保存训练梯度以便回放。

批处理115486、正式训练116166已运行，结束后必须全部审计通过再运行analyze74.py比较90组配对，只有超过同批最强Adam及提示才扩大。正式未完成，不计入34870条累计。盘余24GiB；完成后清理61条预检并保留清单，当前不删。


### 第74轮完成：Lion未超过匹配Adam

[305条完整结果](round74_optimizer/summary.md)、[90组配对](round74_development_pairs.json)。最佳Lion均为0.01：GraphMAE67.84%低于Adam68.44%、随机初始化68.68%；GraphMAE2为68.48%低于Adam68.52%，与Direct68.56%也无优势。本批未中心化强根68.76%还更高。更小学习率总体明显欠佳，无动量sign也未超过Adam。本轮不扩大，不据此推断Lion所有预算/数据集都无效。

[首次恢复诊断](round74_recovery_diagnostic.json)：独立GPU重训的Adam token最大差1.72e-5超初始容差，原型score最大3.04e-6、全部预测一致。初版严格参数恢复审计停下，保留日志；不把重训token称完全相同。修改审计以逐步保存梯度的独立CPU官方实现回放验证更新规则，保持回放容差2e-6，新增原型score 5e-6与预测一致核对，未改训练源码/结果。180条更新回放最大3.58e-7，[更新审计](round74_optimizer_update_audit.json)通过；240条独立几何最大7.03e-6，140邻接引用通过。回放不独立证明梯度正确，但与公式原语、冻结/几何检查共同约束实现。

全部305条同步后本地累计35175条SHA通过。[清理059](cleanup_059.json)删除61条完成预检约40.08MB，正式与失败日志保留。

## 第75轮草案：低置信支持样本降权

[设计草案](round75_design_draft.md)已阅读GCE原论文3.2/4.1节。保持强Prompt共享参数与位置，只替换支持LOO监督项，q=0.3/0.7，匹配平均权重及固定置换权重控制；不假定真实支持标签错误，不做截断删样本。原文干净数据不总获益，作为待检验假设。

[9项损失梯度核验](round75_loss_audit.json)在ProG环境通过，包括q=0恢复CE、两个q的GCE梯度等价p_y^q加权CE，以及均值/置换梯度独立公式核对。当前仅loss75.py原语，完整驱动/矩阵、真实GCN预检尚待实现，无准确率结果，无运行训练任务。目标active。


### 第75轮已接入真实图训练

[固定设计](round75_design.md)、[配置](round75_config.json)。两权重、新支持114100–114104，GCE/平均权重/置换权重×q0.3/0.7，保持同一768参数、Adam0.01、100步及原强中心化提示。新增损失改变训练内部梯度，不是末端预测融合。带原无KD/新KD及随机初始化强参照、Frozen/GraphPrompt和Direct，共245条正式。

预检53条3步包含q0恢复；保存每步支持logits、真实权重、损失、监督logit梯度和token总梯度。独立双精度解析核对损失/梯度、CPU Adam回放最终token，配合冻结/teacher及CPU几何传播审计。循环置换固定seed+750000非零位移，无位置自映射。所有预检审计通过才进入正式。批处理117585已启动，当前尚无准确率结论，不计入35175累计。

第75轮53条预检全部通过，正式245条训练子进程118282已启动。预检[损失/更新审计](smoke75_loss_audit.json)及[独立几何审计](smoke75_geometry_audit.json)已同步，本轮尚无正式结论。


### 第75轮完成：GCE未超过同批最强对照

[245条完整结果](round75_gce/summary.md)、[72组配对](round75_development_pairs.json)。GraphMAE的GCE q0.3/0.7传播均70.56%，比零初始化70.36%仅+0.20pp；均值缩放70.44%、置换70.36–70.44%，随机初始化71.28%更高。GraphMAE2 GCE q0.7为71.20%，零初始化70.88%，但新KD与随机初始化均71.36%；均值71.12%、置换71.16%，没有清晰超越强替代。Direct70.08%虽较低，不能因此忽略更强提示。当前配置不扩大，保留相对零初始化局部正向。

[损失/更新审计](round75_gce_loss_audit.json)120条通过，Adam回放最大3.58e-7；[几何审计](round75_gce_geometry_audit.json)180条最大5.48e-6，110邻接引用通过。245条已同步本地，核验完成后总数35420。[清理060](cleanup_060.json)删除53条预检约34.58MB，正式记录保留，盘余23GiB。

## 第76轮：GraphMAE随机初始化固定确认

[设计](round76_design.md)、[配置](round76_config.json)。此前第73轮20+20确认仅GraphMAE2，GraphMAE的开发增益尚未做新20支持确认；第73/74/75轮GraphMAE随机候选分别较零初始化+1.84/+0.24/+0.92pp，这些属于开发选择证据。

现固定GraphMAE、新支持115100–115119、确认query后500，完全复用第73轮源码和0.1实际扰动匹配规则，随机+k50为主候选，不再调参。保留零初始化、新KD、语义/置换、原型及Direct，共340条。批处理119080已运行，无新增预检；两权重已有第73轮真实图预检及正式100步。第76轮尚无结论，同图重复查询限制不变。


### 第76轮340条确认完成：GraphMAE初始化收益不明确

[完整结果](round76_initial/summary.md)、[20支持配对](round76_initial_pairs.json)、[几何审计](round76_initial_geometry_audit.json)。随机0.1+k50为67.20%，零初始化67.03%，+0.17pp [-0.41,0.80]，10胜10负；新KD66.99%，差+0.21区间跨0。语义66.90%、置换66.66%，随机的对应差异区间仍跨0。Direct68.06%，随机-0.86pp [-1.64,-0.05]。

本轮未确认开发阶段的较大提升，不替换GraphMAE默认零初始化，不继续该权重初始化强度搜索。加上GraphMAE2两组结果，记录为首组局部正向、后续未稳定确认，不能称通用初始化收益。200条独立几何最大6.85e-6，140邻接引用及所有340条预测通过服务器审计。

第75轮本地累计35420条已完成SHA核验。第76轮服务器通过后累计35760条，正式文件正在同步，当前不提前称本地35760已核验；同步完成后运行refresh_local_audit.py。第76轮完全复用源码，没有新增预检需清理，仍为60次清理。当前无训练任务，持续目标active。

第76轮文件同步及本地刷新已完成，累计35760条预测、2115个几何/邻接引用SHA核验通过。


## 第77轮：加性余弦间隔（预检启动）

[固定设计](round77_design.md)、[配置](round77_config.json)。新读CosFace第3.1节公式4，CVF直读403后使用arXiv原文。借鉴人脸识别训练间隔，改用图支持LOO原型，保持教师门控的768提示参数、冻结GCN及查询读出，不训练分类权重。

支持正确类logit减margin/0.2，固定margin0.1/0.3；查询和teacher不减间隔。解析梯度等于普通CE乘a=1/(1-p+exp(-margin/0.2)*p)，该a随正确类p增大，不把它说成专门强调困难样本。匹配detach均值缩放和固定无自映射置换控制，保留原无KD/新KD和随机强参照及Direct。

[27项原语核验](round77_loss_audit.json)通过，含零margin恢复、常规/极端logits及三模式解析梯度。53条真实图3步预检启动，批处理120249；通过预测、冻结teacher、独立CPU几何/传播、逐步损失/梯度及Adam回放审计后自动进入245条正式，新支持116100–116104。训练源码与init73/loss77联合SHA，真实前向和记录流程复用第75轮，不改变已封存旧实验。本轮尚无正式结论。

第77轮53条预检已全部通过，审计JSON同步完成，245条正式训练已启动；正式结果尚未完成。


### 第77轮245条完成：间隔未超过最强替代

[完整结果](round77_margin/summary.md)、[72组配对](round77_development_pairs.json)。GraphMAE margin0.3为67.96%，零初始化67.80%、均值缩放/置换67.92%，额外仅0.04pp；margin0.1与均值控制均67.84%。GraphMAE2 margin0.3为69.56%，零初始化69.40%、均值69.36%、置换69.32%，但新KD69.64%及Direct70.28%更高。停止该配置，不以弱参照的小增量扩大。

[损失/更新审计](round77_margin_loss_audit.json)120条通过，回放误差3.58e-7；[独立几何](round77_margin_geometry_audit.json)180条最大5.78e-6，110邻接引用通过。正式文件已同步，本地累计待刷新为36005。[清理061](cleanup_061.json)删除53条完成预检34.58MB，所有正式记录保留。

## 第78轮草案：按支持类别进行梯度投影

[设计草案](round78_design_draft.md)。检索协方差分类发现第38轮已有Simple CNAPS类读出确认失败，不原样重复。新读PCGrad第2.2/2.3节及算法1，拟将每类支持CE作为伪任务，在共享768token上投影冲突梯度；同GCN、同早层位置，无终端融合。负内积本身不证明优化有害，论文的完整条件在此未验证。

[4项投影算子审计](round78_operator_audit.json)通过，独立NumPy递推、无冲突恢复、零梯度和相反梯度均核对。真实GCN类梯度均值恢复、随机分组/范数控制、完整训练矩阵与轨迹审计尚未实现，不计作准确率实验。当前无训练任务，继续active。

第77轮本地刷新完成，累计36005条封存预测、2115个几何/邻接引用SHA核验通过。


### 第78轮真实图预检通过，正式已启动

[固定设计](round78_design.md)、[矩阵](round78_config.json)：类别梯度均值、类别PCGrad、随机均衡分组PCGrad、范数匹配均值四模式，原型/k50双读出，原无KD/新KD及随机提示强对照、Direct，共205条。新支持117100–117104，41条3步预检全部通过。批处理122076，正式运行中。

每步6次组梯度加1次直接CE梯度用于检查均值恢复，正则梯度单独加入；投影使用CPU双精度，参考梯度不修改。每步记录组梯度/顺序/实际更新，独立NumPy投影及Adam回放通过[投影审计](smoke78_projection_audit.json)，另有[几何审计](smoke78_geometry_audit.json)及[教师/冻结审计](smoke78_transfer_audit.json)。计算并非与原1次反传等价，不把潜在提升称免费。当前没有正式准确率结论，36005累计不变。

### 用户新增项目管理要求

已补齐[本项目README](README.md)与[持续推进目录入口](../README.md)。本项目本就在“持续推进”，未重复搬迁或复制checkpoint。README列明Cora已确认配方、CiteSeer强Prompt与Direct边界，失败与待验证轮次不单独标成有效项目；新独立项目进入该目录时须附证据、限制、复核方式和继续入口。所有README相对链接已验证存在。


### 第78轮205条完成：梯度冲突较少且无明确增量

[完整结果](round78_project/summary.md)：GraphMAE类别PCGrad、梯度均值、范数控制均68.76%，随机分组68.84%；GraphMAE2四者均69.56%，随机初始化和Direct均69.64%。类别冲突处理未提高同批强根，不扩大。

[冲突诊断](round78_conflict_diagnostic.json)：GraphMAE五支持中四个全部100步无投影，另一个13步触发26次；GraphMAE2一组无投影，其余分别9/22/18/23步触发，最小类梯度余弦约-0.076。缺少严重冲突与低收益一致，但这不是因果证明，也不声称所有梯度严格相同。随机分组0.08pp变化不作为类别机制收益。

[独立投影回放](round78_project_projection_audit.json)80条通过，最大3.58e-7；[几何审计](round78_project_geometry_audit.json)140条最大6.35e-6，90邻接引用通过。全部同步，本地累计36210条及2115个几何/邻接引用SHA核验通过。[清理062](cleanup_062.json)删除41预检26.73MB，正式轨迹保留。

## 第79轮草案：门控教师温度

[设计草案](round79_design_draft.md)，检索DINO并核对官方DINOLoss教师softmax温度；只借鉴锐化/平滑，不移植EMA、多视图和logit中心化。与第57轮训练温度位置不同，本轮拟只改变共享早层提示的教师混合门控，不变查询/支持CE温度。

温度0.1/0.4，带原幅度、门控整体Frobenius范数匹配、原门控仅幅度匹配三模式。原始教师score已有除0.2，转换为score*0.2/t；教师argmax不变，不能把直接教师变化误当提示收益。[9项原语检查](round79_gate_audit.json)通过，包括恢复0.2、概率和/argmax不变、稀疏/稠密传播一致及范数匹配。未实现真实图训练驱动与逐节点独立门控审计，尚无准确率结果。当前无训练任务。


### 第79轮已接入真实门控训练

[固定设计](round79_design.md)、[配置](round79_config.json)。温度0.1/0.4，各带raw/matched/amplitude三种门控，768提示参数、零初始化、100步Adam0.01，原强训练/读出保持。正式245条支持118100–118104，53条预检含原温度0.2恢复。

已保存门控与概率分别SHA、门控范数比、实际最终隐藏扰动RMS；CPU独立重建实际门控，再核对冻结GCN前向、原型及近邻传播。amplitude分支概率SHA仍对应原概率，避免元数据假称锐化；范数匹配仅矩阵Frobenius，不称任意token扰动严格相等。批处理123933启动，全部预检通过才正式；本轮暂无准确率结论。

第79轮53条预检全部通过，[CPU独立门控/几何审计](smoke79_geometry_audit.json)与[教师/冻结审计](smoke79_transfer_audit.json)已同步；245条正式训练子进程124557已运行，尚无正式结论。


### 第79轮开发通过，进入固定20支持确认

[245条结果](round79_gate/summary.md)、[72组配对](round79_development_pairs.json)、[独立门控/几何审计](round79_gate_geometry_audit.json)：GraphMAE raw温度0.1传播69.88%，原强提示68.56%、新KD68.20%、随机68.68%、Direct69.44%。但原门控仅幅度调整69.76%，同范数锐化69.12%；开发不能认定锐化比放大门控更有效。GraphMAE2 raw69.76%、原69.08%、幅度/匹配均69.56%。温度0.4原幅度明显变差（GraphMAE59.48%、GraphMAE2 64.76%），不再搜索。

180条独立重构最大6.73e-6，110邻接引用及teacher/门控SHA通过。文件已同步，累计本地待刷新36455。[清理063](cleanup_063.json)删除53完成预检38.59MB，正式保留，盘余23GiB。

[固定确认设计](round79_confirm_design.md)：主配置GraphMAE raw0.1+k50、GraphMAE2次要参照；新支持119100–119119，confirm500，两权重各18方法与Direct合计740条，仍保留原门控幅度对照与范数匹配锐化。训练源码不改，不重复预检。批处理125295已运行，确认结果尚未完成；同图重复query限制持续保留。paired79.py比较raw，paired79_amplitude.py单独检查幅度收益，不能只报告有利分支。

第79轮开发本地刷新完成：36455条正式预测与2115个几何/邻接引用SHA核验通过，确认740条继续运行。


### 第79轮首组740条确认完成：保留门控调整候选

[确认结果](round79_confirm/summary.md)、[raw配对](round79_confirm_pairs.json)、[幅度配对](round79_amplitude_pairs.json)。主配置GraphMAE raw温度0.1+k50为69.84%，零初始化强根68.74%，+1.10pp [0.55,1.63]，16胜3负1平；对新KD+1.30 [0.64,2.03]。幅度控制69.69%，对原+0.95 [0.38,1.46]；raw对幅度+0.15 [-0.14,0.45]。故可保留准确率改进候选，但不能断言锐化优于幅度。Direct69.49%，raw+0.35 [-0.19,0.94]，尚未明确超过。

GraphMAE2次要raw69.40%，原68.92%，+0.48 [0.17,0.77]；对幅度68.99%高0.41 [0.23,0.58]，但对新KD+0.36区间跨0，Direct差-0.09区间跨0。同范数raw匹配GraphMAE69.51%，比raw低0.33 [0.03,0.64]。

[机制诊断](round79_confirm_mechanism.json)显示门控范数比平均1.736；GraphMAE raw/amplitude/matched的最终隐藏扰动RMS约0.0058/0.0059/0.0038，GraphMAE2约0.0083/0.0082/0.0057。仅描述实际运行，不能把近似扰动当相同机制。480条独立几何重构最大8.29e-6，320邻接引用通过；正式已同步，本地待刷新37195。

按用户项目管理要求，新增[候选方案说明](候选方案_教师门控锐化.md)并加入项目及父目录README，文件仍留在现有“持续推进”研究项目，不重复复制checkpoint。候选有首组支持证据但尚无稳定最优结论。

[第二组固定复核](round79_confirm2_design.md)已启动，批处理126515，新支持120100–120119，全部源码/配置/query不变，仍740条；分别报告两组，不用合并掩盖潜在失败。盘余22GiB，无新增预检需清理，当前清理63次。

第79轮首组确认本地刷新完成，累计37195条预测与2115个几何/邻接引用SHA核验通过；新增候选和README链接均已核验。


## 第79轮第二组确认完成：保留CiteSeer温度0.1配方

120100–120119共740条全部通过服务器和本地审计；累计37935条。480条门控/几何独立重建最大误差7.5102e-6，20条Direct与320邻接引用通过。无新增预检清理，清理编号仍063。

GraphMAE raw温度0.1+k50为69.18%，原强提示68.04%，+1.14pp [0.58,1.74]，15胜3负2平；对原KD68.11%为+1.07 [0.58,1.63]；对范数匹配68.79%为+0.39 [0.06,0.74]；对幅度控制68.77%为+0.41 [-0.02,0.88]。Direct69.15%，差+0.03 [-0.29,0.35]。GraphMAE2次要参照69.18%对原68.56%，+0.62 [0.15,1.09]，14胜3负3平；对幅度68.81%为+0.37 [-0.01,0.75]；Direct差+0.03 [-0.44,0.59]。

与首组分别报告：两组均支持相对原Prompt的增益，未稳定超过Direct或幅度控制；保留准确率配方，不把收益全部归为锐化方向。区间只反映同图支持变化，查询反复使用。证据：round79_confirm2_pairs.json、round79_confirm2_amplitude_pairs.json、round79_confirm2_geometry_audit.json。后续固定迁移PubMed，不继续选择温度；设计见round80_design_draft.md。


## 第80轮固定PubMed迁移封存

实现与审核设计见round80_design.md。70条正式矩阵与16条3步预检，门控改动与KD目标分离；VAT和最终原型保持原方案。待预检，不预报增益。


## 第80轮：PubMed固定迁移停止

70条正式全部完成，GraphMAE40/GraphMAE2 30（Direct仅存一份），45个提示的CPU教师/门控/KD目标/最终几何重建最大误差2.8611e-6。GraphMAE原VAT根73.96%，raw锐化73.44%（-0.52pp）；范数匹配74.24%，幅度控制74.20%，无VAT74.20%，Direct68.00%。GraphMAE2原根71.96%，raw71.92%，匹配71.88%，幅度71.28%。不扩大raw迁移；匹配对最强无VAT参照仅+0.04，暂不作为强增量。CiteSeer局部收益保留。证据round80_development_pairs.json、round80_gate_GraphMAE_gate_audit.json及GraphMAE2对应文件。

两权重3步预检原温度恢复预测完全相同，最大分数误差1.4305e-6/9.5367e-7。cleanup_064仅删除16条已完成预检，共6,342,969字节；正式记录保留。下一轮Sparsemax门控草案见round81_design_draft.md，仅设计未运行。


## 第81轮Sparsemax门控封存

依据ICML2016概率单纯形投影，固定2score映射，145条开发与29条预检。必须比较第79轮softmax温度0.1强根。实现、独立审计与边界见round81_design.md；尚未预检，不预报有效。


## 第81轮开发通过，固定范数匹配候选进入确认

145条开发已全审计：100条几何重建最大误差1.2458e-5，5条Direct、60邻接引用通过；103组原语与独立二分误差1.78e-15。GraphMAE matched68.00对当前强根67.44，+0.56；幅度67.56、raw67.80。GraphMAE2 matched68.12对当前强根67.32，+0.80；幅度67.20、raw68.00。Direct68.28，仍未胜Direct。

概率零比例均值71.57%，最终门控零比例0；候选门控范数是softmax.1的1.548倍（匹配版本缩回），不称稀疏图传播。GraphMAE raw/matched/amplitude实际隐藏扰动RMS约0.00832/0.00601/0.00768；GraphMAE2约0.01187/0.00890/0.01067。仅诊断，不替代query增益。

固定matched为主候选、GraphMAE主权重，20新支持123100–123119共580条确认已启动，设计round81_confirm_design.md；不按确认结果重新挑主方法。cleanup_065清理29条smoke81，共19,710,992字节，正式数据保留。


## 第81轮首组确认：主权重失败，次要权重有待复核增益

580条全部服务器审计通过：400几何重建最大误差1.4425e-5，20条Direct和240邻接引用通过。GraphMAE matched67.00% vs当前强根67.28%，-0.28pp [-0.69,0.15]，7胜12负1平；对幅度67.09亦-0.09，主检验未通过。虽胜旧温度0.2根66.25%，不能绕过最新强对照。

GraphMAE2 matched67.04% vs当前强根66.50%，+0.54 [0.12,0.98]，13胜2负5平；对幅度66.42为+0.62 [0.21,1.03]；raw66.92，与matched差+0.12，区间下限约0，勿夸大。Direct66.81，与matched差+0.23 [-0.15,0.59]。区间仍反映同图支持采样，查询重复使用。

以新20支持124100–124119单独固定复核GraphMAE2，300条已启动，详见round81_confirm2_design.md。不将次要发现事后改写为原主检验通过。


## 第81轮第二组复核未复现：停止稀疏映射替换

GraphMAE2单权重300条均服务器审计通过，200几何最大误差1.0252e-5。matched69.29%对当前softmax.1 69.29%，差约0.00pp [-0.37,0.34]，11胜7负2平；对幅度69.33差-0.04 [-0.37,0.30]；raw69.30；Direct69.19，差+0.10 [-0.26,0.46]。首组次要收益未复现，停止，不替换第79轮配方。仅优于更旧方案不能覆盖该失败。

首组580及第二组300分别保留，不合并掩盖第二组无增益；开发、确认和复核全部沿用封存源。无新增预检，清理仍065。下一轮已检索并核对Shampoo矩阵算法，草案round82_design_draft.md，仅设计未运行。


## 第82轮预条件优化封存

Shampoo、对角L/R、逐元素AdaGrad及Adam匹配学习率.01/.03。41预检、205正式，固定第79轮softmax.1强根。源码/审计和实现约定见round82_design.md。尚未启动，不计有效方案。


## 第82轮开发完成：Shampoo未胜，AdaGrad参照进入确认

205条全部服务器四审计通过：180几何重建最大误差7.9275e-6，80个训练更新重放最终token误差1.1921e-6；9原语误差6.94e-18，Adam.01恢复两权重预测相同、分数差9.54e-7。

GraphMAE原Adam.01+k50 70.64%，Shampoo.01/.03为24.92/52.48，对角.03 62.44，AdaGrad.03 70.36，Adam.03 70.28。GraphMAE2原70.00%，Shampoo.01/.03为28.04/60.44，对角.03 69.28，AdaGrad.03 70.96，Adam.03 69.72；Direct70.24。所测Shampoo配置停止，但低步长可能更新不足，不作全优化器无效结论。

以准确率为目标，固定参照中的GraphMAE2 AdaGrad.03为候选，GraphMAE为次要，新20支持126100–126119共500条运行中。设计round82_confirm_design.md，原代码不改。cleanup_066删除41条smoke共31,694,706字节，保留正式记录和审计；上一轮300条仍同一rsync进程传输，未因耗时重启。


## 第82轮确认失败，停止AdaGrad替换

500条服务器审计通过：400几何最大误差8.5235e-6，160个更新重放最大token误差1.7882e-6，20Direct/200邻接引用通过。GraphMAE2 AdaGrad.03 68.28%对原Adam.01 68.69%，-0.41pp [-0.90,0.00]，5胜12负3平；对Adam.03 68.42为-0.14，Direct68.55差-0.27。GraphMAE 68.06对原69.26，-1.20 [-1.99,-0.36]，5胜14负1平。开发单次增益未复现，停止替换。

Shampoo步长诊断发现约17–65倍更新范数差，支持训练不足为待测解释，见round82_step_diagnostic.json；不直接据此宣称方向有效。首次临时诊断误读sealed顶层导致KeyError，已修正为prompt字段并保存可重跑diagnose82_steps.py，未修改训练源码或结果。第83轮方向/范数分离草案见round83_design_draft.md，尚未实现。第81轮最后300条同步已完成，本地39030核验通过；第82轮205条同步也完成，等待本地刷新，再同步500确认。


## 第83轮实现封存并启动

145正式与29条三步预检，四种原语已通过。用候选自身梯度生成Adam影子范数，比较Shampoo/对角/原梯度方向，保留真实Adam和未匹配Shampoo；零方向明确no-op。代码及审计约定见round83_design.md。训练启动PID136613、round83.log；四审计通过才自动正式。第82轮确认500已同步完成，待本地总核验39735；清理编号仍066。


## 第83轮开发审计通过，580条固定确认启动

145条开发完成：120几何重建最大误差9.656e-6，50更新重放参数误差3.577e-7、范数误差1.422e-14，5Direct/60邻接引用通过。GraphMAE graftSh68.96%对Adam69.24%，不增；GraphMAE2 69.48%对68.84%，+0.64pp，对角与普通梯度均69.40%、Direct69.48%。恢复Adam与根均值完全相同。

固定GraphMAE2 graftSh为主、GraphMAE次要，新支持128100–128119共580条，不改源码。不能把相近的简单方向收益归为Shampoo独有贡献，设计见round83_confirm_design.md。cleanup_067只删除29条已完成预检，正式证据保留。


## 第83轮首组确认结果与复核准备

580预测已完成，480几何审计最大误差9.7752e-6、Direct20/邻接240通过，最终200训练更新重放仍运行，未称全部审计完成。GraphMAE2 graftSh70.53对Adam70.12，+0.41pp [0.13,0.73]，14胜5负1平；对角70.47、普通梯度70.34，均未可靠胜出；Direct70.24，差+0.29 [-0.02,0.61]。GraphMAE70.20对70.18基本持平。

第二组GraphMAE2新支持129100–129119、300条设计已封存，round83_confirm2_design.md；需首组最终审计通过才启动。当前尚未启动第二组。信息检索IDF加权教师的下一轮草案已写入round84_design_draft.md，尚未实现。开发145本地核验完成，累计39880；确认580同步会话84407仍运行，必须等待退出后刷新。


首组最终更新审计现已通过：200记录，参数误差4.7684e-7，范数误差1.4211e-14。服务器累计40460；第二组300固定复核已启动round83_confirm2.log，训练源未修改。首组同步84407仍在运行，本地39880。


## 第83轮第二组确认通过：保留GraphMAE2步长匹配配方

300条服务器全部审计通过，240几何最大误差7.6294e-6，100更新重放参数误差4.7684e-7、范数误差1.0659e-14；20Direct/120邻接引用通过。graftSh69.52%对Adam69.18%，+0.34pp [0.04,0.63]，14胜4负2平。对角69.33、普通梯度69.36，与候选差区间均跨0；Direct69.65，差-0.13 [-0.53,0.24]。

两组相对Adam均有小幅收益，按准确率目标保留限定配方，不主张矩阵方向独有或胜Direct。文件保留在既有持续推进项目，新增保留方案_步长匹配优化.md，并更新父目录与本项目README；没有复制大权重。服务器累计40760，本地40460已核验，最后300条rsync1655仍运行，等待退出后刷新。第84轮IDF原语通过，不计query结果。


第83轮第二组同步已结束，独立本地总核验返回40760 2115，累计40760条正式本地及服务器均通过。当前无训练运行；第84轮已完成特征原语，主训练及审计驱动待实现。


## 第84轮IDF教师实现封存

34条三步预检、150正式。候选只改teacher输入，GraphMAE2沿用第83轮保留优化器，GraphMAE用Adam；原主干输入不动。raw/匹配范数/幅度/列置换与两种Direct齐备，设计round84_design.md。训练及审计未完成前不报有效。


## 第84轮IDF停止

150条服务器四审计通过：120几何最大误差5.8413e-6，50更新重放参数误差7.1526e-7、范数误差1.0659e-14。GraphMAE IDF raw68.72%对强Adam70.72%，-2.00pp；匹配范数68.56、仅幅度70.76、列置换70.52。GraphMAE2 raw68.72对强graft70.60%，-1.88；匹配68.52、幅度70.64、置换70.64。Direct原70.80、IDF68.52，降低2.28pp。

IDF语义加权本轮未胜幅度/置换控制，停止扩大，不把幅度对原仅0.04pp视作可靠新方向；不据此声称所有文档频率权重无效。cleanup_068删除34条已完成预检26,375,705字节，正式保留。服务器累计40910，本地40760，正式同步31564待结束后刷新。下一轮Bagging教师草案round85_design_draft.md，仅设计未实现。


第84轮同步及本地核验完成：40910 2115，累计40910正式结果。当前无训练运行，下一轮Bagging教师待实现；目标保持active。


## 第85轮Bagging教师实现封存

分层8bootstrap仅取原支持，固定无标签池排除原全部支持。原语恢复最大误差1.1921e-7通过。34预检、150正式，新支持131100–131104；保留新强根、单次采样、范数/幅度和Direct。设计round85_design.md，尚未结果，不计有效方案。

### 第85轮完成：Bagging教师停止扩大

150条正式、seeds131100..104全部通过独立审计并完成本地同步；累计41060条，local refresh输出41060 2115。8次类内bootstrap概率均值在GraphMAE为68.32%，当前Adam根71.16%；GraphMAE2为70.88%，当前grafted根71.52%。范数匹配69.28/70.96，纯幅度71.12/71.60。单次bootstrap64.60/65.12；Bagging修复一部分单次重采样损失，但仍不及完整支持教师。Direct原71.24→Bagging69.16。GraphMAE2幅度仅+.08pp不作为新候选。

审计：120几何最大误差5.3346e-6；50更新最大token误差8.34465e-7、范数误差1.0658e-14；10 Direct和60邻居引用通过。cleanup_069仅删34条已完成预检24,849,483字节，正式保留。

下一轮固定平方根教师，设计见round86_design.md，借用视觉小样本分布校准中的幂变换组件；尚无正式结果。

第86轮实现、原语检查及源清单封存完成，PID144080，round86.log。原语sqrt/square/identity与NumPy误差均0。预检34→四项审计→正式150串行执行；当前正式结果尚未产生，不提前计入总数。

### 第86轮完成：平方根教师停止扩大

150条正式、seeds132100..104已通过全部审计、本地同步与SHA核验，累计41210条，refresh输出41210 2115。GraphMAE平方根64.24 vs当前Adam68.88；GraphMAE2平方根64.32 vs当前grafted69.12。等范数64.20/64.20，纯幅度69.04/69.08，平方66.32/66.84。Direct原69.24→平方根63.64。平方根及平方均不保留；幅度在GraphMAE仅+.16pp不作为可靠新方案。

120几何最大误差6.6459e-6，50更新token误差7.1526e-7、范数1.0658e-14；10 Direct、60邻居引用通过。cleanup_070只删34预检24,806,116字节，正式证据保留。下一轮梯度中心化草案round87_design_draft.md已写，尚未实现/运行；当前无训练运行，目标保持active。

### 第87轮启动：提示梯度中心化

作者实现gc_loc=True对应矩估计前中心化，本轮固定此位置。一个冻结GCN和6×128token、原门控及最终读出不变。隐藏轴gch、类别轴gcc、范数控制gcn、恒等gci预检；125正式、29预检。原语NumPy误差≤4.44e-16，首步隐藏轴相对变化GraphMAE .3733、GraphMAE2 .1396，类别轴约3e-8，未使用查询标签。PID146005，round87.log；先预检四审计通过再正式。设计round87_design.md，当前未有正式结论。

### 第87轮正式完成：梯度中心化停止扩大

125条开发全部通过审计。GraphMAE隐藏轴69.52 vs当前Adam69.80（−.28pp）；GraphMAE2隐藏轴68.88 vs当前grafted69.12（−.24pp）。类别轴69.80/69.20，范数控制69.80/69.12，Direct68.68。仅类别轴GraphMAE2+.08pp不视为可靠新方案。未扩大确认。

100几何误差最大1.37091e-5；40更新重放token误差7.15256e-7、范数1.0658e-14；5 Direct、50邻居引用通过。cleanup_071仅删29预检21,384,748字节。正式已同步，本地refresh61170待完成再改总数，服务器累计41335。

诊断round87_gradient_diagnostic.json显示：类别轴在首步近似恒等，但100步平均梯度相对变化为GraphMAE .0441、GraphMAE2 .2187，不能据首步把整个方法视为无操作。隐藏轴平均变化.1168/.0977，未带来准确率提升。下一轮更新方向中心化草案round88_design_draft.md，当前未实现、未运行。

第87轮本地refresh完成：41335 2115，总数已更新。当前无训练运行，第88轮仅草案，持续目标active。

### 第88轮启动：更新方向中心化

实现与设计round88_design.md。保持原梯度/优化器状态，处理预条件后的方向；dcr投影、dcm匹配原方向范数、dca纯幅度、dci预检恢复。原语NumPy误差≤2.22e-16；20步恒等恢复Adam2.98e-8、grafted0，包含零梯度动量情况。源已封存，PID147942，先29预检及四审计，再125正式seeds134100..104。暂无正式结论。

第88轮开发125条通过全审计。GraphMAE dcr65.92、dcm66.12、dca65.60、Adam65.68、gsh67.00；GraphMAE2 dcr66.60、dcm66.68、dca66.68、gsh66.68。Direct67.32。仅GraphMAE dcm对当前配方+.44pp进入20支持确认，尚低于同批gsh，不能称最优。100几何误差5.722e-6；40更新token误差3.5763e-7、范数1.0658e-14，全部通过。开发正式同步完成，本地refresh73016待核验；服务器41460。

确认设计round88_confirm_design.md，固定dcm，260条GraphMAE、seeds135100..119、确认查询后500；PID149477，round88_confirm.log。原Adam、gsh、幅度和Direct继续保留。当前无确认结论。

### 第88轮确认完成：更新方向中心化不保留

20支持GraphMAE，dcm69.29 vsAdam69.51，−.22pp、95%支持级bootstrap[-.42,-.01]，6胜13负1平；对gsh69.70为−.41[-.67,-.15]，对纯幅度69.48为−.19[-.37,-.01]。dcr69.26、Direct69.56。开发+.44未复现，不展开第二确认，不新增保留项目。仍是同图重复查询的支持采样证据。

260条确认全审计通过：200几何最大误差9.656e-6；80更新重放token误差4.7684e-7，范数1.4211e-14；20Direct、100邻居引用通过。正式已同步，最终本地refresh待完成；服务器累计41720。cleanup_072删除29预检21,792,893字节，正式保留。

下一轮AdaBelief残差矩方向草案round89_design_draft.md已检索论文并写入，未实现未运行；先核对epsilon与bias correction位置。已有保留配方不变。

本地最终核验完成：41720 2115。累计41720正式结果，README/state已更新，当前无训练运行，目标active。

### 第89轮启动：AdaBelief提示更新

作者实现确认epsilon累加残差矩+分母，本轮固定1e-8（视觉参数）、不rectify/AMSGrad/额外WD。固定门控与读出，abr原方向/abm范数匹配/aba纯幅度/abi恢复；一个冻结GCN和768提示参数。原语方向与NumPy误差6.66e-16，identity恢复Adam2.98e-8、grafted0。PID150604，round89.log，29预检四审计后125正式136100..104。暂无正式结论，详见round89_design.md。

第89轮125开发全部通过审计：100几何误差8.8215e-6；40更新token4.7684e-7、范数1.0658e-14；5Direct、50邻居引用通过。GraphMAE abr69.92、abm70.76、aba70.08 vsAdam70.56/gsh70.84；GraphMAE2 abr71.00、abm70.84、aba70.76 vsgsh70.64。Direct71.12。固定主候选GraphMAE2 abr、次候选GraphMAE abm进入确认，尚未保留。

确认500条，137100..119，phase confirm后500查询，PID152127、round89_confirm.log。开发正式已同步，本地refresh待完成；服务器41845。cleanup073仅删29预检，保留正式和审计。

### 第89轮确认完成：AdaBelief不保留

20支持137100..119，主候选GraphMAE2 abr68.68 vs当前gsh68.82，−.14pp，95%支持级区间[-.48,.20]，5胜9负6平；次候选GraphMAE abm68.46 vsAdam68.64，−.18[-.47,.07]，8胜10负2平。两者对Direct68.99也无优势。GraphMAE abm对gsh68.85为−.39[-.63,-.14]，对幅度68.80为−.34[-.60,-.08]；GraphMAE2对幅度68.75为−.07、区间跨0。不扩大第二确认，限定本配置，不泛化为所有AdaBelief无效。

500确认全部审计通过：400几何最大9.2983e-6；160更新重放token3.5763e-7、范数1.3434e-14；20Direct、200邻居引用通过。正式同步已结束，最终本地refresh54398待完成；服务器累计42345。cleanup073删29预检22,092,710字节，正式保留。

下一轮来源NIPS2016结构化稀疏论文，提示通道组正则草案round90_design_draft.md，带随机组和等范数L2控制；平滑正则不声明精确稀疏/加速。当前仅设计无训练，原有效配方保持。

本地最终refresh完成42345 2115。累计42345正式结果已核验，README/state更新；当前无训练，持续目标active。

### 第90轮启动：提示通道组正则

实现round90_design.md：.1平滑组范数、固定随机分组、额外L2梯度范数匹配、零系数恢复；一个冻结GCN和768提示参数，原损失/门控/读出保留。原语解析梯度误差1.39e-17，有限差分/零点通过。5步无查询诊断正则对基础梯度：GraphMAE更新后约.088—.141，GraphMAE2约.029—.049。PID153457，29预检全部审计后125正式seeds138100..104，暂无准确率结论。

第90轮125开发全审计通过：100几何6.2585e-6，40更新token5.9605e-7/范数1.0658e-14；分组、额外解析梯度、逐步参数、5Direct和50邻居引用均通过。GraphMAE通道70.24/随机70.32/L2匹配70.12 vsAdam70.08/gsh70.12；GraphMAE2通道70.12/随机70.28/L2匹配70.36 vsgsh70.32，Direct70.44。仅GraphMAE展开确认，按用户准确率目标固定随机组主候选，不预设语义通道优越。

确认设计round90_confirm_design.md，260条139100..119，PID154978，round90_confirm.log。开发已同步，refresh待完成、服务器42470；cleanup074仅删29预检。暂无确认结论。

### 第90轮确认完成：组正则不保留

20支持GraphMAE随机组68.91 vs当前Adam69.10，−.19pp，95%支持级区间[-.44,.06]，6胜13负1平；对gsh68.99为−.08、区间跨0。通道68.92、L2匹配68.89亦无明确收益。随机组对Direct68.51为+.40[.02,.80]，但仍低于当前提示，不据此升级候选。停止扩大，限定此平滑强度/设置。

260确认全审计通过：200几何9.7752e-6，80更新重放token5.9605e-7/范数1.0658e-14；解析正则梯度及20Direct、100邻居引用通过。正式同步完成，最终本地refresh30512待核验，服务器42730；cleanup074仅删29预检21,755,406字节。下一轮CLR门控草案round91_design_draft.md，已记录公式等价与原论文全文未读取的来源边界；尚无代码或训练。

本地refresh完成42730 2115，累计42730条正式核验。README/state已更新，当前无训练，目标active。

### 第91轮启动：CLR教师门控

实现与设计round91_design.md，训练以raw_scores*2减类别均值计算CLR，CPU独立以log_softmax计算。原概率门控.1保持；clr原门控、clm等范数、cla纯幅度、cli恢复。原语移位不变/CLR等价/零分支/等范数/恢复通过，最大误差1.11e-15。PID156193，round91.log，先29预检四审计再125正式140100..104。单GCN单token，暂无准确率结论。

第91轮125开发全审计通过：100几何6.1095e-6，40更新token5.9605e-7/范数5.5178e-14；5Direct、50邻居引用通过。GraphMAE2等范数CLR70.28 vs当前gsh68.92（+1.36pp），原CLR68.28、纯幅度68.16，Direct69.32；GraphMAE等范数62.96 vsAdam69.56失败。固定GraphMAE2 clm进入20支持确认，失败权重不扩大。

确认设计round91_confirm_design.md，260条141100..119，PID157708、round91_confirm.log。开发正式同步已完成，refresh待核验、服务器42855；cleanup075只删29预检。暂无确认结论。

### 第91轮确认完成：CLR完整替换暂不保留

20支持GraphMAE2 clm69.03 vs当前gsh68.94，+.09pp，95%支持级区间[-.38,.58]，9胜10负1平；未达到稳定当前强根增益。对旧Adam68.17为+.86[.39,1.43]、16胜3负1平；对纯幅度67.93为+1.10[.38,2.01]；对原CLR68.55与Direct68.75区间跨0。保留这些局部阳性，但不替换gsh、不因击败旧Adam就称新最强。

260确认全审计通过：200几何7.391e-6，80更新token3.5763e-7/范数7.2164e-14；20Direct、100邻居引用通过。正式同步完成，最终本地refresh90017待核验，服务器43115。cleanup075只删29预检22,474,989字节。下一轮CLIP-Adapter残差混合原则迁移到概率/CLR单门控，round92_design_draft.md已写，仅设计未实现。

本地refresh完成43115 2115，累计43115正式核验，README/state更新；当前无训练，目标active。

### 第92轮启动：概率/CLR门控残差混合

实现round92_design.md：固定alpha=.5，一个门控乘同一token矩阵；原混合/等范数/纯幅度/节点置换控制；alpha0/1仅预检。原语端点精确恢复、混合误差1.11e-16。GraphMAE2当前gsh及完整CLR和Adam均保留。PID158934，21预检四审计后85正式142100..104，暂无准确率结论。


### 第92轮完成：混合门控停止扩大

85正式全审计通过：70几何最大6.1989e-6，30更新重放token3.5763e-7/范数1.0658e-14，5Direct/35邻居引用通过。等范数混合69.80%、原混合69.84%，当前gsh69.96%，完整CLR70.12%、幅度69.96%、置换65.80%，Direct70.56%。主候选未超过当前强根，不扩大；节点置换下降只说明对应关系敏感，不能证明混合收益。

正式同步及refresh完成43200 2115，本地/服务器累计43200。cleanup076仅删除21条smoke92，17,369,920字节，正式全部保留。第93轮检查DEC逆频率目标分布，明确与原型迭代中的Sinkhorn及单纯锐化区分。


### 第93轮启动：DEC逆频率门控

设计round93_design.md，dem等范数主候选、der原式、dea纯幅度、def仅频率、des仅平方、dei恢复预检。原语支持排除/概率重算/温度等价/零分支/范数全部通过，最大4.44e-16。PID160838、round93.log，37预检四审计后165正式143100..104。源/config SHA已封存；暂无正式准确率结论，16GiB空闲。


第93轮165开发全审计通过：140几何7.5102e-6，60更新重放token5.9605e-7/范数1.4211e-14，5Direct及70邻居引用。DEC等范数GraphMAE70.48 vsAdam70.84、GraphMAE2 69.84 vsgsh70.08，停止频率校正扩大。仅平方等范数des在GraphMAE71.04，高于当前70.84（+.20pp），纯幅度70.92、Direct70.40；GraphMAE2 des70.00未胜70.08。按准确率目标固定GraphMAE des追加确认，不能归因DEC。

确认设计round93_confirm_design.md，新20支持144100..119、340条、PID162415、round93_confirm.log。正式开发同步完成，本地refresh83796待核验，服务器43365。cleanup077仅删37预检30,125,164字节，正式保留。


### 第93轮首组确认：小幅均值阳性，尚未保留

GraphMAE仅平方等范数des68.98%，当前Adam68.77%，+.21pp，95%支持级区间[-.09,.52]、11胜7负2平；gsh68.88、纯幅度68.84、Direct68.93，差异均未明确。des对DEC等范数+.45[.18,.70]，不把优于失败候选当成相对当前强根的成功。

340确认全审计通过：280几何1.1325e-5，120更新重放token7.1526e-7/范数1.4211e-14，20Direct/140邻居引用。因开发+.20和首组+.21同向，固定des追加一次20支持复核，不调温度/范数。设计round93_confirm2_design.md，145100..119、340条、PID163362、round93_confirm2.log。现有配方不替换；第二组尚无结论。同图查询重复使用边界不变。首组同步37690进行中，服务器43705、本地已核验43365。


### 第93轮第二确认完成：不升级仅平方变体

GraphMAE des69.39 vs当前Adam69.38，+.01pp，95%支持级区间[-.29,.32]，8胜10负2平；对gsh69.49为−.10、纯幅度69.34为+.05，均区间跨0。对DEC等范数69.05为+.34[.10,.59]，对Direct68.93为+.46[.05,.82]；记录局部阳性，但未稳定优于当前提示。首组+.21、第二组+.01，不升级默认配方、不继续扫描温度。

340第二确认全审计通过：280几何9.5367e-6、120更新重放token8.3447e-7/范数1.0658e-14，20Direct/140邻居引用。同步已完成，本地refresh71229待核验44045；首组refresh已完成43705。cleanup077仍仅删预检37条。下一轮round94_design_draft.md借鉴自步学习渐进筛选，teacher94.py原语已实现，尚未接入训练。

本地refresh完成44045 2115，README/state同步更新。第94轮合成原语恢复误差0、独立逐类求和4.7684e-7，选择数量/支持排除/重复确定性通过；尚未实际训练。


### 第94轮启动：渐进筛选教师原型

固定设计round94_design.md：正序25/50/100原门控与等范数、纯幅度，以及反序/固定50/随机同数量等范数；全部100恢复仅预检。一个冻结GCN及768token，原Adam/Grafted、中心化+k50保留。每步掩码、中心、非支持索引及SHA封存；CPU独立逐类重建，NumPy更新重放。PID164677，round94.log，42预检四审计后190正式146100..104。代码与配置SHA已封存，暂无准确率结论。

第94轮42预检全部通过：36几何8.7023e-6、16更新token1.4901e-8；身份恢复两权重预测一致，分数误差9.5367e-7/1.1921e-6。正式190已启动；预检准确率不作为效果证据。


第94轮190开发全审计通过：160几何8.7619e-6，70更新重放token7.1526e-7/范数1.0658e-14；10Direct和80邻居引用通过。GraphMAE反序70.72 vs当前Adam70.28（+.44），正序等范数70.48，gsh70.48，随机70.36；GraphMAE2正序70.16 vs当前gsh70.04（+.12），固定70.16并列、随机69.96。Direct正序70.80 vs原70.48，暂未被提示超过。

固定GraphMAE cuv、GraphMAE2 cum进入20支持确认（正序为预定主候选，GraphMAE按准确率目标选择反序，不声称易到难优越）。round94_confirm_design.md，新147100..119，共760条，PID166295、round94_confirm.log。保留完整开发对照及Direct正序/原式配对。正式开发同步完成，refresh89615待核验44235；确认尚无结论。

本地refresh完成44235 2115，README/state已更新。cleanup078仅删42预检42,050,666字节，正式全部保留；剩余15GiB。确认PID166295仍运行。


### 第94轮确认配对结果：暂不升级

760预测已完成，GraphMAE反序相对当前Adam+.17pp，95%支持级区间[-.04,.39]、11胜4负5平，但相对同批gsh为−.24，随机/幅度/固定对照差异也不明确。GraphMAE2正序相对gsh+.01[-.12,.14]、6胜7负7平；Direct正序相对原式+.04[-.07,.14]。不把开发最高分当成成功，暂不升级、不继续按确认最高分换候选。几何与更新审计尚在进行，完整结论待审计通过。同步16932进行中。

第95轮草案round95_design_draft.md已核对Mean Teacher原文EMA公式，只借用指数平滑原则，改为教师原型映射欠松弛，不复现其SGD权重教师或一致性损失。teacher95.py与check95.py合成原语通过，恢复精确、逐类重建最大5.0664e-7；三步阻尼为预定主候选，六步阻尼/六步无阻尼作预算控制。尚未接入训练。


第94轮确认绝对数值：GraphMAE反序68.78 vsAdam68.61/gsh69.02，GraphMAE2正序68.78 vsgsh68.77，Direct正序68.70 vs原68.66。640几何重建通过，最大1.0848e-5；优化器重放PID167415仍在运行。正式同步已完成；待全审计后更新有效计数。

第95轮已接入固定训练与独立CPU审计，round95_design.md、源码/config SHA已封存，计划38预检四审计后170正式148100..104。dar/dam/daa三步原门控/等范数/幅度，das六步阻尼及dap六步无阻尼等范数，身份仅预检。沿用现有强根、不用第94轮未确认变体。


第94轮760确认全审计通过：640几何1.0848e-5，280更新token7.1526e-7/范数1.4211e-14；40Direct/320邻居引用通过。配对结论维持不升级。

### 第95轮开发完成：阻尼失败，六步对照进入确认

170正式全审计通过：140几何6.7353e-6，60更新token5.9605e-7/范数1.0658e-14。GraphMAE三步阻尼等范数66.52 vsAdam67.96，GraphMAE2 67.72 vsgsh68.44；Direct阻尼67.12 vs原68.64，阻尼不扩大。六步无阻尼dap68.40/69.00，相对各自当前+.44/+.56，但GraphMAE相对同批gsh68.36仅+.04；六步阻尼67.96/68.60。按准确率目标固定dap，不把迭代次数收益归因阻尼。

round95_confirm_design.md，新20支持149100..119、680条，PID169151、round95_confirm.log。全部开发条件保留，源未改；未设六步同教师Direct，不能声称超过该对照。正式同步3585进行中，服务器45165、本地核验仍44235。cleanup079只删38预检46,582,639字节，正式全部保留。


### 第95轮确认配对：六步收益未复现

GraphMAE dap相对当前Adam−.01pp，区间[-.24,.28]、6胜9负5平；GraphMAE2 dap相对gsh+.06[-.12,.31]、7胜9负4平。相对幅度、六步阻尼也无明确增益，不升级、不继续按确认最高分切换变体。GraphMAE相对原三步Direct+.52[.05,.99]仍记录为局部阳性，不替代强提示比较。560几何重建已通过7.9870e-6，优化器重放PID170945继续。

补充同教师Direct：round95_direct6_design.md，独立源码不改已启动训练，原开发5及确认20共25条。Direct身份/None权重、封存后评分、sourceSHA、独立六步中心重建全部通过，确认20重建最大3.3379e-6。补测不是最初矩阵预注册。配对脚本paired95_direct6.py，网络连接中断后安全重跑只读分析。

本地sync3585仍在传输第95轮开发，目录由17MB/22结果增长到32MB/37结果，确认进程活跃，不因耗时重启或并发重复传同目录。远端剩余13GiB；清理079已做，正式证据保留。待同步完整才能增加本地核验计数。

六步Direct补充配对完成：原三步对六步−.19pp，区间[-.45,.01]；dap提示相对六步Direct，GraphMAE+.33[-.11,.76]、GraphMAE2+.08[-.26,.42]，均区间跨0。原先相对三步Direct的局部阳性不能解释为超过同教师Direct，停止升级决定不变。


第95轮680确认全部审计通过：560几何7.9870e-6，240更新token7.1526e-7/范数1.4211e-14。加上25补充Direct，服务器累计45870正式有效结果；本地仍44235，尚未完整同步，不混用计数。原sync3585仍活跃；串行续传脚本sync95_remaining.py（session36644）等待原PID37012退出并核验170条后，依次同步确认/补充，再refresh预计45870，避免重复并发传同目录。

第96轮round96_design_draft.md已读取RMSNorm原文，改为传播后门控行重加权，明确与LayerNorm零均值等价、epsilon破坏严格尺度不变性。gate96.py/check96.py已实现原语，首次上传连接中断导致远端文件未齐，重传后再验证；尚未训练，不作效果结论。

第96轮原语首测均匀6类的逐位零断言失败：1/6及跨41节点均值存在浮点残差，经行权重放大仍应按容差检查。仅修订check96.py：均匀6类绝对容差1e-12，另加4类可精确表示均匀值验证真正零分支逐位为零；gate96实现不变，未运行正式训练。

第96轮修订后原语通过，显式逐行计算最大误差4.44e-16，范数/置换/恢复/零分支及零均值LayerNorm等价检查通过。尚未接入训练。


### 第96轮启动：门控节点行RMS归一化

round96_design.md已固定，145正式150100..104，33预检150199。rmm等范数主候选，rmr原幅度、rma纯幅度、rmp置换权重，rmi恢复仅预检。同一个冻结GCN和768token，当前三步教师/优化器/读出不变。源码/config SHA已封存；PID171836、round96.log。预检四审计完成才正式，目前无准确率结论。

旧同步PID37012仍活跃且目录已增至85MB，串行续传PID37377正常等待。为保持refresh45870预期，旧续传完成前不在本地同步第96轮正式目录；服务器仍保留全部新预测。

第96轮33预检全部审计通过：28几何8.6427e-6，12更新token1.4901e-8/范数1.0658e-14；两权重身份恢复预测一致，分数误差均1.1921e-6。145正式已开始运行。


第96轮145开发全部审计通过：120几何1.0252e-5，50更新token5.9605e-7/范数7.3053e-14。GraphMAE rmm68.80与当前Adam68.80相同，不扩大；GraphMAE2 rmm69.08 vs当前gsh68.88（+.20），置换rmp69.16，Direct69.24。原归一化68.60、纯幅度68.16。不能称节点RMS对应关系优越，固定原主候选rmm进入确认并保留置换。

round96_confirm_design.md，新20支持151100..119，GraphMAE2共300条，PID173376、round96_confirm.log。源和超参不变。cleanup080仅删33预检25,503,017字节；正式证据保留。服务器有效累计46015，本地核验仍44235；旧同步目录已增长102MB，sync3585/串行36644继续，不能因耗时当成停止。第96轮正式目录待旧串行refresh45870后同步，避免预期计数冲突。


### 第96轮确认配对：RMS门控不升级

GraphMAE2等范数rmm相对当前gsh−.07pp，95%支持级区间[-.42,.26]、6胜13负1平；相对旧Adam+.15、幅度+.38、置换+.18、Direct+.34，区间均跨0。停止扩大，不按其他确认条件最高分更换主候选。240几何通过1.0252e-5，更新重放PID174060进行中；300确认尚未同步。

第97轮round97_design_draft.md已读取DropConnect原文，拟只对有效提示矩阵tanh(T)训练时屏蔽元素，明确整图共享掩码与原文每样本掩码不同、全矩阵eval非精确集成。element主候选，channel结构对照，gain仅训练确定性平方范数控制。drop97/check97已实现原语，尚未接入训练。

第96轮300确认最后审计通过：100更新token3.5763e-7/范数5.8953e-14，累计服务器46315；仍不升级。第97轮原语解析梯度/掩码重现/eval关闭/零提示/keep1恢复及枚举期望通过，期望误差0，尚无训练。


### 第97轮启动：有效提示矩阵DropConnect

固定设计round97_design.md，dce元素屏蔽主候选、dcc通道、dcg训练gain、dci恢复仅预检，保留两优化器根/Direct/Frozen/GraphPrompt。原教师/门控/768token/100步/中心化+k50不变。额外保存首末步参数与全部掩码/梯度；audit97_masks独立逐支持/逐类留一原型重算CE梯度，检查实际训练作用。源/configSHA已封存，PID174520、round97.log，29预检五审计后125正式152100..104。暂无准确率结论。

旧第95轮同步已增至146MB/150结果，仍按原队列运行；本地计数不能提前增加。

第97轮29预检五项审计全部通过：16首末步梯度最大7.4506e-9，24几何7.3910e-6，10更新token1.4901e-8/范数7.1054e-15；两权重身份恢复预测一致、分数9.5367e-7。125正式已启动。


### 第97轮开发完成与固定确认

125正式五审计通过：100几何最大5.7817e-6、40更新token7.1526e-7/范数1.4211e-14、60首末梯度6.5193e-9；5Direct/50邻接引用。GraphMAE元素67.28、通道67.32、gain67.36低于当前Adam67.40；GraphMAE2元素/gain67.48低于gsh67.56，仅通道67.72（+.16pp）。主元素候选失败；固定开发筛选通道候选，20新支持153100..119、confirm查询、13方法260条，不调参。详见round97_confirm_design.md；同图重复查询限制不变。服务器累计46440，本地44235，旧同步仍运行。

第97轮确认已运行，批进程176450（训练176451），round97_confirm.log；启动时检查原源清单与确认清单SHA。cleanup081已仅清理29预检21,338,485字节，正式保留。


### 第97轮确认结束：不升级

260确认五项审计通过，200几何最大9.2387e-6、120首末步梯度5.1223e-9。通道69.50% vs当前gsh69.58%，−.08pp，支持bootstrap95%CI[-.24,.09]，4胜11负5平。比gain−.14、元素+.04、Direct+.34，均区间跨0；比旧Adam+.40[.15,.64]是局部阳性，不能代替当前强根比较。不保留新配方。服务器累计46700，本地44235，旧同步仍未结束。下一轮DCL原型损失草案已写round98_design_draft.md，已读原论文摘要/引言，公式段需补读，尚未实现/运行。


### 第98轮启动：类原型解耦分类目标

来源[DCL第3节/附录A.7](https://arxiv.org/html/2110.06848v3)，记录原文监督分类略逊CE的反例。仅替换支持LOO损失；固定两权重各自当前优化器，比较dcl、平均权重CE、置换权重CE；预检另含原CE恢复和逐样本逆权重梯度恒等。完整可微原型链原语误差6.66134e-16，极限数值通过。源/config已封存；PID177863 round98.log，33预检五审计后125开发154100..104。暂无准确率结论。旧R95gate同步310MB已结束，原串行队列37377继续后续文件，未刷新本地数量。

第98轮33预检全部通过：28几何最大9.53674e-6、12更新token1.49012e-8、20首末梯度7.45058e-9；dci恢复两当前根，dcw恢复dcl预测一致、最大分数误差1.43051e-6。125正式已启动。


### 第98轮开发完成与确认启动

125五审计全过：100几何7.62939e-6、40更新token7.15256e-7/范数1.27676e-14、60首末梯度8.84756e-9。GraphMAE主dcl68.16低于当前68.24；GraphMAE2 dcl68.32对gsh68.16+.16pp，但scale68.32、perm68.36，Direct69.44。固定GraphMAE2主候选及全部对照155100..119、260确认，PID179722 round98_confirm.log；不能先归因去耦。cleanup082仅删除33预检24,857,223B。服务器46825，本地44235。round99_design_draft.md拟将已有配方筛查到DGI/GraphCL/Bandana/S2GAE；权重目录已确认，但尚未加载/运行。


### 第98轮确认完成；第99轮跨权重启动及审计修订

第98轮260确认五审计通过：200几何5.96046e-6、80更新token4.76837e-7/范数1.49880e-14、120首末梯度6.51926e-9。DCL对当前gsh平均0.00pp[-.10,.12]，5胜7负8平；对scale−.04、perm−.07，区间跨0。对旧Adam+.66[.30,1.07]、Direct+.76[.24,1.36]是局部阳性，不能当作DCL增量，停止扩大。服务器47085，本地44235。

第99轮DGI/GraphCL/Bandana/S2GAE strictload成功，17方法/权重，65预检后325正式。原PID180987在预检CPU几何阈值停止，正式未开始；独立全预检诊断见round99_audit_amendment.md，所有预测一致，最大分数误差3.2139e-4。正式前固定v2阈值并额外要求预测逐节点一致，不改训练源码和结果；原日志保留。PID182242 round99_v2.log重新审计后启动正式。新同步队列39329/session14591只在旧队列95完成45870后同步96至98并刷新47085；99不可提前加入该精确计数。

第99轮v2预检四审计通过：56几何重建且预测逐节点一致，最大分数3.21388e-4；4优化器重放token7.45058e-9/范数7.10543e-15。325正式已实际启动，最近DGI seed156102。无正式准确率结论。


### 第99轮开发：逐条审计边界与三权重确认

325评分/身份审计通过，20优化器重放token3.57628e-7/范数2.9976e-14。几何v2正式在argmax相等检查停止；全诊断280行显示只有Bandana/tr1/mean的156100节点2375、156101节点3219各一处近并列差异（约1e-6）。最大均值score误差2.62618e-4、邻居权重3.45558e-5、传播score3.07560e-4、cutoff7.51019e-6，均在v2数值界内。保留要求不放宽：audit99_geometry_rows.json（实际前缀round99_gate）记录278通过、2失败，不能称全325通过，暂不计入累计；不清理本轮预检。

DGI/GraphCL/S2GAE各70几何记录全部通过。开发DGI gsh50.36对tr1 40.20（+10.16pp）；GraphCL tr1 58.04对tr2 49.84（+8.20）；S2GAE tr1 59.12对tr2 49.76（+9.36）。均低于Direct69.40；Bandana最优34.64且数值边界，暂不扩大。固定三候选20新支持157100..119、全部17方法、980条确认，PID184200 round99_confirm.log，启动有逐权重审计守卫。完整设计round99_confirm_design.md。旧全审计累计服务器47085、本地44235保持；R99开发部分通过单独记录。

第100轮草案round100_design_draft.md已阅读LogitNorm公式4及余弦/温度区别；拟固定初始支持logits范数目标，比较可微归一、停止梯度分母、统一动态温度及原CE。原论文OOD收益不等于ID准确率收益。仅草案，未实现/未运行。


### 第100轮实现启动；第99轮确认审计未完成

LogitNorm适配按round100_design.md封存，解析梯度误差6.93889e-18，epsilon0径向误差1.45283e-17；同前向的停止梯度分母确实改变梯度（最大.02047）。原CE/零向量边界通过。PID185513 round100.log，29预检后125开发。暂无准确率结论。

第99轮980确认预测已生成，评分/身份通过，几何v2再次失败，已启动逐条审计和全差异定位（exec35104/63819），暂不计全审计累计，不擦除失败或直接放宽标准。此前Bandana两记录失败仍保留。


### 第99轮确认完成：保留三种限定权重配方

逐条几何确认838通过/2失败，均DGI mean（157107/tm1节点2642，157108/tr2节点3071）各一查询近并列；60优化器重放token2.38419e-7/范数3.77476e-14。开发与确认共1305行中1301通过，4失败继续排除；round99_scoped_audit.json逐seed证明所有paired99候选/对照均合格，不宣称全批通过。旧累计47085与本轮作用域1301分开记录。

DGI gsh47.04 vs锐化Adam36.62，+10.42pp[7.80,13.02]；GraphCL锐化56.65 vs原48.47，+8.18[6.91,9.46]；S2GAE锐化55.92 vs原46.97，+8.95[7.37,10.57]。后两者还超过幅度控制3.64/3.80pp，区间为正。Direct69.11更高。保留限定checkpoint适配，不替换整体强方案；详见保留方案_跨权重适配.md，项目及上级README新增入口。第99轮预检暂留以保全审计差异证据。


### 第100轮开发完成：GraphMAE固定确认

125条五审计全通过：100几何9.29832e-6、40更新token5.96046e-7/范数1.06581e-14、60首末梯度5.58794e-9。GraphMAE lnr67.08对Adam66.88 +.20pp；lnd/lng66.92，gsh68.36和Direct68.28更高。GraphMAE2 lnr67.84低于gsh68.12，不扩大。固定GraphMAE 159100..119、260确认PID187512，全部强对照保留，round100_confirm.log。cleanup083仅删smoke100的29行21,329,138B，smoke99继续留证。普通全通过批累计47210，另R99作用域1301通过/4排除，禁止混淆；本地44235仍同步中。


### 第100轮确认停止与统计修正；第101轮启动

第100轮260确认五审计通过：200几何6.19888e-6，80更新token7.15256e-7/范数1.42109e-14，120首末梯度4.19095e-9。LogitNorm对当前Adam仅+.06pp[-.07,.18]，9胜4负7平，停止扩大；其余主要比较区间均跨0。累计全通过批47470，另R99作用域1301。

发现全局轮次数字替换误改paired98/100分位点，已保留旧文件并生成独立v2重算；详见statistics_98_100_correction.md。第98轮正确当前根区间[-.10,.10]，旧Adam[.30,1.03]、Direct[.24,1.31]；此前旧上界撤回，以v2为准，停止结论不变。第99轮统计未受影响。

第101轮阅读Circle Loss第3节，借鉴独立正负相似度权重，固定gamma5/m.25并匹配常数同间隔、置换权重、原CE恢复。预检先暴露softplus阈值20近似影响高精度导数；封存前改为50，解析误差2.08167e-17、常数CE归约1.38778e-17。源码/配置已封存，PID189108 round101.log：29预检五审计后125正式160100..104。暂无准确率结论。


### 第101轮开发完成与500确认；第102轮原语

第101轮125五审计通过：100几何5.72205e-6、40更新token5.96046e-7/范数1.50990e-14、60首末梯度9.31323e-9。GraphMAE cir69.80 vsAdam69.32 +.48pp，cip69.76；GraphMAE2 cir69.88 vsgsh69.80 +.08，cip69.92，Direct70.44。固定两权重20新支持161100..119、500确认PID190931，现预测完成、审计中。paired101显式写QUANTILES=(.025,.975)，无轮次数字替换。cleanup084仅删29预检21,352,926B；全批累计47595，另99作用域1301。

第102轮已读IMP原文算法1，草案round102_design_draft.md明确固定2簇/不学习方差与原文区别；teacher102/check102原语恢复原根误差≤1.77636e-15，支持重排不变、非空簇/无支持泄漏/退化相同特征通过。尚未接入训练，暂无真实图结果。下一步保留同新教师Direct、原Direct以及门控范数/幅度/随机簇对照。


### 第101轮确认结束：不替换；第102轮启动

500确认五审计全部通过：400几何1.02520e-5、160更新token8.34465e-7/范数1.79856e-14、240首末梯度1.39698e-8。Circle GraphMAE对Adam+.23pp[-.03,.46]，15胜4负1平，但对gsh−.12[-.40,.18]；GraphMAE2对gsh+.05[-.11,.22]，对置换0.00[-.09,.09]。局部正均值保留，但区间跨0且无强对照明确优势，不替换已有根。全批累计48095，另99作用域1301。

第102轮多原型已实现，原语重跑恢复误差≤1.77636e-15。独立重建器验证CPU最远对/支持分组、3步软中心、非支持索引、12分量score并绑定SHA。40预检（重复中心恢复）后180开发，4Direct/5门控变体，源码/config封存；PID192619 round102.log。分组离散数值边界1e-6预先固定，其他几何阈值延用原配方。暂无准确率结论。


### 第102轮停止；第103轮收缩双原型启动

第102轮180条四审计通过：140几何最大误差8.10623e-6，60优化器重放token5.96046e-7/范数1.35447e-14；20Direct教师重建2.86102e-6。GraphMAE双原型max64.48 vs当前Adam68.16（−3.68pp），范数匹配64.36、logmean65.80、随机64.72；GraphMAE2 max63.40 vs当前gsh67.68（−4.28pp），logmean65.64。旧Direct68.04，新max62.48/logmean65.16/random64.48。所有新结构都明显低于当前根，停止扩大，保留失败证据。累计普通整批合格48275，另R99作用域1301合格/4排除，不能合并宣称整批全通过；本地44235已核对，旧同步仍进行。

第103轮借鉴Morris与Lysy 2012统计学分层收缩（已读原文第1节），针对每类5点分成两小簇的潜在高方差，仅作固定0.25收缩修正。详细公式、原文区别及停止条件见round103_design.md：支持子中心向全类中心靠拢，且每次软更新后向双中心均值靠拢；不是原文James–Stein估计，不承诺风险保证。6×128 token、冻结共享GCN、当前优化器与readout保持。未收缩/随机/幅度对照和5个Direct均保留。原语a=0恢复误差1.55431e-15、a=1恢复误差0，封存后启动PID194446 round103.log：45预检四审计后205开发163100..104。目前无准确率结论。

cleanup085仅删除smoke102的40条52,803,439B，已通过审计的正式记录保留。服务器最近可用9.8GiB；旧同步round95_confirm本地201MiB且进程存活，避免重复传输与过早扩充本地精确计数。


### 第103轮开发通过，820条确认启动

205开发四审计全部通过：160几何8.34465e-6，70优化器token5.96046e-7/范数1.42109e-14，25Direct教师误差4.05312e-6。收缩max GraphMAE69.08低于Adam69.16，GraphMAE2 69.56低于gsh69.80；因此不宣称主结构有效。按准确率目标保留开发正增益变体进入确认：GraphMAE随机分簇mxp69.40（+.24pp），GraphMAE2 logmeanexp mxs69.92（+.12pp）；幅度对照分别69.20/69.84，相应Direct69.88，旧Direct69.68。

确认预注册round103_confirm_design.md固定两候选、164100..164119、新query后500但同图重复使用；全部原对照保留，共820条，PID196024 round103_confirm.log。源码封存、显式CI分位点(0.025,0.975)，确认前不替换根。开发普通整批累计48480，另99作用域1301/4排除；本地44235仍等待旧顺序同步。cleanup086删45条预检60,909,239B，正式证据保留。


### 清理087与第104轮互补熵启动

第103轮820确认预测已完成；计分/教师引用与640几何重建通过（最大8.34465e-6），280优化器重放仍运行，未提前确认增益。新一轮不混入尚未确认的收缩教师。

第104轮借鉴[Complement Objective Training](https://arxiv.org/html/1903.01182v2)第2.1/2.2/3.1节，改为一次合并CE与负类条件负熵/5，区别于原文交替两更新。保留全类熵、错配排除标签、标签平滑0.1、当前根和Direct。原语梯度最大3.47e-18，被排除logit不变性、均匀负类恢复、极端有限通过；源码封存后PID197182 round104.log，33预检五审计后145开发165100..104。设计见round104_design.md，无准确率结论。已检索本地历史避免重复第17轮SWA、第49轮PolyLoss、第78轮PCGrad。

cleanup087针对服务器重复副本：round83_confirm与round89_confirm，共2610文件、1,443,848,087B，所有服务器文件与本地逐文件长度及SHA256一致，并在删除前再次核对服务器未变，方才删除远端副本。完整正式结果保存在本地同名目录，archive087_verified_manifest.json及archive087_verification.json/cleanup_087.json记录证据。不是丢弃正式结果，也不是预检清理；以后远端累计完成数是历史审计累计，不代表所有历史文件都仍在线。服务器可用约9.8GiB；旧同步95/98仍存活且增长，本地累计仍44235。


### 第103轮确认未复现；第104轮预检通过

820确认四审计通过：640几何最大8.34465e-6、280更新token7.15256e-7/范数1.42109e-14。GraphMAE随机收缩候选对Adam−.19pp，95%支持bootstrap区间[−.36,−.04]，4胜10负6平；对幅度−.18[−.35,−.02]。GraphMAE2 logmeanexp候选对gsh−.12[−.37,.14]，6胜11负3平；对Direct−.19[−.60,.28]。开发+.24/+.12未复现，按预注册停止双原型修正，不继续调系数。

保留局部事实：两候选相对未收缩双原型分别+2.74[1.48,4.12]与+3.07[1.91,4.42]，只是恢复弱方案损失，未超过当前单中心根；GraphMAE2 logmeanexp对收缩max+.19[.07,.33]也不构成当前根升级。完整逐支持配对见round103_confirm_pairs.json。普通累计全批合格49300，另99作用域1301/4排除；本地已核验累计44235，待同步正式新批不能计入。

第104轮33预检五审计通过：28几何7.25687e-6、12更新1.49012e-8，20首末梯度5.82077e-9；原CE恢复预测相同，分数误差≤2.38419e-6。145正式165100..104已启动，PID197182/训练198133。清理088须等正式全部审计通过后仅删除smoke104，当前未执行。


### 第104轮首组确认阳性，第二组复核；第105轮启动（2026-09-24）

104开发145五审计通过：120几何5.99027e-6、50更新token7.15256e-7/范数1.06581e-14、80首末梯度8.38190e-9。GraphMAE与当前根71.04持平，不扩大；GraphMAE2互补熵71.80 vsgsh71.64 +.16pp，固定300确认。cleanup088删除33预检24,891,202B，正式保留。

首组166100..119共300五审计通过：240几何5.48363e-6、100更新token3.57628e-7/范数1.42109e-14、160首末梯度6.51926e-9。互补熵相对gsh+.13pp[.04,.23]，9胜2负9平；相对全类熵+.12[.05,.19]，错配排除+.10[.02,.18]，标签平滑+.12[.06,.20]；对旧Adam+.29[−.09,.69]、Direct+.02[−.29,.31]不明确。固定168100..119第二组300，源/系数/对照完全不变，PID201754 round104_confirm2.log。仅列待复核候选，候选方案_互补熵.md及两级README记录，尚不替换根。普通全批累计49745，另R99作用域1301/4排除；本地44235。

第105轮读Gneiting与Raftery2007第3.1节Example1，借用概率天气预报二次评分；Brier1950原文链接未取得，未声称已读。CE+Brier固定1为主，另纯(K/2)Brier、支持logit梯度范数匹配CE、错配辅助目标和原CE恢复。原语梯度最大1.388e-17、均匀概率纯Brier恢复CE3.47e-18、极端/零分支通过。源码封存，PID200364 round105.log，33预检五审计后145开发167100..104。预检已过、开发预测已完，正式审计仍在运行；暂读主候选GraphMAE70.12等于Adam，GraphMAE2 70.96低于gsh71.04，未见主方案增量，待审计完整后归档。


### 第105轮结束：不扩大

145正式五审计全部通过：120几何最大6.4373e-06，50更新token5.96046e-7/范数1.06581e-14，80首末梯度6.51926e-9。GraphMAE CE+Brier70.12等于当前Adam70.12，低于gsh70.80；GraphMAE2主候选70.96低于当前gsh71.04。纯Brier70.08/70.84；错配辅助70.24/71.04，前者仅弱根+.12但低于同权重gsh，不扩大。停止此固定损失方案，不把概率评分性质当准确率证据。cleanup089删33预检24,900,838B，正式保留；普通全批累计49890，另99作用域1301/4排除。第104轮第二组300仍由PID201754运行，下一步以其五审计和配对区间决定候选处置。


### 第104轮第二组不升级；第106轮跨权重迁移

第二组300五审计通过：240几何7.39098e-6、100更新token3.57628e-7/范数1.06581e-14、160首末梯度5.58794e-9。互补熵对当前gsh+.05pp[−.07,.19]，7胜5负8平，未达到固定升级要求；对全类熵+.05[−.05,.17]、错配排除+.08[−.01,.18]、平滑+.10[约0,.22]、Direct+.32[−.09,.71]。保留首组+.13[.04,.23]为条件阳性，停止GraphMAE2此配置继续扩张，不合并两组追逐显著性，不升级当前根。候选说明与两级README同步更新。普通全批历史累计50190，另99作用域1301通过/4排除；本地44235待旧同步，两个数量不能混用。

第106轮固定DGI/GraphCL/S2GAE迁移互补熵及所有损失对照，DGI使用保留嫁接Shampoo、其他使用保留Adam。三checkpoint SHA与99严格加载审计一致，无新预训练；来源仍ProG-adapted未完全源等价。49预检五审计后215开发169100..104，源/config已封存，PID203417 round106.log。几何阈值在运行前明确继承99的跨权重v2，并额外严格要求CPU argmax一致；失败不隐去，不根据结果调宽。目标只检查新权重适用范围，不能反向修改104的停止结论。


### 第106轮预检全过，正式预测完成

49预检五审计通过：42几何4.40776e-5且所有重建argmax一致；18更新token1.49012e-8/范数1.06581e-14；30首末梯度4.47035e-8；三权重原CE恢复分数最大1.23978e-5、prediction一致。215正式预测已完成，PID204928几何审计运行中，尚不计入合格累计，cleanup090未执行。

补充跨领域检索见screen_spherical106.md：2007概率预测文献Example2球面评分，经自身代数和值/梯度检查可归约到两倍logit的q=.5 GCE乘.5；其负对数等于两倍logit CE乘.5，最大梯度误差1.04084e-17。此前75只测q=.3/.7，不推断精确组合已失败；仅暂缓此同族算子，真实图未运行，不增加正式记录或称作准确率失败。


### 第106轮开发结束与860确认；清理091及入口修正

215开发五审计通过：180几何最大6.38962e-5且CPU argmax完全一致、75更新token5.36442e-7/范数4.46310e-14、120首末梯度3.72529e-8。DGI互补熵45.28 vs保留gsh44.48 +.80pp；GraphCL互补熵53.52低于Adam53.84，但原CE gsh55.60 +1.76；S2GAE互补熵55.48低于gsh55.80，后者对Adam54.40 +1.40。Direct70.28明显更高。按准确率目标固定DGI互补熵、GraphCL/S2GAE gsh三候选，后两者是优化器收益，不能归因互补熵。170100..119全部对照保留共860确认，PID205467 round106_confirm.log实测存活。普通全批历史累计50405，另99作用域1301/4排除，确认批未计入。

cleanup090仅删49预检37,260,694B。cleanup091逐文件验证round81_confirm/round82_confirm/round88_confirm/round90_confirm的3860文件与本地长度及SHA完全一致，删除前再检服务器未变，然后仅清理远端重复目录，共1,557,052,824B；本地正式证据完整，清单archive091_verified_manifest.json。服务器实测9.7GiB可用。原95/98慢速队列持续增长，未重复启动或扩大本地精确计数。

检查发现旧脚本r=Path('.')时r.parent仍为'.'，导致第99/104轮几条父级README说明追加进项目README，真正父README未收到。本轮改用Path.cwd().parent并断言绝对路径，已补写真正持续推进/README.md并读回验证；历史误置说明保留且追加链接校正。新增当前进展摘要_20260924.md提供阶段快照入口，区分已保留/条件阳性/停止/运行中及本地远端资产位置。


### 2026-09-24：R106确认审计失败与R107实现修复
R106确认860条预测完成；转移、几何和更新审计通过，720条几何最大误差0.0001094043且argmax一致，300条更新重放最大token误差6.56e-7。独立CPU损失审计在GraphCL/170109/cps末步失败，最大梯度误差0.000251358，超过原阈值。暂不将860条计入全批验证，不调整阈值；运行逐条诊断确定影响范围，保留原失败日志。
R107借鉴RAdam（ICLR2020）早期自适应方差修正，采用原式SGD回退、范数匹配、仅幅度及前5步跳过对照，和原Adam/GraphMAE2 graftedShampoo比较；primitive独立NumPy重放通过。第一版smoke因无边界代号替换将probability误改为probrrility而退出，正式实验未启动。v2仅修复字段名，保留原代码/manifest/log/3条smoke，另建smoke107_v2与v2源码封存；8步smoke覆盖第6步自适应分支，通过审计后才运行145条开发实验。
原95/98归档队列仍在运行，暂不并发复制99以后正式目录，避免干扰精确数量审计。历史普通全批验证仍50405，R99另记1301验证/4排除，本地完整44235。


### 2026-09-24：R106范围审计、S2GAE第二组复核与R107停止
R106确认逐条CPU损失诊断239组通过、1组失败；失败GraphCL/170109/cps的mean与k50共享训练状态，保守同时排除。其余转移/几何/更新通过；[范围审计](round106_confirm_scoped_audit.json)明确858验证、2排除、whole_batch_passed=false。独立GPU重算该失败快照两步误差最大1.95578e-8，提示设备数值路径可能有关，但尚未定位具体算子，不据此解除CPU排除。原阈值和源文件保持不变。
[范围配对](round106_confirm_pairs_scoped.json)：DGI cpr相对gsh+.48pp[-.35,1.30]，相对全类熵−.01、错配+.01，无明确升级；GraphCL gsh相对Adam−1.75[-3.47,−.32]，停止。S2GAE gsh57.81相对Adam55.87 **+1.94[.24,3.78]**，相对Direct−11.33[-12.80,−9.71]。三个主比较都20种子完整合格；仅GraphCL对标签平滑的辅助比较19种子。
按封存门槛启动[第二组复核](round106_confirm2_design.md)：S2GAE 172100..172119，同样15方法共300条，原源码/超参数与5审计完全不变，PID209152，round106_confirm2.log。若第二组gsh对Adam区间仍>0才升级；不合并追显著，收益归于优化器迁移，不归于COT。
R107 v2开发145条全部审计通过：几何120条最大5.00679e-6，更新50条token最大3.57628e-7/norm2.13163e-14。GraphMAE：Adam68.36、gsh67.96、RAdam原式60.40、范数匹配68.28、仅幅度60.36、早期跳步60.40；GraphMAE2：gsh68.04、Adam68.44、原式65.16、范数匹配68.28、仅幅度67.76、早期跳步65.00。Direct69.12。范数匹配对GMAE2当前gsh有+.24，但仍低于同组Adam；原式显著退化，不扩展本配置。这个结论仅针对lr=.01/100步，不能泛化为RAdam整体无效；高学习率/更长步数未运行。参考[RAdam原论文](https://www.microsoft.com/en-us/research/uploads/prod/2020/04/1908.03265.pdf)及[作者代码](https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py)，当前原式使用SGD回退，并单列早期跳步，区别于作者默认degenerate_to_sgd=False。
历史普通全批验证更新50550；R99另1301/4排除、R106确认另858/2排除，不折入普通整批数。本地完整44235，95/98原归档队列仍运行。cleanup092删除通过审计的33条smoke107_v2共28,060,520B，保留清单、审计和v1失败smoke证据；正式实验不删除。


### 2026-09-24：S2GAE优化器迁移第二组完成，不升级
R106第二组172100..119共300条全部通过：转移20/120，几何240最大7.24792e-5且严格argmax，更新100最大token5.36442e-7/norm1.56541e-14，损失160快照最大梯度1.11759e-8。gsh55.43相对Adam53.96 **+1.47pp[-.25025,3.23000]**，14胜6负；相对Direct−12.98[-14.31,−11.69]。相对互补熵+.87[-.38,2.22]、全类熵+1.50[.12,2.91]、错配+1.28[-.15,2.83]、平滑+1.78[.26,3.32]。
第一组+1.94[.24,3.78]与第二组正均值仍保留，但第二组CI跨0，预先封存升级条件不满足；S2GAE继续使用原Adam.01，不合并两组追显著，不重复扩大相同配置。DGI互补熵、GraphCL gsh此前已不升级；跨checkpoint保留文件同步更新边界。
历史普通全批验证50550+300=50850；R99范围1301/4排除、R106第一确认范围858/2排除仍单列。

### 2026-09-24：R108短程步长解释与匹配网格
继续核对[RAdam作者README](https://github.com/LiyuanLucasLiu/RAdam#questions-and-discussions)关于仍需学习率调参的说明。R107原式低而范数匹配恢复，提出短程累计更新不足的待验解释，不能把lr=.01结果泛化为整个优化器无效。
[设计](round108_design.md)固定RAdam原式SGD回退、仅幅度、Adam与gsh四类各.01/.03/.1，100步，两个checkpoint、173100..104，共265条；同一冻结两层GCN与768提示参数、门控与损失不改。复用已核验optimizer107递推；新方法分派、源码manifest已封存。173199八步57条预检已通过原4审计，正式进程PID210269，round108.log。另加独立audit108_rates.py检查方法代号与result学习率/优化器映射，作为附加检查，不修改运行中源码或原manifest。
开发只在超过该批最强Adam/gsh学习率和保留根至少.20pp后确认；若仅基线调参变好，按调参收益单列。幅度对照胜过原式时可保留幅度候选但不称方向修正收益。尚未完成，不计265为验证结果。


### 2026-09-24：R108开发完成，RAdam匹配网格未胜，确认GraphMAE调参分支
265条正式记录全部通过封存计分、转移、几何、更新及附加映射检查。几何240最大6.85453e-6；更新110最大token2.26498e-6、norm2.10942e-14；方法学习率映射220通过。
GraphMAE：RAdam.01/.03/.1=66.04/69.88/69.84；对应幅度66.08/69.92/69.92；Adam69.84/69.88/69.80；gsh70.16/70.20/69.48。GraphMAE2：RAdam67.68/70.16/70.60；幅度70.56/70.76/70.48；Adam70.44/70.76/70.88；gsh70.80/70.68/70.48。Direct70.32。增大学习率恢复原式表现，但尚未超过匹配最强基线；不确认RAdam，不将恢复当作独立方向收益。
按预定普通基线调参分支，固定GraphMAE gsh.03：对原Adam.01+.36pp，但对gsh.01仅+.04；不能把+.36全归于调参。GraphMAE2最佳Adam.1对保留gsh.01仅+.08不达门槛。
[确认设计](round108_confirm_design.md)：174100..119，GraphMAE，17方法340条，PID211994，round108_confirm.log。保留Adam三lr、gsh.01/.03和RAdam/幅度.03、Direct等，不再挑lr；原源码和100步不变。主要比较gsh.03对Adam.01并同步比较gsh.01，只有主区间>0才考虑第二组。
历史普通全批累计50850+265=51115；R99范围1301/4排除、R106范围858/2排除单列，本地完整44235。cleanup093删除57条通过预检共50635297B，manifest、审计留存，正式数据未删。旧95/98同步队列继续，未在本地插入99以后正式目录。
下一候选[109草案](round109_design_draft.md)读Dozat作者Stanford报告算法8，拟借鉴Nadam；正式OpenReview版验证受阻，未核对即不声称完全复现。当前只形成草案，未实现/未验证/不计结果。


### 2026-09-24：R108确认停止，R109 Nadam实现启动
R108确认174100..119共340条全部通过封存计分与4附加检查：转移20Direct/140邻居，几何280最大8.34465e-6，更新120最大token2.14577e-6/norm1.06581e-14，方法lr映射240通过。固定gsh.03相对Adam.01 **+.09pp[-.29,.50]**，9胜10负1平；相对gsh.01+.03[-.21,.27]，相对Adam.03−.02[-.38,.33]、Adam.1−.07[-.46,.31]，相对Direct+.11[-.27,.50]。开发+.36未明确复现，不升级GraphMAE保留根，不继续扩大这套配置。RAdam/幅度.03相对该候选亦无明确优势。历史普通全批51115+340=51455；其他范围计数不变。

R109由草案推进至[固定设计](round109_design.md)和实际实现：Nadam无warmup、beta1=.9特例，不冒充报告原实验的beta1=.99调度。两个偏差校正分别使用当前t与下一t+1；原式/匹配方向范数/仅幅度三模式，三lr .003/.01/.03，Adam/gsh同三lr；共享冻结2层GCN及768token、门控.1、100步损失不变。
[原语审计](round109_primitive_audit.json)：两根×三lr×四模式，100步含零梯度，独立NumPy最大参数误差1.19209e-7，恒等根恢复与缺失梯度不更新通过。所有新驱动、源码、配置在round109_manifest.json封存；真实图175199八步69条预检，通过封存计分/教师/几何/完整更新/方法lr映射后自动运行175100..104双权重325开发，PID213015，round109.log。尚无正式准确率结论，不计325条。
本轮候选按同批最强Adam/gsh和当前根>.20pp筛选，再固定新20支持确认；若仅幅度控制胜出则归为幅度调整。原式与幅度差异不足不声称Nesterov方向收益。


### 2026-09-24：R109 Nadam开发完成，不进入确认
325条全部通过封存计分、转移、几何、更新、lr映射。几何300最大6.67572e-6；更新140最大token1.19209e-6/norm1.42109e-14；方法映射260，5Direct/150邻居通过。
GraphMAE：原根Adam70.28，最佳基线gsh.03=70.56；Nadam原式.003/.01/.03为69.48/70.36/70.32，范数匹配69.52/70.24/70.28，幅度69.64/70.28/70.20。GraphMAE2：原根gsh.01=70.12，最佳gsh.03=70.44；原式69.68/70.12/70.08，匹配69.76/70.04/70.00，幅度70.28/70.04/70.44。Direct70.84。无候选超过同批最强基线至少.20pp，不确认Nadam；GraphMAE2幅度.03仅追平gsh.03，不当作方向收益。GraphMAE gsh.03既有R108确认失败，不因新五支持再次开发正均值就重开同配置确认。
历史普通全批51455+325=51780，R99范围1301/4排除、R106范围858/2排除仍单列，本地完整44235。

### 存储归档与第110轮原语
cleanup094：round91_confirm/round94_confirm/round93_gate/round92_gate共3085文件，本地全量文件SHA与远端一致后，远端再次完整SHA/文件集合检查，删除远端重复副本1,406,706,804B。正式证据保存在本地，见archive094_verified_manifest.json与archive094_verification.json。此操作不减少历史验证计数，不宣称这些正式文件仍在服务器。
cleanup095：69条smoke109通过审计后删除63,116,000B，保留manifest与审计。服务器可用约8.6GiB。原95/98同步队列仍运行，没有复制99以后正式目录到本地。

下一轮[110草案](round110_design_draft.md)：借鉴R-Drop原文第2.1节及作者README示例，在一个共享提示矩阵的两次独立连接扰动间加入支持LOO预测一致性。保持主干冻结及单token，不是终端融合；同两次前向平均CE无KL是关键计算/采样对照，另有节点置换KL与单掩码参照。论文CE求和与示例CE平均不同，本地拟明确采用平均CE+支持均值对称KL系数1。
loss110.py/check110.py已通过[损失原语](round110_loss_primitive_audit.json)，独立中心化logsumexp公式及双端梯度最大误差8.88178e-16，同分布KL及梯度为0、交换对称、极端logit有限。真实图接入、双掩码随机流/快照/完整更新审计尚未实现，未封正式种子或启动；不计准确率证据。当前无GPU训练任务，下一步实现110真实图双前向与独立快照梯度审计。


### 2026-09-24：R110真实图双前向实现与启动
第110轮已由损失原语推进至run_round110_rdrop.py真实图实现，[封存设计](round110_design.md)、round110_manifest.json。一个共享768token/冻结2层GCN，训练双连接掩码（keep.9）对应双支持LOO概率，平均CE+对称KL系数1，最后完整token单前向推断。rdr为主候选，rdc同两前向/同随机流无KL，rdp只置换KL节点对应而CE真标签，rds单掩码，rdi无掩码恢复只预检。
两随机流CPU seed+1100000/1100001，全部掩码/置换及SHA保存；首末步更新前token及双LOO分数封存。audit110_masks.py独立CPU逐类LOO+logsumexp目标与双端梯度；另有教师/几何/邻居、全更新NumPy与查询计分封存检查。主干无训练梯度。
固定176199三步33预检，全部5审计通过后自动运行176100..104双权重145开发，PID215060 round110.log。查询分割、100步、原Adam/gsh.01根保持；主候选需同时超过当前根与双CE无KL至少.20pp再固定确认，不用错配下降作收益证明。当前仅预检，暂无开发准确率结论，不计新增正式记录。

R110预检已完成：33条全部检查通过，几何28条最大1.1086464e-05，更新12条最大token1.4901161e-08，双视图快照梯度20项最大5.5879354e-09；identity恢复两根预测一致。145条正式开发已启动，暂不计数。


### 2026-09-24：R110完成，KL未胜；R111确认双CE次要发现
145条开发全部五审计通过：5Direct/60邻居，几何120最大6.79493e-6；更新50最大token5.96046e-7/norm1.06581e-14；双掩码快照梯度80项最大6.75209e-9。
GraphMAE rdr69.92低于原Adam69.96与rdc69.96；gsh70.04、单掩码70.00、置换KL70.12，Direct70.84。GraphMAE2 rdr70.88 vsrdc70.92、原gsh70.64、Adam70.72、单掩码70.80、置换KL70.84、Direct70.84。双权重KL均较无KL双CE−.04pp，RDrop主机制未达超过双CE门槛，不升级、不宣称一致性有效。
按用户准确率优先目标，将GraphMAE2 rdc列为开发后选择的次要候选：对原gsh+.28、对单掩码+.12、对Adam+.20、对Direct+.08。这个选择不是R110预先指定的主发现，需另固定支持确认；不把无KL增益归给RDrop。
[第111轮固定设计](round111_design.md)：177100..119，GraphMAE2全部15方法100步300条，再原gsh与单掩码rds各200步双读出80条，总380条。复用110原源码/超参数/掩码流，只改变预算控制的steps；所有源/config与pair脚本已封存。主要rdc100对gsh100，其次对gsh200/rds200和其余控制；200步对照只粗略匹配GCN前向次数，不是严格FLOP或墙钟，且更新次数不同。每档独立5审计与跨目录source/checkpoint/split/support/query绑定。PID216976，round111.log运行；尚不计380。
历史普通全批51780+145=51925；R99范围1301/4排除、R106范围858/2排除仍另记。本地完整44235，原95/98归档队列仍运行。cleanup096删除通过5审计的33预检25,007,480B，证据manifest与审计保留；正式数据未删除。


### 2026-09-24：R111持续审计；R112干净视图JS候选原语
本轮重新检查R111进程216976存活，100步300条已完成预测，转移与几何240条（最大7.86781e-6）已通过，独立全更新仍在运行；尚未终止，不重启。之后由原批脚本自动执行快照梯度检查，再跑200步80条预算对照。未提前统计380全部验证或解释主差值。

已检索并读取[AugMix原文](https://arxiv.org/html/1912.02781v2)JS部分和[作者cifar.py](https://raw.githubusercontent.com/google-research/augmix/master/cifar.py)训练实现，形成[112草案](round112_design_draft.md)：只借鉴cleanCE+12JS(clean,masked1,masked2)，不移植图像混合操作链，不称完整AugMix或鲁棒性保证。相对110的双扰动对称KL，改变为含完整提示预测的有界三视图JS，但不同公式不是有效性证据。Mixup初轮已有失败，未重复同配置。
loss112.py/check112.py已在ProG环境执行。[原语审计](round112_primitive_audit.json)用独立熵差公式核对值及三个logit端梯度，最大6.21725e-15；相同三视图恢复CE且JS梯度0，极端logit有限并满足三分布JS上界log3。采用稳定logsumexp混合分布，相比作者代码省略概率下限截断，明确数值差异；本图有限cosine分数一般不触发截断的判断仍待实测。
候选拟保留三视图CE均值的同前向预算对照、节点错配JS与原强根；若正式验证需要扩大，补300步原根粗预算控制。当前仅损失原语和草案，真实GCN/首末快照梯度/掩码审计及种子manifest尚未封存，不计准确率结果。历史普通全批51925，范围计数、本地归档与清理096状态不变。


### 2026-09-24：R112三视图真实图实现启动
R112由原语推进至run_round112_js.py，继承110共享冻结2层GCN与一个768token，clean完整提示+两个连接掩码分别构造支持LOO分数。cleanCE+12JS为jsr，三视图CE均值jsc，真标签cleanCE+两扰动视图节点错配JS为jsp；jsi无掩码恢复两根仅预检。最终完整提示单前向，无查询标签训练和预测融合。
[封存设计](round112_design.md)、round112_manifest.json：178199三步29预检后178100..104双权重125开发，100步/Adam或gsh.01原根不变。掩码seed+1120000与置换seed+1120001独立CPU流；首末三视图分数、token、完整掩码/置换及全梯度都保存。五审计覆盖计分封存、教师引用、几何/邻居、全部更新、三视图首末梯度。CPU目标用独立熵差公式，与训练logsumexp实现不同；实测混合概率>1e-7才声称该图快照未触发官方截断。
PID218892 round112.log已启动，预检中，无准确率结论。主JS须超过原根与三CE至少.20pp才固定确认；不因正对照弱或错配下降就断言JS有效。若扩大还需300步粗前向预算控制。
R111两档预测已完成，100步300条五审计已过，200步80条转移/几何已过（80几何最大7.86781e-6），更新PID218791仍存活，之后快照梯度与paired111由原批脚本执行。没有重启或修改该实验。最终380未全部审计前暂不增加正式全批数，历史普通仍51925。归档95/98旧队列仍存活，本地完整44235与范围计数不变，清理最新096。


R111最终完成：100步300+200步80均通过原5审计，跨目录配对已完成。双CE100对gsh100+.06pp[-.07,.21]，对单CE100−.01[-.11,.09]，对RDrop100−.01[-.05,.03]；对gsh200+.21[-.02,.48]，对单CE200约0[-.17,.18]，对Adam100+.33[-.08,.72]，对Direct+.14[-.41,.70]。仅相对错配KL+.32[.03,.61]，不能证明超越强基线。开发+.28未复现为明确收益，未满足主要区间>0，不升级或重复扩大该配置。两次前向没有显示超越200步单掩码的特有效果。历史普通全批51925+380=52305，范围审计另列不变；R112预检仍进行中。


### 2026-09-24：R112完成，正确JS停止；错配JS次要确认
125条全部五审计通过：5Direct/50邻居，几何100最大5.72205e-6；更新40最大token7.15256e-7/norm1.06581e-14；三视图首末梯度60项最大2.23517e-8。审计的实际混合概率全部>1e-7，因此该批首末快照未触发作者实现的下限截断；不将这个结论扩大到所有中间步或其他任务。
GraphMAE：Adam69.76、gsh69.68、正确JS69.68、三CE69.76、错配JS69.76；GraphMAE2：gsh69.44、Adam69.20、正确JS69.48、三CE69.52、错配JS69.84；Direct69.76。正确JS在两权重均未超过三CE，停止主候选，不声称AugMix式正确对应一致性有效。
按准确率目标，将GraphMAE2错配JS列为开发后选择次要候选：对根+.40pp、正确JS+.36、三CE+.32、Direct+.08。它打乱了扰动节点对应，所以不能解释为原机制成功，也尚未证明其等同或优于熵正则；只检验固定损失控制能否提高准确率。
[第113轮设计](round113_design.md)：179100..119，GraphMAE2原112源码，100步13方法260条，再gsh300步双读出40条，合计300。主要jsp100对gsh100；辅助正确JS、三CE、Adam、Direct及gsh300粗前向预算，固定10000支持bootstrap。PID221068，round113.log。未通过全部审计及配对前不计确认或升级；主区间>0才考虑第二组。
历史普通全批52305+125=52430；范围R99=1301/4排除与R106=858/2排除继续单列，本地完整44235不变。cleanup097删除29条通过预检21,468,014B，保留清单与五审计。另检索几何中位数联邦聚合路线后核对第40轮已完成确认且无明确增益，没有原样重复。


### 2026-09-24：归档队列完成至98，Sparsemax门控原语
原95同步队列已正常完成，sync95_remaining_complete.json记录45870条传输SHA通过；随后96–98队列正常完成，sync98_remaining_complete.json及local_transfer_audit.json记录47085条、邻居引用10310与几何引用2115核验通过。旧PID37377/39329已结束，不重启。
cleanup098归档95四目录：round95_gate/confirm/direct6_dev/direct6_confirm，共2120文件、1,235,042,294B，经本地完整SHA与远端再次文件集合/SHA复核后，仅移除远端副本，本地正式证据留存。首次verify_archive098.py继承“文件数>100”的不适用于5/20条小批的防误选条件而中止，未删除；v2改为各目录精确result数170/680/5/20并保留全量SHA检查，才通过并删除。不是放宽科学实验阈值。服务器可用约8.9GiB。
新本地sync99_scoped.py PID44109开始顺序复制99开发/确认，原98完成标记作为前置。refresh_local_audit_v2.py另输出传输完整性与已知范围排除，99的1301通过/4排除不能混称1305全科学验证。预期全部传输48390条，其中99范围单列；脚本尚运行，当前已完成快照仍47085。新脚本同时为之后106的858/2排除预留明确计数，不修改原失败证据。

新读[Sparsemax原文](https://proceedings.mlr.press/v48/martins16.pdf)第2.1–2.3节/算法1，形成[114草案](round114_design_draft.md)。借鉴语言注意力概率单纯形投影，只改早层提示门控softmax(2score)到sparsemax(2score)，不改分类CE或新增终端融合。计划原幅度、范数匹配、仅幅度与identity控制，一个768token与冻结GCN。稀疏不构成准确率保证，也不因换算子主张原创。
gate114.py/check114.py在ProG执行通过；排序解对独立逐行二分阈值最大8.88178e-16，概率和/非负、argmax、平移与类别置换、均匀/onehot极限及门控范数和原根恢复检查通过。当前仅原语和草案，真实图及几何审计未接入、种子与manifest未封存，不计准确率实验。
R113两档预测已完成，100步组已通过5审计，300步组几何40已通过（最大6.19888e-6），更新PID222999仍存活；之后由原批脚本继续空mask检查与paired113。不重启、不提前计300全批；历史普通全批52430保持。


### 2026-09-24T01:09:02 第113轮停止；第114轮Sparsemax接入

第113轮300条正式记录完成。100步260条与300步40条的预测汇总、传输、几何、优化器及适用的mask审计通过。置乱JS候选相对同批100步grafted Shampoo为−0.71个百分点，95%支持种子bootstrap区间[-1.17,-0.29]；相对正确JS−0.74，三视图CE−0.88，300步根−0.83，Direct−0.57。开发集收益未复现，停止，不替换根方案。普通全审计历史累计52730条；R99的1301条/4排除及R106首确认858条/2排除仍单列。

第114轮借鉴Sparsemax概率单纯形投影；原语独立二分校验误差8.88e-16。已经接入冻结共享主干的单token门控，设计、源码、配置已封存，真实图准确率待跑。详见round114_design.md；不宣称稀疏性必然提高准确率。

归档：95/98队列完成，本地47085条封存SHA通过。99同步旧进程消失且无日志/完成标记，本地0条，原因未确定；确认无残留进程后重试，未计入归档完成。cleanup098已核验删除95远端重复副本1,235,042,294字节，本地完整保留。


### 第114轮正式核验完成：不升级稀疏门控

125条正式结果通过封存/传输、教师、独立二分投影与CPU几何、NumPy全步更新核验。几何重建最大误差1.1891e-5，40组优化器复算token最大误差5.9605e-7。普通全审计历史累计52855条，R99与R106的部分排除批次仍单列。

|权重|Adam根k50|grafted根k50|Sparsemax原始|等范数|仅幅度|Direct|
|---|---:|---:|---:|---:|---:|---:|
|GraphMAE|70.36|69.88|69.28|69.40|69.80|68.88|
|GraphMAE2|69.32|69.64|69.56|69.44|69.52|68.88|

无候选达成超过同批最强Prompt根至少0.20个百分点的预定筛选条件，不启动20种子确认。局部观察：GraphMAE不传播读出从68.48到68.96；GraphMAE2等范数不传播读出从67.96到68.72，但都低于既有k50基线，不据此替换保留方案。收益被传播读出覆盖仅为观察，不能推出因果机制。

归档99重试完成：sync99_scoped_complete.json确认本地48390条文件完整性记录，含范围核验1301条及4条排除记录；文件完整性不等价于全实验科学核验。sync105_remaining.py PID44430正在顺序同步100–105轮3105条，预期总计51495条，完成前不计数。服务器最近余量8.8GiB；本轮未删除新正式文件。

后续检索发现[ACL2019 Sparse Sequence-to-Sequence Models](https://aclanthology.org/P19-1146/)及[作者entmax实现](https://github.com/deep-spin/entmax)。可检验固定alpha=1.5的较温和稀疏门控，沿用幅度/等范数/identity控制。当前仅阅读摘要与定位来源，尚未完成公式/作者代码核对、原语实现或真实图运行；不得写作已验证方案。


### 第115轮启动：1.5-entmax门控

来源：[ACL2019 Sparse Sequence-to-Sequence Models](https://aclanthology.org/P19-1146.pdf)，已读3.2–3.4节、式13及算法1/2；作者[activations.py](https://raw.githubusercontent.com/deep-spin/entmax/master/entmax/activations.py)已保存原始文件与SHA。只借用概率映射，不移植分类损失或训练alpha。

输入仍为2倍既有教师score，固定alpha=1.5，概率为阈值截断后的平方；门控全节点中心化后作原图传播。一个6×128共享token注入冻结两层GCN第一层激活前，没有额外可训练分支。GraphMAE用Adam .01，GraphMAE2用grafted Shampoo .01。对照为原始映射、等门控范数、仅幅度、softmax恒等恢复；保留同批两根、Frozen、GraphPrompt和Direct。

原语与独立二分最大误差6.66e-16，与作者原始实现2.22e-16。当前仅验证固定前向，门控无梯度；不声称完整作者训练复现。

CiteSeer5shot，181100–181104开发支持种子，每个100步，125条；181199三步冒烟29条。数据划分沿用既有dev/confirm，重复查询造成探索性选择偏差，后续支持seed bootstrap不是独立图确认。依次封存、传输、CPU独立二分概率与几何、NumPy全步优化器审计。保存实际sparse_probability/SHA和零比例；gate_probability保留softmax参考。

预先固定：k50候选超过同批两Prompt根中更高者至少0.20个百分点，才启动20支持种子确认；多个达标选准确率最高，平局raw、matched、amplitude顺序。无达标停止。保留不传播读出的局部变化，不以它替代最强基线。

源码、配置、设计已封存，正式准确率尚未产生。归档100–105队列PID44430仍在推进，已同步到102，不重启。


### 第115轮完成并保留审计修正；第116轮确认启动

125条开发完成。GraphMAE k50：Adam70.44、grafted70.76、Entmax原始70.72、等范数70.56、幅度70.24、Direct71.24，停止此权重分支。GraphMAE2：Adam70.72、grafted71.36、原始71.56、等范数71.52、幅度71.28、Direct71.24；原始候选恰达+0.20个百分点门槛，进入20支持种子确认，不视为已确认提升。

原几何审计失败保留：181102/node2922的GPU softmax前两类完全并列，CPU参考差2.98e-8，Entmax保存概率同样并列；同一差异出现在三个门控变体。diagnose115_geometry.py与round115_argmax_diagnostic.json记录逐项概率。audit115_geometry_v2.py采用同设备保存概率的精确argmax一致、CPU概率regret<=1e-6，原概率/门控重建容差不变，另新增所有最终查询预测argmax完全一致。该修正是事后审计变化，原脚本、失败日志、manifest均未改写。v2全部100几何条通过，最大重建误差7.39e-6；40组全步优化器重算通过，token误差7.15e-7。普通历史核验累计52980条，包含本轮明确标注的修正审计；99/106范围排除批次仍单列。

第116轮PID227512运行，GraphMAE2、182100–182119，固定115原训练源码，100步260条；主比较原始Entmax k50减grafted根，CI下界>0才继续第二组。预注册与新manifest见round116_design.md / round116_manifest.json。尚无确认结论。

cleanup099：96/97四个远端重复目录共2010文件，经过本地/远端全SHA及删除前再次验证后移除778,859,343字节；本地正式证据保留，远端可用9.3GiB。100–105同步已传完各目录，PID44430后续正在做整体SHA核验，未收到完成标记前仍不提升本地核验计数。


### 第116轮停止；117邻域统计混合原语通过

第116轮260条全部封存、传输、几何v2及优化器核验通过：200条几何最大误差9.30e-6、80组更新token误差3.58e-7。主比较Entmax原始−grafted根为−0.09个百分点，95%CI[-0.39,0.24]，8胜12负，不进行第二组。相对Adam+0.47[0.08,0.86]、Direct+0.42[0.05,0.78]仅是辅助比较，不能替代最强根；相对等范数+0.07、仅幅度−0.20也无可靠胜出。保留原grafted根。普通历史核验累计53240条；R99与R106范围核验单列。

归档100–105完成：sync105_complete.json与local_transfer_audit_v2.json确认本地51495条文件完整性记录，已知排除4条，不把文件传输完整性当全实验验证。106–116新队列sync116_remaining.py PID44946同步3910条，预期55405条文件，包含99四条+106两条已知排除。

第117轮已读MixStyle论文及作者实现，提出邻域统计混合训练。详见round117_design_draft.md。数值原语通过，真实图训练尚未接入；旧原语数值误差与诊断输出边界均保留。


### 第117轮真实图实现与启动

设计、配置及源码已封存；183199三步冒烟29条，全部审计通过才自动运行183100–183104/100步的125条正式比较。增加有效扰动梯度快照及独立闭邻域统计重算，推理不扰动。尚未获得准确率结论。详见round117_design.md和round117_manifest.json。


### 第117轮完成；第118轮仅标准差混合确认

125条正式记录全部通过封存、传输、几何、更新及统计扰动审计。几何100条最大误差7.27e-6；40组优化器重算token误差8.34e-7；90个梯度快照（60有效扰动）最大差2.14e-8。快照最大标准差比例69.14、最大扰动RMS0.00825，仅作为数值诊断，不宣称所有步骤同界。普通历史核验累计53365条，另99范围1301/4与106范围858/2仍单列。

|权重|Adam k50|grafted k50|both|mean|std|Direct|
|---|---:|---:|---:|---:|---:|---:|
|GraphMAE|68.92|68.04|68.20|68.28|69.12|68.12|
|GraphMAE2|68.56|68.64|68.64|68.64|68.44|68.12|

仅GraphMAE标准差混合达到+0.20个百分点预定门槛，GraphMAE2停止。第118轮PID230731运行，固定GraphMAE/184100–184119/100步/260条，沿用原117训练与扰动配置；主比较std-k50减Adam-k50，CI下界>0才开展第二组。已封存round118_design.md、配置及manifest，尚无确认结论。

归档106–116队列PID44946仍在推进，最后观察到108开发完成；没有完成标记前不提升本地51495条传输核验数。本轮未删除正式文件。


### 第118轮确认停止；第119轮损失原语

118确认260条全部完成：200几何最大误差7.44e-6、80优化器重建token误差7.15e-7、189梯度快照含120有效扰动，最大梯度误差1.49e-8。主比较std混合−Adam为−0.11个百分点，95%CI[-0.25,0.02]；对grafted−0.30[-0.90,0.18]，对Direct−0.03[-0.69,0.63]。开发+0.20未复现，停止，不升级根。普通历史核验累计53625条，99/106范围批次仍单列。

检索Lookahead发现60轮已试，未重复。新读Bi-Tempered原文并完成loss119.py/check119.py，合成梯度误差3.13e-17以内，详见round119_design_draft.md；尚未接入图训练，不计准确率证据。

106–116归档PID44946仍活跃，已传至115，整体SHA核验未完成，当前本地已核验51495条计数不变。


### 第119轮接入真实图；归档106–116完成

Bi-Tempered三配置已接入相同共享768token与冻结GCN；首末步梯度用独立Brent概率根和解析Jacobian核验，保留不变推理。源码/配置/设计已封存，冒烟29条通过五类检查才启动185100–185104的125条正式实验，无准确率结论。详见round119_design.md。

归档sync116_complete.json已确认55405条传输完整性记录，known_excluded_rows=6（99四条+106两条），不把这些排除计为实验通过。普通历史核验53625条包括尚未归档的117/118共385条；两个计数口径不同。


### 第119轮冒烟通过、正式运行；清理100完成

29条冒烟通过五类检查，包括CE身份恢复及16个独立loss梯度快照（最大误差4.19e-9）。125条正式任务PID232275运行，最后观察GraphMAE2/185102。暂不增加实验完成计数。

cleanup100在完整本地/远端SHA逐文件核验后，删除98/99远端重复目录4110个文件，释放1,323,077,064字节；本地包含99四条既有排除记录在内全部证据保留，未更改范围验证结论。远端余量9.9GiB。

sync116完成55405条/6排除的传输核验；新sync118_remaining.py PID45524同步117/118共385条，完成后预期55790条/6排除，尚未计数。


### 第119轮完成：Bi-Tempered不扩大

125条正式结果通过全部检查：100几何重建最大误差5.96e-6，40优化器重算token误差5.96e-7，60损失梯度快照误差3.03e-9。快照loss梯度/CE梯度范数比0.489–1.168，不能概括成只缩小梯度。普通历史核验累计53750条，99/106范围批次仍单列。

GraphMAE k50：Adam69.32、grafted69.52、both69.40、bounded69.56、heavy69.44；最高候选只比最强根+0.04。GraphMAE2：Adam69.04、grafted69.68、both69.68、bounded69.68、heavy69.60。Direct68.72。无候选达到预定+0.20门槛，停止，不做确认，不替换保留配方。

后续继续从其他领域检索并设计新机制，先核对本地已试记录；没有第120轮已运行方案。归档117/118两目录已传输，整体SHA仍待完成标记；119正式文件尚未归档。


### 检索去重与第120轮Yogi原语


更正114前的检索记录：第81轮确实已测同一Sparsemax(2s)及范数控制，不能称114为首次。114的GraphMAE2参考已变为83轮graftedShampoo，但GraphMAE存在重复验证；这属于检索遗漏，不是新机制，既有实际结果保留。后续候选检索同时覆盖小写/英文与正文标题。

归档117/118完成：sync118_complete.json确认55790条文件完整性、6条既有排除。119新增同步PID45740，完成后预期55915条/6排除，不能提前计数。普通科学核验累计53750条保持不变。


### 第120轮真实图启动

Yogi两eps与三方向控制、额外Adam eps1e-3参照已封存。186199三步49冒烟通过后运行186100–186104/100步205正式，复算全步方向/更新及最终几何。尚无准确率结论。见round120_design.md。

119归档完成，sync119_complete.json确认本地55915条文件完整性、6条既有排除，文件传输状态与科学验证范围保持分列。


### 第120轮冒烟通过、正式205运行

49条冒烟通过封存、教师/传输、几何与全步更新核验，两eps的identity都恢复对应根的最终预测。PID234595当前运行205条正式任务，已进入GraphMAE2/186100；尚未完成正式审计，不增加科学核验计数。


## 第120轮结论：Yogi 未达开发门槛（2026-09-24）

205条正式记录全部完成预测、教师、邻域、几何和优化器审计；几何180项最大误差8.24035e-6，更新80项最大token误差4.17233e-7。GraphMAE最佳Yogi 68.48%，低于gsh 68.76%；GraphMAE2最佳68.60%，相对gsh 68.56%仅+0.04个百分点；Direct 68.76%。两checkpoint均stop，不进入确认。历史普通完整审计累计53955条；R99的1301与R106的858条分范围验证另列，6条排除保持不变。完整原始产物同步进行中，不能把传输验证等同科学审计。


## 第121轮启动：Fisher向量带符号幂归一化迁移（2026-09-24）

原论文§3.2式15已读取并保存，方案见`round121_design.md`。共享训练根，在最终全图中心化后、原型/k50构建前比较alpha=0.5/0.75，平方对照及alpha1恒等恢复；不是新增可训练Prompt模块。与R86输入教师幂变换位置不同。冻结/GraphPrompt与两优化器Prompt均纳入，正式345条，smoke85条；进程236314。必须先完成smoke全部审计，再启动正式。第120轮205条完整产物本地同步完成：传输56120条，仍有历史6条明确排除；普通完整审计累计53955条。


### R121冒烟审计修订v2

原始几何审计在alpha1的GPU传播得分逐位相等条件失败；所有16项比较的embedding SHA与预测一致，mean得分完全一致，8项k50得分最大差4.76837e-7。保留原始manifest、代码、round121.log；新audit121_geometry_v2检查相同embedding、邻居索引/权重、预测完全一致，重复GPU传播分数atol1e-6/rtol0。独立几何重建阈值未改。正式尚未启动时修订，run_round121_batch_v2进程237240重新审计既有smoke，通过后才跑正式；细节见round121_audit_revision.json。原语21项通过，正行缩放归一化误差2.22e-16。


### R121运行状态

smoke85条通过v2全审计：geometry80项最大9.53675e-6；更新10项最大token误差1.49012e-8，16个alpha1读出比较通过。正式345条预测已生成，正在末端几何与更新轨迹审计；当前不记入已审计累计，不提前筛选准确率。


## 第121轮结论与第122轮启动（2026-09-24）

R121正式345条全部审计通过：geometry320项最大1.03415e-5，更新40项最大token误差2.38419e-7；教师5项、邻域160引用通过。普通历史完整审计累计54300条。预定主候选GMAE/tr1幂0.5为68.48%，未超过较强gsh68.72%；GMAE2/gsh幂0.5/0.75均68.80%，低于原gsh68.88%，主检验stop。

事后对照GMAE/gsh幂2达到69.64%，比gsh68.72%高0.92个百分点，低于Direct69.80%。另立R122固定假设确认，禁止改写R121主结论。R122配置、设计、源SHA已封存：188100..188119，confirm查询子集，420条，进程239050；平方gsh为唯一主要候选，比较gsh/tr1/Direct及同幂Frozen/GraphPrompt，带平方根形状对照。只有对gsh配对bootstrap下界>0才保留匹配增益；未赢Direct不称最优，跨轮适应性与同图种子相关仍存在。

cleanup101完成：本地与远端全部2440文件逐文件SHA一致，远端删除前再验证文件集和SHA，删除R100/R101四正式目录的重复副本共938711972字节。本地完整产物、失败记录、审计均保留。R121正式产物正在本地同步；传输校验与科学审计分开记录。


### R122阶段状态及新增来源

420条预测完成，教师20项与180邻域引用通过，geometry360项最大误差7.74861e-6；优化器审计PID239756正在运行，不能提前记为完整通过或解读确认结果。盘余8.8GiB。新增`literature_followup122.md`核对Distribution Calibration作者代码，基类统计输入条件尚未落实，保留未运行状态。


## 第122轮结论：平方读出确认未复现，停止（2026-09-24）

420条全审计通过，更新60项最大token误差2.98024e-7。20新支持种子confirm主要比较平方gsh−原gsh为−0.12个百分点，95%支持bootstrap区间[−0.41,+0.16]；相对tr1 +0.17[−0.28,+0.65]，相对Direct +0.07[−0.39,+0.53]。开发的+0.92未复现，停止平方候选；不能用明显优于弱Frozen/GraphPrompt替代匹配增益。普通完整审计累计54720条。数据与源码保留，继续下一文献借鉴方向，目标仍active。


## 第123轮：AdamP条件投影前置诊断，当前配置无作用（2026-09-24）

已核对[AdamP原文](https://sanghyukchun.github.io/home/media/papers/heo2021iclr_adamp.pdf)的尺度不变前提及[作者实现](https://raw.githubusercontent.com/clovaai/AdamP/master/adamp/adamp.py)的channel/layer判据顺序。本地Prompt不满足论文尺度不变假设，测试限定为经验投影迁移到graftedShampoo，不称为官方AdamP复现。

`round123_design_draft.md`、`projection123.py`与`probe123.py`已封SHA，服务器不读查询标签重放R122的20条gsh轨迹，每条100步。20次首步零参数触发channel判据但方向变化为0，后续1980步均none；非零投影轨迹0/20。原语NumPy对照覆盖零、channel、layer、none，最大误差2.22e-16。原tokens回放误差通过既定3e-5界。保留`round123_projection_probe.json`和`round123_decision.json`；当前默认条件投影停止扩大，不重跑等价准确率。它是2000步无查询诊断，不增加正式预测计数，不声称已测试介入后的准确率，也不否定其他参数化下AdamP。

R121完整本地传输已完成56465条/6历史排除；R122的420条已传输，整体SHA刷新仍需检查最终marker。普通科学审计累计保持54720。后续继续借鉴其他领域方法，避免把不触发的分支当成新机制。


R122本地同步终态：sync122_complete.json通过，全量传输56885条、已知排除6条；科学审计计数与范围不变。R123原tokens回放最大误差实际为0。


## 第124轮启动：AdaMax梯度最大尺度优化（2026-09-24）

已核对论文AdaMax扩展与PyTorch官方数值定义，详见`round124_design.md`。核心以历史衰减梯度绝对值最大值替代二阶RMS，不对u做偏差校正；epsilon置于|g|+eps内。CPU双精度状态，tokens float32；保留原式、范数匹配、仅幅度三控制，同共享768参数、同冻结GCN、100步、lr.01。不泛化为完整调参后的优化器优劣。

原语8组NumPy递推、零/缺失梯度、范数与恒等恢复通过；与torch.optim.Adamax双精度100步比较最大误差8.326672684688674e-17。代码/配置/设计/审计在round124_manifest.json封存。PID240390，round124.log；29条smoke全部审计通过后才自动运行189100..104双checkpoint共125条正式开发。尚不计准确率证据。


R124 smoke29条全审计通过，geometry24项、updates10项及2个恒等根恢复通过；正式125条正在运行，不提前计入已验证累计。


## 第124轮结论：AdaMax开发未达门槛（2026-09-24）

125条完成全部审计：geometry100最大5.48363e-6，更新40项最大token误差4.76837e-7，教师5项和邻域50引用通过。GMAE原式64.12%、匹配64.32%、幅度64.00%，未超过Adam64.32%；GMAE2原式63.68%、匹配64.28%、幅度65.04%，相对gsh64.88%仅+.16个百分点，不达+.20门槛；Direct65.32%。两checkpoint均stop，不启动确认，也不把幅度控制弱正向解释为AdaMax方向收益。普通完整审计累计54845条，R99/R106范围与6历史排除不变。完整产物同步PID46956。


## 第125轮草案与原语：退火梯度噪声

已读2015原论文第3节式1，见`round125_design_draft.md`。采用首步干净梯度RMS定标，与原文绝对eta不同；标准差按(1+t)^−.275衰减。控制包含原式、匹配梯度范数、仅梯度幅度、恒定噪声、固定随机方向、恒等。不把梯度幅度对照误称参数更新方向保持，因为Adam/gsh状态亦受影响。

服务器`check125.py`通过六模式独立NumPy公式、相同seed复现、强度0恒等、零初始尺度、缺少全局RNG副作用检查，最大公式误差1.33227e-15。源码与结果封存round125_primitive_manifest.json。仍未接入真实图、未封正式配置与支持种子，不计准确率结果；下一步实现噪声快照与梯度/优化器独立回放后再启动。复查几何中位数/低秩文献路线发现R40、R36/37已有失败记录，不原样重复。


## 第125轮真实图接入与启动（2026-09-24）

`run_round125_noise.py`已实现反向后、优化器前注入。保存每步标准噪声与clean/noisy/used梯度，CPU独立随机流；首末步tokens/干净梯度快照由独立CPU图前向CE+正则梯度核对。新增audit125_noise并继承教师/几何/邻居/优化器回放；几何额外要求最终argmax与预测一致。正式设计与manifest已封存，不能再修改运行中来源。

190199三步57条smoke，全部审计通过后自动跑190100..104双checkpoint265条。两强度.1/.3×五模式，主要六候选r/m/a，c/f仅解释性对照；PID242259，round125.log。暂无正式准确率结果，不增加54845已审计累计。R124完整同步已通过57010条传输/6历史排除，科学与传输计数仍分开。


### R125冒烟通过与cleanup102

smoke57条全部完成审计：geometry52最大1.10865e-5；更新24最大token1.49012e-8，两个零强度恢复预测一致；噪声22组/44梯度快照，NumPy噪声误差5.20418e-18、CPU干净梯度误差6.51926e-9。正式265条进程243139已运行，父进程242259；全审计后才看门槛。

cleanup102已执行：R102gate/R103gate/R103confirm共2895文件，全量本地SHA一致后远端再验文件集与SHA，删除远端重复副本1825131815字节。本地证据与失败边界完整保留；当前服务器盘余9.8GiB。不减少历史科学审计计数。


## 第125轮完成：退火梯度噪声未达门槛

265条全审计通过：geometry240最大9.26853e-6，更新110项最大token误差5.96047e-7；噪声100组/200干净梯度快照，公式误差1.73473e-17、梯度误差4.19096e-9。GMAE最强候选幅度.3为69.24%，低于gsh69.32%；原式.1/.3为68.96/68.80%。GMAE2原式/匹配.3为69.40%，比gsh69.28%仅+.12个百分点，未达+.20门槛。Direct69.24%。两权重stop，不将弱微增推广为噪声有效。普通完整审计累计55110条；R99/R106范围与6历史排除不变。完整产物同步PID47313。

## 第126轮草案与尺度原语

已核对AdaCos原文第4节式13–15；CVF403后改读作者arXiv。见round126_design_draft.md，拟将人脸分类权向量替换为支持LOO原型，仅改变训练尺度，保留查询温度.2。不同于R77固定margin，不重复该失败配置。

scale126/check126服务器验证五模式（完整自适应、仅partition、仅角度、固定初始尺度、原尺度5）通过：15步独立NumPy递推误差0，解析CE梯度与冻结尺度一致，大尺度logsumexp稳定、非正尺度显式拒绝。源码/结果已封round126_primitive_manifest.json。目前没有真实图接入或准确率结果，下一步实现尺度快照与梯度审计再封正式配置。


## 第126轮真实图接入与启动

run_round126_adacos已实现支持LOO CE的动态尺度；首步s0，后续no_grad递推，保留原查询温度。保存每步基础logits、CE与尺度统计、首末步参数/梯度快照。新增audit126_scale独立NumPy尺度和CE、CPU真实梯度复算，全部实际梯度回放优化器。round126_manifest.json封存代码/config/设计/审计。

191199三步33条smoke，完整通过后自动运行191100..104双checkpoint145条正式；PID244619，round126.log。固定原尺度5恒等与固定s0控制保留，候选须超过同批tr1/gsh/acf最强至少.20pp。暂无正式准确率结论，不增加55110条普通完整审计累计。


R126 smoke33条全审计通过：geometry28最大8.94070e-6；updates12最大token1.49012e-8，两根恒等恢复预测一致；scale10组20梯度快照，CE误差1.08931e-7、梯度误差1.21072e-8，smoke尺度范围1.6176..9.9508（不代表100步范围）。正式145条子进程245602运行中。R125本地同步终态57275条传输、6历史排除，通过全量SHA刷新；科学累计仍55110。


## R126数值失败诊断与独立恢复批（2026-09-24）

原145条计划中途停止，不能记为完整完成。失败位置GMAE2/191101/acp（仅partition更新）。新增只读观测复现记录round126_failure_diagnosis.json：第79步尺度1.6785241217e12，128×128矩阵特征值最大1.0383e18、最小−44.77，触发原Shampoo正定断言。保留原run日志、config/source/manifest和全部已封存预测，不放宽检查、不补ridge、不把失败分支幸存种子子集选为候选。

恢复计划另封round126_recovery_design/config/manifest：原driver与审计脚本不改，另目录完整重跑原5支持和两checkpoint，但移除不稳定的acp对照，125条。仅acd/aca进入原门槛，比较tr1/gsh/acf；不混用失败批数据。PID246099，round126_recovery.log，已观察仍在运行。科学累计仍55110，恢复全部审计前不增加。

原失败批及诊断目录同步到本地forensics子目录，PID47579，逐文件全SHA校验；独立保存、不进入常规正式传输或科学累计计数，原远端证据暂保留。失败归因为该局部动态尺度的数值不稳定，不能等同完整AdaCos方案效果失败。


## R126恢复批结论与R127确认启动

恢复125条全审计通过：geometry100最大5.96047e-6，更新40最大token8.34466e-7；尺度30组/60快照，CE误差3.10794e-7、梯度误差2.32831e-9，尺度范围1.61690587..27.06114275。普通科学审计累计55235条。原失败97条预测/235文件已本地全SHA归档forensics，明确不在普通/常规传输计数中；不是97条都验证失败，而是整批未完成科学审计、仅作为失败过程证据保留。

GMAE acd69.72%对固定acf69.60%仅+.12，stop；GMAE2 acd69.60%对gsh69.28%+.32，对acf69.24%+.36，达到预定门槛。主候选固定GMAE2/acd，R127192100..119确认查询，260条，PID247155，round127.log。必须同时赢gsh和acf的配对区间才保留动态收益，另报告Direct；不在确认结果中重选variant。设计/config/manifest已封，原driver/audits不改。

R126数值诊断补充：失败acp的每步尺度比始终大于1.374，最后10步约1.4045；这是记录到的反馈放大，非仅孤立特征值断言。该对照固定cos(pi/4)分母，而完整动态使用当前角度分母，不能把两者数值行为混为一谈。

恢复批完整同步PID47700，期待常规原始传输57400/6历史排除，forensics97另列。


### R127期间资产与后续范围复核

确认批260预测生成、geometry200最大7.03335e-6，更新审计PID247861观察存活，暂不计完整通过。新增checkpoint_inventory_after127.json和checkpoint_remaining_probe.json：复核3类本地权重strictload/有限无标签输出，见checkpoint_followup_after127.md。R18/R48/R67已测试这些权重，后续若做则是当前强配置迁移，不称新权重发现；未运行新准确率实验。R126恢复125条全量本地同步完成57400条传输/6已知排除，forensics97独立保留。


## R127确认完成：完整AdaCos未满足保留条件

260条全审计通过：geometry200最大7.03335e-6，更新80最大token3.57628e-7，尺度60组120梯度快照，CE误差2.17010e-7、梯度误差2.09548e-9，尺度1.61387..17.72673。主要acd−gsh +.05pp[−.06,.17]；acd−acf +.12[−.04,.31]，均跨0，停止当前动态尺度配置，不启动第二组。对Adam+.26[−.04,.57]、Direct+.17[−.32,.71]亦不确立优势；对仅角度+.24[.08,.42]只是辅助，不能替代双主比较。普通科学累计55495条，原失败forensics97单列。完整同步PID47894，预计常规传输57660/6排除。

## R128现有强Prompt的跨checkpoint迁移筛选启动

设计见round128_design.md：SimGRACE/Edgepred_GPPT/Edgepred_Gprompt三迁移权重与GMAE/GMAE2强参照，同批Direct、原/中心化Frozen/GraphPrompt、tr1/gsh Prompt配对。不是新权重发现，也不重跑R67旧配置；明确检验当前低温门控/post-center/gsh组合的迁移。193199三步51条smoke全部审计后运行193100..104正式255条；原driver126不改，不使用AdaCos，审计阈值保持。PID248427，round128.log。必须胜过同批强权重与Direct，而非只胜弱Frozen才晋级，配置/manifest已封存。


## R128完成与R129角度间隔启动

R128 smoke51和正式255条全部审计通过，正式geometry200最大1.64509e-5，updates25最大token3.57628e-7；预检geometry40最大9.11951e-6、updates5最大1.49012e-8。三迁移权重SimGRACE/GPPT/Gprompt的最佳k50分别63.20/63.60/66.04%，同批GMAE Adam68.92%，全部不满足全局强参照+.20pp门槛，停止扩大。详见round128_pairs.json。普通科学完整审计累计55750；R99/R106范围排除边界、R126原失败97条forensics仍单列。R127全量本地同步57660/6排除通过，R128同步PID48062预期57915/6排除。

R129来源ArcFace原文III-A式3，设计见round129_design.md。同一冻结GCN与768参数Prompt，只替换支持LOO训练目标，角度margin.1/.3；完整梯度、逐样本停止间隔梯度、批次平均扣减三类分别检验，允许有效控制按用户效果优先晋级但不混淆机制。官方当前实现含no_grad变换，此处明确采用论文公式完整链式梯度适配。原语独立解析最大梯度误差1.38778e-17，margin0值/梯度精确恢复，非单调区间显式拒绝。

已封存配置/manifest并启动PID250229 round129.log：194199三步49条smoke全审计后，194100..104双checkpoint225条正式。仍须胜同批两权重四强Prompt与Direct最高均值至少.20pp；尚无正式准确率结论，不增加累计。


### R129预检恒等审计边界修订（原记录保留）

原smoke129训练/教师/几何/更新均完成，但跨运行m0与原根token逐位相同断言失败，原PID250229退出，未启动正式。diagnosis记录：identity最大token3.07337e-7、score1.43051e-6，预测全一致；单独重复原tr1/gsh最大token2.77535e-7、score1.43051e-6，预测亦一致，说明原GPU训练跨运行本身非逐位重现。

另封round129_recovery_design/manifest，v2保持原代码及数值/实际梯度/优化器/几何审计，新增每步保存logits上m0值/梯度精确恒等；跨运行token atol1e-6、score atol5e-6、rtol0并要求预测精确相同。此为有诊断支持的浮点复现边界修订，非隐去失败或改变算法。PID251443 round129_recovery.log先复核原49条smoke后才启动原225条正式，不重写原manifest。

R128本地同步全SHA刷新57915条常规传输、6已知排除通过。cleanup103对R104/R105的710文件、本地/远端全SHA一致后远端二次检查，清理290300812字节重复副本；本地完整正式证据保留。


R129恢复smoke49五项审计全部通过：geometry44最大8.65757e-6，updates16最大token1.49012e-8；angular14轨迹28梯度快照，CE误差2.42698e-7、真实梯度误差1.60654e-8，截断0，目标最小余弦.9741396。该高余弦是预检支持LOO训练量，不是查询准确率证据。正式225条已生成，仍待全审计通过才做门槛决策；累计暂维持55750。


## R129全审计完成：停止当前角度间隔组合

225条全部通过：geometry200最大9.65596e-6，updates70最大token7.15256e-7；angular60轨迹120快照，CE误差3.69495e-7、实际梯度误差4.93601e-8，截断0，目标最小余弦.8524191。前次分析在末项审计仍运行时发现文件未生成，仅为尚未结束；随后进程正常完成，全审计通过后重新分析，无重启。

同批最高参照Direct70.64%，所有候选未达到+.20pp门槛，stop：GMAE完整角度.3为70.24%，其两控制69.36/69.28%；GMAE2最高均为平均扣减70.40%，仍低Direct。记录条件局部差异但不进入20支持确认，不把收益归因角度几何。详见round129_pairs.json。普通科学累计55975；同步129启动，预期常规58140/6排除。


## R130固定特征目标点积回归：封存与启动

新读NC-FSCIL4.1/4.2式1/4、视觉跨域固定分类器与最近ETF研究，详细适配/区别见round130_design.md。不同于R24的token Gram约束，本轮在共享冻结GCN最终节点表示上增加支持点积回归，仍只训练768token；初始支持均值最近ETF固定后不再更新，query保持原型/k50，不添加终端分类融合。aligned/原始centers为主候选，错位/随机等角作诊断；lambda1/10，基础CE保留。

原语独立NumPy全图中心化梯度最大误差5.20418e-18，SVD/Gram特征分解目标一致、ETF Gram/迹上界/重复确定性/有限差分通过。源码/config/manifest已封存。PID253513 round130.log：195199三步57预检全部审计后运行195100..104双权重265条正式；两权重四强Prompt与Direct仍为全局晋级参照。没有新准确率结论，普通科学累计维持55975。

R129本地完整常规同步终态见sync129_complete.json，58140条传输/6历史排除；R126失败forensics97仍独立保留、不计普通。


R130入口工程修复：初始入口误保留smoke129目录，source/config保护拒绝恢复，原PID253513退出，未生成R130预测。smoke129 config本地/远端SHA c98706dead1b3bccbd53df7cdd67a39da05bf8fb6b529b76ff24da245025f379一致，原批未改写。保留原入口/manifest/日志，另封run_round130_v2与entry_v2_manifest，仅修正目录至smoke130；driver/矩阵/审计不变，PID253720 round130_v2.log。该工程失败不属于方法效果失败。


### R130审计入口遗漏纠正：不以旧脚本passed替代覆盖

v2入口实际仍调用audit129_angular，checked0未覆盖frame，故不能认定57预检完成全部审计；前面的“先全审计再正式”为计划，实际入口未满足。已核对真实PID253720与子进程254922（旧transfer审计），终止旧入口；其正式预测已全部生成，但专用目标/真实辅助梯度未审计，因此该批不计普通科学完整记录，也不按效果失败处理。

原run130/entryv2/source/config/manifest/log及round130_gate全部保留；本地forensics嵌套独立同步，常规传输计数排除。v3另封入口和说明，改为实际audit130/geometry/frame，先重新完整审计原smoke13057条，并核对frame应18轨迹36快照；之后完整全新round130_recovery265，不混用旧批。PID255100 round130_v3.log。训练driver/方法/数值容差均未改。科学累计55975、常规传输58140保持。


### R130新专用审计的零权重传播边界

v3真实frame审计在独立GPU训练的零权重传播score检查停下，recovery正式尚未启动。diagnosis：GMAE token8.38190e-8、mean score1.07288e-6、k50 score.00659275，预测均一致；两权重近邻集合10/16行变化。不可把.0066描述为单纯末位分数误差，保存此邻域敏感性。

新增同输入CPU单线程完整配对训练diagnose130_identity：实际原tr1/gsh函数与weight0各3步，参数、全节点表示、独立构造k50后的传播score均逐位一致，两权重通过。v4改用此严格功能恒等验证，保留GPU跨运行参数/原型score原小容差和预测一致；每条GPU仍通过原独立几何与邻接审计，不放宽目标/梯度/更新阈值。专用frame_v2新增覆盖断言，smoke18轨迹36快照、正式80/160；PID256371 round130_v4.log运行中，原失败全部保留。见round130_entry_v4_note.md。

旧入口正式265条为完整预测但缺专用审计的工程失败批，本地forensics/round130_gate已全SHA归档（详round130_forensics_sync.json），明确不进入普通科学或常规传输计数；不是265条均被判定效果失败。普通55975、常规58140/6历史排除未变。


R130 v4现已真实通过smoke专用18轨迹/36快照覆盖：DR值最大7.37368e-8、总梯度4.17233e-7、目标重建3.42791e-7；其余geometry52/update20通过。正式恢复265条实际运行，PID256371新鲜观察存活。补充audit130_coverage逐项验证两批完整预设键集合、每条预测SHA/argmax、所有frame轨迹步数/快照和audit覆盖计数，再由finalize130分析，避免空集合passed替代验证。该补充不修改已封存训练/审计版本。


## R130恢复完成与R131固定候选确认

恢复265全审计通过：geometry240最大1.03713e-5；updates90最大token5.96047e-7；frame80轨迹/160快照，DR误差9.37143e-8、实际梯度5.96047e-7、目标重建5.47740e-7。另audit130_coverage核对57预检/265正式完整键集合、SHA/argmax、每轨迹100步和所有审计覆盖数，passed。原失败入口265条留在独立forensics，不混入。普通科学完整累计56240。恢复批同步已启动，预计常规58405/6历史排除。

按预设资格选中GraphMAE2原始中心ftc1+k50=70.60%，同权重gsh70.12、开发最强GMAEgsh70.20、Direct70.08。alignedETF70.48不胜中心对照，因此不称ETF收益。GMAE随机目标诊断70.80高于主候选，但预先规则未纳入主候选；不事后替换，后续若有价值需新固定确认。

R131固定主候选GraphMAE2/ftc1，196100..119确认，两权重原/中心化对照、Adam/gsh、四目标weight1和Direct，共740条。双主比较同模型gsh+GMAEgsh，配对区间下界均>0才进入第二组；Direct另报，辅助比较不替代主条件。配置/源码/明确覆盖数/配对脚本已封存，PID258377 round131.log；原driver不变。详见round131_design.md。

cleanup104完成1815文件/1335972665字节的本地远端全SHA匹配及远端二次验证，删除R108/R109/R110远端重复副本，本地全证据保留，磁盘观察9.6GiB可用。


R131期间后备检索：新读VICReg第4.1节及官方实现，形成variance_covariance_design_draft.md。只借鉴单视图方差/协方差辅助训练，不含双视图invariance，不复现完整VICReg；与R32/33固定白化读出区分。服务器原语解析误差8.32668e-17、有限差分/平移不变/零输入有限通过。当前尚无真实图训练和准确率，不新增正式计数。当前确认优先。


R130恢复备份终态58405条常规传输/6历史排除，全SHA刷新通过；R12697/R130265工程失败forensics均另列。R131740条已生成，geometry640最大1.26958e-5通过，余下更新/真实辅助梯度/覆盖及配对尚待完成，父PID258377仍存活，不能提前计完成。

后备VC真实零Prompt梯度诊断（不更新、不查询准确率）：GMAE联合辅助/CE梯度范数比106.99、方向余弦−.450；GMAE2比119.49、余弦−.108。原草案lambda.1/1会明显主导初始梯度，后续拟据此改.001/.01；这不是准确率失败或有效证据，单一支持初始比例亦非全轨迹保证。原草案封存保留，修订说明见variance_covariance_probe_note.md，真实训练尚未运行。


## R131确认完成：固定中心DR停止

740条完整通过全部审计与覆盖计数：geometry640最大1.26958e-5，updates200最大token7.15256e-7，frame160轨迹320快照，DR误差1.14069e-7、梯度1.06870e-7、目标重建4.56551e-7。普通科学累计56980，原失败forensics另列。R131完整同步启动，预期常规59145/6排除。

固定主候选GMAE2 ftc1相对同权重gsh −.22pp CI[−.49,.03]，6胜9负5平；相对开发最强GMAEgsh −.59[−.9903,−.26]，2胜13负5平；相对Direct −.11[−.54,.32]。两个主条件未通过，停止当前固定中心DR，不启动第二组，也不在辅助比较中换主候选。相对GMAE2 Adam+.01[−.36,.40]，对alignedETF−.08[−.17,.02]均无新增结论。开发+.40没有在此确认维持，不能报告稳定提升。

后续继续VC草案：原语和真实初始梯度诊断完成，但完整训练/审计接入未实施。正式候选拟.lambda .001/.01以避免未标注初始辅助梯度压倒CE；需重新封存完整设计和强对照矩阵后运行。当前没有在跑的GPU正式训练，不把草案写成实验完成。


## R132方差/协方差辅助训练：正式接入并启动

新driver132在冻结共享GCN/原768token的支持CE上增加单视图25V+R、25V或R，固定lambda.001/.01；初始r0无偏方差尺度detach、全图无标签统计、原gate/读出不改。与已确认失败的固定中心DR分开，不叠加失败项。详细公式、源文献与初始梯度选尺度依据见round132_design.md；新增vc132与已解析校验草案在值/梯度上逐位一致。

每步保存协方差/三个辅助标量/基础logits和CE/实际总梯度，首末token快照。新增独立NumPy标量、CPU真实协方差与总梯度，独立CPU原根/零权重全路径恒等。入口显式调用audit132系列并在每阶段检查完整键集合、来源/预测SHA、argmax、每轨迹长度及非空覆盖。smoke预期49条geometry44/update16/VC14与28快照；正式225条geometry200/update70/VC60与120快照。

配置/source/manifest已封存，PID260325 round132.log：197199三步49预检生成完成，当前在严格恒等/专用审计；全部通过才运行197100..104正式。尚无真实图正式准确率结论，普通科学56980不变。R131全量备份终态59145常规传输/6历史排除通过，失败forensics另列。


R132 smoke49已全部审计与覆盖通过，CPU两根零权重参数/表示/k50分数逐位恒等；VC14轨迹28快照：{"checked": 14, "snapshots": 28, "max_scalar_reconstruction_error": 3.094085158750204e-06, "max_gradient_error": 9.313225746154785e-09, "max_covariance_reconstruction_error": 2.86102294921875e-06, "min_weighted_aux_to_ce": 0.0, "max_weighted_aux_to_ce": 0.3235649153408792, "passed": true}。正式225已实际开始，父PID260325存活；预检准确率不进入正式结论，普通累计暂不增加。


## R132完成：停止当前方差/协方差配置

225条正式及49预检全部审计/完整键覆盖通过：geometry200最大1.01328e-5，updates70最大token5.96047e-7，VC60轨迹120快照，梯度最大1.72877e-8、协方差重建4.57764e-5。标量最大绝对误差.000697664出现在GMAE/197100/vcv10第89步未用于方差单项损失的covariance_value，约7017.865；按原atol1e-5+rtol1e-5容差归一化误差.00994，全体最大归一化.01564，非放宽阈值。详round132_scalar_diagnosis。

最强根GMAE tr1 70.16%，最佳候选方差单项.01为70.24%仅+.08；GMAE2方差.00170.08与其gsh持平。联合/协方差.001分别GMAE67.36/67.40、GMAE2 67.80/67.92；.01更低。未过+.20pp，停止当前配置，不调查询后再追尺度。普通科学累计57205，完整同步启动，预期常规59370/6历史排除，失败forensics另列。

## 后续转向Cora固定门控锐化迁移

核对当前Cora强方案与真实run_round53_focal源码：scale2是gamma2导数系数均值stopgrad乘CE，不是简单乘2；后层896token、平滑岭KD、原始几何k50必须保留。重新读DINO官方teacher温度代码，只借鉴固定锐化思想。Cora草案在平滑概率Q上固定平方归一化，仅门控改变、KD仍Q；raw/matched/amplitude控制，避免把CiteSeer原始教师、中心化、早层配置整套搬来。

原语NumPy误差1.52656e-16、power1精确恢复通过；真实198199支持无query诊断：门控范数比1.465964、熵1.37858→.74290，两checkpoint strictload/有限输出。未训练Prompt或测准确率，不能计提升；完整真实训练与审计待接入，详cora_sharpen_design_draft与probe_note。


### R133 启动：Cora 固定门控锐化（2026-09-24）

迁移 DINO 概率锐化到 Cora 原强方案的门控，保留后层注入、动态 scale2 CE 和原教师 KD；新增 raw、等范数 matched、纯幅度 amplitude 三个分支。设计见 `round133_design.md`。正式145条、冒烟33条。原legacy与power1在CPU单线程3步token、embedding和k50精确一致。首版冒烟审计遗漏传播前的首次支持标签钳制，误差0.00169456，门控阻止正式启动；保留原审计/日志，新增 `audit133_v2.py` 与入口v2修复审计，训练和容差不变，复核原封存冒烟后才允许正式训练。PID264327是v2实际句柄。

R132本地同步已完成，累计regular transport 59370条，其中已知科学排除6条；普通科学审计累计57205，另有forensics不计入。


### R133 完整结果：Cora锐化未超过原强根，停止扩大


| checkpoint | 原scale2 k50 | raw锐化 | matched锐化 | amplitude控制 |
|---|---:|---:|---:|---:|
| GraphMAE | 74.20 | 72.96 | 73.68 | 73.96 |
| GraphMAE2 | 70.44 | 70.52 | 70.20 | 71.16 |

Direct 69.44%。最强候选GraphMAE amplitude比全局强根低0.24pp，五种子0胜4负1平，停止确认。GraphMAE2 amplitude同权重+0.72pp属于局部正向，但远低于GraphMAE原根，不作全局成功。真实锐化没有优于纯幅度控制。普通科学验证累计57350条；当前本地全量同步PID49924，目标regular transport59515条（已知排除6），尚不能在同步完成前宣称全部本地落盘。

存储维护cleanup105：对R112/R114/R115完整远端文件逐一与本地SHA/字节数匹配后，远端再次复核再删除副本，915文件、360136802字节，本地完整证据保留；manifest和清理凭证均保存。没有删除失败现场。

下一候选检索DIST相关性蒸馏：本地已做R25 DKD、R65距离RKD、R66角度RKD，避免重复。下一步是核对DIST官方实现与Cora KD约束，尚未实现或训练。


### R134准备进展与R133备份收尾

R133完整同步已完成，`sync133_complete.json`核验regular transport59515条（已知科学排除6）；普通完整科学审计57350条，forensics另计。远端剩余8.1GiB。

DIST已按官方子模块固定commit读取并保存；原语官方loss/gradient、NumPy和gradcheck均通过。Cora无查询诊断显示both相关梯度相对KL在GraphMAE为2.66倍、GraphMAE2为15.73倍，下一步补固定初始梯度范数匹配控制后再封存训练矩阵。草案见 `round134_design_draft.md`，诊断不是准确率结果；尚未完整训练。


### R134启动：完整相关性KD对照

设计已封存 `round134_design.md`，45条冒烟/205条正式；raw both与初始token梯度范数匹配的both/inter/intra，另有无KD/双KL控制。所有候选只替换KD，原Cora门控/后层/scale2 CE/读出不变。梯度匹配是每run T=0计算的固定系数，不利用query标签；全轨迹回放、首末独立相关KD梯度和初始梯度重构已接入。启动PID266344，先legacy恒等与冒烟，通过全部45/16轨迹/32快照后才正式。


### R134完成：DIST未超越最强根，停止扩大

205/205审计通过，70条完整更新回放、140首末快照、100组k50；最大几何/损失重建误差3.81470e-6，token5.96046e-7，gradient1.75089e-6。`round134_pairs.json`：GraphMAE原75.52，raw DIST75.24、范数匹配both75.00、inter75.08、intra74.84；无KD73.64、双KL75.60。GraphMAE2原74.00、raw75.00、匹配both73.80、inter73.88、intra72.84；双KL74.20，无KD71.44。Direct72.04。

最优双KL相对全局原根仅+0.08pp，2胜3负，未达0.20pp门槛；DIST本身未超越强根。GraphMAE2 raw同权重+1.00pp作为局部观察保留，但梯度匹配后无效，不归因为相关性方向独有，也未超过GraphMAE。停止当前配置。普通科学累计57555条；全量同步PID50321目标59720传输记录（6已知排除），forensics另列。

后续本地检索排除了已完成的R40几何中位数、R58跨checkpoint集成和R26/78梯度冲突路线。拟将已有CiteSeer正向优化器grafted Shampoo迁移到Cora真正强根，保留后层+scale2+KD，和同影子Adam范数的对角/原梯度方向对照；只借鉴方向/幅度分离，不把CiteSeer结果当Cora证据。


### R135启动：Cora优化器迁移

固定设计 `round135_design.md`，原R83优化器源码不改，保持Cora原late+scale2+KD+未中心化k50。gsh矩阵方向、gdi对角方向、ggd原梯度方向各以自身影子Adam范数归一；osh原矩阵未归一参照。37冒烟/165正式；50正式更新轨迹与100首末梯度快照，NumPy逐步特征分解回放与范数核验已接入。PID267617，零梯度/动量原语和CPU根恒等已过，待冒烟审计后正式。R134全量本地同步已完成：`sync134_complete.json`明确59720传输条目、6已知排除，科学累计57555。


### R135开发完整通过，固定GraphMAE gsh进入确认

165条全审计通过，50更新轨迹/100梯度快照/80近邻重构；37条smoke及gad原Adam恢复最大mean分数误差1.43051e-6通过。开发GraphMAE gsh75.96对原scale2 75.64（+0.32pp，3胜1负1平），Direct73.44（+2.52）；达到原封存门槛。**同批旧普通CE token75.88，比候选仅低0.08，尚不能宣布超过最强方案。** GraphMAE2 gsh75.56、对角75.92、原梯度75.96；简单方向几乎同样好，不能宣称完整矩阵机制独有。

确认设计 `round135_confirm_design.md` 已封存：选定GraphMAE gsh，不事后换选GraphMAE2；新20支持201100–201119、confirm后500查询，保留两ck token/scale2/gsh/gdi/ggd和Direct等，共580条，期望160更新轨迹/320快照。主要判断要同时超过两ck的token/scale2及Direct，配对10000bootstrap95%下界均>0；不仅比较scale2。源码不变，原37条smoke复用，实际PID268563运行中。确认未完成，不计有效累计。普通科学累计57720；开发同步PID50613目标59885传输条目。

cleanup106完成：R116confirm/R117gate/R118confirm共1555文件、603694022字节全部远端/本地SHA匹配且远端再次核验后移除副本，本地完整证据保留。


### R135确认完整完成：开发增益未复现，停止当前配置

580/580全部审计通过：160条矩阵更新轨迹、320首末梯度快照、280邻域重建；最大重建3.81470e-6、token4.76837e-7、gradient8.94070e-8。见 `round135_confirm_audit135.json`、`round135_confirm_pairs.json`。

固定GraphMAE gsh76.31%，原scale2 76.45%，差−0.14pp，95%配对支持bootstrap[-0.69,0.38]，9胜9负2平；旧CE75.88%，差+0.43[-0.07,0.97]；Direct72.09%，差+4.22[3.13,5.40]。虽超过直接教师与更弱checkpoint，但未明确超过本来强根，不能凭弱参照升级。GraphMAE2 gsh74.10与本权重原scale2 74.10相同，对角74.09/原梯度74.11；不另事后换选。停止本次grafted迁移，不进入第二组确认。

本轮普通科学累计58300；R135开发备份已核验59885传输记录（6已知排除），确认全量同步PID50813目标60465，未完成之前不声称全部本地落盘。范围科学2159与forensics继续单列，计数口径不变。

后备R136读取Muon作者文章/官方固定commit并写入 `round136_design_draft.md`；官方适用隐藏权重而非embedding，本项目token仅为实验性迁移。CPU double五次矩阵多项式与官方bf16区别明确；9原语case的NumPy误差9.26e-15、SVD谱多项式重建5.50e-14，零/秩亏/转置/输入不变通过。尚未接入真实训练，下一步先做Cora初始实际梯度与零空间诊断，不把NS5说成精确正交或把原语通过当准确率证据。


### R136接入并启动完整训练

真实Cora初始梯度诊断完成（不读query标签），NS原方向范数约2.12–2.30而Adam约29.93；NS放大浮点近零类和方向，占比约1.5e-5，明确不宣称严格零空间不变。矩阵优化器新增原幅度、匹配Adam范数、无momentum、无NS、纯幅度及identity，输入梯度不修改。设计 `round136_design.md` 已封存：45冒烟/205正式，70更新轨迹/140首末快照，源码与独立NumPy审计分开。扩大门槛纳入两ck token/scale2/gsh最高者，避免只超过一个较弱根。PID269647启动，尚无正式结论。

上一批确认本地备份已完成：`sync135_confirm_complete.json`验证60465条regular transport、6已知排除；普通科学累计58300不变。


### R136开发通过并固定确认

205条完整审计，70轨迹/140快照/100近邻；最大重建3.81470e-6，token5.36442e-7，gradient5.96046e-8。GraphMAE nmm77.36，原scale2 76.88、旧CE76.72、gsh76.12；相对最强+0.48pp（4胜1负）。无momentum NS77.12、无NS动量76.92，raw NS74.48、纯幅度71.08；GraphMAE2 nmm75.84低于本权重原scale2 76.48。Direct73.32。结果见 `round136_pairs.json`；开发不替代确认。

固定GraphMAE nmm，20新支持203100–203119、confirm后500查询共660条；`round136_confirm_design.md`包含两ck token/scale2/gsh七强参照（含Direct），以及无NS/无momentum对照。主差异区间下界须全部>0；不事后换选。PID270411已启动，200更新轨迹/400快照/320近邻必须全部通过。普通科学累计58505，确认不提前计数；开发同步PID51061目标60670。

同时检索SOAP语言模型优化论文与官方preliminary源码作为后备，发现其首次梯度只初始化特征基而不更新参数，不能直接叫100次参数更新；需明确预算与初始化对照。当前只做来源核对，未修改当前R136或启动SOAP训练。


### R136固定确认660条完成：未超越强根，停止扩大

660条、200完整更新轨迹、400首末梯度快照、320近邻均独立审计通过；最大重建5.48363e-6、token5.96046e-7、gradient8.19564e-8。固定GraphMAE nmm77.79%，原scale2 77.74%，增量+0.05pp，95%支持配对bootstrap[-0.66025,0.71]，11胜8负1平。旧CE77.25%，差+0.54[0.00,1.10]；gsh77.17，差+0.62[0.14,1.11]；Direct72.71，差+5.08[3.92,6.31]。虽然胜过部分参照，但最强原方案的区间跨0，主判断失败，不进入第二组确认，不升级主根。无momentum NS77.21、无NS动量76.29；这些对照关系保留，不替代最强根比较。

开发+0.48未可靠复现，两个阶段分开记录。详见 `round136_confirm_pairs.json`、`round136_confirm_audit136.json`。普通科学累计59165；确认完整本地备份 `sync136_confirm_complete.json`已核验regular transport61330条，其中6已知排除，2159范围核验与forensics仍单列，未混算。

cleanup107：R119/R120/R121远端副本经本地完整SHA/字节比对、远端再次核验后删除，共1665文件/1228572000字节，本地证据全部保留，清理清单已保存。

SOAP后备进入独立原语验证：已读原文算法3/4、固定官方源码commit和21次原生CPU合成调用；独立NumPy回放20次参数更新与两次QR刷新通过，参数误差8.59e-8、基残差5.67e-7。明确验证实际合法基而非要求简并特征向量位精确一致；原始Q与排序保存。见 `soap137_replay_note.md`。尚无图训练/准确率，下一步先真实Cora梯度诊断，再处理首次no-op与99/100更新预算对照。


### R137 SOAP完成：开发小幅上升，未达到扩大门槛

165条正式、50条更新轨迹、100份首末梯度快照、80近邻及30条SOAP全状态回放全部通过。SOAP状态最大误差9.31323e-10。GraphMAE自适应SOAP73.52%，原scale2 73.40%，旧CE73.32%，固定基73.28%，单位基及Adam99均73.36%；Direct71.44%。最优+0.12pp，支持配对差[0.8,-2.2,0.4,-0.4,2.0]，未到预设0.20pp，不进入扩大确认。GraphMAE2 SOAP67.64%，本权重原scale2 72.88%，固定基71.24%，单位基/Adam99 72.84%，跨checkpoint不稳定。不能称可靠提升，原方案保留。详见round137_pairs.json、round137_gate_audit137.json。

原float32真实梯度probe独立回放差2.23163e-4失败；原生同运算复算误差0，投影舍入被小分母放大。保留旧文件/日志，不放宽容差；新v2优化器状态double、真实token仍float32并逐步回舍入，六条无query诊断最大误差1.16415e-10后才训练。方法负结果与数值适配过程分开记录。

普通科学累计59330；本轮local sync137正在核验，目标61495 regular transport，6已知排除，未完成前不宣称全部落盘。下一方向Prodigy已固定官方源码和逐步更新顺序，见round138_design_draft.md；尚未全文算法审查、原语/图训练，当前无远端训练进程。


R137本地备份完成：sync137_complete.json核验61495条regular transport、6已知排除。165正式证据完整落盘，普通科学累计59330；未删除任何本轮远端文件。


### R138 Prodigy从原语到图训练

已核对原文算法4与固定官方实现，记录旧/新d及epsilon顺序差异；官方commit 3efb213ee8af5a6bf76f28726398433a847b38e9，源码SHA见round138_design_draft.md。8组合合成全状态独立回放通过（120调用118更新，含零梯度）；两ck×四模式真实Cora梯度100调用100更新，最大误差7.27596e-11，无query准确率评估。默认与bias-corrected、固定d0与固定.01控制齐备，CPUdouble状态每次回写float32 token。

正式设计round138_design.md已封存：33 smoke→审计→165 dev→完整审计，PID272511。开发205100–205104，保留两ck token/scale2强根、Direct；50更新轨迹/100首末快照/40Prodigy状态/80近邻须全通过。尚无正式结果，不提前增加累计59330。


### R138开发通过，进入固定20支持确认

165正式全审计通过：50轨迹/100快照/40Prodigy状态/80近邻；最大重建4.05312e-6、token5.96046e-7、gradient7.45058e-8。GraphMAE pbc76.68%，同ck scale2 75.64%、token75.48%、默认Prodigy76.16%、固定.01 75.64%；全局原强根GraphMAE2 token/scale2均76.40%，Direct74.64%。最佳相对最强+.28pp（2胜3负）、Direct+2.04，达到扩大门槛但一致性有限。GraphMAE2 pbc73.52明显弱于自身原方案。

固定GraphMAE pbc，confirm支持206100–206119共660条，PID273253，配置/设计/分析器已封存；七主参照含两ck token/scale2/pf1及Direct。200完整轨迹/400快照/160Prodigy状态/320近邻需全审计。bootstrap所有主差异区间下界>0才进入第二组；不能事后换候选。普通科学累计59495，确认不提前计数；本地sync138 PID52036目标61660仍在核验。

cleanup108核验并删除R122confirm/R124gate/R125gate远端重复副本：1985文件、1684310848字节；本地完整SHA/字节匹配后，远端再次检查所有文件才删除。完整本地证据保留，磁盘剩余8.5GiB。


R138确认入口修正：原PID273253因训练文件名漏.py在训练前退出，无预测结果；旧入口/manifest/日志保留，run_round138_confirm_v2.py与新manifest仅修正调用文件名，训练源码和实验选择不变。新PID273326已启动，后续以v2日志为准。开发sync138_complete.json核验61660 regular transport、6已知排除，正式科学累计59495；确认仍不提前计数。


### R138固定确认失败：停止扩大Prodigy

660正式全审计通过：200更新轨迹、400首末梯度快照、160全状态回放、320近邻；最大几何重建5.48363e-6、token5.36442e-7、梯度6.70552e-8。固定GraphMAE pbc75.65%，原scale2/pf1均76.67%，差-1.02pp、支持配对95%区间[-2.07,-0.07]，6胜14负。旧CE76.28%，差-.63[-1.67,.35]。Direct73.06%，差+2.59[1.79,3.47]不构成超越最强根证据。默认Prodigy74.77%；GraphMAE2 pbc71.43%，其原scale2 75.15%。

primary_pass=false，开发+0.28没有在新支持/confirm查询复现；停止第二组确认，不改选其他候选，不升级现有方案。结果round138_confirm_pairs.json。普通科学累计60155。确认本地sync138_confirm PID52290正在进行，目标62320 regular transport、6已知排除；当前已核验61660，不提前宣称全部备份。

后备R139 Schedule-Free已读取原文算法1与官方Reference/fast实现，固定commit70785b53e778d0e872c0bbb75ff4ee54ee10c291。8组合合成100步x/y/z/v及train/eval回放通过，最大8.88178e-16；fast会原地改grad，未来记录必须在step前。进一步用两ck×sf0无warmup/sfw10步warmup/sfz在z求梯度三模式，真实Cora100调用无query准确率诊断通过，最大状态误差9.99201e-16。optimizer139.py保持CPUdouble x/y/z/v，模型梯度点为float32舍入y，最终token是float32舍入x；已独立核验，不能把训练y当评估x。见schedulefree139_graph_probe.json。尚未正式精度对照，计划补足匹配warmup Adam和同轨迹y读出后封存正式矩阵。


### R139完整对照启动

正式185条（37 smoke前置）已封存round139_design.md，PID274022。加入10步warmup匹配Adam，以及直接复用sf0轨迹的最终y读出；sf0/sfw/sfz按最终平均x评估。审计明确训练梯度在y、评估在x/y，sfy必须逐项与sf0原梯度/状态精确一致。60来源轨迹中10条为复用，独立训练50条，不能说60条独立训练。尚无正式结果，累计60155不变。

R138确认完整本地备份已完成：sync138_confirm_complete.json核验62320 regular transport（6已知排除），其660正式证据已落盘。


### R139完成185条：Schedule-Free未超过匹配强根

完整审计185条、60来源轨迹/120首末快照/40SF状态/90近邻通过；含10条sfy复用轨迹，实际独立训练50条。重建最大4.29153e-6、token4.76837e-7、gradient5.96046e-8。GraphMAE标准sf0平均x75.56%、warmup sfw75.72%、beta0平均sfz75.56%、同轨迹最终y75.92%；原scale2 76.68%、普通CE76.36%、匹配warmup Adam76.76%，Direct73.08%。GraphMAE2 sf0/sfy73.84，sfw/sfz73.76，原scale2 75.36。最佳SF相对匹配强根-.84pp，不扩大；warmup Adam对原只+.08，也不升级。

汇总器原query_size断言因字符串替换误写600；实际config/query及训练审计始终500。analyze139_v2.py只修正断言，旧版保留，实验与审计不改。普通科学累计60340；sync139 PID52542正在核验目标62505 regular transport（6已知排除），当前已完成62320。

后备Sophia-H已读算法1/3和官方G版实现，见round140_design_draft.md；明确不能用真实标签平方梯度冒充Hessian。下一步固定本步detach监督系数、验证真实HVP及预算/裁剪控制；尚未实现或精度实验。Distribution Calibration为旧有待数据候选，本次不重复计新方向。


R139完整本地备份已完成：sync139_complete.json核验62505 regular transport、6已知排除。普通科学累计60340，范围核验2159与排除/forensics仍单列。


### R140 Sophia-H：二阶与全轨迹诊断完成，尚无准确率

六场景真实GCN二阶反传通过，固定detach监督系数的CPUdouble有限差分最大相对误差1.73125e-09；GPUfloat32 HVP相对差2.07717e-07。原float32 one-hot与double不兼容失败已保留，新probe版本仅修正诊断原型dtype。Sophia-H不定二次型原语及两ck×五尺度/符号配置的100步轨迹更新回放通过，实际probe10条轨迹（第一批6、尺度批4），均无query评价。

小rho.01约99.9%裁剪，接近动量sign；rho1仍94%，rho10/100降到约71–81%。这是预条件退化诊断，不是精度证据；负随机估计不等于真实对角负曲率。下一步在正式对照保留小尺度、较大尺度、坐标置换、纯sign和Adam预算参照；完整HVP点需独立CPU验证。详情round140_probe_note.md。普通科学累计60340不变。

cleanup109：R127confirm/R128gate/R129gate远端1795文件642461181字节经完整本地SHA及远端二次核验后删除，本地证据全部保留；远端余6.9GiB。


### R140 Sophia-H正式流水线启动

run_round140_sophia.py、optimizer140.py、独立replay140/audit140和配置/设计已封存，PID275444。41冒烟前置完整审计后才205正式：三rho(.01/10/100)、rho10坐标置换、动量sign、Adam100/110和旧CE；70全更新轨迹/140快照/400实际HVP/100近邻。每个HVP都在CPU冻结当步系数的局部目标重算，生成探针与置换也独立再生；不把随机负估计说成真实负曲率。预算分别记录100普通反向+10HVP与Adam110，不假称严格等算力。累计60340不变，尚无正式精度结论。


### R140完成205条：H版曲率未提高最强准确率

205正式、70更新轨迹/140梯度快照/400实际HVP/100近邻全部通过。HVP最大绝对误差6.55651e-7、相对8.09006e-7；几何3.81470e-6、token6.55651e-7、梯度4.84288e-8。GraphMAE rho100最佳75.72%，rho.01/10均75.52%、置换75.48%、sign75.48%；旧CE76.28%、scale2 76.08%、Adam110 76.16%，Direct73.96%。相对全局强根-.56pp，不扩大。GraphMAE2置换74.84优于其正确曲率74.32–74.64，但仍低于Adam110 75.56；不能据此称二阶结构有益。

普通科学累计60545。本地sync140_complete.json已核验62710 regular transport、6已知排除，正式证据全量落盘；无运行GPU任务。

后备R141加权GNB已核对Sophia-G官方采样标签接口，并推导混合支持CE+KD的sqrt权重伪梯度。4组微型问题穷举标签验证期望平方梯度与独立GGN对角最大8.88178e-16，错误权重反例保留。详round141_design_draft.md；尚无图训练/查询准确率。R140负结果不改，不能用新的曲率名称重标旧结果。


### R141加权GNB真实诊断通过并启动正式

两ck×g01/g1/g10/原gp1/sign共10条100步无query诊断，完整采样/状态独立回放通过；为匹配主要非退化尺度，正式改用rho10置换gp10，新optimizer141_v2/source/replay另存并两ck补测通过，旧版保持。未读取query准确率。平均裁剪g01约99.97–99.98%、g1约97.3–98.5%、g10约81.7–88.0%；负Hessian问题在此改为非负GGN，但不预设它会提升。

round141_design.md与manifest已封存，PID276489：41冒烟→完整审计→205正式。70轨迹/140首末快照/400实际伪梯度/100近邻须全通过；每次概率与原uniform/伪标签、sqrt混合权重、原目标梯度及曲率都独立检查。计数断言在封存前按实际矩阵复核为140快照，未运行错误计数版本。普通科学累计60545不变，无本轮正式结论。


### R141开发通过；固定候选进入20支持确认

205条/70轨迹/140梯度快照/400实际伪梯度/100近邻全部通过。GraphMAE g10=75.96%，强根scale2=75.68%，旧CE75.44%，Adam110=75.56%，Direct73.48%；对最强根+0.28pp，配对[1.0,0.4,-0.8,-0.8,1.6]。置换75.84%、sign75.88%，相对优势仅0.12/0.08pp，尚不能归因正确曲率。GraphMAE2最佳g01/sign74.80%。

固定 GraphMAE/head_g10_qeps-50，seed210100..210119、test后500，完整820条；PID277308，run_round141_confirm.py及分析/配置/设计已SHA封存。主判据对Direct及两ck的CE/scale2/Adam110共7参照配对区间下界均>0；机制对置换/sign单列，不事后更换候选。预期280轨迹/560快照/1600实际伪梯度/400近邻。同图重复query限制保留。

普通科学累计60750；sync141_complete.json通过，本地regular transport62915、已知排除6。远程剩余5.8GiB，确认尚在运行，未计入完成数。


### R141确认预测完成，审计与同步进行中；R142原语准备

固定820条预测已齐，父PID277308、独立CPU audit141子PID277691仍运行。尚未跑固定确认分析，不把已生成预测计作审计完成；sync141_confirm.py正在全量同步，预期regular63735，已确认数仍60750。

cleanup110完成：R132/R133/R134共1429文件、1403968318B；本地完整逐文件SHA/字节一致，远程二次核验后删除远程副本，本地保留。首次verify_archive110因R133审计存放在根目录而停止，无删除；v2改为读取实际完整审计报告后通过。远程剩余4.9GiB（含新确认数据）。

后备R142参考ACL2023 f-DISTILL（https://aclanthology.org/2023.acl-long.605/）及官方commit8e2ff7c，区分类别散度与原论文序列采样方法。新divergence142_draft.py和check142.py对KL/RKL/JS独立解析梯度、finite difference、相等分布及对称检查通过；最大梯度差3.47e-17，遗漏学生熵的反例梯度差0.100038。round142_design_draft.md计划只替换KD并加入初始梯度幅度匹配，未封存正式矩阵、未做图训练、无新准确率结论。


### R141确认停止扩大；R142散度实验正式运行

R141确认820条完整通过：280轨迹/560梯度快照/1600伪梯度/400近邻。固定g10 GraphMAE 76.40%，原scale2 76.81%（-0.41pp，95%配对支持bootstrap[-1.17025,0.33]，9胜11负）；Adam110 76.67%，旧CE76.35%（+0.05，[-0.59,0.68]）；Direct72.65%（+3.75，[2.76,4.67]）。置换76.28%，sign76.43%，也无稳定正确曲率优势。主判据失败，不换候选、不扩大、不替换强根。普通科学累计61570。R141确认全量本地同步以sync141_confirm_complete.json为准，运行期间不宣称备份完成。

R142仅替换全节点KD为RKL/JS：原幅度与初始化token梯度范数匹配，保留无KD/KL2控制及两ck旧CE/scale2/Direct。无query诊断显示JS梯度范数只有KL约18.5–20.3%，故必须区分尺度与形状；3步CPU旧根tokens/embedding/k50完全一致。配置seed211100..104、205正式、70轨迹/140快照/100近邻封存。父PID278270，45条12步冒烟通过16轨迹/32快照/22近邻，已进入正式；尚无正式准确率结论。详round142_design.md及manifest。


### R142完成：不升级全局强根；R143标准化进入实现准备

205条、70全更新轨迹/140梯度快照/100近邻全部通过，最大几何3.81470e-6、token5.36442e-7、梯度7.45058e-8。GraphMAE：原scale2 76.36%，旧CE75.88%，KL2 76.24%，RKL原/匹配均76.12%，JS原74.64%/匹配76.16%；无KD73.92%。全局开发门槛失败，不扩大。GraphMAE2局部RKL匹配75.16%对本ck根74.56%为+0.60，但KL2已75.12%；记录局部阳性与简单权重解释，不能声称散度整体优于KL。Direct72.32%。普通科学累计61775；sync141_confirm与sync142均完成，本地regular63940、已知排除6，全证据已落盘。无运行GPU任务，远程4.6GiB。

后备R143借鉴CVPR2024 Logit Standardization in Knowledge Distillation。已核对原文Algorithm1与官方KD.py；原文分母K而官方torch.std默认K-1+1e-7，采用官方代码口径并保留差异。仅改全节点KD两端logits，教师Q取log Q；门控仍Q、监督CE和原读出不变。官方函数快照损失误差0、梯度3.47e-18；独立解析梯度1.56e-17，finite difference及uniform边界通过；detach标准差前向相同但梯度差0.1446。

真实Cora无query初始化（seed212199）通过：both相对原KL梯度范数GMAE1.193倍、GMAE2 10.033倍，方向余弦0.427/0.637；teacher-only方向余弦0.993/0.996。尤其GMAE2大幅放大，正式必须配初始token梯度范数匹配及停止梯度/单侧对照。round143_design_draft.md、round143_primitive.json、round143_initial_probe.json；尚未实现完整图训练驱动/正式封存，不记准确率。


### R143正式实现封存并启动

run_round143_standard.py、audit143.py、identity143.py及配置/设计/分析SHA封存，PID279574。Cora seed212100..104，两ck；zbr原式both、zbm初始梯度范数匹配、zss停止标准差梯度但共用zbm权重、zpm学生单侧匹配、ztm教师单侧匹配、KL2对照。zss不按自身重标权重，确保与both在相同参数时前向相同以隔离导数。另保留CE/scale2/Direct和原读出对照，共205。

先CPU旧根3步完全一致及45条12步冒烟（16轨迹32快照22近邻），再205正式（70轨迹140快照100近邻）。独立CPU标准差使用显式K-1公式，KL用概率log比，不复用待验证函数；所有实际梯度独立Adam回放，首末目标梯度重建。封存前发现分析审计文件后缀仍指142，已改143并再SHA封存；未运行错误分析版本。累计61775保持，尚无本轮正式结论。


### R143完成205条：全局提升未通过，保留GraphMAE2局部观察

205条、70轨迹/140梯度快照/100近邻独立审计通过。GraphMAE旧CE77.00%、scale2 76.80%；teacher-only匹配76.76%，both原式76.36%、both匹配75.96%、stop共权重76.16%、student-only75.92%、KL2 76.28%。全局最佳新配置仍比最强根低0.24pp，停止扩大。

GraphMAE2 both原式76.16%对本ck旧CE75.72%为+0.44、对scale2 75.44%为+0.72；匹配75.48%、stop75.52%、student75.56%、teacher75.40%，说明原式局部阳性并不能排除强梯度幅度作用。该局部观察保留，未完成20支持确认；不是全局最佳，也不是已证实机制。Direct73.00%。累计普通科学61980。sync143.py全量备份进行中，以sync143_complete.json为完成依据。

cleanup111已完成：R135/R136/R137共1336文件891317082B经本地逐SHA与远程二次SHA后仅删远程副本，本地证据保留。服务器现余5.2GiB。当前无GPU训练进程，下一方向尚未选择。


### R144时序KD设计封存并启动

R143备份sync143_complete通过：regular64145、排除6，正式证据齐。下一方案借鉴ICCV2019 On the Efficacy of Knowledge Distillation第5.3节后段停止KD；不与同文“提前停止教师训练”混淆。相关原型Sinkhorn、CE学习率schedule已有实验，不重复。

R144保留冻结主干/896 late tokens/teacher与gate/scale2监督/Adam100/k50，只控制KD时序：前50/80步开启、后50步开启、全程.5/.8及原全程1。恒定控制匹配累计标量权重而非实际梯度积分；不重置动量，因此关闭KD不等于清除历史影响。schedule原语12/100步边界/总量/反序检查通过，设计round144_design.md和源码/config/分析SHA封存。

PID280648，先旧根3步完全一致、41条12步冒烟（14轨迹84快照20近邻），再185正式seed213100..104（60轨迹360快照90近邻）。每条除首末，还检查49/50、79/80步切换两侧真实梯度，全部Adam更新独立回放。累计科学61980保持；尚无正式准确率结论。


### R144完成185条：时序控制未产生可靠开发增益

185条/60轨迹/360含切换边界的梯度快照/90近邻全部通过。几何最大3.81470e-6、token5.36442e-7、梯度5.21541e-8。GraphMAE原scale2 76.32%、CE76.12%；early50 75.48、early80 76.20、late50 76.36、constant50 76.28、constant80 76.24。GraphMAE2原scale2 76.64%、CE76.28%；early50 75.92、early80 76.64、late50 76.68、constant50 76.60、constant80 76.64。最佳late50仅+.04pp，不扩大；早停KD没有胜出。Direct75.16%。累计普通科学62165，sync144_complete通过regular64330/排除6，本地备份完成，无运行GPU任务。

### R145后备：DoRA内部有效权重适配完成原语及真实GCN检查

再次排除已做R26/R78 PCGrad及R36/R37低秩原样重跑。R145核对DoRA原文4.1–4.3与NVlabs官方commit7e2f10abbe8efe212c8fca1d983ae1d04ef13a18；Linear[out,in]行范数、官方simple分母detach、保留bias。它是Prompt与权重适配在第二层共享forward，不是终端融合，也不是纯Prompt。原W0冻结但有效权重改变，命名明确。

矩阵原语独立解析梯度最大6.66e-16，官方展开形式forward最大3.20e-14；full数值梯度通过，detach验证指定surrogate梯度。真实两ck×4模式无query检查全通过，零初始化GCN输出完全相同，显式邻接传播最大1.49e-8，所有原模型参数未变、无梯度。实际bias范数0.06453/0.06786非零，所以不能把幅度适配直接简化为整个带bias输出缩放。联合参数DoRA/full1280、LoRA1152、幅度1024；首步A梯度0为B零初始化的预期结果，B/幅度/Prompt均有梯度。

详round145_design_draft.md、round145_primitive.json、round145_initial_probe.json。下一步实现Prompt/adapter 2×2及普通低秩、幅度、完整分母梯度对照，多参数更新回放尚未完成；没有图训练准确率或正式封存矩阵。


### R145多参数训练与审计完成，正式流水线启动

新增joint145/replay145用于Prompt+有效GCN权重适配。最初CPU三步诊断发现Parameter(initial.to(cpu))共享存储，使保存的初始向量被更新；无query/正式未启动。已另存joint145_v2、frame/audit/identity_v2，clone初值后两ck×8模式16条三步无query轨迹全通过（梯度最大2.23517e-8、Adam重放7.45058e-9），不改原失败源。

rank1八模式为dp/fp/lp/mp联合与da/fa/la/ma仅adapter；原scale2/旧CE/Direct/frozen/GraphPrompt完整保留。90条正式训练轨迹保存初值、布局、全部梯度及首末快照，由独立CPU手写有效权重/邻接传播/bias/目标与NumPy Adam审计；不是只验证预测文件。所有adapter用实际DeltaW均方正则，token正则照旧。

round145_design.md、config和13个源码/设计/分析文件SHA封存。PID282325，先identity旧根及53条12步冒烟（20轨迹40快照26近邻），再seed214100..104的245正式（90轨迹180快照120近邻）。仅adapter胜出不会称为Prompt收益；联合胜出也需比较LoRA/幅度。累计62165不变，尚无正式精度结论。


### R145开发通过，固定full-norm联合方案进入确认

245条/90轨迹/180梯度快照/120近邻全审计通过。GraphMAE fp=75.80%，旧CE/scale2均75.44%，+0.36pp；五组差[.6,-.4,-.8,-.2,2.6]，2胜3负，保留单组较大增益驱动均值的局限。dp75.64%、LoRA75.40%、幅度75.52%；adapter-only最佳full66.48%，不能替代Prompt。GraphMAE2 dp/fp74.72%、LoRA74.64%、原scale2 73.84%，本ck局部优势亦保留。Direct72.92%。

固定GraphMAE/head_fp_qeps-50、seed215100..119、确认query后500、100步全配置不变；980条完整矩阵，360轨迹720梯度快照480近邻，PID283290。run_round145_confirm/config/design/analyzer启动前SHA封存；主参考Direct和两ck旧CE/scale2共5项，配对bootstrap10000次、seed145000，区间下界全>0才第二组确认。机制对LoRA/幅度/detach/单adapter单列，不替代强根判据。

普通科学累计62410；sync145_complete通过，本地regular64575、排除6。确认待完成，不计入。启动前245开发占247MB，远程剩余4.7GiB，容量满足本轮预计约1GB。

后备R146已读PiSSA原文第3节及作者仓库推荐的官方peft pissa_init快照。说明DeltaW=BA-B0A0可达2r而非r，需同范数随机双非零因子控制与普通LoRA参照；源码commit见pissa146_source.json。仅文献/设计round146_design_draft.md，未原语、未图训练，不更改当前确认。


### R145确认停止扩展；R146进入PiSSA轨迹验证

固定GraphMAE/full-norm DoRA+Prompt在20组新support（215100..215119）确认均值76.55%，原scale2 76.67%，差−0.12pp，配对bootstrap95%区间[−0.56,0.32]，10胜10负；旧CE75.94%，差+0.61[0.12975,1.05]，局部增益保留。普通LoRA76.70%、幅度适配76.77%，也没有证据将增益归于完整分母梯度。Direct72.28%。主判据未过，不进行第二20组，不改选确认中的对照。所有980条、360轨迹、720快照、480近邻通过独立审计；sync145_confirm_complete为passed，本地regular65555/排除6，科学累计63390。仍是同一Cora图和反复使用的query半集，区间仅反映support重采样。

R146矩阵原语12项通过，独立解析梯度最大误差1.0658e−14，rank1 PiSSA/随机双非零的DeltaW实测可达rank2。两ck×4初始化的真实GCN无query初值验证8项全过，输出bit-exact、底座未改、梯度有限。实际第二层top1能量仅GraphMAE5.32%、GraphMAE2 3.23%，不能称一个主方向覆盖大部分权重。PiSSA首步因子梯度较随机/零B大是初始化诊断，不是准确率证据；保留同因子范数随机双非零、rank1 LoRA同参数量、rank2 LoRA最大更新秩对照。新joint146/replay146独立CPU目标与Adam重放正在验证，正式训练尚未开始。


### R146开发通过并启动固定确认；R147后备原语

R146最初冒烟在审计末尾失败：计数白名单误留R145方法名。保留原源码/round146.log/round146_smoke_failure.json，另存audit146_v2与entry/manifest_v2；复用同一批封存预测重新审计，训练源码未变。v2启动stdout误接/dev/null，故开发控制台日志缺失；逐任务result/sealed、配置SHA、独立audit和done文件均完整，按这些工件验证实际完成状态，不补造日志。

53条冒烟通过20轨迹40快照26近邻；正式245条通过90轨迹180快照120近邻，梯度最大5.076e−7、Adam最大4.768e−7。GraphMAE2 PiSSA+Prompt75.88%，原CE74.84%、scale2 74.80%、随机75.44%、LoRA-r1 75.72%、r2 75.40%。全局最强原根GraphMAE CE75.64%，固定候选+.24pp、3胜2负（差[1.4,1.4,-.8,2.,-2.8]），Direct73.96%。GraphMAE PiSSA75.72%、随机75.64%，仅adapter远低于联合。开发通过但不能声称稳定提升或PiSSA独有收益。

固定GraphMAE2/head_pp_qeps-50，新support216200..216219、query确认后500，980条/360轨迹/720快照/480近邻，100步配置不变。启动前confirm源码/设计/配置/分析SHA封存，PID286005，日志round146_confirm.log。主比较Direct与两ck旧CE/scale2，10000次bootstrap seed146000，下界全>0才下一组确认；同ck随机/LoRA-r1/r2/单adapter单列。科学累计63635；本地sync146进行中，目标regular65800/排除6，未完成前仍只确认65555。

cleanup112已完成：R138/R139/R140开发的1386文件、1367067925字节，经完整本地SHA及第二次远程SHA后仅删远程重复件；本地保留，服务器释放后4.7GiB。后备R147已读VeRA原文第3节和固定PEFT源码，6项矩阵原语通过；无真实GCN/准确率结果，详round147_design_draft.md。


R146开发本地同步追加核验：sync146_complete passed，regular65800、排除6。PID286005确认仍在运行（已观察至GraphMAE/support216215），当前远程4.3GiB；确认尚不计入科学累计63635。


### R147 VeRA真实GCN与轨迹验证通过，等待R146固定确认

两ck×rank16/128真实GCN初值4项输出bit-exact、原模型所有参数不变且无梯度。b有首步梯度，d首步梯度为0（b初始为0的预期），不把此现象误判实现失效。两ck×10模式的20条三步CPU轨迹全过，独立传播/保留bias/目标梯度及全部Adam重放；无query标签或准确率。联合双缩放r16/r128参数1040/1152，固定d仅b联合1024；仅adapter144/256或128。rank128双缩放与LoRA-r1联合1152参数匹配，表达集合不等价。

joint147/replay147、frame147、audit147实现中保留LoRA-r1/r2同split对照；随机A/B seed固定0，两ck同shape使用相同矩阵，support只影响teacher/gate和训练。只有第二层适配，不宣称论文跨层共享效益。拟285条开发（110轨迹220快照140近邻）、61条冒烟（24轨迹48快照30近邻）。暂未正式训练/封存，等待R146确认结束，避免遗漏可能成功的新强基线。

R146确认已生成980条，PID286005/独立审计子PID286439仍在运行；本地sync146_confirm执行中，未计入科学累计63635。后续先固定analyze146_confirm，再决定R147正式矩阵。


### R146确认未过全局升级，保留同checkpoint增益；R147正式启动

固定GraphMAE2 PiSSA+Prompt在216200..216219确认74.30%，对GraphMAE scale2 75.46%为−1.16pp，95%区间[−2.11,−.23]；对GraphMAE旧CE75.19%为−.89[−1.75,−.01]，不继续第二20组。对自身GraphMAE2 scale2 73.57%为+.73[.17,1.37]、13胜7负，对自身旧CE73.38%为+.92[.38,1.51]，这是有效的局部提升，保留而不宣称全部失败。相同ck LoRA-r1 74.48%、r2 74.46%、随机74.02%，不能把局部提升全部归于主奇异初始化。确认内GraphMAE LoRA-r1 76.27%为探索性对照，不在确认后改选该方案；下一轮同split保留LoRA控制。

980条/360轨迹/720首末快照/480近邻全部审计通过。科学累计64615。sync146_confirm已经传输完毕、正在本地全库refresh（PID55683，父55632/session73416）；尚未看到完成JSON，regular确认值仍65800，目标66780/排除6，不能提前宣称本地核验完成。

R147 official_init两rank与固定PEFT函数逐位一致，identity147旧根token/embedding/k50全相等。16个源码/配置/设计文件SHA封存；PID287200、round147.log，先61条冒烟（24轨迹48快照30近邻）再285条开发（110轨迹220快照140近邻）。本轮只固定随机seed0与rank16/128，不按query重新抽随机映射；原scale2仍主根，匹配LoRA和单adapter完整保留。后续若正增益再确认及随机映射复核。启动时远程剩余3.7GiB。

后备文献线索：已检索ICML2024 LoRA+，官方页面https://proceedings.mlr.press/v235/hayou24a.html及作者仓库nikhilgsh/loraplus，拟考察A/B不同学习率；尚未精读或实现，不算新验证结果。


追加核验：sync146_confirm_complete passed，本地regular66780/排除6；R147冒烟61条/24轨迹/48快照/30近邻通过，PID287200已进入正式开发并观察至GraphMAE2/support217101。


### R147停止全局扩展；R148 LoRA+分组学习率启动

R147正式285条/110轨迹/220快照/140近邻全过。GraphMAE双缩放r16/r128为73.80/73.60，固定d仅b为74.92/74.96，原scale2 75.20、旧CE74.68；全局最高适配74.96仍低于强根.24pp，不确认。GraphMAE2双缩放73.72/73.40高于本ck scale2 72.80，但LoRA-r1 74.04/r2 74.52更高，局部提升保留而不归于随机缩放独特收益。Direct71.76，仅adapter均较差。科学累计64900，sync147_complete通过，本地regular67065/排除6。

已读ICML2024 LoRA+原文第4节末及第5节，作者实现固定commit a0c44bfec0eff97b9574469366bc6b0ccc28838d；来源/文件SHA见loraplus148_source.json。R148仅改变rank1 A/B学习率，Prompt固定.01。eq、升B4/16、降A4/16、整体升4/降4、反向4，以及升B4/降A4仅adapter，十模式区分因子比例和整体幅度。先前普通LoRA没有一致跨批优势，此轮测的是优化策略，不把已有低秩组合改名当新机制。

抽取官方优化器函数的10配置四步toy轨迹与本地分组Adam逐位一致；真实两ck×10三步CPU轨迹独立梯度/Adam重放全过，最大5.9605e−8/8.9407e−8。identity148保留旧根token/embedding/k50精确性，eq与旧rank1 LoRA匹配。源码/配置/设计SHA启动前封存，PID288883、round148.log；先61条12步冒烟24轨迹48快照30近邻，再support218100..104的285条开发110轨迹220快照140近邻。100步、物理DeltaW正则、原teacher/gate、标签封存顺序不变。正式准确率尚无结论。

为存储余量，cleanup113正在核验R142/R143/R144开发归档；核验完成前未删除任何文件。


追加核验：R148冒烟61条/24轨迹/48快照/30近邻通过，PID288883正式开发已观察到GraphMAE2/support218102。cleanup113远程第二次SHA核验后删除1486份重复文件、690139587字节，本地完整保留；远程余3.8GiB。


### R148开发通过，固定d4进入20组确认；R149梯度初始化后备

285条/110轨迹/220快照/140近邻全部审计通过。GraphMAE d4（A=.0025，B=.01，Prompt=.01）75.20%，原scale2 74.84%，+.36pp，差[.6,0,1,.6,-.4]为3胜1平1负；普通LoRA eq74.72、整体降4 l4 74.88、升B4 u4 74.24、升B16 73.76。Direct71.68。GraphMAE2适配最高反向r4 74.68，旧CE72.60、scale2 72.52，显示不能把所有局部收益归为B更大学习率。

固定GraphMAE/head_d4_qeps-50，support218200..218219、confirm后500query，1140条完整矩阵/440轨迹/880快照/560近邻。入口/配置/设计/分析启动前SHA封存，PID290170，round148_confirm.log。100步与全部配置不变；主比较Direct及两ck旧CE/scale2，10000次bootstrap seed148000区间下界全>0才第二组20support。eq/l4/u4/r4/da机制对照单列，确认后不改选。科学累计65185，sync148_complete通过，本地regular67350/排除6；确认未计入。

后备R149已读LoRA-GA梯度初始化原文及固定作者源码，发现arXiv正文/gamma与算法/代码/sqrt(gamma)尺度差异，按代码明确选择，完整SVD/规范符号改写不声称原随机SVD逐位复现。矩阵12项无query检查通过；真实GCN梯度探针正在运行。rank1差分可rank2；小步SGD一阶匹配不等于Adam匹配。详round149_design_draft.md，目前无正式训练/精度结果。


R149真实GCN无query6项追加验证通过：两ck×互异/重叠/随机初值全bit-exact，底座参数未改且无梯度；实际梯度top2能量GraphMAE90.29%、GraphMAE2 63.92%，互异方向一阶截断重建误差1.828e−7/3.108e−8。这只是固定support219199的初始化诊断，不是准确率或Adam收敛证据。R148确认PID290170仍活跃，已观察到GraphMAE2/support218210；远程余2.8GiB，下一轮前继续检查容量。


### R148确认正均值但区间未过；R149梯度初始化流水线启动

R148固定GraphMAE d4在218200..218219确认76.76%，scale2 76.51%，+.25pp，95%区间[−.62,1.02]，12胜6负2平；旧CE76.23%，+.53[−.43,1.39]。不开展第二组确认，保留均值正增益但不认定稳定升级。eq76.59、d16 76.80、l4 77.35、r4 76.86；其中l4为确认中的探索性控制线索，不能事后改选为R148确认成功。1140条/440轨迹/880快照/560近邻全部审计通过，科学累计66325；本地sync148_confirm尚在全库核验，regular仍67350确认，目标68490/排除6。

R149原始18条三步轨迹全通过，加入l4后另存joint149_v2/frame_v2/audit_v2/identity_v2，20轨迹全过，未改任何运行中源码。初始化梯度在CPU按原目标重算，只用support标签和固定teacher。独立审计先重建该梯度，再在被核验的保存梯度上SVD，避免不同设备近简并向量不一致；该容差及符号规范不构成上游逐位复现声明。

固定gd/op/rp（互异方向/重叠方向/同范数随机）联合与ga/oa/ra仅adapter、LoRA-r1/r2、d4/l4完整同split，rank1差分最高rank2。support219100..104，285条110轨迹220快照140近邻；先seed219199的61条12步冒烟24轨迹48快照30近邻，旧根identity通过后正式。源码/config/design/analyzer启动前SHA封存，PID292253、round149.log。此轮将l4作为新开发候选/对照，若胜出归为步长策略而非梯度初始化。额外初始化反向明确计数，不宣称同计算量速度优势。

cleanup114已完成：round138_confirm/round141_confirm/round145_gate共4296文件、4346938846字节，完整本地SHA及第二次远程SHA通过后仅删除远程重复件，本地保留。远程剩余6.5GiB。


追加核验：sync148_confirm_complete passed，本地regular68490/排除6。R149冒烟61条24轨迹48快照30近邻全过，PID292253进入正式开发，已观察至GraphMAE/support219102。


### R149停止扩展；R150借鉴扩散模型OFT启动

R149全285条110轨迹220快照140近邻审计通过。GraphMAE梯度互异73.44、重叠73.04、同范数随机73.24，普通LoRA-r1/l4均73.88，原旧CE74.20、scale2 73.84。GraphMAE2互异72.72高于自身scale2 71.88，优于随机72.16、LoRA-r2 72.56，但低于全局强根，局部正收益保留，不扩展确认。此前R148 l4对照较高均值没有在本轮新support重现为全局优势。Direct71.24。科学累计66610；sync149_complete通过，regular68775/排除6。

下一轮R150来源NeurIPS2023 OFT扩散模型适配，已读原文3.2–3.3及作者固定commit59ab7eae36a963ef8152ee5f1da460f10310ff53。正确对应Linear[out,in]的W_eff=W Q^T而非左乘输出旋转，原bias保留；保持权重行Gram不代表节点几何或准确率保留。独立上三角坐标是本地紧凑参数化，不能声称与作者冗余全块Adam逐位等同。

block4/8正交Cayley及首步梯度匹配的I−2S线性控制，分别有/无896参数Prompt，再带LoRA-r1/l4。联合1088/1344参数，adapter-only192/448。4项矩阵、8项实际GCN初值检查、20条三步独立CPU轨迹均过；梯度最大3.7253e−8、Adam重放7.4506e−9。未将矩阵性质当精度收益。

round150_design/config/analyzer及18个源码/配置/设计SHA封存，PID294092，round150.log；先seed220199的61条12步冒烟24轨迹48快照30近邻，过后自动support220100..104正式285条110轨迹220快照140近邻。原teacher/gate/100步/物理DeltaW正则与封存评分流程保持，l4沿用匹配低学习率。此时尚无正式精度结论。


R150追加核验：61条冒烟/24轨迹/48快照/30近邻全通过，PID294092已进入正式开发，观察至GraphMAE/support220104；尚未给出正式结论。


## R150 完成与 R151 开始（2026-09-24）

R150 正式285条、110训练轨迹、220首末快照、140近邻重建均通过独立审计；本地全量 SHA 传输核验已通过（69060常规条，另6排除）。累计远端科学审计66895条。OFT 最佳 GraphMAE c8 76.80%，最强旧根 GraphMAE scale2 77.44%；全部候选最高是 GraphMAE2 线性 n8 77.12%，差−0.32百分点，2胜3负，按预封存门槛停止。GraphMAE2 OFT c8 76.72% 相对本 checkpoint scale2 75.48% +1.24百分点，但线性 n8 77.12% 更高；保留局部正结果，不归因为正交约束优势。Direct73.44%。详见 `round150_pairs.json`、`round150_gate_audit150.json`、`sync150_complete.json`。

R151 借鉴 ICML2024 ETHER+，拟测恒等初始化的单侧/双侧权重适配与共享第二层 Prompt。固定分母、单独适配、LoRA和最强旧根均保留；设计见 `round151_design.md`。当前仅实现和无query核验，尚无正式准确率。目标持续 active。


### R151 预检通过并启动

官方前向/梯度四例、真实GCN三步20轨迹独立回放全部通过（最大梯度差4.47e−8），代码、设计、配置与分析器已封存为 `round151_manifest.json`。服务器 PID296077 已启动旧根恒等→61条smoke及审计→285条正式及审计流水线；保持同一进程，不因轮询超时重启。远端剩余5.8GiB，正对已完整落地本地的 R145确认/R146开发/R147开发重复文件执行SHA归档检查，尚未删除。


### R151 smoke 已审计；第115次远端重复归档清理完成

R151 smoke61条、24轨迹、48首末快照、30近邻重建全部通过；正式285条矩阵正在同一PID296077下运行，尚未宣布有效性。cleanup115已再次核验远端SHA后删除 R145确认980/R146开发245/R147开发285的3766个远端重复文件，共1593496518字节；完整本地文件不变，远端可用空间7.1GiB。清理凭证 `archive115_verification.json` / `archive115_verified_manifest.json` / `cleanup_115.json`。


## R151 正式完成；R152 FourierFT 完成无query验证（2026-09-24）

R151 ETHER+：285条、110轨迹、220首末快照、140近邻全部审计通过。GraphMAE单侧76.08%、双侧74.92%，同批旧CE76.28%、scale2 76.00%；全部候选最高LoRA低LR76.24%，仍差−.04百分点，未过预封存+.20门槛。GraphMAE2单侧75.08%对本ck旧CE74.44%有+.64局部提升，但同ck LoRA75.28%更高。停止全局升级，不否定局部改善。`round151_pairs.json` / `round151_gate_audit151.json`记录完整。累计科学审计67180条；`sync151_complete.json`本地69345常规条、6排除传输核验通过。

R152 借鉴ICML2024 FourierFT，官方代码及原文已落地；将少量频谱系数的权重增量与第二层Prompt共同训练。128/256频点×64/128尺度，保持范数的位置置换、单独适配及LoRA对照；详见 `round152_design.md`。六组原语、20三步真实GCN独立余弦基/梯度/Adam回放通过。首次probe元数据字段不一致已在v2修正，原证据保留，无query训练污染。下一步运行旧根恒等、61条smoke、285条正式及全量审计。


R152 v2正式流水线已启动，父PID297979。先恒等/61条smoke独立审计，通过后自动进入285条正式矩阵；本轮尚无有效性结论。封存清单 `round152_manifest.json`，不会修改运行源代码。


## R152 开发完成并进入固定对照胜者确认；R153来源核验

R152正式285预测、110轨迹、220首末快照、140近邻均通过，`round152_gate_audit152.json`。FourierFT最好GraphMAE2 f4 75.56%，原k50强根75.24%；真正开发胜者为GraphMAE2 LoRA-r1 l1 76.20%，同低LR l4并列，由预定候选顺序选l1。相对最强旧k50根+.96，4胜1负；相对Direct74.64% +1.56。不能归因为FourierFT。原型读出亦完整保留：GraphMAE scale2均值75.80%，最高Fourier原型76.00%；既定选择空间是k50，确认阶段不换读出。累计科学审计67465条。

已锁定222200..222219共20个新support，确认query后500节点，全矩阵1140条/440轨迹/880快照/560近邻。`round152_confirm_design.md`、配置、分析器、manifest已封存；父PID299169确认存活，子任务正在运行，不修改源码、不重选胜者。

后备R153借鉴SVFT的固定奇异向量稀疏系数适配。论文与作者源码差异、零增量迁移边界见 `svft153_source_audit.md`。首版原语检查的128维随机基带宽1范数差5.72e−6来自float32范数归约；同张量double范数差1.47e−8、矩阵重建差5.66e−7，诊断保留 `round153_diagnostic.log`。v2仅将范数审计改为double累加且阈值收紧到1e−6；不改变训练原语。目前仅无query原语，无正式训练。

R152本地sync152进程仍在进行，完成前不声称传输核验通过。


R153 v2八组原语（16/128维×SVD/随机正交基×带宽0/1）全部通过：零增量精确恢复、作者稀疏索引一致、显式秩一和梯度核验通过。证据`round153_primitive.json`；尚待真实GCN轨迹核验。


## R153 固定奇异基方案启动；R152确认进入审计（2026-09-24）

R153 两checkpoint×10方法的20条三步无query轨迹全部独立通过，最大梯度差1.3411e−7、Adam回放差7.45e−9。纯对角/带宽1分别128/382系数，配相同维数随机正交基、适配单独及LoRA，详`round153_design.md`。源码、协议与分析器已封存`round153_manifest.json`，父PID300054已启动恒等→61条smoke及审计→285条正式及审计流水线。

R152固定GraphMAE2 LoRA确认的1140条训练已完成，父PID299169下子进程299848执行`audit152_v2.py round152_confirm`；全量审计尚未结束，不提前分析或计入完成数。R152开发本地SHA核验已完成，69630常规条、6排除。

cleanup116已对R146确认980/R148开发285/R149开发285的3866文件完成本地全量SHA及删除前远端二次SHA核对，仅移除远端重复文件1690168902字节。完整本地证据保留，远端可用7.0GiB；凭证`archive116_verification.json`/`archive116_verified_manifest.json`/`cleanup_116.json`。


## 近期方案速览入口（R145起）

新增[近期方案对照](近期方案对照_145起.md)，由已落地配对JSON生成，区分开发赢家、确认结论及对照胜出。固定确认区间与全局/单checkpoint边界保留；刷新命令`python3 refresh_recent_progress.py`。当前R152确认与R153正式训练均结束，各自独立CPU审计仍运行，未提前判精度或计入科学完成数。


## R152 确认停止；R153 正式停止（2026-09-24）

R152确认1140条、440轨迹、880首末快照、560近邻独立审计全部通过。固定GraphMAE2 l1为72.96%，同批GraphMAE scale2 74.93%，−1.97百分点，95%支持bootstrap区间[−3.11025,−.87]，6胜13负1平，不能升级全局强根。相对自身GraphMAE2 scale2 72.58%仅+.38[−.08,.86]、9胜8负3平；相对自身旧CE72.14% +.82[.27,1.44]局部正结果保留。GraphMAE l4 75.24%为未预选参考，不换赢家。详`round152_confirm_pairs.json` / `round152_confirm_audit152.json`。

R153正式285条、110轨迹、220快照、140近邻审计全部通过。GraphMAE奇异基对角74.24%、带状73.80%，均低于原scale2 75.40%；GraphMAE2对角72.76%、带状72.84%，相对自身scale2 71.52%有+1.24/+1.32，但随机带状73.32%更高，不能归因于预训练奇异方向优势。所有候选最高GraphMAE LoRA低LR75.48%，只比强根+.08，未达+.20开发门槛。停止此配置全局升级，保留局部正结果；详`round153_pairs.json` / `round153_gate_audit153.json`。

两轮均已终止且结论完成，累计科学审计68890条。同步按R153开发→R152确认顺序串行执行，最终预期71055常规传输条、6排除；完成标记前不提前宣称本地SHA核验完成。近期方案速览已刷新。目标保持active，后续继续设计验证。


R153本地全量传输SHA已完成（69915常规条、6排除）；R152确认的sync152_confirm已串行启动，完成后预期71055。后备R154已读LoRA Dropout预印本输入/输出维屏蔽公式，明确无inverted缩放时权重增量期望为q²BA，须与确定性幅度控制分开。尚无正式实现/训练，详`loradrop154_design_draft.md`，不与R97 token DropConnect混为一项。


## R154 LoRA行列屏蔽启动；R152确认同步完成（2026-09-24）

R154已完成六组全部64掩码穷举与有限差分验证；真实两checkpoint×十模式20条三步轨迹独立回放通过，mask流逐位一致，最大梯度差2.2352e−8。实现行列屏蔽p=.1/.3、反向缩放和确定性q²增量控制，Prompt在同一第二层共享训练；推理固定期望权重增量，并不声称复现论文随机预测集成。完整设计`round154_design.md`，预封存`round154_manifest.json`。父PID301565启动恒等→61条smoke及审计→285条正式及审计流程。

来源审查排除了检索页关联的船舶事故仓库：它不是LoRA Dropout作者实现。本轮只依据已读预印本公式做本地迁移，来源边界写入设计。R12隐藏屏蔽、R97有效token屏蔽与本轮增量行列屏蔽分别记录。

R152确认sync已通过，本地累计71055常规传输条、6排除；科学审计仍68890条，R154尚未计入。两个计数口径不混用。证据`sync152_confirm_complete.json`。目标继续active。


## R154 完成并停止；R155 rsLoRA 秩缩放启动（2026-09-24）

R154正式285条、110轨迹、220快照、140近邻全部独立审计通过。GraphMAE未缩放屏蔽p=.3为75.40%，p=.1为75.08%，低于scale2 75.52%；所有候选最高仍LoRA低LR75.44%，差−.08，未过开发门槛。GraphMAE2反向缩放p=.3为74.80%，对自身scale2 74.24%局部+.56；未缩放p=.3为74.68%，确定性幅度74.64%，不能凭该小差异断言随机正则优势。原文测试MC集成未运行，结果限于固定期望权重推理。证据`round154_pairs.json`/`round154_gate_audit154.json`。累计科学审计69175条；sync154仍在执行，常规传输当前已核验71055条，完成后预期71340，6排除。

R155已读rsLoRA原文和作者说明，并核对固定PEFT缩放表达式。rank4/16，1/sqrt(r)对1/r，同rank同初始化，联合Prompt及单独适配；旧rank1两个LR与旧强根都保留。rs可由普通LoRA调整alpha完全表达，不主张新算子，只测试准确率。六原语和真实GCN20条三步独立回放通过（最大梯度2.2352e−8）。设计`round155_design.md`与SHA清单`round155_manifest.json`已封存。父PID302819启动恒等→61条smoke及审计→285条正式及审计；无新准确率结论。


## R155 开发完成、固定LoRA对照进入确认；R156优化器来源核验

R155 285条、110轨迹、220快照、140近邻独立审计通过。GraphMAE rs rank4/16为74.96/75.28%，ordinary为75.44/75.60%，均未超过旧scale2 k50 75.72%。固定开发胜者低LR rank1 LoRA l4 75.92%，+.20，3胜1负1平，按预设规则进入20新支持确认。旧scale2原型读出亦75.92%，l4原型76.48%，如实保留，不能说已全面提升；不改预设k50选择空间。GraphMAE2 rs16 74.84%对自身根74.00%局部+.84，但同ck低LR rank1也74.84%。科学累计69460条。

固定GraphMAE l4，support225200..225219，全矩阵1140，确认query后500，代码/配置/分析器已封存；父PID303934启动确认，详round155_confirm_design.md。R155开发sync155正在执行，之前sync154已完成，常规传输71340、6排除。

后备R156已读LoRA-FA原文及PEFT固定源码，区别冻结A与B梯度Gram伪逆变换，并识别legacy Adam eps位置和GPU bf16路径差异。四个CPU double四步官方类对NumPy回放通过（最大8.14e-16），仅原语不含query。详lorafa156_design_draft.md；还未接入真实GCN。


R155开发本地sync155全量SHA通过：71625常规条、6排除。cleanup117对R148确认1140/R150开发285/R151开发285的4266文件，经本地全量与删除前远端二次SHA后清除远端重复1935153451字节；本地不删。凭证archive117_verification.json、archive117_verified_manifest.json、cleanup_117.json。远端6.7GiB可用。确认父PID303934仍存活运行，不重启、不提前分析。


## R156 LoRA-FA真实GCN核验完成并启动（2026-09-24）

R156完整Gram逆、对角逆、原梯度冻结A的rank4/16控制，配单独适配及旧LoRA，共20条三步实际GCN轨迹全部通过独立回放。分别保存原始梯度和实际更新梯度，检查首末原梯度及全步变换/Adam；最大原梯度差2.98e-8，Adam回放差7.45e-9。独立本地float32更新对官方CPU四步最大差1.98e-7，明确CPU double预计算伪逆、禁用bf16及legacy Adam eps位置，未冒称GPU逐位复现。

正式设计round156_design.md、源代码/config/analyzer的round156_manifest.json已封存；父PID304727已启动恒等→61条smoke及审计→285条正式及审计。恒等已通过。R155确认父PID303934仍在完整CPU审计中，尚不分析、不计入完成数，原运行代码不改。科学累计69460，常规本地传输71625、6排除保持不变。


## R155固定确认通过；R156开发晋级（2026-09-24）

R155首批固定确认1140条、440轨迹、880梯度快照、560邻域审计通过。预先锁定GraphMAE低学习率rank1 LoRA联合Prompt+k50达到77.33%，对最强旧scale2 76.87%提升0.46pp，支持集bootstrap95%区间[0.09,0.82025]，13胜4负3平；对旧CE76.51%为+.82[.30,1.34]，Direct72.76%为+4.57[3.33,5.83]。全部预设主比较通过；rsLoRA不是胜者。详round155_confirm_pairs.json。第二批保持同方案、全1140矩阵、支持225300..225319，父PID305835已启动，源/config/analyzer封存round155_confirm2_manifest.json，不再选择winner。

R156正式285/110轨迹/220快照/140邻域审计通过。GraphMAE FA完整预条件rank16+k50为75.00%，对同批scale2 74.48%为+.52pp（3胜2负），低LR LoRA同为75.00%；rank4 74.64%，rank16对角/无预条件均74.56%。GraphMAE2完整rank16 74.08%，低于对角/无预条件74.20%。按预设顺序锁定GraphMAE f16进入20新支持226200..226219固定确认，不能将当前收益解释成FA相对LoRA优势。见round156_pairs.json及round156_confirm_design.md。

科学累计70885条；本地sync155_confirm（目标72765）随后sync156（目标73050）串行进行，未完成前不更新传输通过数。区间仅为同图支持采样，query多轮复用，两个通过信号都不代表跨数据集泛化。

R156固定确认已启动，父PID306070；R155第二确认父PID305835已核实正常推进。两批均1140条，均等待完整审计，不提前解析选择结果。


## R157跨数据集迁移准备与cleanup118

已实查远端CiteSeer、PubMed数据和GraphMAE/GraphMAE2 checkpoint存在。发现旧审计器固定7类、旧训练入口会按数据集切换教师，故不能直接据命令可运行就视为方案迁移。新R157固定Cora教师/损失/LoRA学习率和k50，独立审计类别参数化；CiteSeer Prompt自然为6×128，LoRA256，总1024可训参数，checkpoint仍冻结且以有效权重增量计算。详round157_design.md。新源文件和清单封存，父PID306551启动Cora与CiteSeer各21行、12步的smoke及全审计（共42，不计科学正式任务）；正式105行尚未启动。既有CiteSeer R73中心化/directcenter强基线须同种子比较，完成前不作最强提升宣称。

cleanup118已完成：R152确认1140、R153开发285、R154开发285，共4266文件1950081761字节，经完整本地字节/SHA相等与删除前远端二次SHA后清除远端重复，本地保留。凭证archive118_verification.json、archive118_verified_manifest.json、cleanup_118.json。R155第二确认PID305835已转CPU完整审计，R156确认PID306070仍训练；不重启。sync155_confirm已进入本地全量SHA刷新，sync156顺序等待；科学70885及已确认传输71625暂不变。

R157原Cora smoke已审计通过；CiteSeer训练完成但审计在初始KD梯度重建处发现遗漏torch.zeros(7,128,requires_grad=True)，这是审计形状错误，不是性能失败。原文件与日志保留，新audit157_v2.py、probe157_v2.py及manifest_v2仅将该处改为数据类别数，对既存两数据集结果重审，不改训练。统一观察session34795。sync155_confirm全量本地SHA已通过：72765常规、6排除；sync156随后执行，目标73050。

R157新版审计对Cora和CiteSeer均通过：各21行、6轨迹、12梯度快照、8邻域；凭证round157_smoke_Cora_audit157.json及round157_smoke_CiteSeer_audit157.json已取回。固定CiteSeer迁移105条正式已启动父PID307688，仍不换方法或重选checkpoint，待完整审计。入口SHA另记round157_launch_sha.json。


## R155第二确认：均值正向但未重复通过（2026-09-24）

固定GraphMAE低LR rank1 LoRA+k50第二批1140/440轨迹/880快照/560邻域审计通过。准确率76.51%，对旧scale2 76.18%为+.33pp，95%支持bootstrap[-.25,.91]，11胜7负2平；对旧CE75.87%为+.64[.03,1.28]，对Direct71.85%为+4.66[3.69,5.73]。首批+0.46[.09,.82025]仍保留；第二批primary_pass=false，不能称已重复验证稳定优胜，不追加第三批Cora追逐显著性，不切换本批winner。详round155_confirm2_pairs.json；近期对照表已加入第二批，避免只展示首批成功。

R156确认1140/440/880/560及R157正式105/30/60/40均已完整审计通过，锁定分析取回中；尚未在本文解读准确率。科学累计73270条。本地sync156全量通过73050常规、6排除；sync155_confirm2父PID59137目标74190，完成前不改传输累计。

R157既有强基线65行父PID308200已启动，原R73源码及init文件原样，Directcenter/posttdnone/posttdboth/ins1、原型和k50，同种子227100..104。详round157_legacy_design.md。原geometry/teacher/封存预测审计加新旧split/graph/checkpoint SHA配对核验；旧方法没有全步梯度记录，此边界明示，不混同新版全轨迹审计。


## R156确认停止；R157迁移相对强基线停止；R158来源初审

R156固定FA rank16确认75.91%，对GraphMAE scale2 76.45%为−.54pp，支持bootstrap[-1.53,.40]，8胜11负1平；旧CE75.69%为+.22[-.62,1.07]。全部主比较未通过，停止晋级，保留开发正向结果。

R157固定Cora低LR LoRA迁移CiteSeer：62.40%，相对同教师/同loss的scale2 56.12%为+6.28pp，5/5胜；但普通LR LoRA65.28%、direct ridge65.88%均更高。补充65行原R73强基线后，Directcenter71.52%、GraphMAE ins1+k50 70.76%、GraphMAE2 ins1+k50 72.20%；固定迁移比Directcenter低9.12pp，0胜5负。明确停止原样迁移，不将匹配弱基线改善包装成总体准确率升级。原型及其他方法得分见round157_legacy_pairs.json。

旧geometry审计首轮因未列入未中心化tdnone结果而找不到额外对照文件；新独立文件audit157_legacy_geometry.py仍重建最终权重forward、teacher、center、初始化、全部kNN/diffusion，但不执行该未包含的跨方法token身份比较，输出明确uncentered_companion_identity_checked=false；训练结果不改、不重训。新版终态60行、teacher/Direct5行、封存预测65行及新旧split/graph/checkpoint哈希均通过。没有全步Adam回放，故明确归入旧终态审计证据。科学累计73335；同步队列sync155_confirm2后sync157_bundle，后者目标75500常规、6排除，未完成不提前认定传输通过。

后备R158检索ICLR2025 LoRA-Pro及作者实现，commit fe2602725b21f6b1c689462d73553f3e50b3d565。已识别权重空间Adam后因子回映射、Sylvester解、作者要求外层SGD、硬编码scale2和bf16二阶矩；不是将FA伪逆原样挪用。官方快照lorapro158_official.py、草稿lorapro158_design_draft.md保存；完整论文及原语对照待完成，目前无训练/准确率结论。


## R158 LoRA-Pro原语通过，尚未进入正式训练

已读取论文定理2.1/2.3和附录C算法1/2，并核对官方首步零B分支、权重空间Adam、Sylvester、外层SGD和bf16二阶矩。rank1/4×保留/禁用量化共4组4步无query原语完成；保留量化分支与AST提取官方函数的最大方向差2.762e−11，SciPy独立Sylvester差3.695e−13。一阶gauge抵消成立，但有限步交叉项范数最大0.011544；禁止宣称离散更新完全等于全参数Adam。禁用bf16时四步方向差最大0.05260，作为明确的本地精度变体保留。详lorapro158_source_audit.md、round158_primitive.json。

首版project闭包读取自增step，使首步第二次映射访问None Gram逆；未发生训练/准确率评估。原版保留，pro158_v2.py修正首步条件，新check158_v2.py通过，不改测试阈值。下一步接入真实GCN与独立全轨迹回放，再预注册正式矩阵。

sync155_confirm2本地全量SHA已通过74190常规、6排除；sync157_bundle父PID59463仍在执行，目标75500，未提前计入。远端当前3.3GiB空闲，GPU空闲；后续清理仍须完整本地及远端二次SHA，不删尚未备份证据。科学累计73335保持不变。


## R158真实GCN回放通过并启动；本地备份75500通过

两个checkpoint×10方法共20条三步CPU真实GCN轨迹，前向/首末梯度/全步NumPy-SciPy更新均通过。最大原梯度差2.2352e−8，最终参数回放差7.4506e−9，独立优化方向差3.7482e−13。p1/p4完整Pro、q1/q4官方二阶矩量化分支、z1/z4去Sylvester控制、a1/a4单独适配、旧l1/l4；所有新方法固定gamma1及A/B外层SGD.0025，Prompt Adam.01。文档round158_design.md、20文件round158_manifest.json已封存，父PID310395启动恒等→61行12步smoke及审计→285正式及审计。目前没有正式准确率结论。

科学累计73335保持不变。sync157_bundle全量本地SHA通过，75500常规、6排除；R155第二确认/R156确认/R157正式及旧强基线均完整保留本地。cleanup119准备核对R155首确认、R155开发、R156开发远端重复，尚未在本文认定删除完成。

R158恒等与61条12步smoke完整审计已通过，父PID310395已核实进入正式矩阵。cleanup119已删除4266远端重复文件2853446636字节（约2.85GB），完整本地及删除前远端二次SHA通过，本地全部保留；凭证archive119_verification.json、archive119_verified_manifest.json、cleanup_119.json。正式结果尚未完整审计，不提前分析。


## R158开发晋级固定确认；R159随机前缀原语核验

R158正式285/110轨迹/220快照/140邻域全部审计通过。GraphMAE全精度rank1 Pro p1+k50为76.32%，对最强旧CE75.68%为+.64pp（3胜1负1平），scale2 75.60%，Direct74.00%；量化q1同76.32%，无Sylvester z1 76.24%，旧因子Adam l1/l4为76.04/76.00%。进入确认的是整体p1，未证明Sylvester或全精度独有优势。科学累计73620。

固定GraphMAE p1，支持228200..228219，全1140矩阵，代码/配置/分析器在round158_confirm_manifest.json封存，父PID311459已启动；详round158_confirm_design.md。sync158父PID60037目标75785常规、6排除，完成前本地计数仍75500。近期对照表已加入R158开发。

R159读取DyLoRA EACL2023论文第4节/算法1及固定作者源码。rank4/16所有截断秩、冻结与非冻结算子均通过官方对照，最大前向7.106e−15、梯度1.422e−14。明确零基rank接口、双sqrt缩放与alpha/b一致，并证明均匀nested前缀的期望带谐和分量权重而非完整BA。dense Adam两步例子显示被detach的早期分量仍移动0.00670058；不把梯度冻结等同参数冻结，也不冒称已复现完整作者训练。详dylora159_design_draft.md和round159_primitive.json；尚无图训练结果。


## R159真实GCN轨迹通过并启动（2026-09-24）

两checkpoint×10方法20条三步轨迹，通过独立随机秩/子集序列重建、外积求和前向、首末梯度和全步NumPy Adam回放；最大梯度差2.9803e−8，最终参数差7.4506e−9。n4/n16随机前缀R/b、u4/u16相同b随机子集、c4/c16前缀缩放1、f4/f16固定全秩、旧l1/l4。推理固定完整秩gamma1，不按query挑秩，正则使用完整推理DeltaW平方范数；未抽中分量仍可能受正则/Adam动量影响，不声称冻结。

round159_design.md、20文件round159_manifest.json封存，父PID312030已启动恒等→61条12步smoke→285条正式与审计；当前尚无正式准确率结论。R158固定确认父PID311459已实查正常推进，仍不提前解析/重选。科学累计73620。sync158全量SHA已通过75785常规、6排除，本地同步无待办。

R159随后已观察到smoke审计通过：61条、24轨迹、48快照、30邻域；父PID312030继续原计划进入正式，不重启。


## R158确认停止；R159正向结果保留但重复胜者不追加确认；R160原语

R158固定GraphMAE rank1 Pro，确认1140/440轨迹/880快照/560邻域审计通过。77.38%，对旧scale2 77.33%为+.05pp，95%支持bootstrap[-.51,.60]，9胜9负2平；对旧CE76.98%为+.40[-.18,1.00]。对低LR LoRA l4 77.91%为−.53[-.90,−.20]，对无Sylvester z1 77.47%为−.09[-.17025,−.01]；量化q1得分完全相同。停止Pro扩大，不在确认后换成对照winner；保留开发+.64证据。

R159正式285/110/220/140审计通过。嵌套前缀n16与随机子集u4同71.84%，相对旧scale2 71.04%为+.80；预设全候选winner仍为GraphMAE l4 71.88%，+.84（3胜2负），不是DyLoRA。机器开发gate=true照实保留。该方案与R155已做两批固定确认的方法、checkpoint、学习率、读出完全相同，因此人工执行决定复用已存在R155证据而不再跑第三批同方案Cora确认；这是明确的去重复决定，不把机器gate改写成失败，也不改选次优动态方法冒充预选winner。动态前缀比固定R16 71.00%有局部正向收益，作为后续候选记录，未声称重复验证。

R160借鉴联邦学习FedPara的低秩Hadamard乘积，参考固定PEFT LoHa实现。已读取原文2.2命题1/2，明确从头联邦重参数化与本地冻结GCN增量适配不同。官方AST HadaWeight三秩r1/2/4有限差分梯度、r²外积展开、单零因子no-op且梯度存活、双零死梯度检查通过，最大展开差1.777e−15。下一步必须有相同参数量的LoRA rank2r及固定随机乘积分支控制，不能以更高代数rank当准确率保证。详loha160_design_draft.md及round160_primitive.json，尚无图训练结果。

科学累计75045。本地sync159（76070）后sync158_confirm（77210）串行进行，完成前已通过传输累计仍75785常规、6排除。


## R160 LoHa真实GCN回放通过并启动；cleanup120完成

完整LoHa r4/r8、冻结第一乘积控制、同预算普通LoRA r8/r16、单独LoHa、旧l1/l4，共两个checkpoint20条3步轨迹通过独立初始化、forward、首末梯度和全步Adam回放。最大梯度差2.2352e−8、参数回放差2.9803e−8。PEFT式三个kaiming因子+下因子V2零，gamma1；不把普通LoRA同预算称为同初始化/同梯度。方法共用第二GCN权重和中间Prompt，没有终端融合。

round160_design.md与20文件round160_manifest.json封存，父PID313758启动恒等→61条12步smoke→285正式全审计。暂未取得正式准确率结论。R159全量本地SHA通过76070常规、6排除；R158确认sync158_confirm父PID60479仍同步，目标77210，不提前计入。科学累计75045保持。

cleanup120已完成：R155第二确认1140/R156确认1140/R157正式105，共5936文件4051434105字节（约4.05GB），完整本地及删除前远端二次SHA核验后删除远端副本，本地完整证据保留。凭证archive120_verification.json、archive120_verified_manifest.json、cleanup_120.json。


## R160 LoHa开发停止扩大；R161 LoKr原语检查

R160正式285/110轨迹/220快照/140邻域审计通过。GraphMAE完整LoHa h4/h8为74.00/74.08%，低于旧CE74.44%；固定第一随机乘积f4/f8为74.36/74.32%，也未超过。GraphMAE2 h8 73.84%对自身scale2 73.20%有+.64局部正向，但同预算普通LoRA r8 73.96%更高。不能把代数rank更高归为精度收益。

全候选开发winner仍GraphMAE l4 75.00%，对最强旧CE+.56，3胜2负，机器gate=true照实保留；按本轮预先声明的重复方案规则，不追加第三批相同Cora l4确认，引用R155两批确认边界。LoHa方向停止扩大，保留GraphMAE2局部结果。科学累计75330。

R161已读LyCORIS ICLR2024 LoKr小节与固定PEFT源码，拟C⊗(UV)在第二GCN联合Prompt。factor4/8、内部rank4适配272/192参数，普通rank1 LoRA为256，明确只是接近预算。PEFT默认C零与LyCORIS式V零的首步可学习参数不同，两者均no-op；原语六案例检查通过，包括官方factorization/make_kron、einsum索引、有限差分梯度、矩阵rank乘法和重排rank1结构约束。详lokr161_design_draft.md、round161_primitive.json。目前尚未接入真实GCN，不宣称准确率或完整官方训练复现。

R158确认全量本地SHA通过77210常规、6排除；sync160已启动目标77495，未完成前不更新已通过数。


## R161 LoKr真实GCN回放通过并启动（2026-09-24）

两checkpoint×10方法共20条3步轨迹，通过独立初始化、固定行列置换重建、einsum Kronecker forward、首末梯度及全部NumPy Adam回放。最大梯度差2.2352e−8、参数差2.9803e−8。factor4/8、内部rank4，PEFT C零与LyCORIS V零分开；另有固定通道置换、无Prompt、旧l1/l4。共享第二GCN有效权重与中间Prompt，不变teacher/readout，不做预测融合。

round161_design.md、20文件round161_manifest.json封存；父PID315118启动恒等→61条12步smoke→285正式与审计，尚无正式准确率结论。若开发winner是R155相同l4，按预注册去重复，不追第三批Cora确认；新配置才锁定确认。sync160已完整本地SHA通过77495常规、6排除；科学累计75330，正式R161尚未计入。


## R161结果与R162 GLoRA准备（2026-09-24）

R161正式285/110轨迹/220快照/140邻域审计通过。GraphMAE LoKr k8 77.48%，对scale2 77.16%为+.32；旧LoRA l4 77.56%仍更高。全候选winner为l4，对最强旧根+.40，3胜2负，对Direct74.44%为+3.12。机器gate=true保留；按预注册重复配置规则，复用R155两批确认，不开启第三批同图l4确认，也不事后改选LoKr。保留LoKr局部正向，停止本配置扩大。科学累计75615。证据round161_pairs.json、round161_gate_audit161.json。

R162已读GLoRA原文及作者固定提交390467ddf529d747168709ce2f007a85a0399cca，PEFT固定实现亦保存。确认是Hadamard权重缩放，不是论文记号容易误读的矩阵乘法；C/D/E偏置分支可合成自由偏置。服务器rank1/4原语通过作者前向、梯度有限差分、PEFT初始化、对角展开和偏置化简检查，并验证图聚合前后常量不能直接互换。拟完整分支/仅权重/合成偏置/仅乘法/普通LoRA加偏置等匹配对照；详glora162_design_draft.md。尚未真实GCN训练，不宣称准确率。

sync161 PID61120正在全量传输与本地SHA刷新，目标77780常规、6排除；完成凭证前仍沿用77495已通过数。


sync161 completed: 77780 regular SHA-verified rows, 6 known excluded; sync161_complete.json retained.


## R162 GLoRA真实GCN接入与启动（2026-09-24）

20条两checkpoint×10配置的3步实际GCN轨迹均通过独立回放，最大梯度差9.5367e−7、参数差7.4506e−9。共享第二GCN的有效权重及聚合后bias，与原中间Prompt共同训练；完整A/B/C/D/E、仅权重、合成自由偏置、仅乘法、普通LoRA加偏置及旧l1/l4分别保留。偏置分支以物理Deltabias正则，避免重复参数化引入不同参数范数惩罚。原语已证明C/D/E可合并；不将额外分支数量当作机制贡献。

已封存round162_design.md、21文件round162_manifest.json。PID316518依次执行恒等检查→61条12步smoke→285条100步正式与独立审计。支持232100..104，预选k50，完整批次审计前不分析；若winner为旧l4，复用R155两次确认，不追加第三次。新配置winner达门槛后锁定20新支持确认。目前没有GLoRA正式准确率结论。

科学累计仍75615，本地SHA77780常规、6已知排除；R162尚未计入。


## R162开发晋级与R163来源核查（2026-09-24）

R162全285/110轨迹220快照140邻域通过。GraphMAE g1+k50为76.60%，最强旧scale2 75.64%，+.96（3胜1平1负）；Direct73.48%。完整g1高于仅权重w1 76.12、等效自由偏置e1 75.92、普通LoRA加偏置b1 75.88、旧l1 75.68/l4 75.60。锁定g1进入20新支持232200..219确认，全1140条，主比较和机制对照已预注册round162_confirm_design.md/manifest；未完成确认前不称稳定提升。科学累计75900。

R163已读LoRA-SB全文和固定作者源码，验证非零首步初始化、BRA、SVD投影与有限差分。中心Adam并不等于全权重Adam再投影，保留反例；bf16正交性也仅近似。方案草案包含非零/零初始更新、sign/raw-gradient/随机子空间和旧LoRA，详lorasb163_design_draft.md。尚未接入实际GCN。

sync162 PID61550目标78065常规/6排除，完成凭证前沿用77780。


R162固定确认已启动PID317579，1140条及全部独立审计。


## R163 LoRA-SB图模型接入与启动（2026-09-24）

20条真实GCN三步轨迹全部通过，最大梯度差8.9407e−8、参数差7.4506e−9。初始训练loss的全权重梯度独立重算，SVD由NumPy重做并比较不受符号影响的子空间及物理截断矩阵；全部Adam回放、首末梯度与前向通过。初始化仅使用支持标签及固定teacher，未使用query标签。rank4/16中心分别16/256参数，外加Prompt896；signstep在100步之前已注入一次非零截断更新，零中心对照用于分离此效果。

21文件round163_manifest.json与round163_design.md封存。PID318059启动恒等→61条12步smoke→285正式及审计，支持233100..104。GLoRA固定确认PID317579继续运行，不据部分结果改配置。科学累计仍75900，R163及R162确认尚未计入。

sync162已全部本地SHA通过78065常规、6排除。cleanup121完成第一阶段完整本地匹配：R158确认1140、R159开发285、R160开发285，共4266文件3240712208字节；远端二次SHA及删除正在执行，完成凭证前不报告删除完成。


cleanup121已完成：4266文件3240712208字节远端重复副本已删除；本地完整保留，删除前远端第二次完整SHA通过。凭证cleanup_121.json。


## R163停止扩大；R164 EVA准备（2026-09-24）

R163正式285/110轨迹220快照140邻域审计通过。GraphMAE sign首步s4/s16为76.72/76.16%，旧scale2 76.88%；同sign子空间零中心z4 76.92%，随机r4最高77.04%但仅+.16，未达预设+.20门槛。GraphMAE2 s4 76.16%对自身scale2 75.44%有+.72局部改善，但旧CE75.64和同批普通LoRA76.24需一并考虑。全局停止扩大，不改选原型或另开确认；保留局部正向。

R162确认1140/440轨迹880快照560邻域审计通过，锁定分析正在获取（session51379），科学累计77325（此前75900+R163 285+R162确认1140）。未读取配对分析前不判定确认成败。

R164已读EVA全文与固定PEFT源码。官方PCA先中心化激活；拟全部节点中心化/未中心化、support-only和随机正交初始化A，B零且A/B共同训练。单层固定rank1/4不声称跨层rank重分配，完整SVD不声称增量过程复现。服务器原语通过官方PCA/独立SVD子空间、尾能量、no-op和梯度有限差分。详eva164_design_draft.md，尚无实际GCN结果。

sync163 PID61906目标78350常规/6排除，尚待完成；R162确认尚未同步。


R162固定确认判定STOP：GraphMAE g1 77.22%，旧scale2 77.41%，差−.19百分点，CI[-1.11,.56]，10胜9负1平；旧CE76.92，差+.30[-.55,1.09]；Direct73.10，差+4.12[3.17,5.22]。l4 77.34、仅权重w1 77.24、合成偏置e1 77.25、LoRA加偏置b1 77.27均不低于完整g1。primary_pass=false、mechanism_pass=false，不改选次优或补挑读出。保留开发+.96但不称稳定改善。证据round162_confirm_pairs.json。


## R164 EVA真实GCN通过并启动（2026-09-24）

20条两checkpoint×10配置真实GCN三步轨迹全部通过，最大梯度差5.9605e−8、参数差2.9802e−8。独立重建全图或support激活统计，并用NumPy SVD校验初始化；初始化统计中心化不改变正式forward。rank1/4两因子均训练，共256/1024适配参数加896Prompt；共享第二GCN与中间Prompt，无终端融合。全图激活包含无标签query特征，符合当前transductive设定，support-only作为对照。

21文件round164_manifest.json及round164_design.md封存；PID319728启动恒等→61条12步smoke→285正式及全量审计，支持234100..104。尚无正式准确率结论。新winner完整审计后才决定20支持确认；旧l4不追加第三批。科学累计仍77325，未计R164。

sync163已全量SHA通过78350常规、6排除；R162确认同步PID62047启动，目标79490常规、6排除，完成凭证前不提前更新数值。


## R164开发平局修正并固定确认；R165原语（2026-09-24）

R164正式285/110轨迹220快照140邻域审计通过，科学累计77610。GraphMAE u1未中心化rank1和旧l4均74.76%，scale2 74.56%；中心化c1 74.68、support s1 74.72，随机r1 74.44。GraphMAE2 c4/s4均74.36%对自身scale2 72.64%为+1.72，但仍不及跨checkpoint最强根。

原analyze164.py用float均值最大值，因约1e−14舍入将实际同分的l4置前。原脚本和原pairs保留；新增analyze164_v2.py/round164_pairs_v2.json按整数正确数总和排序，不改候选、数据或预注册顺序，因此u1按原顺序胜出。准确名称是未中心化EVA初始化变体，不是标准中心化EVA优势，且与旧LoRA平局。锁定GraphMAE u1+k50进入20新支持234200..219确认，PID320910，配置/25+文件清单已封存；未完成确认不称稳定提升。

R165已核查CorDA全文与固定PEFT源码；官方SVD(WC)后右因子乘C逆，冻结残差减回B0A0保持起点。rank1/4×top/tail四案例官方重建/no-op/有限差分通过，证明C倍乘会改变因子坐标尺度，虽BA不变却可能影响Adam。草案加入协方差尺度归一化、官方尺度、tail及普通SVD同学习率对照，详corda165_design_draft.md；尚未接入真实GCN。

R162确认同步PID62047仍运行，同步结束后再顺序启动R164开发同步；不并发运行全量本地SHA刷新。


## R165 CorDA图模型接入与启动（2026-09-24）

20条两checkpoint×10配置真实GCN三步轨迹全部通过，最大梯度差1.1921e−6、参数差5.9605e−8。协方差、阻尼、逆矩阵、NumPy独立SVD和初始因子校验通过；训练用W+(BA-B0A0)维持起点，首末梯度和完整Adam轨迹独立回放。原始统计尺度、均值对角归一化尺度、尾部分量及普通SVD同学习率分别保留；均与共享第二GCN及原中间Prompt联合，不做读出融合。

20文件round165_manifest.json与round165_design.md封存，PID321388启动恒等→61条12步smoke→285正式及审计，支持235100..104。分析预先使用整数正确数总和排序并按预注册候选顺序解平局。新配置胜出且达门槛才确认，旧l4不追加第三批。尚无正式准确率；科学累计仍77610。

EVA未中心化u1固定确认PID320910仍运行。R162确认全量本地SHA已通过79490常规、6排除；sync164 PID62377顺序启动，目标79775，完成前不提前更新。


## R164确认通过旧根比较但无额外初始化优势；R165停止扩大（2026-09-24）

R164固定u1完整1140/440轨迹880快照560邻域审计通过：GraphMAE u1 76.51%，旧scale2 75.68%，+.83 CI[.25,1.39]，13胜6负1平；旧CE75.04%，+1.47[.78,2.18]；Direct72.08%，+4.43[3.46,5.63]。所有预设旧根主比较通过，保留这项准确率提升，不归为有效性失败。

但同批旧l4 76.79%，u1差−.28[-.58,.04]；中心化c1 76.81%，差−.30[-.65,.03]；support-only76.69%，随机r1 76.45%，相对随机仅+.06[-.42,.48]。mechanism_pass=false，没有额外初始化收益证据。当前保留为对旧根正向的可用备选，不提升为优于已知LoRA的方法，不事后改选c1，也不因primary_pass再追同图显著性。samegraph/query重复使用的边界保留。证据round164_confirm_pairs.json。

R165 CorDA正式285/110轨迹220快照140邻域通过。GraphMAE normalized top1/top4 70.60/70.80、官方尺度70.88/70.32，tail1 72.56，均低于scale2 73.08。GraphMAE2 top1 70.68对自身scale2 69.24为+1.44，但同批l1 70.72更高。全候选仍旧l4 73.28、+.20，按预注册复用R155两次确认，不追加第三批；CorDA停止扩大。科学累计79035。

R166已读LoRA-RITE原文HTML及固定作者PyTorch重实现，独立NumPy回放rank1/4×escape开/关五步通过，最大参数差1.3323e−15；两倍尺度测试物理乘积相同。源码relative_epsilon=True触发未定义p_new，保持默认False；不覆盖源码或夸大为完整论文复现。详rite166_design_draft.md，尚未实际GCN。

sync164已完整SHA通过79775常规、6排除；sync165_bundle PID62675同步R164确认及R165开发，目标81200/6，未完成前不更新。


## R166 LoRA-RITE真实GCN回放通过并启动（2026-09-24）

20条两checkpoint×10配置真实GCN三步轨迹通过，最大梯度差2.2352e−8、参数差9.3132e−9。因子训练float32，作者RITE CPUdouble proxy与状态逐步更新再回float32；独立NumPy回放覆盖矩变换/escape/QR/pinv/逆平方根与量化。Prompt继续独立Adam，防止误入成对矩阵预条件。预条件前后均显式有限检查，特征值检查全部记录，异常即失败，不静默nan_to_num。

22文件round166_manifest.json和设计封存。PID323104启动恒等→61条12步smoke→285正式与审计，支持236100..104，rank1/4、escape开/关、两学习率、同初始化Adam及旧l1/l4。尚无正式准确率，不计科学累计（仍79035）。

cleanup122第一阶段完成：R161/R162/R163开发各285，共2136文件1072737712字节，与本地全部SHA匹配；远端二次核验及删除session89884正在执行，凭证前不报告完成。sync165_bundle PID62675仍待完成，目标81200常规/6排除。


cleanup122已完成：2136文件1072737712字节远端重复副本删除，删除前远端二次完整SHA通过，本地全部保留。R166 smoke61完整审计通过，正式285正在运行。


## R166停止扩大；R167顺序累积适配准备（2026-09-24）

R166全285/110轨迹220快照140邻域审计通过。GraphMAE noescape rank1为73.96%，旧scale2 73.80%，+.16，2胜2负1平，未达+.20预设门槛。escape rank1 73.92，匹配Adam rank1 73.64；GraphMAE2低学习率rank4 72.88对自身72.48为+.40，但旧l1 73.32更高。停止扩大，保留局部正向，不改阈值追确认。科学累计79320。

R167已读ReLoRA全文及固定作者源码，确认合并后B零保持函数、动量重置保留Adam step、默认所谓清零实为随机剪99.9%。四类语义检查及rank累积反例通过；草案将merge与仅动量/学习率重置分离，详relora167_design_draft.md。明确是从checkpoint借鉴顺序累积，不称全模型预训练复现；全尺寸累计更新存储成本单列。尚未实际GCN。

sync165_bundle PID62675仍运行，当前传输通过数仍79775；完成后再同步R166，顺序刷新全量本地SHA。


sync165_bundle已完成81200常规SHA/6排除；sync166 PID63069已顺序启动，目标81485/6。


## R167数值修订、完整开发及固定确认（2026-09-24）

原ReLoRA独立幅值动量剪枝31步smoke在末梯度审计失败（差2.8610e−5、容差2e−6），正式未启动。GraphMAE2 m1/n1留下一个一阶矩非零而二阶矩零的位置，参数幅值1312.78/7338.82；严格清零z1为.341。原20条31步探针虽混合容差通过，最大参数差实际.000488，不代表数值稳定。原代码、日志和失败数据保留，不计正式行数。

v2让两种矩共用二阶矩99%阈值掩码，保持Adam step；明确是本地paired-moment变体。20×31真实GCN回放最大参数差1.4901e−7、梯度差1.4901e−8；阈值未放宽。21文件新manifest封存，PID325506已结束：61条31步smoke通过，正式285/110轨迹220快照140邻域及25/50/75全部事件审计通过。

预定整数计数选出GraphMAE2 f4 76.24%，对同批最强旧scale2 75.84%为+.40，4胜0负1平；Direct74.72%。f4是连续rank4普通Adam，不包含合并重启。GMAE n1合并常率76.00%，GMAE2非合并动量剪枝o1 76.20%；没有把这些局部表现混作胜出机制。科学正式累计79605。

按准确率目标保留并固定f4，20新支持237200..219、100步、后500query、全控制1140条确认PID326451已启动。配置/设计/源码/开发结果一起哈希封存；主比较两ck旧CE/scale2及Direct，旧l1/l4和v4用于说明增量，不事后换胜者。此阶段尚无确认结论。证据round167_pairs_v2.json、round167_confirm_design.md。

sync166已完成81485常规SHA/6排除；sync167 PID63447正在同步R167v2正式，目标81770/6，完成前不更新通过数。


## R168 RandLoRA源码语义与图轨迹准备（2026-09-24）

新读ICLR2025 RandLoRA全文及作者提交371d82f4ef713fc4c35d6cb581d37eef558dbb31；固定随机A/B，仅lambda/gamma训练，第二层权重适配与原Prompt共同训练。原作者基矩阵展平维度顺序与直观逐基求和不同；严格保留源码并用重排B后的独立einsum核验。rank16/32/64原始前向与自定义backward匹配，最大double梯度差5.6843e−13，非零随机系数下实际秩128；初始lambda零、gamma梯度零均符合设计，不以满秩保证准确率。

20条真实GCN三步轨迹通过独立Adam与首末梯度回放，最大参数差1.1176e−8、梯度差3.3528e−8。拟比较三个基矩阵rank×两学习率、单基低秩控制、旧LoRA；21文件manifest封存。完整设计round168_design.md；尚未正式运行，不计科学行数。

R167固定确认1140预测已生成，原PID326451的审计子进程327012仍在运行，未分析未晋级。sync167完成81770常规SHA/6排除。cleanup123已完成R162确认/R164开发/R165开发本地对照：4266文件2536167726字节全部一致；二次远端SHA及重复副本删除正在执行，尚不报告删除完成。


## R167固定确认结束；R168正式运行（2026-09-24）

R167全1140/440轨迹880快照560邻域审计通过。固定GraphMAE2 f4 73.24%，相对自身scale2 72.31%提升+.93，区间[.47,1.44]，16胜2负2平；相对自身CE71.92%为+1.32[.75,1.98]，相对Direct70.29%为+2.95[1.86,3.87]。保留同checkpoint准确率提升。

但更强GraphMAE scale2 75.42%，固定候选差−2.18[-3.79,-1.03]，2胜17负1平；同GraphMAE2旧l1 73.27%也略高，差−.03[-.33,.30]。主比较未全部通过，停止扩大，不事后改选GraphMAE s1 76.04%追确认，不称重启机制有效。科学累计80745。完整round167_confirm_pairs.json。

cleanup123完成：4266远端重复文件2536167726字节经本地完整SHA及删除前第二次远端SHA后删除，本地全保留；远端可用4.2GiB。R168 PID327329启动，61条smoke完整审计已通过，正式285运行中。sync167_confirm PID63874正在同步R167确认，目标82910常规SHA/6排除；现已通过仍81770。


## R168旧LoRA胜出并确认；R169真实GCN准备（2026-09-24）

R168正式285/110轨迹220快照140邻域完整审计通过。GraphMAE RandLoRA最佳低学习率k64为76.48%，对scale2 76.24%为+.24，但旧l1为77.20%胜出，+.96，2胜2负1平（单支持+4.4拉动），旧l4为76.76%。固定旧l1，不冒称RandLoRA收益，也不更换候选；它的因子Adam.01区别R155 l4的.0025。20新支持238200..219、100步、后500query、全控制1140条确认PID328388已启动且仍在运行。科学累计81030。

R169已读KRAdapter全文和固定作者代码，发现论文列式与README行式、初始化及缩放不一致；README merge转置错误、unmerge的weigh拼写错误已实际验证，均不调用。三类定义分开实现，并加入参数预算接近的rank11/rank12 LoRA控制。20条真实GCN三步回放通过，最大参数差2.9802322e-08、梯度差2.9802322e-08。尚未正式运行，详round169_design_draft.md。

sync167_confirm已完整SHA通过82910常规/6排除；sync168 PID64158顺序同步R168开发，目标83195/6。


## R169正式启动（2026-09-24）

21文件源码/配置/设计manifest已封存。PID329113启动恒等验证→61条12步smoke→285正式。smoke完整24轨迹48梯度快照30邻域审计通过，正式运行中，尚无准确率结论。R168固定确认PID328388仍处于独立审计，不以训练完成代替审计完成。

cleanup124第一阶段完成：R164确认、R166开发、R167v2开发共4266文件2938936271字节，与本地完整字节/SHA一致；远端二次核验和删除正在执行，未见凭证前不报告完成。sync168仍在刷新全量本地SHA，已通过数仍82910/6。


cleanup124已完成：4266文件2938936271字节远端重复副本，经删除前第二次完整SHA后删除；本地完整证据保留，远端可用4.8GiB。R168确认审计子进程328947实时在运行，R169正式仍在运行，无重启。


## R168固定确认停止；R169旧l4重复胜出（2026-09-24）

R168确认1140/440轨迹880快照560邻域审计通过。固定GraphMAE l1为76.08%，同批scale2 76.26%，差−.18，95%区间[-.84,.35025]，9胜11负；旧l4 76.16%也更高。相对旧CE75.32%为+.76[.08,1.37]，相对Direct72.48%为+3.60[2.88,4.32]，保留这些改善但不超过强根。RandLoRA低学习率k16/k32/k64为76.27/76.25/76.26%，与强根近似；不事后改选。停止扩大固定l1，机制比较亦未全部通过。

R169开发285/110轨迹220快照140邻域审计通过。GraphMAE论文p1/p2均74.60%，README r2 75.04%，toy t1 75.16%，相对旧CE74.88%为+.28，但l1 75.40%、l4 75.84%更高。最高l4对旧根+.96，按预设复用R155两次确认，不追第三次或事后选择t1。停止扩大KRAdapter，保留toy局部改善。科学累计82455。

sync168完整SHA完成83195常规/6排除。sync169_bundle PID64427顺序同步R168确认与R169开发，目标84620/6，尚未完成。R170开始读ICLR2025正弦低秩全文，拟用相同初始导数的线性控制排除纯幅值效应，尚未实现。


R170来源核对完成：旧项目页404，经作者当前主页定位Sine-Low-Rank提交6dd03a0fe187960b098049b04cd9f286cbb25f6a；实际src为sin(weight product)/sqrt128，不是sin(node output)。作者forward四组rank1/4×freq5/20通过独立前向及梯度，最大梯度差7.1054e−15；零B初始化与缩放线性控制初始梯度一致。下一轮八新配置包含一一匹配线性控制，详round170_design_draft.md。尚无真实GCN与准确率结果。


## R170正弦权重更新真实GCN通过并启动（2026-09-24）

20条两checkpoint×十配置真实GCN三步轨迹通过独立CPU首末梯度、前向与全部NumPy Adam回放，最大参数差7.4506e−9、梯度差2.6077e−8。频率5/20×rank1/4均配相同初始化、相同初始导数的线性更新控制；因子.0025、Prompt.01，权重更新物理正则相同。

20文件round170_manifest.json与正式设计封存，PID330600启动恒等→61条12步smoke→285条100步正式，支持240100..104。整数正确数选胜者及既有+.20门槛不变；确认前不因配置是否正弦而调整选择。正式尚未审计，不计科学行数，累计仍82455。

sync169_bundle PID64427仍在运行，已完成本地SHA数仍83195常规/6排除，目标84620/6。远端启动前可用4.6GiB。


R170 smoke61完整审计通过（24轨迹48快照30邻域）；原PID330600已进入正式285，不重启。


## R170停止扩大；R171权重平均原语通过（2026-09-24）

R170全285/110轨迹220快照140邻域审计通过。GraphMAE sine20 rank4为75.96%，匹配线性75.92%，强根scale2 76.44%，差−.48，未达门槛。其余GraphMAE sine/linear配对差均−.04；GraphMAE2 sine5rank1 75.04%对自身74.96%仅+.08且线性同为75.04%。停止扩大，不把微小非线性差值解释为可靠增益。科学累计82740。

R171已读Model Soups全文与作者固定源码，实际执行uniform平均语句；物理更新与LoRA因子平均不等价的符号反例通过，单模型合成前向误差5.6843e−14。预定同ck同支持的两学习率成员与三种混合权重，加入因子平均、仅Prompt/仅W及端点控制；原理不是末端概率融合。区别R17同轨迹Prompt平均，详round171_design_draft.md。尚无真实GCN。

sync169_bundle已完成84620常规SHA/6排除。sync170 PID64887正在同步R170正式，目标84905/6。


## R171真实GCN合成与成员审计通过并启动（2026-09-24）

两checkpoint共4条实际成员三步轨迹、8首末快照以及20合成结果通过独立回放。最大参数差7.4506e−9、梯度差2.2352e−8。原probe的trajectories字段沿用旧模板为20，实际表示合成结果，已单列round171_probe_interpretation.json澄清，不能算20次独立训练。

20文件manifest已封存，PID331919启动。正式285预期30实际轨迹（20成员+10scale2）、60快照、100合成、140邻域；smoke61预期8轨迹16快照20合成30邻域。成员只保存一次，所有合成和端点引用SHA；对每个成员独立验证checkpoint/图/支持/标签/步数、teacher/gate及全部Adam/梯度。禁止跨ck或跨支持混合，最终重新计算单模型表示与原型读出。尚无正式准确率。

sync170已完成84905常规SHA/6排除；科学累计仍82740。新增verify171_member_transport.py用于后续本地成员引用核验，避免通用sealed/neighbor传输核验漏掉新增文件。


## R171共享数据与跨设备取证修正（2026-09-24）

原PID331919的smoke失败：两成员重复计算teacher/gate出现1.4901161e−8差异，触发逐位一致检查，正式未启动。v2改为预先计算一份teacher/gate供两成员共用，保留检查；4成员/20合成CPU回放通过，GPU smoke训练通过。v2 PID332562在审计又因错误要求CPU/GPU raw传播特征哈希一致而停止。独立诊断确认只有raw不同，model/edges/support/labels/seed/steps全部相等。原两版本、manifest与失败数据保留，不计正式条数。

v3保存单份shared.pt（raw/teacher/gate），以SHA绑定成员与合成，再用既有容差验证CPU独立传播与teacher/gate。全部其他来源字段仍严格一致，没有放宽阈值；旧训练源码保持不变，新增成员核心仅接受共用teacher/gate。22文件manifest封存，PID333228启动；smoke61已完整通过：8真实训练轨迹16快照、20合成、30邻域，4成员文件/2共享输入。正式285运行中。

verify171_member_transport.py已扩展核对共享输入SHA，不仅核对两个成员；后续本地传输审计必须执行。科学累计仍82740，本地SHA84905常规/6排除。


## R171通过开发门槛并锁定确认；R172来源语义检查（2026-09-24）

R171v3正式285完整审计通过：30实际训练轨迹60首末快照、100合成140邻域、20成员文件10共享输入。GraphMAE pw（只平均W，Prompt保留l4）76.60%，对强根scale2 76.24%为+.36，3胜2负，Direct73.16%。同批l1 76.52%、l4 76.44%，增量仅+.08/+.16，不能据此称可靠合并收益。全部平均w25/w50/w75为76.52/76.32/76.28%，没有事后切换到其中一个。

固定pw做20支持241200..219确认，PID334320已启动且运行。预期1140预测、120实际轨迹240快照、400合成560邻域、80成员40共享输入；所有主比较及端点/控制报告，详round171_confirm_design.md。科学累计83025。

R172已读TIES原文和固定源码，九个裁剪/符号/均值原语与独立NumPy一致；明确记录阈值并列、K=1原错误及全局平票偏负分支。拟只合并物理deltaW、保留l4 Prompt，并用裁剪平均和范数匹配控制隔离幅值效应，尚无图训练，详round172_design_draft.md。

sync171 PID65364正在同步R171v3正式，且新增成员/共享输入传输核验后再刷新全量SHA，目标85190/6，已通过数仍84905/6。


## R171固定确认停止；R172 TIES真实GCN通过并启动（2026-09-24）

R171确认全1140、120实际轨迹240快照、400合成560邻域及80成员40共享输入审计通过。固定GraphMAE pw为75.63%，旧scale2 75.46%，+.17 CI[-.39,.74]，9胜9负2平；旧CE74.82%，+.81[.07975,1.52]；Direct71.43%，+4.20[3.03,5.39]。保留均值正向与相对较弱基线提升，但主比较未全部通过。端点l1同为75.63%、l4 75.65%，完整平均w50 75.77%，不能事后改选w50。停止扩大pw，不声称可靠合并增益。科学累计84165。

R172真实GCN两ck共4成员轨迹8快照、20合成通过；最大梯度差2.9802e−8。独立NumPy外积/排序/符号选择/非零平均回放，记录实际裁剪数、符号冲突、未解平票、范数缩放及输出范数。23文件manifest封存，PID335198启动恒等→smoke61→正式285；尚无正式准确率，详round172_design.md。

sync171完整成员/共享输入及全量SHA通过85190常规/6排除；sync171_confirm PID65587同步R171确认，目标86330/6。cleanup125已完成4266文件2636910340字节远端重复副本清理，删除前本地与第二次远端SHA均通过，本地完整证据保留。


R172 smoke61完整审计通过：8实际轨迹16快照20合成30邻域；PID335198正式285运行中。


## 172轮保留价值汇总与R172开发审计完成（2026-09-24）

按用户要求完成[172轮方案保留价值说明](172轮方案保留价值说明.md)及[逐轮JSON清单](172轮保留价值清单.json)。按本次“当前可用＋历史有效参考”口径共31轮：25轮证据归并11组可用配方（5组主线、6组限定备用），6轮历史有效归档；另3轮不稳定观察、1轮仅诊断、136轮不新增方案、1轮待确认。各轮只能计一次；11为用途配方包数，不是独立机制或全部CLI配置数。未以原创性淘汰；负结果、弱基线阳性与审计工具均保留，不删除资产。

本次检查R172原PID335198已经结束，日志与正式audit172均完成：285条、30实际训练轨迹、60快照、100合成、140邻接。执行原analyze172.py，固定开发选择GraphMAE/t20+k50=75.44%，scale2+k50=75.16%，+0.28pp；Direct74.00%，+1.44pp；相对根2胜3负，尚未固定20支持确认，不计有效方案。全局k50筛选passed不代表胜所有读出（同批scale2原型75.52%）。未重新启动训练或切换候选。

服务器普通科学审计累计84165+285=84450；sync171_confirm_complete.json确认本地常规传输86330条、已知排除6条。R172本次只同步配对分析及正式审计JSON，完整结果尚未计入本地全量传输核验。持续研究目标仍active；本次文档汇总完成，后续入口为R172固定确认设计与结果完整归档。


## 五组优先主线独立整理完成（2026-09-24）

按用户要求，5组主线已分别放在本地持续推进一级目录，保留独立方案说明、相关工作、源码依赖快照、关键证据、历史复现计划及SHA清单。[统一索引](/Users/shq/Research/GP/持续推进/五组优先主线索引.md) · [核查记录](/Users/shq/Research/GP/持续推进/五组优先主线核查.json)。

- [A1 Cora：平滑岭教师条件Prompt、整体CE缩放与概率传播](/Users/shq/Research/GP/持续推进/主线01_Cora_条件Prompt与概率传播/方案说明.md)；[相关工作](/Users/shq/Research/GP/持续推进/主线01_Cora_条件Prompt与概率传播/相关工作.md)。
- [A2 CiteSeer：中心化修正原型直接教师](/Users/shq/Research/GP/持续推进/主线02_CiteSeer_中心化修正原型教师/方案说明.md)；[相关工作](/Users/shq/Research/GP/持续推进/主线02_CiteSeer_中心化修正原型教师/相关工作.md)。
- [A3 CiteSeer：锐化教师门控Prompt与限定权重步长匹配](/Users/shq/Research/GP/持续推进/主线03_CiteSeer_锐化门控Prompt与步长匹配/方案说明.md)；[相关工作](/Users/shq/Research/GP/持续推进/主线03_CiteSeer_锐化门控Prompt与步长匹配/相关工作.md)。
- [A4 PubMed：原型教师条件Prompt与限定权重隐藏VAT](/Users/shq/Research/GP/持续推进/主线04_PubMed_修正教师Prompt与VAT/方案说明.md)；[相关工作](/Users/shq/Research/GP/持续推进/主线04_PubMed_修正教师Prompt与VAT/相关工作.md)。
- [A5 Actor：固定自身消息与中心化几何校正](/Users/shq/Research/GP/持续推进/主线05_Actor_固定结构与几何校正/方案说明.md)；[相关工作](/Users/shq/Research/GP/持续推进/主线05_Actor_固定结构与几何校正/相关工作.md)。

每组5–7篇参考，共19篇去重论文，原始作者页面题名/作者/摘要已检索核对；清楚区分参考思想与本地实际实现，不声称全部论文重新全文精读或完整复现。62份原文件快照SHA一致；主入口与历史配置源码哈希匹配，Python语法、内部链接和默认命令预览通过。本次未新增GPU训练，原项目和完整预测保持原位置。

代码核对补充：PubMed/R39的cons_none_0在两权重上均映射token_kd3_ref50；其中GraphMAE2 76.43%是无VAT修正教师对照，不是R5 token_kd3原教师配置。主线04文档已明确区分，历史结果与原源码未改。


### 2026-09-24 服务器仅保留五组主线清理

按用户要求，已归档核验后删除旧方案 97,863 个文件、35.41 GB逻辑数据及6个旧链接。五组主线独立部署到 `/root/autodl-tmp/retained_mainlines_20260924/`。全部清单内受保护文件删除后SHA256一致，预训练checkpoint、环境、数据、特征、权重来源及五组历史匹配对照保留。

完整清单与本地备份：`/Users/shq/Research/GP/持续推进/服务器清理_仅保留五主线_20260924`。R172保留本地历史，不自动启动后续确认；今后只推进五组入选主线。


### 2026-09-24 五主线全checkpoint网格已启动

用户要求将五组方案在所有支持的数据集预训练checkpoint上运行网格。协议及完整清单：[五主线全checkpoint网格_20260924](/Users/shq/Research/GP/持续推进/五主线全checkpoint网格_20260924/方案与协议.md)。8个节点数据集，按SHA去重63份兼容GCN权重，71个任务，预期5475条正式结果；每类5-shot、5个种子、100步。主线02独立于checkpoint，按数据集执行一次后共享引用。3份MultiGprompt权重架构不兼容、16份图分类权重任务接口不支持，显式列入覆盖缺口。16个跨8数据集的短程工程任务全部通过，不作为效果验证。

服务器目录：`/root/autodl-tmp/five_mainline_grid_20260924`。进展见本地网格目录及服务器 `进展.md`、`summary.json`。当前为运行中，不能把已启动或短程检查称为正式网格完成。旧checkpoint不改写。


### 2026-09-24 五主线网格完成与结论收敛

已完成8个节点数据集、63份兼容独立checkpoint的71个任务、5475条正式结果，零失败。每类5-shot、5个seed、100步；服务器逐行审计、本地预测哈希和完整文件传输核验全部通过，13307个文件约5.97GB完整归档。3份MultiGprompt架构和16份图分类checkpoint的任务接口不支持，未计入成功。

[完整结果汇总](/Users/shq/Research/GP/持续推进/五主线全checkpoint网格_20260924/结果汇总.md)，[逐checkpoint准确率](/Users/shq/Research/GP/持续推进/五主线全checkpoint网格_20260924/全网格准确率.md)。同名不同SHA分别统计；主线02为每数据集一次的直接方法，不重复计独立证据。

核心结论：CE×2比原CE在16/63格为正，8个数据集的checkpoint平均增量均为负；中心化直接教师只在3/8数据集平均为正，在Computers/Photo明显退化；步长匹配Shampoo较同温度Adam为35正/1平/27负；VAT为25正/14平/24负且平均增量小；self-only和几何校正更适合Actor/Chameleon等特定情形，不宜推广为通用替代。代表固定方案相对强未中心化直接教师只有8—13/63格平均领先，不能只根据相对冻结主干的大提升宣称普适优势。

后续仅保留条件性方向与强直接基线，本次未按查询表现修改超参数或启动新搜索；5个支持种子和历史数据不能视为独立外部确认。


### 2026-09-24 scale2机制说明更正

核查focal_loss源码确认：scale2是γ=2的focal梯度系数均值对CE进行动态缩放，不是固定CE乘2。此前“CE×2”“CE权重加倍”的解释有误。既有checkpoint、训练代码、预测和准确率均不变；关于scale2较普通CE的配对差值仍有效，但不能解释为固定倍增系数的效果。详见本轮消融设计。本更正不改写封存实验资料。


### 2026-09-24 五主线消融已启动

按用户要求，为五条主线分别设置模块消融与匹配对照；覆盖上轮8个节点数据集、63份兼容checkpoint、5-shot、5种子、100步。新增4610条正式结果，复用5475条旧结果并逐文件固定引用哈希。16个短程工程任务与全部8数据集的不消融算子一致性检查均通过。完整消融定义及scale2机制纠正见 `/Users/shq/Research/GP/持续推进/五主线消融_20260924/消融设计.md`。原始实验、checkpoint均不改写，运行及结果状态见 `/Users/shq/Research/GP/持续推进/五主线消融_20260924/进展.md`。


## 2026-09-24 五主线消融完成并归档

71/71任务完成、0失败；新增4610条，审计旧引用5475条。8节点数据集、63兼容checkpoint、5-shot、5种子、100步。服务器与本地预测/引用审计通过，全文件SHA通过。

主线01：概率传播的数据集宏均增益为 +0.745 pp（38/63 checkpoint格为正），但在 CiteSeer、Actor、Chameleon 的均值为负。打乱门控对应后下降很大，完整对应的宏均增益 +18.070 pp；这支持节点与门控对应的重要性，不证明机制原创。动态CE缩放相对普通CE为 −0.633 pp，8个数据集均值全部为负，不能继续作为通用增强项。KD为 −0.155 pp，收益依数据集变化。保留条件Prompt和传播作为有条件方向，动态缩放应降级为历史对照。

主线02：两跳传播宏均 +4.509 pp；Cora +22.240、CiteSeer +17.240 pp，但 Actor −3.816、Cornell −10.270 pp。中心化宏均 −6.180 pp，Computers −23.498、Photo −20.700 pp；只在 Cora、CiteSeer、Actor 均值为正。软原型修正在 CiteSeer +4.760、Cora +2.760 pp，整体 −0.591 pp。额外迭代、自适应分配与伪标签对应均无普遍收益。本线保留 CiteSeer 条件下的修正教师，不能推广为所有数据集默认教师。它仍是直接读出方案，不是可学习Prompt。

主线03：锐化门控宏均 +1.271 pp（42/63格为正），7个数据集均值为正、Cornell为零。Adam步长匹配相对无匹配矩阵更新 +11.718 pp，说明这套矩阵更新对更新尺度高度敏感；不能把这一差值全部解释为矩阵预条件的贡献。完整矩阵相对对角仅 −0.013 pp，25正/1平/37负；匹配Shampoo相对Adam +0.454 pp。输出中心化 −1.726 pp，仅Actor均值为正。保留锐化和匹配步长；完整矩阵必要性证据不足，中心化不宜默认打开。

主线04：VAT相对无VAT宏均 +0.132 pp，25正/14平/24负，属于小且不稳定的增益。教师修正 +0.631 pp（46/63格为正），其中 PubMed +1.596 pp；KD −0.035 pp，接近零。VAT相对同epsilon随机噪声 +0.121 pp，但315个checkpoint×seed记录中只有207个首末步RMS匹配，108个不匹配，存在方向和幅度混杂，不能宣称纯方向优势。保留修正教师作为本线较清晰贡献，VAT仅作为有条件小增益。

主线05：self-only相对原图宏均 −2.679 pp，自环100为 −1.401 pp，均非通用增强；self-only在Actor +1.382、Chameleon +1.228、Cornell +8.198 pp，保留结构依赖的适用范围。self-only上的中心化在Actor +2.943、Chameleon +3.121 pp，而Cornell −3.604 pp。中心化之后加对角缩放整体仅 +0.171 pp，Actor −0.007 pp；方差收缩仅 +0.002 pp。Actor收益主要支持固定结构与中心化，对角缩放及收缩的额外贡献很弱。本线属于固定结构/几何对照。

全部正负模块比较与2×2交互已归档；5个seed不是独立图，VAT有108/315个实际RMS不匹配记录。见[消融结论](../五主线消融_20260924/结论与完成状态.md)。


## 2026-09-24 锐化与概率传播组合验证启动

固定2×2已有三格，补齐普通温度0.2无传播；覆盖8节点数据集、63份兼容checkpoint、5个原支持seed，新增315条读出，不新增训练。历史训练100步；固定原A3输出中心化、普通CE、Adam，KD关闭。复用1970条固定参照，比较组合与两个单组件、原A1/A3完整方案及两个直接教师。v1近邻重建smoke分数不一致，正式未启动；v2复用原封存近邻图并保持严格重放阈值，保存失败证据。设计见[组合设计](../锐化与传播组合验证_20260924/组合设计.md)。


## 2026-09-24 锐化×概率传播组合验证完成

63任务、315新增读出、1970历史引用全部审计及本地SHA通过，新增训练0。组合相对仅锐化+0.562 pp、仅传播+1.271 pp，但交互−0.151 pp；相对未中心化直接教师−9.695 pp。32/63格胜两个单组件，仅3/63格还同时胜两个直接教师。不支持普遍协同或升级为已确认新方法。数据集等权，5支持seed不是独立图。

[完整组合结论](../锐化与传播组合验证_20260924/结论与完成状态.md)。
