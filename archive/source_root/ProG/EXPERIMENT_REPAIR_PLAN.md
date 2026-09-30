# ProG 实现修复与统一评测方案

日期：2026-09-11。状态：待实施方案，不代表下述修复已经完成。

本文针对 `results/cora_5shot_full_matrix.csv` 的生成链和当前本地代码。目标是建立足以判断预训练和 prompt 真实增益的实验依据，而不是让某个方法必然超过基线。当前矩阵保留为历史诊断数据；修复后另建结果目录，不覆盖、不混合统计。

## 1. 当前结论与证据边界

入口为 `scripts/run_cora_5shot_matrix.py → scripts/run_prog_cell.py → bench.do_config_bench → NodeTask.run`。worker 传入 200 epochs、seed=42、num_iter=1，但入口配置不等于实际训练预算。

| 已确认问题 | 实际行为 | 后果 |
| --- | --- | --- |
| None 的语义不清 | GNN 和线性头全部更新；矩阵仅保留无预训练的 None | 63.86% 是从头监督训练基线，不能视为随机冻结特征性能；缺少预训练后完整微调基线 |
| 冻结实现不一致 | 部分分支明确冻结，部分仅将 GNN 排除出 optimizer | 梯度计算、训练模式、随机层行为没有统一 |
| prompt 未进入预测路径 | EdgePrompt 节点调用不传 prompt；GraphPrompter 在 GNN 的节点返回路径被跳过 | 对应数值不能代表这些方法的有效实现 |
| SelfPro 查询标签泄漏 | 用测试节点真实标签构造测试类别原型 | 无法作为合法 few-shot 结果 |
| PSP 训练、评估模型不同 | 训练 prompt logits，评估 GNN＋answering | 测试指标没有衡量训练得到的预测器 |
| 输入与分类方式混杂 | 全图节点分类、子图图分类、中心读出、线性头和原型混合 | 跨列差值不等于 prompt 净增益 |
| 预算不同 | 通用路径按训练 loss 早停；SUPT/GraphTOP 固定训练；All-in-one 改写外层轮数 | 同为 200 不等于同样训练量 |
| 隐式超参数采样 | bench 从候选值随机选择学习率、weight decay、batch size | 单次随机配置不能代表充分调参结果，也难以解释失败 |
| 元信息不足 | cell JSON 不记录实际训练轮数、更新计数和完整有效配置 | 无法从 CSV 恢复历史真实预算 |

主要代码依据：[worker](scripts/run_prog_cell.py)、[bench](bench.py)、[优化器与冻结](prompt_graph/tasker/task.py)、[节点调度](prompt_graph/tasker/node_task.py)、[GNN 前向](prompt_graph/model/gnn.py)、[SelfPro](prompt_graph/tasker/strategies/self_pro.py)、[PSP](prompt_graph/tasker/strategies/psp.py)。这些是当前代码的静态审计结论；未验证历史执行时的源码哈希与当前完全一致。

现有 [FAIR_PROTOCOL_V3.md](FAIR_PROTOCOL_V3.md) 的“所有冻结分支显式冻结”不能继续作为已全量验证的承诺。实施时需更新为实际适用范围，并链接新协议；不能因为文档写了已修复就跳过验证。

## 2. 要回答的三个研究问题

1. **预训练是否提供有效表示？** 相同架构、输入和分类规则下，预训练相对随机骨干有什么增益？预训练后完整微调是否优于从头训练？
2. **prompt 是否提供额外增益？** 同一个 checkpoint、数据划分、输入、读出和训练政策下，加 prompt 是否优于对应无 prompt 基线？
3. **完整方法是否有使用价值？** 保留其必要机制后，精度、参数量、时间、显存和稳定性表现如何？

三类问题分开报告。完整方法排行榜不能替代机制消融；统一骨干的移植实验也不能直接断言作者原方法无效。

## 3. 哪些必须统一，哪些允许不同

| 因素 | 必须统一的部分 | 允许不同的部分及要求 |
| --- | --- | --- |
| 任务与标签预算 | 同一数据版本、同一 support/query、每类 k 个支持标签；禁止 query 标签进入训练、原型和选择 | 有验证标签的实验单独命名、单独统计 |
| 可见信息 | 同一 transductive 或 inductive 规则 | 不跨设置直接归因 |
| checkpoint | 同一配对比较必须使用同一文件哈希 | 比较预训练方法时 checkpoint 本就不同；记录预训练种子和预算 |
| 骨干 | 受控比较固定架构、维度、层数和预处理 | 作者原生架构放完整方法轨道，并标注容量差异 |
| 冻结 | frozen 组参数和运行状态保持不变；eval 模式不妨碍对输入 prompt 求梯度 | FullFT 独立分组；部分层解冻必须另起方法名 |
| 输入/读出/head | prompt 净增益比较必须匹配 | 完整方法可用原生子图、中心读出、原型等；必须附对应消融基线 |
| 训练政策 | 固定预算或验证选择政策在同一比较内一致 | 原生交替训练可保留，但显式计数并另做预算消融 |
| 学习率、正则 | 同等调参机会、同一选择规则 | 不要求所有方法相同数值；不同参数化通常需要不同适宜范围 |
| 参数规模 | 记录实际参与训练的参数、变化量和 optimizer 分组 | 不强制完全相同；容量匹配作为补充实验 |
| 随机性 | 共享 split、checkpoint；记录训练和推理种子 | 方法内部抽样可保留，不能挑最好随机结果 |
| 指标 | 相同实现、类别集合、聚合单位和查询集合 | 可以增加方法专有诊断，不取代公共指标 |

不建议强行把所有方法改成“全图＋线性头＋200 step”后称为原方法复现。正确方式是同时提供受控适配版本和保留机制的完整版本。

## 4. 修复清单与优先级

### P0：先恢复评测有效性

**P0-1：隔离查询标签。** 训练上下文只提供 support 的索引和标签；评估预测函数只接收 query 索引，评分器最后才读取 query 标签。SelfPro 的类别中心必须由 support 构建，训练结束后用最终参数重新计算；不得从 query 聚类结果与真实标签建立对应。增加“打乱 query 标签不改变预测”的测试，评分结果可以改变。

**P0-2：接通 prompt。** EdgePrompt 在每层适配位置显式传入 prompt 和类型，节点/图任务分别检查；GraphPrompter 先核对本地作者项目是否存在合法节点任务路径。若存在则按其项目实现，若不存在则标记当前 NodeTask 不支持，不用随意加一次 prompt 调用冒充迁移。PSP 使用同一预测函数训练和评估；其当前简化模块忽略 edge_index，应审查真实结构机制，不能只修评估就宣称完成作者方法复现。

**P0-3：统一实际冻结。** 创建一个 encoder 冻结入口，设置 requires_grad=False 和 eval，并在每个训练阶段维持该状态。优化器参数不得包含 frozen encoder。输入 prompt 需要穿过冻结 GNN 反传，不能给整个前向包 no_grad；仅固定嵌入缓存允许 no_grad。前后检查完整 state_dict，覆盖 buffers。All-in-one 非活动模块仍可能需要传播输入梯度，但不应积累其参数梯度。

**P0-4：消除训练/推理不一致。** 共用 `predict_logits` 或方法等价接口；评估使用同一个 prompt、head、支持集原型和规定的结构。方法原生随机推理可保留，但记录推理种子。禁止把 train/eval 模式切换误当作参数冻结。

**P0-5：统一声明与能力检查。** 每个策略声明支持的任务、backbone、输入、readout、head、需要的标签和优化器。运行前检查兼容性；不支持写 skipped 和原因，禁止静默换成普通 GNN。矩阵不应为了填满而制造无意义组合，MultiGprompt 等自带预训练系统独立列组。

### P1：修复运行器和可追溯性

新增版本化实验入口，直接读取有效配置；保留 bench 的随机搜索作为显式 search 模式，不用于固定配置的 cell worker。配置缺字段报错，禁止在内部再次随机选择 batch size 或学习率。不要继续依赖 All-in-one 对 epochs 的隐藏改写。

训练器返回结构化结果：requested/completed epochs、support passes、每个 optimizer 的 step、停止原因、最终/被选 checkpoint、标签使用量。无训练基线不运行空的 200 epoch 循环。

结果身份由协议、数据/划分、代码、checkpoint、配置和种子共同决定；旧 JSON 存在不能作为跳过新配置的依据。失败不填 0 分，不与成功样本混合求均值。GPU worker 使用固定设备队列，同一设备串行领取任务，避免仅按提交序号分配造成同卡意外并发。

### P2：恢复迁移可信度

逐方法建立“作者 commit → 关键模块 → ProG 对应实现 → 已知偏差 → 验证证据”的清单。用户确定的优先级是作者项目代码优先于论文描述，但不能保留标签泄漏作为合法评测协议；发现冲突需记录修正及影响。

SUPT、GraphTOP 保留已有移植验证并补入新运行器。其他新增 prompt 和 GraphMAE2/S2GAE/Bandana 逐一检查，不将能加载 checkpoint 视为忠实复现。统一 GCN 的目标函数适配版本明确命名为 ProG-adapted；作者原生复现单列。

预训练重点检查输入预处理、目标构造、decoder、mask/re-mask、EMA 时序、负采样和 checkpoint 选择。边重建需明确无向边正反向等价类的划分，防止本应隐藏的目标以反向边形式留在输入；若某种有向协议故意保留反向边，要单独声明，不能混成无向重建结果。

## 5. 推荐任务设置

### 5.1 主任务：同图 transductive、严格标签预算的节点 few-shot

第一阶段使用 Cora 的 5-shot 完成工程验收；随后扩展到 Cora、CiteSeer、PubMed 的 1/5/10-shot。这是分阶段实验建议，不预设所有数据和方法已验证兼容。该范围仅支持对引文网络的结论，普适性结论还需增加不同结构/特征分布的数据。

- 预训练和下游可见全图 x、edge_index，包括 query 的未标注输入；这必须明确写为 transductive。
- support 从官方训练池每类抽 k 个，query 使用固定官方测试池；同一个 split 文件供所有方法共享。未被抽为 support 的训练池标签也不可使用。
- 主轨道不使用目标任务的验证标签，保持 k-shot 严格含义。固定超参数和训练预算在最终运行前锁定；可以来自作者默认配置或独立开发任务，但必须记录来源。
- 如果使用官方验证标签选择配置，另称“k-shot support＋额外验证标签”，披露验证标签数量与试验次数。即使不反传，验证选择也消耗监督信息。
- 无官方 split 的数据，先分层生成一次 train/validation/test 池并持久化，再从 train 池抽 support；小类不足时应报不适用，不从测试池补样本。
- 子图采样仅依赖输入和拓扑。子图之间共享节点、support 子图包含 query 的未标注节点，在此 transductive 设置下允许；query 标签始终不可见。
- 当前 v3 子图通常使用 100–300 节点的目标范围，与 GraphTOP 二跳子图不等价。需要显式保存 hop、截断、中心映射、实际大小分布和缓存哈希。

严格 inductive 节点任务和图分类作为后续独立协议，不与主表混算。Inductive 必须连预训练、子图缓存、特征统计和结构处理一起排除 held-out 图/节点，不能仅删除测试标签就声称 inductive。

### 5.2 受控比较轨道

| 设置 | 必备无 prompt 基线 | 可回答的问题 |
| --- | --- | --- |
| 全图＋中心节点＋线性头 | Random-Frozen-Linear、Pretrained-Frozen-Linear | 表示质量；同输入 prompt 净增益 |
| 统一诱导子图＋mean pooling＋线性头 | Random/Pretrained-FrozenGraph | GPF、GPF-plus、SUPT 等适配版本增益 |
| 统一诱导子图＋mean pooling＋支持集原型 | Random/Pretrained-FrozenGraphPrototype | Gprompt 的额外贡献 |
| 全图＋节点原型 | Random/Pretrained-FrozenNodePrototype | 节点原型方法的对照；需新增基线 |
| GraphTOP 相同子图＋中心读出＋线性头 | 关闭结构改写，保持原图边和相同上下文 | 结构改写的净增益 |
| 完整参数训练 | Scratch-FullFT、Pretrained-FullFT | 预训练初始化收益及 prompt 的精度/成本权衡 |

同输入的线性头与原型头再做交叉对照，用于分解 head 本身的贡献。关闭结构 prompt 时不要意外删除原图边；如作者方法还改变候选边空间，应增加候选空间对照。SelfPro 使用自环输入时必须另配自环无 prompt 基线。

FullFT 是参考基线，不是 prompt 必须超过的公平门槛；若 prompt 精度相近但训练参数更少，可体现价值。反之，参数少也不能自动说明性能损失可接受，需报告实际权衡。

### 5.3 完整方法轨道

保留合法的原生输入、分类器、损失、交替过程、结构采样和专用骨干，匹配标签预算、划分、信息可见性及调参机会。每个方法附“关闭 prompt”消融以及计算成本，和受控轨道分表。主表名称应包含适配状态，避免把简化实现当成原论文方法。

## 6. 训练预算与超参数政策

### 6.1 默认受控预算

Cora 5-shot 工程验收建议 200 个 support passes，每个 pass 使用全部 35 个支持目标，一次联合 optimizer step；不按训练 loss 早停。200 是统一起点，不是已验证的最佳训练长度。正式结果增加预先规定的 50/100/200/400 预算曲线，不能看测试峰值后只保留最好点。

子图 batch 显存允许时使用全部支持图；否则 microbatch 累积梯度，按实际支持目标数加权 loss，每个完整 pass 才 step。原型目标依赖全支持集，不能简单改成局部 batch 原型后宣称等价；需要完整嵌入计算、可微聚合或单独定义的适配版本。对耦合正则、BatchNorm、批依赖随机操作，也不能默认累积与大 batch 数值等价。

同时记录三种预算：support passes、每 optimizer 的更新数、墙钟时间/显存。相同 step 不保证计算量相同，因此不声称只统一 step 就达到了全部公平。

### 6.2 特殊方法

- All-in-one 原生轨道显式配置 4 个 cycle，每 cycle 50 answer passes＋50 prompt passes，合计 400 passes。受控总预算 200 的适配实验可设 2×(50+50)，应标注为预算适配版本，不能称两者同为“200 epoch”。分别记录两个 optimizer 的 100 或 200 次更新。
- FrozenGraphPrototype 等无梯度方法计算最终支持原型后评估，step=0；无需凑齐预算。
- GPPT 除 optimizer step，还记录结构 token 手动更新次数。原型刷新、聚类和采样不记作 optimizer step，但单独统计时间和次数。
- GraphTOP 原生随机推理保持作者机制。主结果预先固定每个训练 run 的推理种子；补充多次推理方差。多次采样平均概率属于集成版本，必须另报，不能静默替换原生推理。

### 6.3 调参

调参政策统一而非学习率数值统一。严格标签预算轨道采用预先冻结的配置；有验证集轨道每方法给相同候选配置数，例如 12 个有效候选，按验证指标选择，记录所有候选和成本。GPPT 等固定作者学习率可作为候选，但不得绕过配置日志硬编码覆盖。

验证模式统一选最大验证 Accuracy 的 checkpoint，同分选更早者；Macro-F1 可作为另一个预先声明的主目标，不临时切换。严格无验证模式只用预定最后 checkpoint。禁止用 query/test 选 epoch、学习率、pretrain 或 prompt 后仍把同一测试集称作独立验证；后续设计应在开发任务决策，再在保留任务确认。

## 7. 种子、统计和结果文件

工程回归：一个合法 checkpoint、2 个固定 split、短训练确认功能，再做完整 200-pass Cora 5-shot。正式受控比较建议 10 个 support split × 3 个训练种子，方法之间配对；零训练随机性方法不复制相同结果冒充独立样本。

预训练比较增加至少 3 个预训练种子。先固定 checkpoint 比较 prompt，再跨预训练种子检验稳健性，避免一开始盲目运行全部笛卡尔积。随机骨干也需要独立初始化种子；不同 checkpoint 下的 head 初始化尽量配对，且拆分数据、模型、loader、推理 RNG，避免前一个模块耗用随机数影响下一个模块。

主指标 Accuracy、Macro-F1；AUROC 为辅助，明确 one-vs-rest 与平均方式。每 run 保存预测/概率和 query 索引，便于重新计算；不在普通调参日志展示可供人工逐轮选优的 test 曲线。

按 split 报告配对增益 Δ，再聚合训练种子；报告均值、标准差和配对置信区间。共享 query 集的多个 support split 不是独立数据集样本，区间主要反映该数据集内的训练/划分不确定性；不能据此推断跨数据集普适性。跨预训练种子采用分层聚合，不把相关重复当独立样本。

每个结果包至少包含：

```text
protocol_id, task_setting, method_variant, run_id
dataset_hash, split_hash, support/query/validation_counts
source_commit, dirty_diff_hash, source_file_hashes
checkpoint_hash, pretrain_seed, split_seed, train_seed, inference_seed
backbone, preprocessing, input_mode, subgraph_hash, readout, head
effective_hparams, selection_policy, requested/completed_budget
optimizer_steps_by_group, manual_updates, support_passes
encoder_state_before/after, trainable/effective_parameter_counts
stop_reason, status, failure_reason, metrics, time, peak_memory
```

预训练元数据另外保存训练图身份、标签使用声明、目标函数、原生/适配版本、mask/负采样政策和 checkpoint 选择规则。旧 checkpoint 缺元数据时标记 legacy-unverified，仅用于诊断，正式结论需补齐证据或重训。

## 8. 验收测试与阶段交付

| 阶段 | 交付 | 通过标准 |
| --- | --- | --- |
| A：实现修复 | P0 各方法修复及能力清单 | query 标签置换不改变预测；训练/评估调用同一预测器；不支持组合明确拒绝 |
| B：梯度与状态 | 小型合成图集成测试 | frozen encoder 完整状态不变；预期参数获得有限梯度并在多步后变化；非活动参数不更新 |
| C：预算与重现 | 新 runner、配置、计数器 | 实际 step/pass 与声明相符；固定 seed 可在声明容差内复现；更换配置不命中旧结果 |
| D：对照验证 | Cora 5-shot 受控小矩阵 | prompt 关闭后与匹配基线前向等价；原型只来自 support；指标能由保存预测重算 |
| E：来源核验 | 每方法移植差异表 | 关键 forward/loss/gradient 与作者实现的小图数值对照；必要随机路径做分布或固定样本验证 |
| F：正式实验 | 多数据、多 shot、配对报告 | 先锁配置再运行；全量报告成功/失败、预算和置信区间；原生与适配结果分表 |

梯度测试应选择能激活机制的非退化输入，不能要求每个参数每一步都非零梯度。对合法 no-op 初始化，检查多步后的有效依赖；hard sampling 等路径按作者梯度估计器验证。测试验证计算机制，而不是断言“某 prompt 必须涨分”。

阶段 D 优先纳入 Scratch-FullFT、Pretrained-FullFT、Frozen、FrozenGraph、FrozenGraphPrototype、GPF、GPF-plus、Gprompt、SUPT、GraphTOP；其他策略通过 A/B/E 后再加入。先用同一已核验 checkpoint 和随机骨干定位下游问题，再扩展预训练矩阵。

本文仅新增方案文档。实施依赖顺序为 A → B/C → D，E 按方法作为进入正式比较的门槛，全部满足后执行 F。不要先重跑 202 个旧 cell，再修复其共同问题。

## 9. 最终应达到的效果

最终交付应让每个分数都能回答：使用了多少标签、看到了哪些图输入、更新了哪些参数、训练了多少、如何选模型、与哪个无 prompt 对照可比，以及对应作者方法的哪种实现。

修复成功的标志不是 GPF-plus 第一、None 降分，或所有 prompt 普遍涨分，而是：

1. 无测试标签泄漏、无断开的 prompt、无训练/评估预测器错配。
2. 能分别量化预训练收益、输入/读出收益、分类器收益与 prompt 额外收益。
3. 同样配置可追溯重跑，预算差异可解释，失效结果不混入排名。
4. 若输入或结构 prompt 在多 checkpoint、split、数据任务上有稳定配对增益，才形成后续方案设计依据；若无增益，也能区分实现失败、优化不足、预训练表示差或机制确实无效。
5. 以 Accuracy/Macro-F1、训练参数、时间和显存共同判断方法价值；所有“不如基线”的结论都注明协议和适用范围。

在完成这些验收前，当前 CSV 适合发现问题和制定修复优先级，不适合支撑“这些 prompt 已经证明没有用”或“某类预训练没有价值”的研究结论。
