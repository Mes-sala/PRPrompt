# ProG v4 实验运行说明

v4 是当前可执行的固定预算 NodeTask 协议，实施范围与剩余工作见 [进度](EXPERIMENT_REPAIR_PROGRESS.md)。完整设计见 [修复方案](EXPERIMENT_REPAIR_PLAN.md)。v3 及更早结果不覆盖，不与 v4 聚合。

## 单个 cell

```bash
python scripts/run_protocol_cell.py --config configs/protocol_v4_cora.json
```

配置必须显式提供 dataset、prompt、checkpoint、gnn、hidden_dim、layers、shot、folds、support_passes、learning_rate、weight_decay、split_seed、train_seed、device、output_root、split_root。未知字段报错；不会忽略 batch_size 然后随机挑选另一个值。当前验证 GCN 和带明确 train/test mask 的节点图；无官方池的图需要先准备版本化划分。

- `checkpoint: "None"` 表示随机骨干。
- `prompt: "None"` 表示完整微调；可以与任意兼容 checkpoint 配对。
- `Frozen` 表示全图线性探测。
- `FrozenGraph` / `FrozenGraphPrototype` 是通用诱导子图的线性头/支持集原型对照。
- `FrozenNodePrototype` / `FrozenIdentityPrototype` 是全图/自环节点原型对照。
- `GraphTOP-Frozen` 使用 GraphTOP 相同 ego 子图与中心读出，保留原图边权，关闭学习改边。
- SelfPro、PSP、EdgePrompt 系列等未完成作者来源核验的方法，需要 `allow_unverified: true` 才能用于诊断。GraphPrompter NodeTask 当前明确不支持。
- 其他未进入新入口能力表的方法返回 skipped，不生成虚假的 accuracy=0。旧策略注册仍存在，不代表已通过 v4 验收。

默认严格标签预算：每类 k 个支持标签；预训练/下游可见测试节点的未标注特征和结构；不使用验证标签，不早停，不在测试指标上选模型。学习率是预定配置，不是已调到最优的数值。

## 预算

全支持集为一个 batch，每 support pass 一次 optimizer 更新。显存不足会失败，不静默缩小 batch 或改变原型语义；microbatch 梯度累积尚未实现。无梯度原型基线只构造原型，不凑满 200 次训练。

All-in-one 必须额外提供 `answer_passes`、`prompt_passes` 和 `prompt_lr`。例如总预算 200、两阶段各 50，执行 2 个 cycle，两个 optimizer 各 100 次。总预算必须整除一个 cycle；实际计数写入诊断。

GraphTOP 保留随机推理，`inference_seed` 与训练种子分离；当前不做多样本集成。SUPT 保留其提示机制及正则配置。

## 数据与复现

划分身份绑定数据输入/标签/官方 mask、shot 和 split_seed。训练种子改变不重新抽 support。骨干、head、prompt 分别设置初始化种子，避免方法构建耗用随机数导致 head 初始化漂移。每个 fold 训练前屏蔽原图非 support 标签；分类指标只在最终预测之后计算。

输出目录是协议、有效配置、数据、checkpoint 和源码哈希的内容地址。重复运行同一身份可复用结果，变更配置或代码不会跳过旧 cell。JSON 是完整制品，CSV 仅是索引摘要。

结果包包含：

- `config.json`：显式配置。
- `identity.json`：数据、源码文件、checkpoint 身份。
- `result.json`：每 fold 真实更新数、参数变化、冻结哈希、指标和耗时。
- `predictions.json`：最终概率、标签及对应 query 节点，支持独立重算指标。
- `failure.json`：失败原因和栈；成功后移除旧失败记录。

AUROC/AUPRC 对缺类别查询集返回 null（内存中为 NaN），不冒充 0。Accuracy 和 Macro-F1 按统一类别集合计算。现有 checkpoint 未携带来源元信息时标记 `legacy_unverified`；能加载不代表忠实复现。

## 矩阵

```bash
python scripts/run_cora_5shot_matrix.py --dry-run
python scripts/run_cora_5shot_matrix.py --workers 2
```

worker 已转入 v4；矩阵输出 `results/cora_5shot_full_matrix_v4.csv`。每 GPU 一条串行队列；None 与所有常规预训练组合，补足预训练后完整微调。未核验/不支持组合写 skipped，不纳入成功分数。矩阵会重新提交 cell，由内容身份决定是否复用，不能仅看旧 JSON 文件名。

该矩阵命令仍是工程固定配置扫描，尚不等于 10 splits × 3 train seeds × 3 pretrain seeds 的正式研究实验。正式多种子与有验证标签的调参轨道仍需独立实施，不能把一次固定配置的结果当作方法上限。

## 验证命令

```bash
python -m pytest -q tests/test_protocol_repair.py tests/test_input_structure_ports.py
```

测试不依赖数据下载；作者来源数值检查在缺少对应 checkout 时会 skip，应保留实际测试摘要。真实 Cora 工程运行另存于 `results/protocol_v4_validation/`，仅用于验证完整预算和参数更新，不作为调参选择依据。
