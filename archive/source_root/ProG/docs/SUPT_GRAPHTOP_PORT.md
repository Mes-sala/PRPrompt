# SUPT / GraphTOP ProG 迁移说明

## 来源与优先级

- SUPT：本地 `../SUPT` 作者仓库，commit `ccc57285d128386e7771d62ed55d01c1f7fcf686`。
- GraphTOP：本地 `../GraphTOP` 作者仓库，commit `7c662da0d53b3bf71ffd52b2c7a59d97d1fd45a2`。
- 论文用于确认方法语义和适用范围；论文与代码不一致时，以作者代码为准。

## SUPT

ProG 提供两个策略：`SUPT-soft` 和 `SUPT-hard`。

- `SUPT-soft` 使用作者 `DiffPoolPrompt`：GCN assignment、列向 softmax 和 basis feature 加和。
- `SUPT-hard` 使用作者 `SAGPoolPrompt`：每个 basis 按 score 选 top-rank 节点，并按命中次数归一化。
- `supt_orth_loss` 的行为保留作者分支差异：`bio` 分支在 hard prompt 上除以 basis 数，`chem` 分支不除；默认使用 `chem`。
- Prompt 作用在输入特征上；预训练 GNN 和线性 head 由 ProG 的 prompt 流程管理。
- 支持 NodeTask 和 GraphTask。GraphTask 使用 ProG 的图分类 split；NodeTask 使用 ProG v3 的 support/query split。

## GraphTOP

ProG 提供 `GraphTOP` 策略，当前限定为 `NodeTask + GCN`，因为作者实现和 edge-weight 语义均针对 GCN。

- 对每个目标节点抽取二跳 ego subgraph；只保留目标节点与子图其他节点之间的可学习边，其他边固定。
- 使用作者 `Rewiring` projector：`fc1 -> ReLU -> fc2`，概率通过 `sigmoid(projector(h_i+h_j)/0.2)` 得到。
- 训练使用作者的 Gumbel-style 两均匀随机数松弛、温度 `0.97(1-epoch/epochs)+0.03`、正熵项和密度平方项。
- 推理保留作者代码的 Bernoulli 随机采样，而不是擅自改成阈值化推理。
- 共享无向边的两条方向使用同一个采样值；目标节点读出与作者代码一致。
- 编码器冻结状态、checkpoint 哈希、split 和最终指标写入 `scripts/run_input_structure.py` 生成的制品。

GraphTOP 的论文公式使用了熵符号和稀疏项的表述，作者代码的实际 loss 计算是本迁移的依据。作者代码还使用独立子图和随机采样，因此不能直接与 ProG 的全图 GPF+ 数值作无条件因果比较。

## 运行

```bash
cd /Users/shq/Research/GP/rep/ProG

python scripts/run_input_structure.py \
  --output-dir results/input_structure/supt_soft_cora \
  --config configs/supt_soft_cora.yaml

python scripts/run_input_structure.py \
  --output-dir results/input_structure/supt_hard_cora \
  --config configs/supt_hard_cora.yaml

python scripts/run_input_structure.py \
  --output-dir results/input_structure/graphtop_cora \
  --config configs/graphtop_cora.yaml
```

配置文件中的 `pre_train_model_path` 应替换为实际 checkpoint 路径。运行器拒绝覆盖既有输出目录，并保存 `config.json`、实现文件哈希、checkpoint 哈希、`metrics.json`、`diagnostics.json` 或 `failure.json`。

## 验证

在临时 Python 3.12 环境中运行：

```bash
python -m pytest tests/test_input_structure_ports.py \
  tests/test_factory.py tests/test_induced_graph.py tests/test_strategy_registry.py -q
```

结果：33 passed。测试覆盖作者 SUPT 两个分支的 forward/gradient 数值一致性、GraphTOP 子图与边采样、正则项、冻结 encoder、标签不影响图输入，以及三个 NodeTask 的端到端小 checkpoint 流程。
