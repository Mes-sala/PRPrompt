# ProG 公平实验协议 v3

本次修订把 ProG 的节点 few-shot 路径分成两部分：保留旧结果用于追溯；新增 v3 路径用于和 CDP 做可比实验。旧的 `Experiment/sample_data/` 和 `induced_graph_*_v2.pkl` 不会被覆盖，也不再作为新实验的默认输入。

## 已修复

- 节点 support/query 使用官方 `train_mask`/`test_mask`；没有官方 mask 时只生成一次随机排列，再拆成互斥集合。生成时写入 `protocol.json`，读取时检查重复、越界、交集和每类 shot 数量。
- v3 诱导子图只使用 `x`、`edge_index` 和拓扑规则。子图不足目标大小时允许变长，不再用中心节点标签寻找同类节点；缓存身份只绑定图输入、节点数和大小参数，标签只在加载后写入中心目标。
- 线性分类头输出裸 logits，交给 `CrossEntropyLoss`；修正旧版 `Softmax → CrossEntropyLoss` 的重复概率变换。
- `Gprompt` 每个训练 epoch 后用最终 prompt 在全部 support 上重新计算类中心，并按样本数累积，避免首个 batch 以均值直接参与累计和不同 prompt 状态混用。
- 此条是 v3 的设计目标，旧矩阵并未全量满足。2026-09-11 起修复标准非 None 骨干的冻结；实际验证范围和剩余限制见 [实施进度](EXPERIMENT_REPAIR_PROGRESS.md)。新实验使用 [v4 运行说明](PROTOCOL_V4_RUNBOOK.md)。新增 `Frozen`（整图＋线性头）、`FrozenGraph`（诱导子图＋mean pooling＋线性头）和 `FrozenGraphPrototype`（诱导子图＋余弦 support 原型）基线。

## 仍需区分的实验

GPF+ 的输入 prompt、GraphPrompt 的读出权重和 CDP 的层内方向动作不是同一容量。因此报告时分别使用：

1. 方法原始配套读出，回答完整方法效果；
2. 同一计算图、同一读出、同一 head、同一训练 budget，只替换 prompt，回答净机制增益。

`None` 仍表示可更新骨干的原始 ProG 基线；`Frozen` 才表示冻结骨干。诱导子图需要单独报告 `FrozenGraph`，否则 pooling 和上下文变化会被误报为 Prompt 收益。

## 运行顺序

```bash
cd /Users/shq/Research/GP/rep/ProG
python -m pytest -q tests/test_data_loaders.py tests/test_induced_graph.py tests/test_strategy_gprompt.py tests/test_strategy_gpf.py
python bench.py --pretrain_task NodeTask --dataset_name Cora --prompt_type Frozen \
  --gnn_type GCN --shot_num 5 --seed 42 --epochs 200 --num_iter 1 --device cpu \
  --pre_train_model_path Experiment/pre_trained_model/Cora/DGI.GCN.128hidden_dim.pth
python bench.py --pretrain_task NodeTask --dataset_name Cora --prompt_type FrozenGraph \
  --gnn_type GCN --shot_num 5 --seed 42 --epochs 200 --num_iter 1 --device cpu \
  --pre_train_model_path Experiment/pre_trained_model/Cora/DGI.GCN.128hidden_dim.pth
```

首次运行会生成 `Experiment/sample_data_v3/` 和 `Experiment/induced_graph/<dataset>/induced_graph_*_v3_<input_hash>.pkl`。已有 v2 缓存不删除。新实验必须记录 checkpoint SHA、split `protocol.json`、子图缓存 sidecar JSON、学习率/weight decay、head 类型和最终 test partition。

当前工作站未安装 ProG 依赖，本次只完成静态编译和代码修改，未运行 benchmark。
