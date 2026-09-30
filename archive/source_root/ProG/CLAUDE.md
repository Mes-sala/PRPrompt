# ProG 项目上下文

## 项目定位

ProG（Graph Prompt Learning benchmark，ProG-V2）：图 prompt 学习工具包，`pre-train → prompt-tune → evaluate` 流程。R3Prompt 项目复用它的预训练编码器和 checkpoint 格式。

## 环境

- 环境 `prog-v2-cu130`（Python 3.11，torch 2.11.0+cu130，torch_geometric 2.8.0，numpy 1.26.4）。R3Prompt 共用此环境。
- 激活：`source /data1/shq/miniforge3/etc/profile.d/conda.sh && conda activate prog-v2-cu130`
- 硬件：2× RTX 5090（Blackwell，CUDA 13.0），必须用 cu130 的 torch 轮子。

## 数据

- `/home/shq/ProG/data/Planetoid/{Cora,CiteSeer}/`：软链自 `/data1/shq/COLA/graphdata`，raw 文件齐全。
- 网络受限：GitHub / HuggingFace 不可达（curl 超时），PyPI / download.pytorch / data.pyg 可通。所以 Planitoid 数据靠手工软链，不能在线下载。

## 预训练（6 种方法）

- 入口：`pre_train.py --pretrain_task <DGI|GraphMAE|GraphCL|SimGRACE|Edgepred_GPPT|Edgepred_Gprompt> --dataset_name <Cora|CiteSeer> ...`
- 批量：`scripts/pretrain_paper_grid.sh --task node --datasets <ds> --device 0`（幂等，已存 checkpoint 会跳过）。
- 保存格式：`Experiment/pre_trained_model/<dataset>/<method>.GCN.128hidden_dim.pth`（`gnn.state_dict()`，无 module 前缀）。
- 已训练：Cora 和 CiteSeer 各 6 个 checkpoint。

## 下游 bench（prompt 方法）

- 入口：`bench.py --pretrain_task NodeTask --prompt_type <GPF|GPF-plus|Gprompt|...> --pre_train_model_path <ckpt>`
- 批量：`scripts/bench_paper_grid.sh`（结果写 Excel `Experiment/ExcelResults/Node/{shot}shot/<ds>/GCN_total_results.xlsx`，列名 `<pretrain>+<prompt>`）。
- 默认每组 10 次随机搜索 × 5 折，`num_iter` 控制。

## 结果对比

- `results/benchmark-gcn/summary.csv`：公开发布的节点/图分类结果（含 Cora，**不含 CiteSeer 节点分类**）。
- `results/cora_gpu0_vs_summary.csv`：本机 GPU0 重跑的 Cora 结果 vs summary 对比。

## 已知坑

- 数据加载 `load4node` 用 `Planetoid`，缺数据会去 GitHub 下载（失败），须提前软链数据。
- `Edgepred_GPPT` / `SimGRACE` 预训练在 Cora 的 summary.csv 里缺失（summary 只覆盖 None/DGI/GraphMAE/Edgepred_Gprompt/GraphCL/MultiGprompt）。
