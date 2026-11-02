# PRPrompt

**Prototype-Routed Graph Prompting：少样本节点分类与图分类的输入层提示。**

冻结两层GCN，只训练类别提示；用支持集建立路由和最终分类原型。图分类采用锁定的全局无标签统计规则，自动选择StatsRoute、ViewMeanStats或RawKernelStats，以及推理增益和软/硬路由。选择器不接收数据集名称、真实标签或编码器表示。

本仓库只包含最终实现、运行入口和此README。历史搜索、实验输出、论文、数据、checkpoint和机器配置不属于此代码发布。旧内容仍在Git历史中；完整研究材料另外本地保存。图分类不包含白化门控或联合门控，节点分类不包含旧隐藏层方案。

## 1. 安装与目录

Python >=3.10；PyTorch >=2.1、PyG >=2.5、NumPy、scikit-learn。推荐复用已安装PyTorch/PyG的ProG环境：

```bash
conda activate <your-ProG-env>
python -m pip install --no-deps -e .
```

新环境需先安装适合运行设备的PyTorch/PyG，再执行 `python -m pip install -e .`。下游实现仅依赖上述Python包，不需要ProG checkout；预训练委托入口另需已有ProG源码。

```text
prprompt/
  backbone.py       两层GCN、归一化邻接；兼容ProG checkpoint键
  graph.py          统计参考、任务路由、输入提示、图分类训练/推理/保存恢复
  selection.py      无标签数据分布规则、统计重建风险、增益与软硬选择
  views.py          等权统计视图路由
  node.py           最终全图/子图输入节点分类接口
  _node_ops.py      节点原型校正与原型读出
  data.py           TU含属性输入、固定初始重排与ProG式任务采样
  utils.py          种子、支持集检查、模型指纹
scripts/
  run_graph.py      图分类任务与结果聚合
  run_node.py       显式节点任务入口
  pretrain.py       可选：调用已有ProG的GraphMAE/Edgepred_Gprompt
```

## 2. 图分类：最终锁定规则

### 2.1 无标签图统计

对图的节点数、平均度、度的总体标准差取log1p，再拼接每个输入通道的节点均值和总体标准差，得到 $3+2D$ 维描述。使用显式声明的无标签参考图池计算均值/标准差并标准化。统计过程在CPU float64上执行，无编码器调用。

### 2.2 自动选择统计方式

从全部原始节点特征的分布判断：

1. 非常量有效通道全部为二值，且有效通道逐节点和为1：**ViewMeanStats**。
2. 存在连续通道，且这些连续通道在各图的均值都近似固定：**RawKernelStats**。
3. 其他情况：**StatsRoute**。

有效性与近似判定固定使用64倍float32机器精度及对应尺度，不通过数据集名称选择分支。

- **StatsRoute**：支持集标准化描述的类别均值作为路由原型，归一化余弦相似度除以0.1后softmax。
- **ViewMeanStats**：每个输入通道建立一个“3个结构统计＋该通道均值/标准差”的5维视图；独立计算路由、支持集中心化/RMS、tanh、再次中心化/RMS；等权平均各视图并重新校准。
- **RawKernelStats**：全图描述间非零平方距离的中位数确定高斯核带宽；对每类支持图的核相似度求平均，除以0.1后softmax。

Stats/Raw路由减去支持图的均值，并以支持节点数加权的RMS归一化。路由原型只用于构造输入提示，与最终GNN表示的分类原型不同。

### 2.3 增益与软硬路由

使用两个固定种子的节点bootstrap统计副本，轮流遮蔽结构组或通道均值/标准差组，评价无标签统计重建风险。硬路由仅在所选家族的硬−软重建误差均值加1.96倍配对标准误小于0时启用；这是启发式规则，不是分类准确率的置信结论。

增益依据全体节点特征RMS与图内变化的比值及shot，从固定集合 `{1,3,10}` 按对数距离选择。训练增益固定1，仅推理时应用选出的增益，并重新计算提示后的支持集分类原型。

路由固定，唯一可训练提示为零初始化的 $C\times D$ 矩阵 $P$：

$$
X'_g=X_g+a\,[T_g\tanh(P)]\odot\sigma_S
$$

其中 $T_g$ 是图路由，$\sigma_S$ 是支持图全部节点的特征总体标准差。冻结GCN前向得到图均值表示，最终按提示后的支持类别表示均值做余弦原型分类，不使用线性头。

### 2.4 API

```python
import torch
from torch_geometric.data import Data
from prprompt import TwoLayerGCN, prepare_graph_reference, fit_graph, GraphFit

# 原始数据的y不传入参考构造或拟合；只显式传支持标签。
clean = [Data(x=g.x, edge_index=g.edge_index) for g in graphs]
reference = prepare_graph_reference(clean)  # 每个图池构造一次，跨任务复用
backbone = TwoLayerGCN.from_checkpoint(checkpoint_path, clean[0].x.shape[1])
fit = fit_graph(backbone, clean, support_ids, support_labels, classes,
                reference=reference, seed=0, device='cuda:0')
scores = fit.scores(clean, query_ids)  # [Q,C]，CPU余弦分数
predictions = scores.softmax(-1).argmax(-1)
torch.save(fit.state_dict(), 'task_state.pt')
restored = GraphFit.from_state_dict(backbone,
    torch.load('task_state.pt', weights_only=True), device='cuda:0')
restored_scores = restored.scores(clean, query_ids)
```

`GraphReference`和路由表绑定原参考图池的内容及顺序；预测IDs必须指向该图池，不能将另一批图按相同ID替换。参考中包含query无标签输入时是transductive协议。支持标签必须编码为0..C−1，并且每类恰有1、3或5图；选择器与路由均不需要query标签。多类接口可运行，当前锁定规则的正式验证范围是下述三个二分类数据集。

默认训练保持ProG图原型配方：Adam lr=.001、weight decay=0、batch128、最多1000epoch、支持损失patience20、保留末状态。损失为支持表示对当前支持类别均值的余弦CE（温度0.1）；每个task/seed新建Prompt和空Adam。用于训练的原型包含当前支持样本，这不是留一或验证损失。`epochs`的小值仅供短测，不代表完整协议。

### 2.5 完整任务入口

```bash
python scripts/run_graph.py \
  --dataset PROTEINS --data-root data \
  --checkpoint checkpoints/PROTEINS.GraphMAE.GCN.128hidden_dim.pth \
  --shots 1 3 5 --tasks 20 --seeds 0 1 2 \
  --device cuda:0 --output results/PROTEINS
```

该脚本使用TU节点属性和节点标签通道，先按seed12345重排图；每个task使用sampling seed `2026102300+task`，query为整图池的floor(80%)，支持图从剩余池按每类k图选取后打乱。支持与query不重叠，严格要求每类k图。不同任务的query会变化。

历史任务文件可通过 `--split-dir <dataset-split-directory>` 精确载入，布局为 `SHOT/TASK/train_idx.pt` 和 `test_idx.pt`；文件必须对应同一图池顺序。新下载的数据、不同PyG版本或重新生成的划分不自动证明与历史实验逐tensor一致。运行器在拟合之后保存状态，再读取query标签计算指标。

每个任务先平均多个训练seed，再报告任务均值与样本SD；20任务×3seed不是60个独立数据划分。不覆盖已有输出目录。

## 3. 节点分类：最终输入层实现

`fit_fullgraph`使用两跳传播后中心化的属性构造支持原型；由非支持节点软分配作三轮原型校正（混合0.5、温度0.05），再以温度0.1得到固定Soft路由、全图中心化并传播一次。仅训练输入维度的类别提示，幅度10、全图节点特征样本标准差、Adam .01、100步、提示L2系数.001、原型余弦温度0.2。多支持使用留一原型；1-shot回退到支持自原型。KD关闭，推理前对全部节点表示中心化。

```python
from prprompt import TwoLayerGCN, fit_fullgraph, fit_subgraph, SubgraphConfig

backbone = TwoLayerGCN.from_checkpoint(checkpoint_path, x.shape[1], device=x.device)
scores, embeddings, state, audit = fit_fullgraph(
    backbone, x, edge_index, support, support_labels, classes, seed=42)
query_scores = scores[query]

# subgraph_for_center(i) -> {'nodes': 唯一全局节点IDs, 'edges': 局部编号边}
scores_sub, pooled, sub_state, sub_audit = fit_subgraph(
    backbone, x, edge_index, support, support_labels, classes,
    subgraph_for_center, config=SubgraphConfig())
```

子图路径保留全图固定Soft路由，但在每个中心的子图内部传播并均值池化；子图构造、补点或裁剪由调用方提供，不能读取待预测标签。两条节点API都访问非支持节点特征，属于transductive；节点训练配方与图分类ProG配方不同。

显式任务文件可执行：

```bash
python scripts/run_node.py --task data/node_task.pt \
  --checkpoint checkpoints/node.GCN.128hidden_dim.pth \
  --device cuda:0 --output results/node_task.pt
```

任务是张量字典：`x, edge_index, support, support_labels, classes`，可选`query`；子图模式还需按中心编号排列的`subgraphs`。接口只把支持标签传给训练函数，不在模型内部采样或调参。

## 4. 预训练checkpoint

下游骨干严格加载ProG两层GCN的 `conv_layers.*` state_dict；隐藏与输出维度均128，LeakyReLU，图均值池化，冻结骨干。权重文件不随代码发布。

已有checkpoint可直接使用。没有权重时，可委托自己安装的ProG checkout执行原生目标，例如：

```bash
python scripts/pretrain.py --prog-root /path/to/ProG \
  --dataset PROTEINS --method GraphMAE --epochs 1000 --seed 0 --device cuda:0
```

`--epochs`必须显式给出；也支持`Edgepred_Gprompt`。此入口调用ProG的`pre_train.py`，沿用其优化器、保存位置和数据处理，不将重新预训练的结果冒充历史checkpoint。要严格复现既有数值，需要相同源码快照、数据、任务及checkpoint SHA；只有名称相同不足以证明身份一致。

## 5. 验证与研究边界

发布前用CPU核对了抽取前后的节点完整100步训练、子图短测、三个真实图池的统计选择结果、路由、支持损失、提示更新、保存恢复及历史任务采样。该检查验证打包等价性，不是新的正式实验，也不证明跨GPU逐位相同。

锁定图方案已在PROTEINS/MUTAG/COX2完成1/3/5-shot、20任务×3seed。全局选择不读取真实标签，但该规则是在已经用于开发的历史图池上设计的；不能称为独立留出集上的无监督泛化确认。与基线按最高预训练比较只是描述性选择，严格配对要求同输入、同任务索引和同checkpoint。

保留负结果：当前COX2 5-shot的自动软路由未保留此前单独硬路由配置的优势；同信息直接统计分类在既有九格均高于当前Prompt。无标签统计重建风险不等于类别判别风险。核心消融中的共享提示还改变参数量，聚类路由在部分家族同时改变算子，不是单因素去标签实验。

## 6. 来源与许可证

实现抽取自本项目最终输入节点路径及2026-11-01锁定的全局统计图路径。正式包去掉历史AST加载、服务器绝对路径、实验控制器、搜索分支和结果文件。`scripts/pretrain.py`只委托外部ProG，未内置或重新定义其预训练目标。

两层GCN与图原型训练的相关算式适配自本地ProG源码。原ProG快照的MIT许可原文保留如下；本发布不为其他作者来源重新声明许可：

```text
Copyright (c) 2012-2024 Scott Chacon and others

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
