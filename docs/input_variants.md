# 输入层 F100Soft 与子图输入 PRPrompt

本目录说明两套代码入口。它们接收调用方提供的特征、图、支持集和预训练骨干，不绑定数据集、任务抽样器或服务器目录。旧实验与旧论文保留原有入口。

| 项目 | 全图输入层 F100Soft | 子图输入 PRPrompt |
|---|---|---|
| API | `fit_fullgraph` | `fit_subgraph` |
| Prompt 注入 | 原始输入特征 | 子图内原始输入特征 |
| 路由计算 | 全图固定 Soft 路由 | 同一全图路由按节点 ID 提取 |
| GCN 传播 | 全图 | 各中心节点的子图 |
| 中心表示 | 中心节点输出 | 子图内节点输出均值 |
| 读出 | 支持集原型 | 支持集子图表示形成的原型 |
| 标签输入 | 仅支持集标签 | 仅支持集标签 |

## 共同模型配方

- 冻结两层 GCN，默认使用 `TwoLayerGCN(input_dim, 128, 128)`；第一层后 LeakyReLU，最后一层不激活。checkpoint 键与两层 ProG GCN 兼容，调用方严格加载权重。
- 路由由全图两跳传播特征经全图中心化生成；支持原型经非支持节点的软分配做三轮校正，混合系数 0.5，校正分配温度 0.05。
- 路由温度 0.1，概率先中心化再经归一化邻接传播；训练期间路由固定。
- 每类一个输入维度 Prompt，零初始化。仅 Prompt 可训练；没有线性分类头。
- 输入变换为 $X'=X+10[G\tanh(P)]\odot\mathrm{std}(X)$。标准差沿全图节点计算，使用 PyTorch 样本标准差并截断到至少 $10^{-6}$。
- Adam 默认 lr=0.01、weight decay=0、epsilon=1e-8，训练100步。损失为支持集原型交叉熵与 $0.001\operatorname{mean}(P^2)$；KD关闭。
- 原型余弦分类温度为0.2；训练时每类支持数均大于1则使用留一原型，否则使用自原型回退。回退不等价于独立验证损失。
- 推理时全体中心表示减去全局均值，再做归一化原型分类；训练表示不做此输出中心化。

默认 seed=42。子图版将模型超参暴露在 `SubgraphConfig`，但不内置搜索、早停或测试集选择。全图入口保留历史输入层函数的数值路径，不用旧的简化 Prompt 算子替代它。

## 从仓库根目录使用

安装依赖后从 checkout 根目录运行；全图历史实现需要仓库中的 `experiments` 参考代码，不能只拷贝 `prprompt` 目录。

```python
import torch
from prprompt.model import TwoLayerGCN
from prprompt.input_variants import fit_fullgraph, fit_subgraph, SubgraphConfig

# x: [N,F]，edge_index: [2,E]；由调用方加载。
# support: [S] 全局节点 ID；support_labels: [S]，编码为0到C-1。
backbone = TwoLayerGCN(x.shape[1], 128, 128).to(x.device)
backbone.load_state_dict(torch.load(checkpoint_path, map_location=x.device,
                                    weights_only=True), strict=True)

scores, embeddings, state, audit = fit_fullgraph(
    backbone, x, edge_index, support, support_labels, classes,
    seed=42, steps=100)

# graphs[center] 包含 nodes（唯一的全局 ID）和 edges（局部编号边）。
# 子图必须包含中心。图构造由调用方决定，不应使用待预测标签。
scores_sub, pooled, state_sub, audit_sub = fit_subgraph(
    backbone, x, edge_index, support, support_labels, classes,
    graphs.__getitem__, config=SubgraphConfig())

assert not torch.isin(query, support).any()
probabilities = scores_sub[query].softmax(-1)
```

两套 API 均返回全体节点／中心的分数，`query` 筛选与评估由调用方负责。支持集的每类至少包含一个真实节点。没有内置子图扩跳、补点、裁剪、数据集加载、少样本划分或结果聚合规则；这使模型实现可以与显式、可审计的实验协议组合。

全图路由及推理中心化会访问非支持节点特征和结构，属于传导式设置。`query_labels_received=False` 仅描述函数接口，不证明外部提供的子图或骨干权重没有使用标签。

## 验证与来源

输入层原始 `model.py` 从服务器原样取回，位于 `experiments/f100_input_20261005/model.py`。子图版从已运行的输入 Prompt、子图均值池化及原型训练代码抽取，移除了机器路径和实验控制器，并改为调用独立骨干接口。

执行 `python -m tools.test_input_variants` 检查一支持与多支持两种分支：全图包装器对历史实现、子图 API 对独立参考训练循环，分别核对分数、表示、Prompt、路由以及冻结骨干。CPU合成图检查通过不等于全部真实数据集重跑，也不保证不同GPU上的逐位一致。

验证记录及源文件哈希见 `provenance/input_variants_20261012/`。服务器源文件及完整实验控制脚本另存本地来源快照，不作为公共 API。
