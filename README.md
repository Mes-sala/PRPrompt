# PRPrompt：面向少样本节点分类的原型路由图提示

**Prototype-Routed Graph Prompting for Few-Shot Node Classification**

PRPrompt 利用支持集锚定的类别原型构造固定的节点相关路由，组合少量可学习类别提示，并将提示注入冻结 GCN 的隐藏计算。最终采用原型分类。

当前论文方案已经固定，材料入口为 [PRPrompt 论文材料](docs/paper/PRPrompt_三数据集_20261002/README.md)。当前范围为 Cora、CiteSeer、PubMed 的全部 1/3/5-shot 条件；所有条件使用同一方案，以原型路由为核心组织主比较及消融。

## 论文与数据

- [中英文 LaTeX 论文](docs/paper/PRPrompt_LaTeX_20261002/README.md)：完整正文、附录及共享数据表，主架构图保留占位符，已在独立 LaTeX 环境编译并逐页核对。

- [论文叙事](docs/paper/PRPrompt_三数据集_20261002/论文叙事.md)：摘要、引言、原型路由方法与结果解释。
- [主比较表](docs/paper/PRPrompt_三数据集_20261002/主比较表.md)：冻结预训练、全量微调、GPF、GPF-plus、GraphPrompt 统一适配与 PRPrompt。
- [原型路由消融](docs/paper/PRPrompt_三数据集_20261002/路由消融.md)：完整原型路由、同配方共享提示、去掉提示。
- [配对比较](docs/paper/PRPrompt_三数据集_20261002/配对比较.md)：完整九个数据集／shot 条件的差值与区间。
- [实现与实验协议](docs/paper/PRPrompt_三数据集_20261002/实现与实验协议.md)：传导式信息范围、原型校正、优化、1-shot 适配及统计细节。
- [可复用数据与验收](docs/paper/PRPrompt_三数据集_20261002/README.md#数据)：最终方案 540 个逐次记录、完整五指标及配对统计。

结果按数据集、shot 和方法分别报告，仅对同一设置的 seed 计算均值与标准差，不计算跨数据集或跨设置平均值。当前范围在已有实验完成后指定，结论限于这些固定图和 checkpoint 下的条件。

## 实现与复现

```text
experiments/boundary_20261002/model.py   最终方案的实际训练入口
experiments/v1_f100/historical.py       冻结实现载入
experiments/f100_boundary_20261002/     同配方共享提示控制
prprompt/                             模型、基线和协议组件
tools/build_prprompt_three_dataset_paper.py 三数据集结果提取与配对统计
docs/paper/PRPrompt_三数据集_20261002/           当前论文工作稿与数据
```

实际训练实现与方案标识见[实现协议](docs/paper/PRPrompt_三数据集_20261002/实现与实验协议.md#可复现入口)。材料提取脚本不启动训练、不修改种子或超参数；默认拒绝覆盖已有交付目录。

服务器复用 **ProG conda 环境**。本地只保留代码、实验结果、表格及文档；checkpoint、模型状态和数据快照保留服务器。模型权重、运行机密、机器配置、Finder 隐藏文件均由 `.gitignore` 排除。详见[本地保留范围与验收](docs/validation/本地保留范围_20261002.md)。

## 归档与来源

[历史归档导航](docs/历史归档导航.md)保留此前材料；现有原始实验记录不因论文材料整理而改写。

第三方源代码沿用原版权和许可证，例如 [ProG 许可证](archive/source_root/ProG/MIT-LICENSE.txt)。本仓库没有为混合来源历史内容重新指定统一许可证。
