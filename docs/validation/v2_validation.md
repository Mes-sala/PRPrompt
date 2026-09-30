# PRPrompt v2 程序验收

日期：2026-09-30。验收对象是当前 `prprompt/`、`experiments/` 代码及三个实验配置。服务器记录中的全部 12 个源码/配置 SHA256 已与提交前本地文件逐个核对一致。详见 [机器可读记录](v2_validation.json)。

## 已验证内容

| 验收项 | 执行范围 | 结果与边界 |
|---|---|---|
| 科学协议测试 | 6 项测试，覆盖全部 16 个主方法及诊断方法 | 全部通过；检查有限输出、骨干冻结、提示梯度、标签重命名、零提示等价、支持集变化、划分隔离、1-shot 不增加标签及预测篡改检测 |
| 流水线集成测试 | 合成图 8 个配对单元 | 运行、封存、评分、续跑及完整汇总通过；缺失单元和状态篡改均被拒绝 |
| checkpoint 计算兼容 | Cora/CiteSeer × Edgepred_Gprompt、Edgepred_GPPT、DGI，共 6 个 checkpoint | 在 17 节点双向环图输入上，新 GCN 与历史 ProG GCN 前向最大绝对差均为 0；不代表原论文预训练完全复现 |
| 真实图 smoke | Cora 的 Edgepred_Gprompt、CiteSeer 的 Edgepred_GPPT；16 方法 × 2 图 × 1/5-shot | CPU 上 64/64 单元完成并通过汇总审计；每次仅训练 2 步、一个抽样和训练种子，不能用于性能结论 |
| 真实图续跑 | 再执行同一 smoke 命令 | 完整性校验通过，所有 metrics 文件时间戳保持不变，无重新训练 |
| CUDA 兼容 | RTX 5090 上的小型合成图，FrozenLinear、FullFineTune、GPFPlus、PRPrompt、LearnedRouterHidden、RandomFrozenPRPrompt | 6 方法通过；最大已分配显存 18,126,336 字节，仅证明这些路径可执行，不是全图 GPU 性能测试 |
| 正式配置展开 | 主表、shot 曲线、诊断配置 | dry-run 分别为 4,680、18,720、2,520 个最终单元；支持内选参拟合另计 |

测试使用已有 ProG 环境：`/data1/shq/miniforge3/envs/prog-v2-cu130/bin/python`。CPU smoke 使用 4 线程。准确执行参数、checkpoint 来源和哈希在 JSON 记录中；[协议测试原始输出](v2_protocol_tests.txt)与[流水线测试原始输出](v2_pipeline_tests.txt)一并保留。

## 尚未验证的结论

**未运行正式精度实验。** 当前没有新版主表成绩、统计优势、跨图泛化结论或端到端 GPU 效率结果。旧版成绩不迁移为新版成绩，smoke 结果不进入论文。验收覆盖当前两层 GCN 实现，不能推广为任意骨干兼容。

正式运行前按[实验计划](../v2/03_实验计划与执行.md)准备目标机器上的数据与权重清单，执行主配置或 shot 曲线配置，并用汇总程序检查完整覆盖。数据、权重、完整预测和机器路径清单继续由 `.gitignore` 排除。
