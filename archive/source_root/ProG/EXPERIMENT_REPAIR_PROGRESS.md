# ProG 实验修复进度

依据：[修复方案](EXPERIMENT_REPAIR_PLAN.md)。开始日期：2026-09-11。

## 当前状态

实施中。保留既有工作区改动、历史 CSV 和 checkpoint。本文件随代码和验证进展更新；未通过验证的项目不标记完成。

| 阶段 | 状态 | 内容 |
| --- | --- | --- |
| A 实现有效性 | 基本完成 | SelfPro 标签隔离、EdgePrompt 前向、PSP 评估一致性、冻结与能力检查；未核验方法仍禁止进入正式排名 |
| B 梯度与状态测试 | 已完成 | 合成图上的参数更新、冻结哈希、标签置换、step 计数和缓存篡改检查 |
| C 运行器与可追溯性 | 已完成 | 显式配置、固定预算、step 计数、结果身份、源码/环境/预测哈希 |
| D 小矩阵集成 | 进行中 | 无下载合成图回归已完成；真实 Cora 已跑固定配置验收，尚未开展正式多种子矩阵 |
| E 作者实现核验 | 待执行 | 未核验方法保持 adapted/unverified 标识，不冒充忠实复现 |
| F 正式实验 | 待执行 | 前序通过后才可开展，不复用旧 202 个 cell |

## 记录

- 初始审计：SelfPro 测试原型使用 query 标签；EdgePrompt 节点前向未传 prompt；PSP 训练/评估路径不一致。
- GraphPrompter 本地 `rep/GraphPrompter` 不存在，正在检查其他来源；在确认合法 NodeTask 实现前应拒绝该组合。
- 现有方案文档不等于修复已完成；现有 FAIR_PROTOCOL_V3 的冻结承诺需修订。

### 第一批代码改动（待测试）

- SelfPro 原型仅使用显式 support 索引/标签，预测 API 不接收 query 标签；缺类时报错。
- EdgePrompt/plus 训练与评估均接入逐层 prompt；移除图级读出前重复且维度可能错误的额外注入。
- PSP 评估改用实际训练的 prompt logits，优化器只管理其参数；仍标识为简化适配，不宣称恢复作者结构机制。
- 所有非 None 的标准骨干明确冻结；All-in-one 交替关闭非活动模块参数梯度。MultiGprompt 专有预训练器额外冻结。
- GraphPrompter NodeTask 明确拒绝执行，避免静默退化。
- 新增统一节点评分入口：显式 softmax、全类别 Macro-F1、缺类别 AUC 返回 NaN。
- 增加可选固定预算运行模式：完整 support batch、禁用早停、All-in-one 显式周期、逐 optimizer step 计数、参数变化/冻结哈希诊断。

### 验证进展

- 第一轮新测试 19 passed / 1 failed：失败定位到 GPPT 的 TaskToken 声明为 2h、初始化却强行改成 h；已改为与实际前向一致的 h，并使用 copy_ 保持参数形状。
- 修复后：`test_protocol_repair.py + test_input_structure_ports.py` **37 passed**。包括真实 optimizer 步数、冻结哈希、SelfPro 标签置换、EdgePrompt 梯度、配置身份与 split 重用、SUPT/GraphTOP 来源数值检查。
- 旧策略回归尝试：6 passed、2 个 MUTAG 下载/DNS 失败；后续慢测试中止，不能宣称全套通过。改用无下载合成图测试和本地 Cora 实验。
- v4 单 cell 入口已实现；旧 matrix worker 转接 v4，输出目录/CSV 改为 v4 后缀；同一 GPU 使用串行队列。支持预训练 checkpoint + None 完整微调，不再遗漏该基线。
- 新增全图节点原型、自环节点原型及 GraphTOP 原图边权对照，正在补充回归。

### 本地真实数据验收启动

- 本地 Cora 和已有 GraphMAE checkpoint 可用，启动 8 组 × 2 个 support split 的完整预算验证：Scratch/Pretrained FullFT、随机/预训练 Frozen，以及 GraphMAE 下的 FrozenGraph、GPF-plus、FrozenGraphPrototype、Gprompt。
- 使用 200 support passes；原型基线不训练。CPU 单线程仅用于控制本地验收开销，不代表性能基准。
- 临时日志 `/private/tmp/prog_real_validation.log`；持久摘要 `results/protocol_v4_validation/validation_summary.json`。
- 初步完成：Scratch FullFT 0.669、随机 Frozen 0.415、GraphMAE FullFT 0.650。仅 2 个 split，不据此选择方法或调整超参数。
- 完成 GraphMAE 配对验收：GraphMAE Frozen 0.611、FrozenGraph 0.6045、GPF-plus 0.417、FrozenGraphPrototype 0.6375、Gprompt 0.6405（均为 2 split、固定 200 pass；仅工程验收）。这些数字说明代码已能区分完整微调、线性探测、子图和原型路径，不能作为正式方法结论。
- 最新回归集：`test_pretrain_ports.py + test_protocol_repair.py + test_input_structure_ports.py + test_induced_graph.py + test_factory.py` 为 **63 passed, 3 warnings**；warnings 来自第三方依赖的版本检查，不是测试失败。
- v4 验收解释：GraphMAE Frozen 相对随机 Frozen 提升 19.6 个百分点；GraphMAE 全量微调比 Scratch 低 1.9 个百分点。GraphMAE 下 Gprompt 相对 FrozenGraphPrototype 仅高 0.3 个百分点，而 GPF-plus 比 FrozenGraph 低 18.8 个百分点。由于只有 2 个 split，这些是实现和方向诊断，不是正式统计结论。
- 最后一轮复跑仍为 **63 passed, 3 warnings**；代码质量检查覆盖新增评分、审计、S2GAE/GraphMAE2/Bandana provenance 和 v4 runner，Ruff 全部通过。

### 预训练移植修复

- S2GAE 的 `um/dm` 边掩码改为先构造明确的待掩码集合，保留边不再通过无向化重新引入被遮蔽目标；新增正反向边交集检查。
- GraphMAE2、S2GAE、Bandana 导出的 checkpoint 旁写入 `.pth.json` provenance：输入哈希、配置、实际完成轮数、停止政策、优化器、源码哈希和 `ProG-adapted / not_fully_verified` 标识。
- 当前预训练端口测试：63 项（包含新元数据与 S2GAE 掩码测试）已通过；MUTAG 下载型旧测试仍受本地 DNS/无网络影响，未计入通过数。

### 当前剩余工作

- 统一 ProNoG、MultiGprompt 和 DHBP 的评分入口已补齐；MultiGprompt 明确以 `train=0` 评估，不用 query 标签更新原型。
- 仍需逐方法完成作者仓库小图数值对照，尤其是 GraphPrompter 节点任务、PSP 简化结构和新增预训练端口；这些在能力表中继续标记 adapted/unverified。
- 正式实验前还需要清理旧 scheduler 对 legacy JSON 的兼容路径，并以 v4 artifact identity 重新生成完整矩阵。
- 当前尚未完成独立的 pretrain-seed × downstream-train-seed 方差实验；v4 验收中观察到的差异主要来自两个 support split，不能当作 seed 方差。

### Checkpoint 是否需要重训

- DGI、GraphMAE、GraphCL、SimGRACE、Edgepred_GPPT、Edgepred_Gprompt 等已有 checkpoint 的网络结构仍与当前 GCN 下游加载接口兼容；下游冻结/评分/预算修复不会改变其权重，因此可以继续用于工程验收。但它们没有 provenance sidecar，正式结果标记为 `legacy_unverified`。
- GraphMAE2、S2GAE、Bandana 的旧端口 checkpoint 不进入正式结果：S2GAE 的边掩码曾存在重新引入隐藏边的问题，且旧文件没有完整训练 provenance。三者应在当前修复后重新预训练并生成 `.pth.json`。
- 如果目标只是比较下游训练代码、验证冻结和输入路径，不必先重训所有旧 checkpoint；如果目标是发布最终预训练排名，则建议所有预训练方法统一重新训练至少 3 个预训练种子，并记录相同数据、GCN 配置、训练预算和 checkpoint 选择规则。
