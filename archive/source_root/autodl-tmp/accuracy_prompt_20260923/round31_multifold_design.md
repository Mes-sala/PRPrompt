# 第31轮扩展：Actor fold1–9稳定性

固定已选GraphMAE仅自环+CE内部token，不重新选幅度、步数或教师。官方fold1–9，每fold5个支持seed=48100+100×fold至+4；每类5-shot只从该fold训练池抽取，查询该fold测试集。各fold重新使用相同冻结权重；query封存后评分，来源同第31轮。

GraphMAE九方法：Frozen、原GraphPrompt、raw、rawridge、仅自环base/CE/KD/随机token/同结构GraphPrompt；GraphCL只运行仅自环base作为跨权重强对照。9×5×(9+1)=450条。沿用已审计run_round31_combine.py，不更改历史脚本。每个fold/权重独立结果目录与config，支持断点续跑；同fold/seed必须graph/split一致，跨权重只允许checkpoint不同。

先分别报告9个fold均值和每fold配对差，再总体等权fold平均。区间采用fold及fold内支持的两层重采样，仅作同图划分敏感性描述；官方fold测试节点重叠，所有结果不构成独立图显著性。保存测试集合重叠统计。若局部增量不跨fold稳定，不因累计平均增益小于零而隐去fold0局部结果，也不将其夸大为普遍优势。
