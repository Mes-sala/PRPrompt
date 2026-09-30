# 第34轮：Actor固定几何与提示跨划分复核

沿用第32轮NLP句向量白化借鉴与相对数值下限，不修改已审计的run_round32_confirm.py。官方fold1–9，每fold5个新支持seed=52100+100*fold至+4，5-shot/class，官方test；这些测试节点在第31轮复核中使用过，不是新独立测试。全图原始特征、原始有向边契约不变，仅自环主干冻结。

GraphMAE八方法：原图Frozen、原始岭、仅自环原几何base、中心base、diag10 base/CE内部token/GraphPrompt/随机token。GraphCL两方法：中心base、diag10 base。共9×5×10=450条。所有训练100步、gamma3及收缩.1固定，不按fold调参。全图无标签统计固定；同几何提示置于第一层激活后、第二层前，共享同一冻结主干，不做末端分数融合。

主要比较GraphMAE diag10 base对原几何/中心/原始岭/原图Frozen，及CE对同几何base、GP、随机。GraphCL用于跨权重直接比较。报告逐fold均值、9fold正向数及fold/支持两层描述区间；查询重叠意味着不能称为9张独立图。只要固定方法提升也保留，但不归为训练Prompt增益。
