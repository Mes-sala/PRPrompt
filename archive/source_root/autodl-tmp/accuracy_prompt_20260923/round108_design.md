# 第108轮：RAdam步长预算与匹配学习率

第107轮lr=.01/100步原式明显退化，而方向范数匹配恢复大部分准确率；据此提出“短程累计更新不足”的待验证解释，不当作已证明原因。[作者README](https://github.com/LiyuanLucasLiu/RAdam#questions-and-discussions)明确仍需调学习率，本轮做有界匹配网格，不因一个lr失败就否定方法。

复用optimizer107原递推，不新增优化器算法。RAdam原式（前5步SGD回退）、仅幅度控制、Adam、graftedShampoo各取.01/.03/.1；原式lr.01对应rrr，.03/.1为rrr3/rrr10；仅幅度为rra/rra3/rra10。Adam是tr1/oad3/oad10，gsh是gsh/gsh3/gsh10。仅幅度参考根GraphMAE=Adam、GraphMAE2=gsh，均在自身梯度流上匹配RAdam范数，不能当作相同轨迹。所有方法100步，不改变训练预算、损失或教师。

提示仍为共享冻结两层GCN、第一层激活前6×128 token、单中心门控.1、零初始化、CELOO+.001正则、中心化及k50读出。无预测融合新模块。固定173100..104，双checkpoint，共265开发；173199八步57预检，包含恢复原根rri。新增学习率必须由result.optimizer_lr和完整NumPy更新重放核验，不只验证参数解析。

开发筛选：按5个支持平均准确率选择各权重最优RAdam学习率，必须超过该批最佳Adam/gsh学习率以及原保留根至少.20pp才进入新20支持确认；也报告幅度对照，若幅度更强可按准确率目标选择幅度，但不能称RAdam方向收益。若只有Adam/gsh调参超过原保留根.20pp，单独作为调参候选。所有比较使用同批、同分割、同checkpoint；确认时固定选择，不重新挑lr。五支持仅筛选，同图重复query不代表跨图泛化。
