# 第60轮确认：恒定Adam学习率0.02

开发220条中Lookahead未超过恒定学习率。准确率优先，转而固定最强简单对照为主候选：GraphMAE/head_lashigh_qeps-50，即scale2损失+恒定Adam0.02+k50。开发76.68%，原scale2 75.28%、旧CE75.32%、普通CE高学习率76.32%、Lookahead同内lr0.02为75.24%。命名中的la仅复用helper，period1/alpha1等价普通Adam，不归因于Lookahead。

新20支持91100–91119，固定既有确认500查询，两权重12方法共480条：Frozen/GraphPrompt及旧CE、原scale2、高lr普通CE、高lr scale2、Lookahead内lr0.02 scale2，各两读出。GraphMAE为主，GraphMAE2为次要迁移。主比较高lr scale2传播对原scale2传播和旧CE传播，另核对普通CE高lr及Lookahead。禁止确认后改lr或候选；若优势清晰，再用第二批新支持复核。所有结果仍来自同图重复查询。
