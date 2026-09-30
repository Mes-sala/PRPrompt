# 第63轮草案：借鉴SIF抑制公共特征方向

来源：[作者SIF仓库](https://github.com/PrincetonML/SIF)、[权重源码](https://raw.githubusercontent.com/PrincetonML/SIF/master/src/data_io.py)、[去主成分源码](https://raw.githubusercontent.com/PrincetonML/SIF/master/src/SIF_embedding.py)。论文为ICLR2017 A Simple but Tough-to-Beat Baseline for Sentence Embeddings。OpenReview原PDF本次受浏览器验证阻挡，已读作者机构摘要及官方实现；不能声称本次阅读全文。源码使用平滑逆频率权重，并从未中心化表示移除第一主成分。

拟将CiteSeer词袋特征视为稀疏特征坐标：按无标签全图特征出现次数估计 $ p_j=df_j/\sum_k df_k $，设置 $ w_j=a/(a+p_j) $，在原行归一化X上乘w后做原Ahat²平滑。先保留原有行均值分母，不再按权重和重新归一化，避免偏离加权平均含义；图平滑后，再可选移除未中心化第一右奇异方向。a候选1e-3/1e-4。需验证原始特征是否确为非负词袋，并记录频率统计来源。

带原强修正教师、只加权、只去PC、两者结合和普通中心化控制；同修正原型/温度/支持查询，不加入上一轮未确认的传播。去PC方向只从无标签特征计算，不用查询标签；特征空间SIF式适配不是原文词向量复现，也不是Prompt。若强教师准确率提高，再把其分数作为提示KD或门控来源测试，主干输入不改，防止预训练输入契约混淆。只保存权重/PC及必要哈希，不保存大型稠密中间副本。尚未固定正式配置/seed或实现。
