# 第92轮：概率与CLR门控残差混合

来源：[CLIP-Adapter原文](https://img.shlab.org.cn/pjlab/files/2021/10/637704385028930000.pdf)。论文把适配特征与原预训练特征残差混合；本轮只借用保留原表示的混合原则，位置改在提示门控，不复现其瓶颈adapter，也不把两门控称两个独立Prompt模块。第91轮GraphMAE2 CLR完整替换确认仅+.09pp、区间跨0，未保留；这不足以证明混合有效，只作为下一步假设。

令Bq为当前温度.1概率门控，Bc为匹配Bq范数的CLR门控。固定M=.5Bq+.5Bc，不学习混合系数；主候选再将M范数匹配Bq。一个冻结GCN、同一个6×128token，混合发生在第一层激活前提示注入的gate处，最终中心化+k50不变。代数上这是一个混合门控乘一个token矩阵，不是多模型终端融合，不声称新表达模块。

拟只在有局部信号的GraphMAE2筛选：原概率/gsh、完整CLR/clm、.5原混合、.5范数匹配混合、原概率门控乘M/Bq范数比、固定节点置换CLR后混合并匹配范数。原Adam/Direct/Frozen/GraphPrompt仍保留。置换用独立CPU seed并保存，不动支持或标签；它打乱门控对应关系，是诊断控制而非可用目标。alpha0恢复原概率，alpha1恢复完整CLR仅预检，不扫描alpha。

拟142100..104，预检142199。所有门控无查询标签；CPU独立重建混合及归一化，NumPy重放更新，原语检查端点和零方向分支。只有超过当前gsh并在新20支持确认稳定才继续第二确认；同图重复查询边界不变。实现已完成，待原语/预检通过后运行。

实现gate92.py，端点来自gate91.py，独立CPU审计用log_softmax而非调用helper。置换为CPU generator(seed+920000)在CLR传播后进行，保存排列、CLR及门控SHA。brr/brm/bra/brp分别为原混合/范数匹配/纯幅度/置换；brz/bro只做alpha0/1恢复预检。GraphMAE2正式85条，预检21条、3步；原GraftedMatrix更新，NumPy独立重放。
