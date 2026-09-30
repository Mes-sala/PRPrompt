# R138正式设计：Prodigy自适应步长

来源与草稿见round138_design_draft.md。已核对[原文算法4](https://arxiv.org/html/2306.06101v2)及固定官方源码。实现以官方源码为准：分子/s共同去掉1-sqrt(beta2)并除d0可约掉；但源码分母eps乘新d，而论文行11写旧d。这一差异明确保留，不声称与论文逐位等价。仅借用Adam版本，理论凸条件不自动适用于当前图Prompt。

实现optimizer138.py使用官方Prodigy+CPUdouble状态，每步回写float32真实token再同步舍入的影子。模型冻结、token896参数、原teacher/gate/KD/动态scale2与k50均不变。prd=默认适配，pbc=适配加偏差修正，pf0=固定d=1e-6且无偏差修正，pf1=固定d=.01加偏差修正；固定通过d_coef=0实现，其余状态仍更新。pf1与标准Adam存在eps缩放约定差异，不宣称位相同。

8组合合成120调用含初始零梯度和中途零梯度、零/非零初始化均通过全状态NumPy回放，实际118更新；不是调用次数等于更新次数。Cora无query诊断seed205199、两ck×四模式100调用均100更新，回放最大7.27596e-11。GraphMAE默认最终d=.0521，偏差修正.0629，GraphMAE2 .1483/.1530；100步足以适配，无需凭query调d0。诊断未评价准确率。

开发Cora两ck、5新support205100..104、官方train每类5个、dev500、100调用。原token/scale2双强根，Direct与基础GP和k50保留；四Prodigy来源mean/k50，共165。smoke205199、12调用33条。必须验证50完整轨迹/100快照、40Prodigy全状态、80近邻；smoke10/20/8，根identity不变。所有状态步按旧/新d顺序重算，首末loss梯度、原型、teacher、晚层forward、来源哈希、support/query、sealed预测准确率同审计。

扩大条件：prd/pbc/pf0/pf1 k50中最佳，同时超过两ck token/scale2最佳及Direct至少.20pp；平手先prd/pbc/pf0/pf1，再GraphMAE/GraphMAE2。若固定控制赢只归因为控制效果，允许按准确率扩大，但不称适配收益。通过后固定20新support确认；同图多次query使用的限制不变。不通过保留完整结果并继续其他方向。
