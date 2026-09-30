# 第88轮草案：更新方向中心化（未实现）

继续借鉴[ECCV2020 Gradient Centralization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460613.pdf)及[作者Adam代码](https://github.com/Yonghongwei/Gradient-Centralization/blob/master/algorithm-GC/algorithm/Adam.py)。作者实现gc_loc=False对矩估计和预条件后的方向处理；第87轮处理矩估计前的梯度，两者不等价。

本轮拟保持输入、门控温度.1、6×128零初始化token、100步、完整CE+L2梯度和当前优化器状态更新不变。GraphMAE由Adam计算原方向D；GraphMAE2由已保留graftedShampoo计算D（已含shadow Adam范数匹配）。然后沿每token隐藏坐标去均值得到H，不处理原始梯度、m/v或L/R。

四种配对：原始强根D；投影H；保持原D范数的H；原方向乘H/D范数比作为幅度控制。预检identity恢复根。零方向显式no-op，记录零分支；当前Adam可能有非零动量，不得假定零梯度等于零更新。保存原始梯度、原方向和实际方向，NumPy从原始梯度重放全部矩更新、投影和最终token。若使用投影方向，数学上各token的均值从0保持0，但需检查浮点误差；不声称tanh(T)后均值也固定。

原实现类权重轴与本项目类别×隐藏轴并不相同，本轮仍是明确适配。R87类别轴仅第一步近似无变化：正式100步平均相对变化GraphMAE约.044、GraphMAE2约.219，不能把整个训练归为无操作。下一轮不重复类别轴，先检验更新后隐藏轴。

拟开发CiteSeer seeds134100..104，预检134199，GraphMAE/GraphMAE2分别相对当前Adam/grafted根。Direct、Frozen、GraphPrompt保留。固定方案过独立审计后，超过强根才20支持确认；全程同图重复查询边界不变。实现前需确定Adam数值恢复方式，不以精度差异冒充优化收益。
