# 第86轮：传播特征平方根教师

来源：[Free Lunch for Few-shot Learning: Distribution Calibration](https://arxiv.org/html/2101.06395)，ICLR2021，公式3；[作者实现](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration/blob/master/evaluate_DC.py)固定beta=0.5。这里只借用逐元素幂变换，不复现其基类统计迁移、协方差或生成样本，不要求机制原创。

## 操作与假设

NormalizeFeatures输入不变，先计算非负 $ R=\hat A^2X $，候选 $ R'=\sqrt R $，再减去全图特征均值。从支持标签构造原型，沿用3次beta=.5、伪分配温度.05的rectified教师，最终门控概率温度.1。变换可能减少大坐标主导程度；这只是待验证解释，不预断其使图特征高斯化。放在传播后避免原始二值坐标平方根退化；0保持0，无epsilon注入。

一个冻结GCN、一个6×128零初始化token，第一层激活前注入，幅度10，CE留一原型+.001tokenL2；无KD。GraphMAE用Adam.01，GraphMAE2用已保留的步长匹配Shampoo.01。100步、最终中心化+k50保持。没有终端多模型融合。

## 固定对照与协议

pwr平方根；pwm把平方根门控范数匹配原门控；pwa原门控乘候选/原范数比；pw2平方变换作为反方向对照；pwo幂1仅预检恢复。保留tr1及gsh，原Direct与平方根Direct、Frozen、GraphPrompt；mean/k50配对。Direct不使用checkpoint。开发CiteSeer seeds132100..104，每类5支持、500开发查询，150条正式。预检132199、3步34条。查询标签仅封存预测后读取。重复使用同图查询，不解释为独立泛化样本。

先原语NumPy对照，再预检四项审计；全部通过才自动开始正式。CPU独立重建平方根/平方、教师门控和GCN读出；NumPy重放更新；幂1预测恢复当前根。封存训练源SHA后不得修改。只有超过对应当前强根才考虑新的20支持确认；同时记录范数控制、直接教师与平方对照，避免只比较弱基线。正式证据保留，预检审计通过后按清单删除。
