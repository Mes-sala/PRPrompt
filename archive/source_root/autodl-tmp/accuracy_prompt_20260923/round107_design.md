# 第107轮：RAdam早期步长修正

已核对[RAdam论文，ICLR2020](https://www.microsoft.com/en-us/research/uploads/prod/2020/04/1908.03265.pdf)及[作者实现](https://raw.githubusercontent.com/LiyuanLucasLiu/RAdam/master/radam/radam.py)。借鉴视觉/语言任务的自适应学习率方差修正，不假定本图支持全批训练具有相同随机梯度分布。HF markdown不可用后转作者论文/代码。Lion第74轮已有匹配学习率实验未胜，不原样重复。

固定beta1=.9、beta2=.999、epsilon1e-8（加在未偏差校正的sqrt(v)之后），作者代码N_sma>=5分支；前三至五步的修正分支尚未开启，第6步开始adaptive。rrr主原式在早期用偏差校正动量SGD，即显式degenerated_to_sgd=True；作者代码默认False，其前5步跳过参数更新，本轮rrs单独控制该语义。两者从第6步采用同一修正公式。CPUdouble维护状态后转换float32更新提示，这是本地数值实现差异。

rrm把RAdam方向范数匹配当前根在该分支相同梯度流上的方向范数；rra把当前根方向缩到RAdam范数；rri恢复当前根仅用于预检。根GraphMAE为Adam，GraphMAE2为嫁接Shampoo，lr.01。匹配后标量修正因子会被抵消，但早期动量方向与Adam不同，且epsilon位置不同；不宣称范数匹配后的RAdam具有独立后期修正机制。影子根不等于另一条独立训练轨迹。

共享冻结两层GCN、6×128单提示矩阵、旧单中心锐化门控、第一层激活前注入、零初始化、CELOO+.001正则、100步与中心化+k50保持不变。保留两个独立根、Frozen、GraphPrompt和Direct。

原语：两种根×5模式、100步含零梯度，独立NumPy递推最大参数误差2.98e-8；原根恢复、缺失梯度无状态更新、第5/6步分支切换通过。这里未声称运行作者完整float32实现逐位相同。

固定171100..171104、145开发；171199八步33预检，确保真实图触发SGD/skip到adaptive切换。保存全梯度、根/RAdam/实际方向SHA及每步rho/修正系数。四审计（计分封存、教师引用、独立前向/邻居、NumPy全更新）。当前根增量为主，幅度和skip对照同步报告；五支持只用于筛选后续20支持确认。105已失败损失和104未升级互补熵均不混入根。
