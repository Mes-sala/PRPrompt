# 第90轮：提示通道组正则

来源：[NIPS2016 Learning Structured Sparsity in Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2016/file/41bfd20a38bb1b0bec75acf0845530a7-Paper.pdf)，公式1—2按通道/滤波器分组的L2范数和。本轮从视觉压缩借用分组收缩，只检验few-shot准确率，不做主干剪枝或计算加速声明。

一个冻结GCN，提示矩阵T为6×128。沿每个隐藏通道聚合6类token，拟在原CE+.001均方L2上加0.1乘平滑组范数均值：

$$ R(T)=\frac1{128}\sum_{h=1}^{128}\left(\sqrt{\frac1{6}\sum_cT_{ch}^2+10^{-6}}-10^{-3}\right). $$

平滑处理使零初始化梯度有限；它不是精确group Lasso，不保证精确零或通道消除，不进行阈值剪枝。约束T而非tanh(T)，先明确后再实现。原门控、Adam/Grafted两个匹配根、100步和中心化+k50保持。

对照：同大小固定随机分组（768元素打乱后128组×6，保存排列）；把普通L2梯度每步范数匹配当前分组正则梯度（系数detach，零范数显式处理）；系数0恢复原根。随机分组和L2匹配均仍保留原.001L2。只做一个预定系数，不按查询标签挑强度。

实现前先合成双精度梯度/有限差分与零点检查；真实无查询首步及后续少步诊断正则梯度/CE梯度比例。保存CE+原L2梯度、额外正则梯度和各步token用于独立解析梯度核验，然后NumPy重放优化。梯度匹配控制是训练启发式，不把detach系数当固定目标函数。

拟开发138100..104、预检138199，CiteSeer两checkpoint。超过对应当前根才新20支持确认；Direct与原gsh/tr1保留；同图重复查询边界不变。实现已完成，原语和预检通过后运行。

实现regularizer90.py；随机排列用独立CPU generator(seed+900000)。基础梯度先反传，额外正则梯度乘.1后加入，再交给原Adam/Grafted。匹配L2的系数detach；零点两者都为0。保存基础/正则/总梯度、每步token、分组索引与SHA。解析梯度用独立NumPy散射重建，逐步token和optimizer重放核验；history分别记录基础与总loss。正式125、预检29（3步），源封存后不变。
