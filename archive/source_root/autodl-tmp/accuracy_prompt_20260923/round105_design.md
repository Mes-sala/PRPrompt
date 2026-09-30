# 第105轮：概率预测二次评分用于支持训练

来源：[Gneiting与Raftery 2007，Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)，已读第3.1节Example1的离散二次/Brier评分及概率预测语境。原文是奖励最大化，我们取负号成为概率向量与one-hot之平方距离。Brier1950原始期刊/NOAA全文链接未成功获取，不宣称已读1950全文。该评分用于统计/天气概率预测；本轮借用其训练目标，不把适当评分性质当作小样本准确率保证。

令支持LOO概率为 $ p_i=softmax(s_i) $：

$$ B=\frac1N\sum_i\sum_c(p_{ic}-\mathbf1[c=y_i])^2 $$

brr主候选为CE+B（系数1固定）；brb为纯 $ (K/2)B $，因均匀概率处其logit梯度与CE相同，控制初始自然尺度而非搜索系数。bra对照为 $ stopgrad(r)CE $，其中 $ r=\|\nabla_s(CE+B)\|_F/\|\nabla_s CE\|_F $，只匹配本步支持logit梯度整体范数，不保证经过共享原型链或Adam/Shampoo后仍匹配token更新。分母0时显式令r=1。brp为CE+B(p,置换one-hot)，只置换辅助目标，原CE标签保持。bri仅预检恢复CE。每步置换CPUgen(seed+1050000)，可能有原位且保持标签计数。

不使用104待确认方案；仍为原单中心锐化教师门控，同6×128 token、共享冻结两层GCN、第一层激活前注入、零初始化。GraphMAE Adam.01/GraphMAE2嫁接Shampoo.01，100步，无KD，正则.001，末层中心化+k50读出。保留两当前优化器根、Frozen、GraphPrompt、Direct和mean/k50。

原语已验CE恢复、所有模式解析梯度最大1.388e-17、纯Brier均匀处梯度恢复误差3.47e-18、正确/错误极端预测和零梯度分支均有限。纯Brier可能在错误且饱和的softmax处缺乏梯度；用CE+B主候选保留CE梯度，这只是设计动机，尚无真实图证据。

固定167100..167104、两权重145开发；167199三步33预检。支持每类5点、开发query固定500，预测封存后才计分。五审计包括独立CPU首末损失/原型链梯度、NumPy优化器全步骤重放。保存各步概率+范数比、置换、首末tokens、全部梯度。主比当前根，报告纯Brier、幅度、错配辅助目标及Direct；五支持只用于固定确认候选。同图反复query边界不变，确认20新支持不能视为独立图泛化。
