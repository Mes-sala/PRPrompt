# 球面概率评分检索与代数检查（未做图实验）

已读[Gneiting与Raftery2007第3.1节Example2](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)的球面评分。对 $ p=softmax(s) $，自身推导并以double值/梯度校验：

$$ p_y/\|p\|_2=\sqrt{softmax(2s)_y} $$

所以最小化1减球面评分等于两倍logit下q=.5的广义CE乘.5；取负对数则恰为两倍logit下CE乘.5。不是新的跨类别耦合。第75轮仅测q=.3/.7，因此此代数关系不证明这个精确q/.1温度组合失败，也不是因不原创拒绝；目前优先把资源给有局部增益的互补熵跨权重验证。本项记为代数筛选、真实图未运行，不记准确率失败，不纳入正式预测计数。
