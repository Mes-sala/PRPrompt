# 第74轮：提示参数的符号动量更新

来源：[论文](https://arxiv.org/abs/2302.06675)、[官方说明](https://github.com/google/automl/blob/master/lion/README.md)、[官方代码](https://raw.githubusercontent.com/google/automl/master/lion/lion_pytorch.py)。已核对双动量更新顺序；HF markdown访问失败后转官方源码。作者提示符号更新通常需更小学习率。

共享冻结GCN及768提示参数不变，位置保持第一层卷积后激活前，教师门控及只读中心化+k50不变。先用零初始化隔离优化器影响；第73轮随机初始化保留为强候选参照。

先用0.9旧动量与0.1当前梯度的和取sign更新参数，再用0.99旧动量加0.01当前梯度更新动量。学习率固定0.001/0.003/0.01，Adam同三档与无动量sign三档匹配。全部100步、同LOO CE与0.001 token均方正则，不加解耦weight decay；保持目标固定，不是完整原论文调参复现。

主要判断需超过同批最好Adam及既有强提示，并报告Direct；若只胜Adam默认而不胜匹配学习率Adam，归因学习率选择。初筛CiteSeer两权重、新支持113100–113104，保留零/随机Adam0.01、新KD、Frozen/GraphPrompt及Direct。完整矩阵、真实GCN轨迹审计和联合源码哈希尚待接入，目前只有优化器原语，未启动准确率验证。


## 固定执行矩阵

两权重各30方法：9优化器/学习率×两读出18项，原无KD/新KD的原始及中心化8项、随机初始化2项、Frozen/GraphPrompt2项；Direct每支持一次。新支持113100–113104共305条，预检113199共61条3步。训练源码+init73+optimizer74联合SHA封存。

每条优化器提示保存逐步梯度，独立CPU按官方Lion/torch Adam回放更新，核对最终token；此回放验证更新规则但不独立验证梯度正确性。冻结、teacher/source哈希、CPU前向中心/原型、近邻权重及传播审计沿用强根。Adam0.01与原强零初始化训练恢复核对token，容许GPU归约微误差，不要求并列近邻位次完全相同。全部预检通过后才进入正式；无query标签训练。
