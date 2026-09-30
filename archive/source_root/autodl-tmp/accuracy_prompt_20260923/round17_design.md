# 第十七轮：训练轨迹的提示参数平均

来源：[SWA，UAI2018](https://www.auai.org/uai2018/proceedings/papers/313.pdf)及[作者实现](https://github.com/izmailovpavel/contrib_swa_examples)。另参考[Mean Teacher](https://papers.nips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)的权重EMA概念，但本轮不引入其一致性教师，不称Mean Teacher复现。

同一个冻结GCN，固定平滑岭教师与激活后gamma10 token；Adam200步lr0.01。比较最后一步token、后100步token在线算术平均、后100步EMA(decay0.95，首个纳入参数初始化EMA)。全部使用同一损失和初始化，不在中途按query选择。已有Adam100和原型教师gamma3同批参照。

平均在未经过tanh的可训练token空间执行，只保留一个最终参数矩阵和一个分类器，推断成本不变；不是多个预测融合。继承Adam而非原SWA的SGD/循环学习率，只是借鉴参数平均。在线累加不落盘每步模型，避免存储增长。

3图×2权重×5支持seed24100..24104×8方法=240。验证实际收集100步、冻结及零提示检查；开发结果后才决定是否在新支持上扩大。若均值仅近似更早结束训练，不声称找到更平坦极小值；准确率优先。
