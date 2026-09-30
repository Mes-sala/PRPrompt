# 第121轮：带符号幂归一化的 Prompt 读出

## 来源与边界

Perronnin、Sánchez、Mensink，ECCV 2010，[Improving the Fisher Kernel for Large-Scale Image Classification](https://europe.naverlabs.com/wp-content/uploads/2010/09/PSM10_0766.pdf)，§3.2、式15：带符号幂映射后做L2归一化。原论文动机是稀疏Fisher向量的内积相似度，并固定alpha=0.5；不能直接断言GCN有相同统计结构。本地保留原PDF、提取文本及SHA。

本轮是现有Prompt训练后的固定读出几何替换，额外可训练参数0；共享原768参数Prompt及冻结GCN，不涉及多模型融合。第86轮在非负A²X教师输入上做sqrt，本轮在最终有正负值的中心化GCN表示上操作，变换位置不同。

## 方案

令已训练表示减去全图均值得$ Z $，在原型构建与k50邻域构建之前：

$$ T_\alpha(Z)=\operatorname{sign}(Z)|Z|^\alpha. $$

零保持零，不加epsilon，不再额外中心化。行L2归一化沿用原读出。固定候选alpha=0.5、0.75；alpha=2作为反向形状对照，同样记录但不参与候选选择；alpha=1冒烟检验恒等恢复。训练损失、随机种子、100步更新完全不变；同一训练根的所有幂变换共享缓存，保存相同tokens及变换前表示，验证SHA。原训练不经过幂映射，因此不宣称幂导数能改进学习。

## 协议与判定

CiteSeer5-shot，GraphMAE/GraphMAE2，开发support seeds187100..187104，冒烟187199、3步。原dev/confirm query分割不变，不访问query标签训练或选幂参数；跨轮已反复查看同一dev集合，结果存在适应性边界。

四个根tr1、gsh、postbase、postgp，分别为Adam Prompt、grafted Shampoo Prompt、中心化Frozen、中心化GraphPrompt。每根恒等/0.5/0.75/2，记录mean和k50；另保留原始Frozen、GraphPrompt、Direct。正式345条，冒烟85条。

主要候选仅GraphMAE/tr1与GraphMAE2/gsh的0.5/0.75 k50。相对同checkpoint未变换tr1/gsh中较强者平均提高至少0.20个百分点才进入20新support seeds确认；相等时优先0.5。正式确认必须比较同幂Frozen/GraphPrompt、Direct，区分通用几何改善与Prompt条件增益。alpha2若有信号只能另立后续预登记，不能本轮选中。mean仅诊断。支持种子bootstrap不代表独立图证据。

## 验证

原语：NumPy与Torch双精度、符号保持、零处理、列置换、正行缩放的归一化不变性、alpha1恢复。端到端：所有sealed SHA和分割、Direct教师、CPU独立GCN/中心重建、相同tokens/预变换SHA、独立float64幂映射、邻居top50与权重、传播、最终argmax、优化器轨迹恢复。平方根会放大近零CPU/GPU重建误差，故先验证重建与保存的原始中心化表示在既定误差内，再在该精确保存值上独立计算幂映射，不能放宽最终得分阈值。smoke全部通过后才自动跑正式。
