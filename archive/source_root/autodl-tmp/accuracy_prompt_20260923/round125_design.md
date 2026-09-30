# 第125轮正式设计：相对首步梯度尺度的退火噪声

继承round125_design_draft.md的来源、公式与局限：[原论文第3节式1](https://arxiv.org/pdf/1511.06807)。采用初始干净梯度RMS的.1/.3倍为噪声标准差，随后按(1+t)^−.275衰减；不使用原文绝对eta网格。没有额外可训练参数，全部候选共享原768参数Prompt结构、冻结两层GCN，训练阶段改梯度，不是读出融合。

固定CiteSeer5shot、GraphMAE/GraphMAE2；支持190100..104，100步，lr.01，原门控.1、无KD、LOO CE与token正则保持。GMAE候选用Adam、GMAE2候选用graftedShampoo。原tr1/gsh/Direct/Frozen/GraphPrompt全部保留。

各强度五种方法：退火原式r、梯度范数匹配m、仅梯度幅度a、恒定新噪声c、固定噪声向量随步退火f；每种记录mean/k50。正式265条。六个r/m/a+k50为候选，须超过同批tr1/gsh最佳至少.20个百分点才进入20新seed确认；相等时r1,m1,a1,r3,m3,a3顺序。c/f仅解释退火与刷新随机方向作用，不事后选为主候选。若仅a胜出，归因为梯度幅度控制；自适应优化器累计状态会改变，不声称参数更新方向不变。

冒烟190199、3步57条，增加gni0零强度恒等，要求两根恢复。CPU独立Generator隔离全局随机流，保存每步标准噪声、clean/noisy/used梯度、sigma与s0；保存首末步tokens与干净梯度快照。新增audit125_noise独立重放随机序列和NumPy变换，再CPU重建首末步CE/正则梯度。原几何、教师、邻域、传播、优化器回放检查全部保留；当前正式几何审计继承版本另增加最终prediction==scores.argmax附加检查。全smoke通过后才正式启动。

所有query标签仅封存预测后用于计分。配对support bootstrap不能作独立图显著性，跨轮反复使用query产生的适应性边界保留。第125轮不承诺噪声会改善，仅测试可复现的干预。
