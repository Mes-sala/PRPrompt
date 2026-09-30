# 第十六轮：Prompt参数的邻域稳健优化

借鉴[Foret等，SAM，ICLR2021](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)及[作者代码](https://github.com/google-research/sam)。原方法寻找附近损失也较低的参数。本轮用于少标签条件token，冻结GCN，只在提示参数上施加L2梯度方向扰动。

同一平滑岭教师+后层gamma10基础提示，损失仍为支持LOO CE+KD+参数正则。每步先求梯度 $ g $，临时扰动 $ \epsilon=\rho g/(||g||_2+10^{-12}) $，再在扰动点求梯度，精确恢复原参数后用第二次梯度执行Adam。代码对范数与精确恢复作断言。固定rho0.05/0.2，100次Adam更新、200次梯度计算；额外提供原Adam200步对照，区分更多计算与扰动收益。不改变教师，不叠加第十五轮损失。

3图×2权重×5新seed22100..22104×8方法=240。对照Frozen、GraphPrompt、直接平滑教师、原型教师gamma3、Adam100、Adam200、两种SAM。少量预检验证恢复和冻结后运行。记录扰动前后loss、实际更新数与梯度计算数，但只以配对query准确率判断效果。没有预训练或新数据复制。
