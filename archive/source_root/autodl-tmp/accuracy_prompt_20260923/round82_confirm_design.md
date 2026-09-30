# 第82轮AdaGrad参照的固定确认

Shampoo在所测学习率.01/.03下明显低于原Adam，暂不扩大该配置；不能据此声称任何步长的Shampoo都无效。GraphMAE原Adam.01+k50 70.64%，AdaGrad.03 70.36%；GraphMAE2原70.00%，AdaGrad.03 70.96%，Adam.03 69.72%，对角.03 69.28%，Direct70.24%。依据准确率目标，参照方法也可成为候选。

固定GraphMAE2 AdaGrad.03+k50为主候选，GraphMAE次要迁移参照，20新支持126100–126119，固定确认500，共500条。保留原Adam.01、Adam.03、AdaGrad.03、对角.03、Shampoo.03及两读出，Frozen/GP/Direct。源码和目标完全不改；AdaGrad累积初值1e-4、分母sqrt(V)，非PyTorch默认参数。主比较AdaGrad.03对原强根，同时报告学习率匹配Adam及Direct。待开发四审计通过再启动。
