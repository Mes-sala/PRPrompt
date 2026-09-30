# 第117轮草案：邻域统计混合训练

借鉴[MixStyle, ICLR2021](https://arxiv.org/html/2104.02008v1)第2节与附录A.1，以及[作者实现](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/ops/mixstyle.py)。OpenReview访问遇到浏览器检查，采用作者arXiv版本；最初猜测的根目录mixstyle.py为404，随后从官方README定位正式路径。

视觉原方法沿每个样本的空间维计算每通道统计量。本图适配改用节点闭邻域的均匀均值/总体方差，以邻域作为局部统计范围，不声称节点邻域是视觉风格或不同域。不变的共享冻结GCN与768token位于第一层激活前；训练时对提示后的预激活按节点混合均值和标准差，最终推理移除扰动。不读取查询标签。

设闭邻域均值和标准差为 $ mu_i, sigma_i $，排列为 $ pi $，固定 $ lambda_i \sim Beta(0.1,0.1) $。混合统计量后对原中心节点作去均值/缩放/恢复；统计量detach，但中心表示保留梯度。训练应用概率0.5，不引入额外可训练参数。比较both、mean-only、std-only和identity，匹配原始干净CE根；初始化/推理无扰动恢复根预测。作者代码使用样本方差默认设置，本适配明确使用总体方差，eps=1e-6。

原语style117.py的方差写法E[x²]−E[x]²导致严格梯度对照误差1.93e-11；独立前向差2.08e-13。未扩大容差，而在style117_v2.py改为显式中心化邻边平方差的加权求和，保持总体方差定义。check117_v2.py的前向/停止统计梯度对照均4.44e-16，identity、lambda=1、孤立常数有限性通过。旧原语与检查保留；round117_diagnostic_unchecked_gradient.json是移除断言后的诊断输出，虽然内部有passed字段，明确不可作为正式通过，正式证据仅round117_primitive_audit_v2.json。

待实现：真实图训练驱动、随机排列/Beta权重与应用掩码的可复现保存、首末步独立梯度重建、全步优化器重算、推理几何核验、噪声尺度报告。拟使用CiteSeer双权重、各5支持种子；源文件与正式种子尚未封存，未运行准确率，不把本草案计入已验证方案。


## 正式运行协议

以上“待实现”段为草案历史。已接入run_round117_style.py、random117.py与独立audit117_style.py；不变第一层预激活位置，单768token，共享冻结主干，Beta=.1、应用概率.5。闭邻域去重均匀统计，中心化方差求和避开抵消误差。快照包含第0/末步及首末有效扰动，独立CPU双精度统计重建后转float32，核对CELOO分数、loss及完整梯度。

固定CiteSeer双权重，183100–183104，100步共125条正式；183199/3步冒烟29条，增加identity恢复。GraphMAE Adam .01、GraphMAE2 graftedShampoo .01；干净根、另一优化器根、Frozen、GraphPrompt、Direct匹配。只比较同批，k50候选超过两个根中最强者至少0.20个百分点才进入20支持种子确认。多候选达标取准确率最高，平局both、mean、std。共享查询集长期复用，仅探索性支持抽样证据。
