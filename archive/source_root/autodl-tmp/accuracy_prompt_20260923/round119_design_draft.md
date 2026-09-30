# 第119轮草案：Bi-Tempered支持损失

已读[NeurIPS2019原文](https://proceedings.neurips.cc/paper/2019/file/8cd7775f9129da8b5bf787a063d8426e-Paper.pdf)第1.1节、温度log/exp定义及one-hot损失，定位[作者仓库](https://github.com/google/bi-tempered-loss)。本原语依据公式独立实现，未完成作者框架的逐位比对，不称官方代码复现。Lookahead检索发现第60轮已有失败配置，未再次运行。第75轮GCE也已失败；本候选保留Bregman概率幂和修正项及重尾规范化，不能简写为GCE，但不同公式本身不是价值证据。

仅替换共享冻结GCN单768token的支持LOO监督loss，门控、位置、推理原型/k50读出不变；无样本删选，不假设支持标签错误。固定t1=.8,t2=1.2，配对t1-only(.8,1)、t2-only(1,1.2)、CE(1,1)控制。后续需在实际梯度流中匹配当前Adam/Grafted根，并核对梯度幅度变化，不能把梯度缩小等同于鲁棒性。

重尾概率通过双精度80次二分求规范化阈值，自定义隐式Jacobian；训练实际参数仍float32。t2=1采用稳定log_softmax避免极小概率的乘零无穷。one-hot损失使用温度负对数减去概率幂和修正项；t1=t2=1恢复CE。

loss119.py/check119.py在ProG环境通过独立SciPy Brent概率根、闭式loss梯度、有限差分gradcheck、平移不变性、CE恢复、极端输入有限性和t1<1有界性；四配置梯度最大差3.13e-17。该证据限于合成损失原语。真实图驱动、全步更新和首末梯度快照审计、配置/种子封存均未完成，没有准确率结果。
