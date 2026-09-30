# 第119轮草案：Bi-Tempered支持损失

已读[NeurIPS2019原文](https://proceedings.neurips.cc/paper/2019/file/8cd7775f9129da8b5bf787a063d8426e-Paper.pdf)第1.1节、温度log/exp定义及one-hot损失，定位[作者仓库](https://github.com/google/bi-tempered-loss)。本原语依据公式独立实现，未完成作者框架的逐位比对，不称官方代码复现。Lookahead检索发现第60轮已有失败配置，未再次运行。第75轮GCE也已失败；本候选保留Bregman概率幂和修正项及重尾规范化，不能简写为GCE，但不同公式本身不是价值证据。

仅替换共享冻结GCN单768token的支持LOO监督loss，门控、位置、推理原型/k50读出不变；无样本删选，不假设支持标签错误。固定t1=.8,t2=1.2，配对t1-only(.8,1)、t2-only(1,1.2)、CE(1,1)控制。后续需在实际梯度流中匹配当前Adam/Grafted根，并核对梯度幅度变化，不能把梯度缩小等同于鲁棒性。

重尾概率通过双精度80次二分求规范化阈值，自定义隐式Jacobian；训练实际参数仍float32。t2=1采用稳定log_softmax避免极小概率的乘零无穷。one-hot损失使用温度负对数减去概率幂和修正项；t1=t2=1恢复CE。

loss119.py/check119.py在ProG环境通过独立SciPy Brent概率根、闭式loss梯度、有限差分gradcheck、平移不变性、CE恢复、极端输入有限性和t1<1有界性；四配置梯度最大差3.13e-17。该证据限于合成损失原语。真实图驱动、全步更新和首末梯度快照审计、配置/种子封存均未完成，没有准确率结果。


## 正式开发协议

以上未接入段落为草案历史。当前run_round119_bitempered.py已接入三配置，并保存首末step的支持logit、loss、token与梯度，独立audit119_loss.py使用SciPy Brent求概率和闭式score梯度，再回传冻结GCN，核对实际token梯度；另报告相同token点CE梯度范数比，不据此宣称鲁棒性。

固定CiteSeer，GraphMAE Adam .01/GraphMAE2 graftedShampoo .01，单768token，185100–185104，每组100步，总125条。匹配两根、Frozen、GraphPrompt、Direct。185199/3步冒烟29条包含t1=t2=1的CE恒等恢复。推理不采用温度化概率，依然原型/k50原读出。

主候选(.8,1.2)，控制(.8,1)、(1,1.2)；任一k50候选超过同批两个根中的最佳者至少0.20个百分点才进行20支持种子确认。多达标选准确率最高，平局both、bounded、heavy顺序；无达标停止。沿用长期复用dev/confirm查询，同图支持统计仅探索性。五类检查全部通过后解读准确率。
