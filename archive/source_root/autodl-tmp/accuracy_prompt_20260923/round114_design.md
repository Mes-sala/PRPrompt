# R114草案：sparsemax教师门控

已读[Martins与Astudillo，ICML2016](https://proceedings.mlr.press/v48/martins16.pdf)第2.1–2.3节、算法1。借鉴语言注意力的概率单纯形欧氏投影，拟只把门控概率softmax替换为sparsemax，不引入论文另提出的分类损失。正则、支持CE、冻结GCN和原型读出不改。

原教师score已含/.2，当前温度.1门控为softmax(2score)，拟比较sparsemax(2score)，两者均全节点中心化后原图传播。仍是同一共享768token在第一层激活前注入，不是多Prompt或终端融合。raw/matched/amplitude/identity明确区分概率形状与总门控范数；稀疏不保证更准确，不声称直接教师预测改善。底层教师与gate_probability审计字段保留原softmax，另记录实际sparse_probability及其SHA，避免字段误归因。

gate114.py使用排序解投影，check114.py以逐行二分阈值作独立验证，检查概率和、非负、argmax、平移/类别置换不变、均匀/onehot极限、范数与identity。该门控固定无梯度，不需移植sparsemax分类损失的Jacobian。后续真实图须独立重算教师、投影与稀疏概率支持，报告零概率比例及门控变化。尚未接入驱动、封存种子或启动准确率实验。第79轮锐化与第91轮CLR是先前参照，当前算子不同但不据此主张原创。


## 正式开发协议

固定 CiteSeer 5-shot，GraphMAE / GraphMAE2，种子180100–180104，100步。GraphMAE沿用Adam .01，GraphMAE2沿用grafted Shampoo .01；单个6×128 token位于冻结共享GCN第一层激活之前。比较Sparsemax原始、等门控范数、仅幅度及softmax恒等恢复；保留同批Adam、grafted Shampoo、Frozen、GraphPrompt、Direct。两种读出共125条正式结果。冒烟180199，3步，增加恒等恢复，共29条。

先通过封存预测、传输/教师、独立CPU二分投影与几何重建、NumPy优化器全步更新审计，再分析准确率。使用相同历史dev/confirm查询划分，跨轮重复使用查询集意味着探索性选择偏差，不能当独立图确认。以k50候选超过同批两个Prompt根中更强者至少0.20个百分点作为20支持种子确认的开发筛选条件；保留全部控制和Direct差值。若有多个候选，选开发平均准确率最高者，平局优先raw、matched、amplitude。无达标则停止本分支。
