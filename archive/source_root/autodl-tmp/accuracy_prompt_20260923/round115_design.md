# 第115轮：固定1.5-entmax教师门控

来源：[ACL2019 Sparse Sequence-to-Sequence Models](https://aclanthology.org/P19-1146.pdf)，已读3.2–3.4节、式13及算法1/2；作者[activations.py](https://raw.githubusercontent.com/deep-spin/entmax/master/entmax/activations.py)已保存原始文件与SHA。只借用概率映射，不移植分类损失或训练alpha。

输入仍为2倍既有教师score，固定alpha=1.5，概率为阈值截断后的平方；门控全节点中心化后作原图传播。一个6×128共享token注入冻结两层GCN第一层激活前，没有额外可训练分支。GraphMAE用Adam .01，GraphMAE2用grafted Shampoo .01。对照为原始映射、等门控范数、仅幅度、softmax恒等恢复；保留同批两根、Frozen、GraphPrompt和Direct。

原语与独立二分最大误差6.66e-16，与作者原始实现2.22e-16。当前仅验证固定前向，门控无梯度；不声称完整作者训练复现。

CiteSeer5shot，181100–181104开发支持种子，每个100步，125条；181199三步冒烟29条。数据划分沿用既有dev/confirm，重复查询造成探索性选择偏差，后续支持seed bootstrap不是独立图确认。依次封存、传输、CPU独立二分概率与几何、NumPy全步优化器审计。保存实际sparse_probability/SHA和零比例；gate_probability保留softmax参考。

预先固定：k50候选超过同批两Prompt根中更高者至少0.20个百分点，才启动20支持种子确认；多个达标选准确率最高，平局raw、matched、amplitude顺序。无达标停止。保留不传播读出的局部变化，不以它替代最强基线。
