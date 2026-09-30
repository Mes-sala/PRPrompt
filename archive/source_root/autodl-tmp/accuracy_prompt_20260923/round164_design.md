# R164 EVA激活初始化草案

来源：[Parameter Efficient Fine-tuning via Explained Variance Adaptation，NeurIPS2025](https://papers.neurips.cc/paper_files/paper/2025/hash/41d33bd41fd44bd9dba0e092047cf213-Abstract-Conference.html)，arxiv2410.07170最新全文eva164_paper.md；固定PEFT提交116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa的lora/eva.py和utils/incremental_pca.py保存eva164_official.py/eva164_pca_official.py。

官方hook调用IncrementalPCA；第一次partial_fit明确减列均值。取components_右奇异向量初始化A，B零。whiten是可选项，源码按singular_values的平方根缩放；本轮不用，以免引入尺度混杂。LoRA A/B均训练，不等同R163固定双侧基只训练中心。

拟第二GCN联合原Prompt，rank1/4，因子Adam.0025、Prompt.01、gamma1、物理DeltaW正则.001，其余teacher/scale2/KD/k50不变。初始化直接CPU double完整SVD再float32，不复现增量收敛过程。仅一个适配层，不声称实现跨层rank再分配。

八个新配置：c1/c4来自全部节点中心化h的主成分；u1/u4来自未中心化h；s1/s4仅来自support节点中心化h；r1/r4随机正交A。另旧l1/l4。全部B零，A范数按正交行匹配；随机QR固定seed+164000。全部节点使用的是无标签激活，属于当前transductive协议，明确包含query特征但无query标签。support-only用于检查无标签全图估计的贡献。中心化只作用初始化统计，不改变正式forward的h。

GCN.lin的输入h与聚合后的Ahat h不同，主方案按真实线性层输入h统计；不混淆成聚合后协方差最优。rank1/4官方完整SVD路径与独立SVD投影相同、尾能量、初始no-op、首步仅B梯度和有限差分通过round164_primitive.json。中心化与未中心化投影差异已验证，不是可随意省略的步骤。

下一步实际GCN轨迹接入、统计矩阵重建、SVD子空间校验和独立Adam回放，之后封存正式配置。当前没有图训练准确率；不由高方差或首步投影推断分类提升。


## 正式预注册

20条真实GCN三步轨迹已通过：统计矩阵重建、NumPy独立SVD并规范奇异向量符号、初始前向、首末梯度和全Adam回放。实际随机QR种子seed+164000，初始化中心化仅用于统计、不进入forward。adapter参数rank1为256、rank4为1024，外加896Prompt；B零。

候选顺序c1,c4,u1,u4,s1,s4,r1,r4,l1,l4，每个GraphMAE优先。Cora5shot234100..104，dev前500，100步，k50固定。正式285/110轨迹220快照140邻域完整审计后，最高候选对两ck旧CE/scale2最强根及Direct均至少+.20才确认。旧l4引用R155、不追加第三批；新配置锁定20新支持确认。先61条12步smoke。
