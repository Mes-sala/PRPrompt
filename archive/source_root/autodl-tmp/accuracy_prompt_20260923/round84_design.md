# 第84轮草案：信息检索IDF加权教师输入

借鉴信息检索中抑制普遍词项的思想，来源：[Manning等，Inverse document frequency](https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html)，平滑实现参考[scikit-learn TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)。本轮是词袋特征加权的迁移，不把预训练权重重新训练或重命名为新模型。

CiteSeer输入是稀疏词特征，拟仅改变教师使用的原始属性几何。用所有节点无标签特征统计 $ df_j=\sum_i1[X_{ij}>0] $，固定 $ w_j=\log((N+1)/(df_j+1))+1 $；这是一种平滑变体，不等同书中无平滑log(N/df)。加权后重新做逐行L1归一化，再Ahat²传播、列中心化与既有修正原型教师，温度0.1门控。GCN输入继续使用原NormalizeFeaturesX，冻结权重不变，768token、Adam.01/100步、最终中心化+k50。只有教师特征变，不能把输入预处理和主干改善混淆。

对照原教师、全1权重精确恢复、固定随机置换列权重（保持权重分布但破坏词频对应）；另报告加权教师Direct与原Direct，防止把更强属性分类器收益误归Prompt。若门控幅度变化大，纳入相对原门控Frobenius范数匹配与纯幅度控制。查询标签绝不参与df、权重、训练或选择；无标签全图统计仍属传导协议。

先验证非负词特征、零行处理、df和权重SHA、全1恢复及CPU独立教师/门控重建，再固定新支持与正式矩阵。此文件仅设计，未实现未运行；不因为前几轮优化器失败就假设该方案一定有效。


## 第83轮复核后的对照更新

第83轮GraphMAE2步长匹配Shampoo在两组20支持均胜Adam，待最终审计后保留为权重限定配方。第84轮必须同时保留该更强参照；GraphMAE继续Adam。IDF可先以两权重各自保留优化器为根，只改变teacher特征，另外保留原Adam参照以分离优化器作用。尚未封存训练矩阵；当前仅features84.py与check84_features.py原语实现通过独立NumPy（误差5.55e-17）、零行、固定置换和全1/行缩放恢复检查。无query实验，不计正式结果。


第83轮最终审计通过后，GraphMAE2主优化器确定为grafted Shampoo.01；GraphMAE继续Adam.01。建议从run_round83_graft.py派生新驱动，features84.py只处理teacher旁路。教师IDF原始门控、相对原softmax.1门控的范数匹配、纯幅度放大和列权重置换各固定一支；Direct需原属性与IDF两个独立参照，不使用预训练权重。新增gate_feature_mode/feature_weights_sha256/teacher_feature_sha256/actual_gate_sha，审计需独立重建这些量。沿用原训练loss与最终读出，不能让新教师特征改变GCN输入x或原backbone哈希。正式配置未封存前可据必要审计范围缩减重复读出，但不能移除新强根、置换或Direct。


## 固定实现与矩阵

130100–130104，CiteSeer两权重共150正式；smoke130199三步共34条，额外全1权重恢复。保留Adam根、grafted Shampoo根，IDF raw/matched/amplitude/permuted四候选各原型与k50，Frozen/GP以及原Direct与IDF Direct各一份。GraphMAE候选用Adam，GraphMAE2用grafted Shampoo；无KD。门控范数基准为原softmax.1门控，不错用.2门控。

训练源与依赖SHA封存。独立reconstruct84.py不调用features84.py，重建df、weights、置换、teacher输入、teacher概率和门控，audit84_geometry重建冻结GCN/最终读出；audit84_updates保留NumPy逐步更新重放，并要求全1分支恢复各权重正确强根预测。先全部预检审计通过后正式100步，不改既有源码。teacher_feature_sha是运行时来源指纹，几何一致性由独立重建与门控/预测检查保证，不宣称CPU与GPU传播逐字节相同。
