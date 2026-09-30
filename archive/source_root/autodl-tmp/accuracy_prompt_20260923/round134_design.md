# R134固定设计：Cora的DIST相关性蒸馏

来源、固定官方commit、原语验证和初始诊断见 `round134_design_draft.md`。本轮以准确率提升为目标，允许无KD/双KL等朴素控制被选中，不要求创新。DIST对应类别间和样本间概率Pearson相关项，公式及eps=1e-8保留官方实现；固定tau=1，学生logits已含原cosine温度0.2，教师为原平滑岭概率，不再次softmax。

原强根不变：Cora冻结2层GCN，后层7×128token注入，原平滑岭门控，动态gamma2 scale CE，Adam0.01，100步，最终不中心化，k50温度0.2/alpha0.9/20步支持钳制。只有KD损失替换。

候选：dbr=原始inter+intra；dbm=二者和的初始梯度范数匹配；dim/dcm=单inter/单intra初始匹配；dk0=无KD；dk2=双KL。匹配权重为T=0时KL梯度范数除以候选项梯度范数，每个support/checkpoint分别计算并固定；不读取查询标签或搜索准确率，所有初始梯度和标量权重封存。阈值仅拒绝非有限值或范数≤1e-12。初始诊断显示GraphMAE2原始both梯度为KL的15.73倍，因此必须报告尺度控制结果。

两checkpoint GraphMAE/GraphMAE2，seed199100–199104。Direct每seed一条；每ck有frozen、graphprompt、base/gp k50，以及token、scale2、dk0、dk2、dbr、dbm、dim、dcm各mean/k50，共41条/seed=205。冒烟seed199199、3步，添加恒等KL分支dki共45条。CPU单线程legacy-vs-new KL及identity三步token/embedding/k50精确恢复。

候选k50必须在开发5种子均值上同时超过本轮跨ck最强scale2 k50和Direct至少0.20pp才进入确认。选均值最大者，平手顺序dbm、dim、dcm、dbr、dk0、dk2及GraphMAE、GraphMAE2。若控制被选中，只归因为KD权重/去KD。确认将使用新support seeds和后500查询，多轮已使用过查询，不能当独立外部测试。保留mean读出结果，不在看到结果后改主读出。

审计覆盖：逐条封存SHA/graph/support/query/精度复算；全量teacher/gate/late forward/原型及k50几何；每条focal训练保存每步系数/梯度并NumPy Adam回放；CPU独立相关项与初始匹配梯度、首末实际总梯度。正式70条轨迹/140快照、冒烟16条/32快照，完整205/45键集合。审计必须非零且匹配期望计数才允许正式。记录失败与修复版本，不覆写已启动源码。
