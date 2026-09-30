# 第50轮草案：已验证读出在九种预训练权重上迁移

第18轮仅比较提示位置/教师，未包括后来验证的10/50邻居概率传播与表示AQE。服务器当前Cora有九种兼容两层GCN128权重：DGI、GraphCL、GraphMAE、GraphMAE2、Bandana、S2GAE、SimGRACE、Edgepred_GPPT、Edgepred_Gprompt。MultiGprompt结构不同不强行加载。无需复制已有checkpoint/数据/环境。

固定旧平滑岭教师late token100步，所有九权重单独冻结、不做集成。拟Cora×9权重×5新支持seed76100..76104×7方法=315条dev500：Frozen、GP、平滑岭教师、原强提示、10NN概率传播、50NNsoftmax概率传播、均匀50邻居表示聚合。后两项在第18轮未测，已有失败并不当作永久否定；本轮仍以同批GraphMAE/GraphMAE2强方法为参照，弱Frozen大增益不能替代绝对强基线。

先各权重3步预检strictload/no-op/frozen/来源SHA，方法代码复用run_round47_matched.py。最多选一个有希望超过强参照的新权重，用20新支持确认；跨权重也必须同支持/query配对。第18轮Edgepred_Gprompt开发曾接近GraphMAE，但不同批次数值不能直接排名，须新批实际验证。

本轮是把已有跨领域启发的成功方案迁移到更多权重，不宣称新论文/新机制。PubMed已有传播负面证据，本轮不扩大它。所有结果仍同图传导；图存稀疏索引/权重，预检及时清理。当前草案未运行，配置在正式启动前冻结。
