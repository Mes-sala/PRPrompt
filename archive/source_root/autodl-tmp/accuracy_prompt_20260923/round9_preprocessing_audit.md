# 第九轮：Amazon输入预处理契约诊断

第八轮使用NormalizeFeatures；实际查阅 `/root/ProG/prompt_graph/data/load4data.py:261-270`，Planetoid明确NormalizeFeatures，而Amazon加载无transform。DGI/GraphCL/GraphMAE当前预训练路径调用load4node。因此Amazon历史权重可能对应原始输入，而第八轮Frozen过弱可能受输入预处理影响。缺少sidecar，不能将当前源码直接当作历史权重生成证据。

第八轮数值仍是给定归一化输入下的真实配对结果，但暂停将其巨大增幅解释为匹配预训练契约的Prompt收益。Cora/CiteSeer/PubMed明确使用归一化，与当前ProG loader一致，不受此项Amazon差异影响。

诊断固定第八轮支持seed13100..13104、2000查询、权重、教师（仍由行归一化特征传播生成）、提示算法与幅度，只把主干输入改为Amazon原始特征。这样区分输入尺度变化与更换教师造成的差异。Graph hash随输入变化，split/checkpoint hash必须保持一致。主干Frozen、GraphPrompt、直接ridge教师、岭条件提示gamma10、随机岭token共5方法，2图×3权重×5seed=150任务。

该输入状态与当前ProG Amazon源码一致，但仍标记权重来源legacy_unverified；不是事后把不利结果删除。两轮完整保存，按配对差异解释。若原始输入显著恢复Frozen，应先修正后续实验默认输入，再决定哪些提示值得扩大。
