# 第八轮：锁定提示迁移到Amazon图

Photo/Computers数据与DGI、GraphMAE、GraphCL权重已在目标服务器，不下载或复制大文件。将第七轮固定的ridge教师gamma10条件提示迁移到这些不同领域的商品共购图；不读取这些图的查询结果来调整参数。

Amazon数据没有官方训练/测试掩码。固定torch seed2026092308无标签随机排列节点，前2000为查询，剩余为支持候选；再用seed13100..13104每类抽5个支持。查询不参与标签训练/参数选择。全图特征和边仍为传导设置。所有方法共享同一支持/查询。支持抽样使用类别标签是few-shot模拟的采样过程，不作为额外训练标签。

2图×3权重×5支持seed×9方法=270最终指标。9方法：Frozen、GraphPrompt、rawprop2、ridge教师、ridge gamma10提示、旧教师gamma10、无KD、打乱门控、随机token。100步、lr0.01和lambda候选都冻结。该批仅在本任务中未评分，历史工作是否使用过Amazon图不作全新数据的保证。

Amazon权重没有sidecar时标为legacy_unverified，仅严格匹配GCN结构、绑定实际SHA256，不补造来源。数据来源差异、支持池大小差异均单列，准确率不能直接与Planetoid横向相减。
