# 第二十三轮：修正教师的商品图迁移

沿用[第二十一轮原型修正借鉴](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460715.pdf)，不重调beta0.5、三次软分配或gamma3。目的为检查论文图上的有效改法能否迁移到不同数据领域，而非继续挑选同图查询最优参数。

第九轮已确认Amazon归一化主干输入会压低Frozen。本轮完全复用第九轮入口：主干输入raw，教师输入独立NormalizeFeatures；Photo/Computers固定2000无标签抽取query节点（seed2026092308），其余作支持池。query已在第八/九轮使用，不能叫未见测试；使用新支持33100..33104，5-shot每类。权重GraphMAE、DGI、GraphCL均现场存在，历史sidecar缺失继续记legacy_unverified。

11方法：Frozen、GraphPrompt、rawprop2原型教师、岭教师、修正教师beta0.5、同beta打乱教师、旧原型gamma3提示、修正教师gamma3提示、固定随机gamma3修正token、原岭教师gamma10早层、平滑岭教师gamma10后层。2图×3权重×5支持=330最终指标。100步Adam0.01，冻结/零提示检查不变。教师特征不随主干输入改变，图哈希基于raw输入和原边，记录两套预处理标签。

提示参数量保持C×128，随机token直接N(0,0.3)且不训练。直接教师单列，不能把它自身的增益称为内部Prompt增益。六个数据/权重组合各1支持3步预检后正式运行，预检不用于挑配置。若新提示仍低于直接教师，保留对Frozen的局部效果但不认作更强分类器。
