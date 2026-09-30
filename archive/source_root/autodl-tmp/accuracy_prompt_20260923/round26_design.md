# 第二十六轮：支持监督与教师蒸馏的梯度冲突

借鉴[Gradient Surgery for Multi-Task Learning，NeurIPS2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)的冲突梯度投影。此处是同一提示上的两个训练目标，不宣称完整复现论文多任务协议。

共享参数为冻结GCN中的类别条件token；仍只在一个位置注入，原型头不变。两种基础为修正教师gamma3早层和平滑岭教师gamma10后层。支持LOO CE梯度记为 $ g_s $，全部节点输入上的教师KL梯度（不读取查询标签）记为 $ g_d $。若内积为负，protect仅移除教师梯度对支持梯度的反向分量：$ g_d'=g_d-\min(g_s^Tg_d,0)g_s/(\|g_s\|^2+\epsilon) $，最终梯度 $ g_s+g_d'+g_{reg} $。sym用原始两梯度分别对称投影后相加；clip仅将教师梯度范数上限设为支持梯度范数。正则梯度单独相加。

比较原sum、protect、sym、clip、双倍KL、无KD；记录每步冲突率、余弦、实际梯度范数。投影的方向性质只针对原始梯度，不等于Adam更新保证，更不保证查询准确率。先验证冲突/无冲突/零梯度的实际函数性质及原sum梯度等价，再3步预检；禁止查询标签参与梯度。

固定100步Adam .01，两种基础×6+Frozen/GraphPrompt/两直接教师=16方法，3图×2权重×5支持seed37100..37104，共480正式指标。第25轮双KL确认结果另行记录，不能拿不同支持结果直接相减。只有超过同轮强对照的候选才新支持确认。
