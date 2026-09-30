# 第二十二轮：修正教师与提示强度/位置配对

继承[第二十一轮的视觉原型修正来源](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460715.pdf)与[第十轮VPT位置借鉴](https://arxiv.org/abs/2203.12119)。目前CiteSeer直接修正教师66.99%明显强于其gamma3提示53.61%，PubMed/GraphMAE修正教师提示74.55%比原提示72.43%有稳定增量。下一问题是更强提示能否利用已经改善的教师，不能假设强度越大越好。

固定beta0.5、三次软分配修正和同一冻结两层GCN。比较原型教师gamma3、修正教师gamma3、修正教师gamma10早层、gamma10后层、gamma10无KD、gamma10固定随机token。最后两者控制训练和蒸馏收益；随机版本使用已有N(0,0.3)参数分布，参数量相同，未按query调随机幅度。直接原/修正教师、Frozen、GraphPrompt和Cora平滑岭后层提示均同批保留。只改变内部注入，单一原型分类头。

3数据集×GraphMAE/GraphMAE2×5支持seed32100..32104×11方法=330。100步lr0.01，无KD方法名token10_ref50不含kd，随机方法token_random10_ref50不会训练。同源程序新文件，不修改历史已审计程序；每一新位置零提示与冻结检查沿用。先3步预检，正式结果之后最多每图固定一个候选，确认必须带上对应gamma3和直接修正教师。
