# R165 CorDA来源核查与草案

来源：[CorDA，NeurIPS2024](https://papers.nips.cc/paper_files/paper/2024/hash/83f95bb0ac5046338ea2afe3390e9f4b-Abstract-Conference.html)，全文corda165_paper.md；固定PEFT提交116a979cad3a6fe8037ec27a8f494ae5cbd4f5fa，src/peft/tuners/lora/corda.py保存corda165_official.py。

官方收集未中心化二阶矩：每次hook先input/abs(max(input))，再X.T@X，按hook调用次数而非token总数平均。第一层激活h非负，max规则与maxabs一致，但不能一般化到全负输入。分解前从.01倍平均对角开始加阻尼，直到C@inv(C)与I的Frobenius差<.05。然后SVD(WC)，右因子乘C逆。顶部或尾部r分量形成B/A，冻结残差W-B0A0，保证起点保持原函数。

原语四例rank1/4×top/tail通过：固定官方eigens函数、全秩重建、减回B0A0的no-op、有限差分、C=I退化为普通SVD。另证C乘4时BA不变但B范数乘2/A范数除2，参数坐标尺度可改变Adam，因此不能将任意统计归一化当作无影响。

拟比较t1/t4：C按mean(diag)归一化后CorDA top；o1/o4：官方单批h/max(h)所得C的尺度，top；b1/b4：归一化C的tail；p1/p4：普通权重SVD、同学习率；l1/l4旧LoRA。共10，rank1/4适配256/1024参数+Prompt896。共享第二GCN有效权重W+(BA-B0A0)，中间Prompt共同训练；新因子Adam.0025，Prompt.01，gamma1和物理DeltaW正则.001，teacher/scale2/KD/k50不变。尾部分量仅称tail，不用LLM知识保持名称暗示图迁移性质。

全部使用无标签h统计，保持transductive协议；不使用query标签。拟CPUdouble分解/反演后float32因子，阻尼/逆残差保存；不宣称PEFT float32逐位复现。全零统计应明确报错而非无限阻尼循环，预检真实h。下一步独立重建统计、分解物理项、初始化、梯度和Adam轨迹，实际GCN尚未运行。


## 正式预注册

20条真实GCN三步轨迹通过，包括NumPy独立统计/反演/SVD、奇异向量符号规范、有效初始权重、首末梯度和全部Adam回放。CPUdouble统计和SVD，float32因子；阻尼从.01起按逆残差<.05至多20次，不收敛则报错。训练checkpoint本体不改，使用W+(BA-B0A0)。

固定Cora5shot支持235100..104，dev前500，100步，k50。候选顺序t1,t4,o1,o4,b1,b4,p1,p4,l1,l4，每个GraphMAE优先；按整数正确数总和比较，平局遵循该顺序。先61条12步smoke再285正式，完整110轨迹220快照140邻域审计后分析。最高者对两ck旧CE/scale2最强根和Direct均至少+.20才确认；同一旧l4复用R155，不追加第三批；新配置锁定20新支持确认，不改选读出。
