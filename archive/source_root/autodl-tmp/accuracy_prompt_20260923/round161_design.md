# R161 LoKr预注册开发

固定Cora5shot、两GraphMAE checkpoint、支持231100..231104、dev前500、100步，原teacher/scale2损失/KD/物理DeltaW正则/k50保持。第二GCN有效权重W+DeltaW，DeltaW=C⊗(UV)，共同训练同一个中间Prompt；没有终端预测融合。factor4/8，C4×4/8×8、D32×32/16×16，内部rank4、gamma1。因子Adam.0025、Prompt Adam.01、wd0。

k4/k8为PEFT默认C零、U/V kaiming；y4/y8为LyCORIS式V零、C/U kaiming；p4/p8与k同初始化，加固定独立行列置换，仅作用DeltaW；a4/a8与k相同、无Prompt；l1/l4原rank1 LoRA对照。CPU初始化seed+161000，置换seed+161001，所有状态保存并独立重建。两种零初始化首步不同，明确不是同一方法的逐位复现；dtype为float32，与固定PEFT线性公式对齐。

适配272/192参数，加Prompt896为1168/1088；旧rank1 LoRA1152，只是接近预算而非精确匹配。无Prompt272/192。p控制保持增量rank/Frobenius参数规模，改变特征分块与checkpoint轴对齐，不把它当额外backbone。单Kronecker重排rank1结构约束仍存在，不据普通rank较高预言精度。

20条3步真实GCN已通过初始化/置换/前向/首末梯度/全部Adam独立回放。先恒等和61条12步smoke，再285正式、110轨迹220快照140邻域。开发候选順序k4,k8,y4,y8,p4,p8,a4,a8,l1,l4，GraphMAE优先。k50最高者对两ck全部旧CE/scale2最强根及Direct各至少+.20才具备固定确认资格。若winner为R155同一l4，引用既有两批确认，不追加第三批同图确认；新配置锁定后20新支持确认。原型结果全部保留但不改预选读出。
