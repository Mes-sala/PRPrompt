# R160 LoHa预注册开发

借鉴FedPara Hadamard参数化和固定PEFT LoHa增量实现，保留原始checkpoint与bias冻结，第二GCN有效权重加入DeltaW=(U1V1)*(U2V2)，与同一中间Prompt联合优化。单GCN共用参数、无终端预测集成。r4/r8，gamma1，三个因子kaiming、V2零，seed+160000；首步只有V2得到适配梯度。A/B所有因子Adam.0025、Prompt Adam.01，wd0；原强根teacher、scale2损失、KD、物理DeltaW正则和k50不变。

h4/h8完整LoHa；f4/f8冻结初始U1/V1，检验训练第一乘积的额外作用；r8/r16普通LoRA下因子零、上因子kaiming，参数预算匹配h4/h8；a4/a8完整LoHa无Prompt；旧l1/l4保持既有初始化/学习率。联合h4/h8可训2944/4992，匹配r8/r16相同；f4/f8为1920/2944，不能称与完整LoHa同预算。相同预算不等于梯度或初始化严格匹配。

Cora5shot、两GraphMAE checkpoint、支持230100..230104、dev前500、100步。20条3步真实GCN轨迹通过独立初始化、前向、首末梯度、全部Adam回放。先恒等及61条12步smoke，再285正式/110轨迹220快照140邻域。候选顺序h4,h8,f4,f8,r8,r16,a4,a8,l1,l4，GraphMAE优先；k50最佳相对两个ck全部旧CE/scale2最强根和Direct各+.20才具备固定确认资格。若胜者为已有两批确认的完全相同l4，不再追加第三批Cora，直接引用R155证据；其余新配置先锁定再确认。保留原型得分但不改预设读出。
