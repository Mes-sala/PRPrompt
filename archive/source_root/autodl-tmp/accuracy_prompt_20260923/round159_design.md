# R159 DyLoRA式随机前缀预注册

Cora5shot，两GraphMAE checkpoint，支持229100..229104，dev前500，100步。最大rank4/16，A kaiming/B0，Prompt0，A/B Adam lr.0025、Prompt.01、wd0。联合共享第二GCN有效权重与中间Prompt输入，原始checkpoint及bias冻结。

n4/n16：每步均匀采样b=1..R，取前b分量，alpha=R，训练DeltaW=(R/b)B[:,:b]A[:b]；u4/u16：相同b序列、均匀无放回随机子集，打破嵌套；c4/c16：前缀但缩放1；f4/f16：全秩固定gamma1；l1/l4：原有rank1双学习率基线。正则统一为完整推理DeltaW=BA的.001平方均值及Prompt正则，不能称当前未抽中的参数严格冻结；Adam动量也会影响它们。该正则为本地图Prompt协议扩展，不冒称论文完全复现。

每步全图共用一组b/分量。CPU生成器seed+159000采样b，seed+159001独立采样子集；保存全部序列并重建。初轮不采用frozen模式，避免把detach误称参数冻结。最终仅完整R推理（gamma1）；不做按query选秩或MC集成。均匀prefix期望不是完整BA，subset期望是BA，故既保留去缩放也保留固定秩控制；不同控制回答不同问题。

所有teacher/scale2损失/k50读出均沿用强根；不将固定R对照勝出归为动态秩。20条3步两checkpoint真实轨迹已通过独立前向、首末梯度、全步Adam与随机序列回放。先恒等与61条12步smoke，再285正式（110轨迹220快照140邻域）。开发候选顺序n4,n16,u4,u16,c4,c16,f4,f16,l1,l4，GraphMAE优先；k50最佳超过全部旧CE/scale2强根和Direct各至少.20pp才锁定进入20新支持确认，否则停止。
