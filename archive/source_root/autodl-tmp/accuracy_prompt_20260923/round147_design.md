# R147 VeRA固定开发协议

借鉴ICLR2024 VeRA的冻结随机矩阵与可训练向量，来源/公式/固定PEFT SHA见round147_design_draft.md、vera147_source.json。随机A/B初始化已与官方函数逐位一致；两ck四初值、20条三步轨迹通过，无query标签。

仅第二GCN层有效W=W0+diag(b)Bdiag(d)A，原W和bias冻结。原896参数late Prompt共享同一forward，非终端融合；不是纯Prompt或论文跨层共享复刻。两rank16/128，seed0固定随机矩阵；b零/d=.1。v16/v128联合训练b,d；b16/b128固定d仅训练b；a16/a128/c16/c128为对应adapter-only。LoRA l1/l2同split控制保留，与R146相同初始化规则support+146000。v128与LoRA-r1联合均1152参数，表达类不等价。

两ck GraphMAE/GraphMAE2，原teacher/gate/scale2 CE+KL、100步Adam.01、原token+.001物理DeltaW均方正则不变。support217100..104，dev前500query。285条/110轨迹/220首末快照/140近邻；先seed217199的61条12步冒烟24轨迹48快照30近邻、旧根identity全过后正式。训练只读取support标签，预测先封存再评分。

保留Direct/frozen/GraphPrompt、两ck旧CE及scale2。R146固定PiSSA确认74.30%低于强根75.46%，未成为新强基线；同ckPiSSA局部增益保留，LoRA对照未被删除。十种适配×两ck的k50开发最高方案相对最强旧根和Direct均至少+.20pp才锁定确认。若LoRA或单adapter获胜也保留准确率结果，但不归为VeRA或Prompt机制；必须单列对其匹配控制。固定d对照与双缩放对比，不能用向量参数少自动解释精度提升。

开发query反复使用；成功仍需固定20新support确认，不把同图support区间当新图泛化。rank和随机seed均固定，不按query搜索。后续如正增益再独立随机映射复核。
