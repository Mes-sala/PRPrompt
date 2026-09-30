# R149 梯度SVD初始化固定开发

来源及尺度/符号差异见round149_design_draft.md、loraga149_source.json。只借鉴梯度方向初始化，不声称新Prompt或完整作者复现。gamma16、外层scaling1、rank1，计算原训练目标（support scale2 CE+全节点固定teacher KL）在零Prompt/原GCN时对第二层W的梯度；额外一次CPU反向不更新底座。保存该梯度和SHA，独立审计重算梯度后对被核验输入做完整SVD，避免近简并方向的跨设备任意基漂移。

gd: A第一右奇异向量/B第二左奇异向量；op: A/B都取第一方向；rp:同因子范数随机正交方向。三者分别联合Prompt和仅adapter(ga/oa/ra)。保留LoRA-r1/l2、R148 d4与整体降低学习率l4，两ck完整同split控制。联合1152参数（LoRA-r2为1408），adapter-only256；rank1差分可rank2。有效W=W0+(BA−B0A0)，保留原bias，原checkpoint不改。

原teacher/gate/100步Adam.01、token及物理DeltaW正则.001；d4/l4沿用各自分组学习率，Prompt始终.01。梯度初始化六模式均计算一次额外反向，其他控制无该开销；不宣称速度或等计算预算收益。独立CPU目标/梯度/全步NumPy Adam重放，通过首末快照检查目标一致；GPU输出封存后读取query标签。

support219100..104、dev前500，285条110轨迹220快照140近邻；先seed219199的61条12步冒烟24轨迹48快照30近邻和旧根identity。两ck十适配×k50中最高，相对最强旧CE/scale2和Direct均至少+.20pp则锁定新20support确认。机制对同范数随机/重叠/LoRA/lr控制单列。若lr控制或adapter-only最好也按准确率保留，不归为梯度初始化机制。

R148固定d4确认76.76对scale2 76.51为+.25[-.62,1.02]，未过；l4确认77.35为事后对照线索，在本轮新dev同split纳入，不能改报为R148确认成功。不得在本轮中途改种子、初始化或步数。开发/确认query跨轮重复使用，仍需谨慎限定同图support证据。
