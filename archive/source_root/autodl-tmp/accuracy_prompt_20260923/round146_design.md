# R146 固定PiSSA开发协议

来源与原语见round146_design_draft.md和pissa146_source.json。只借鉴PiSSA的主奇异方向初始化，不复刻LLM训练或主张原创。

Cora两冻结checkpoint，第二层128×128有效权重W+(BA−B0A0)，保留原bias。pp/rp/l1/l2为Prompt联合principal-r1、同范数random-r1、zeroB-r1、zeroB-r2；pa/ra/a1/a2是相应adapter-only。原W始终不改，但forward有效W改变，明确为Prompt+PEFT。联合参数1152/1152/1152/1408，仅adapter256/256/256/512。

保持原teacher/gate/scale2监督+全节点KD、100步Adam.01、token及物理DeltaW各.001均方正则；所有因子初始化在CPU完成、种子support+146000，原始保存clone。新support216100..104，dev前500query，245条/90轨迹/180首末快照/120近邻。原oldCE/scale2两ck与Direct/frozen/GraphPrompt保留。训练只读取support标签，先封存预测再读取query标签；独立CPU手写传播/bias/目标梯度及全部Adam轨迹验证。

先seed216199两ck53条12步冒烟，含identity对照；20轨迹40快照26近邻全过后正式。八种适配方案×两ck的k50里选开发最高；相对两ck旧CE/scale2最强者及Direct均至少+.20pp才锁定20新support确认。只adapter或随机对照获胜也按准确率目标保留，但不归于Prompt或主奇异方向。若需确认，预先固定所选方法并比较全部强根，不以本轮对照替换确认后候选。R145已确认失败，原scale2仍是强根；LoRA和随机等在本轮同split实测。

比较principal与random/r1/r2才讨论初始化作用。两个非零因子差分可rank2，不以rank1名义宣称同LoRA表达能力；不同首步梯度或因子尺度不是准确率证据。开发query跨轮反复使用，所有提升仍须确认且只反映同图support抽样。
